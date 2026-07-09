#!/usr/bin/env python3
"""
preflight_check.py — nightly execution-path health check for the EWMAC
dynamic-optimisation live system.

Dry-run mode never exercises contract qualification, market-hours parsing, or
market data, so breakage in any of them stays invisible until an order is
attempted (2026-06-09: seven instruments — DAX, EUROSTX, CAC, AEX, SMI, SILVER,
VIX — could not qualify at all due to missing trading classes / currencies).
This script runs the same calls the executor makes, for every UNIVERSE
instrument, and alerts on Discord when something breaks.

Per instrument:
  1. CONFIG    ib_config_futures.csv row exists
  2. ROLLCAL   roll calendar resolves a current contract month
  3. QUALIFY   front month qualifies unambiguously at IBKR
               (and the next month too when inside the passive-roll window,
               since rebalance orders may route there)
  4. HOURS     contract details have a non-empty tradingHours string and a
               timezone we can map (is_contract_okay_to_trade would otherwise
               defer the instrument forever)
  5. MKTDATA   a valid bid/ask arrives — only evaluated while the instrument's
               exchange is open, so run this while the relevant session trades
               to validate market-data subscriptions for it

SCHEDULING — run_dynamic.sh runs this after the PST refresh and before compute
(~6:20-6:45 PM ET): gateway verified up, CME reopened after the 17:00-18:00 ET
maintenance halt, roll calendars fresh, and alerts land before the daemon picks
up the new snapshot. Failures do NOT block compute (a broken instrument fails
safe at order time); the Discord alert is the action item. Note that at that
hour only US sessions are open, so MKTDATA covers US instruments; to validate
European / Asian market-data subscriptions (e.g. before go-live), run it once
manually while those sessions trade:
  ~9:30 PM ET  (Asia open)     python3 ibkr_fut/preflight_check.py
  ~3:30 AM ET  (Eurex open)    python3 ibkr_fut/preflight_check.py

Exit code: 1 if any failure, 0 otherwise (the run_dynamic.sh caller treats
failure as a warning, not an abort).
"""

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ibkr_fut.instrument_universe import UNIVERSE
from ibkr_fut.live_dynamic import (PASSIVE_ROLL_DAYS, _connect, get_roll_info,
                                   ib_spec, load_ib_config, qualify)
from ibkr_fut.algo_execution import _IB_TZ_MAP, _valid, is_contract_okay_to_trade
from ibkr_fut.risk_check import _send_discord

CLIENT_ID_PREFLIGHT = 7      # daemon=5, compute=6
MKT_DATA_WAIT_SECS  = 12     # per chunk, matches pre_trade_checks' 10s + margin
MKT_DATA_CHUNK      = 30     # concurrent market-data lines per batch

# Known/accepted MKTDATA failures, muted from the Discord alert (still printed).
# The executor fails safe regardless (no order without a live bid/ask), so for
# instruments we've decided not to fix, the nightly alert is pure noise.
# Structural checks (CONFIG/ROLLCAL/QUALIFY/HOURS/SPEC) still run and alert.
#   IBEX_mini — no MEFF market-data subscription, by choice (2026-07-08).
MKTDATA_MUTED = {"IBEX_mini"}


def check_contracts(ib, ibcfg, universe: dict) -> tuple[list, list]:
    """
    Structural checks (1-4) for every instrument. Returns (failures, open_now)
    where failures is [(instr, stage, detail)] and open_now is
    [(instr, qualified_front_contract)] for instruments whose market is open.
    """
    failures, open_now = [], []
    for instr in sorted(universe):
        spec = ib_spec(ibcfg, instr)
        if spec is None:
            failures.append((instr, "CONFIG", "no ib_config row"))
            continue

        current, nxt, days = get_roll_info(instr)
        if current is None:
            failures.append((instr, "ROLLCAL", "no roll calendar"))
            continue

        months = [current]
        if nxt and days <= PASSIVE_ROLL_DAYS:
            months.append(nxt)            # rebalance orders may route here

        front = None
        for m in months:
            c = qualify(ib, spec, m)
            if c is None:
                failures.append((instr, "QUALIFY",
                                 f"{spec['symbol']} {m} on {spec['exchange']}"))
            elif m == current:
                front = c
        if front is None:
            continue

        try:
            cds = ib.reqContractDetails(front)
        except Exception as e:
            failures.append((instr, "HOURS", f"contract details error: {e}"))
            continue
        if not cds:
            failures.append((instr, "HOURS", "no contract details"))
            continue
        cd = cds[0]

        # SPEC (BUG-5): the CSV multiplier / priceMagnifier drive sizing
        # (compute_targets) and the pre-trade divergence gate, but IB knows both
        # authoritatively. Validate (don't replace — the backtest uses the same CSV
        # value, they must stay in parity): a silent CSV drift would mis-size every
        # position in that instrument with no other alert. Zero extra IB round-trips
        # — reuses the reqContractDetails already fetched above.
        ib_mult = float(front.multiplier) if front.multiplier else None
        cfg_mult = spec.get("multiplier")
        if ib_mult is not None and cfg_mult is not None and \
                abs(ib_mult - cfg_mult) > 1e-6 * max(1.0, abs(ib_mult)):
            failures.append((instr, "SPEC",
                             f"multiplier CSV={cfg_mult} != IB={ib_mult}"))
        # IB reports priceMagnifier=0 when unset; effective value is 1.
        ib_pm = float(cd.priceMagnifier or 1)
        cfg_pm = float(spec.get("price_magnifier") or 1)
        if abs(ib_pm - cfg_pm) > 1e-6 * max(1.0, abs(ib_pm)):
            failures.append((instr, "SPEC",
                             f"priceMagnifier CSV={cfg_pm} != IB={ib_pm}"))

        tz = cd.timeZoneId or ""
        if tz not in _IB_TZ_MAP:
            failures.append((instr, "HOURS",
                             f"unmapped timezone '{tz}' (would default to UTC)"))
        if not (cd.tradingHours or "").strip():
            failures.append((instr, "HOURS",
                             "empty tradingHours (would defer forever)"))
            continue

        if is_contract_okay_to_trade(ib, front):
            open_now.append((instr, front))

    return failures, open_now


def check_market_data(ib, open_now: list) -> list:
    """
    Check (5): valid bid/ask for every open-market contract, in chunks so a
    large universe doesn't exhaust market-data lines. Returns failures as
    [(instr, stage, detail)].
    """
    failures = []
    for i in range(0, len(open_now), MKT_DATA_CHUNK):
        chunk = open_now[i:i + MKT_DATA_CHUNK]
        tickers = [(instr, c, ib.reqMktData(c, "", False, False))
                   for instr, c in chunk]
        ib.sleep(MKT_DATA_WAIT_SECS)
        for instr, c, t in tickers:
            if not (_valid(t.bid) and _valid(t.ask) and t.bid < t.ask):
                if instr in MKTDATA_MUTED:
                    print(f"    [MKTDATA] {instr}: no valid bid/ask "
                          f"(bid={t.bid} ask={t.ask}) — muted, known/accepted")
                else:
                    failures.append((instr, "MKTDATA",
                                     f"no valid bid/ask while market open "
                                     f"(bid={t.bid} ask={t.ask}) — check subscription"))
            ib.cancelMktData(c)
    return failures


def main():
    ap = argparse.ArgumentParser(description="Execution-path preflight check")
    ap.add_argument("--no-discord", action="store_true",
                    help="Print failures only; skip the Discord alert")
    args = ap.parse_args()

    print("=" * 80)
    print(f"  EXECUTION PREFLIGHT  |  {date.today().isoformat()}  "
          f"|  {len(UNIVERSE)} instruments")
    print("=" * 80)

    ib = _connect(CLIENT_ID_PREFLIGHT)
    if ib is None:
        msg = "[PREFLIGHT] FAILED: cannot connect to IB Gateway (port 4002)"
        print(msg)
        if not args.no_discord:
            _send_discord(msg)
        sys.exit(1)

    ibcfg = load_ib_config()
    failures, open_now = check_contracts(ib, ibcfg, UNIVERSE)
    print(f"\n  Structural checks done — {len(open_now)} markets open now, "
          f"checking market data…")
    failures += check_market_data(ib, open_now)
    ib.disconnect()

    n_checked = len(UNIVERSE)
    if failures:
        print(f"\n  ✗ {len(failures)} FAILURES ({n_checked} instruments checked, "
              f"{len(open_now)} market-data checks):")
        for instr, stage, detail in failures:
            print(f"    [{stage:<7}] {instr}: {detail}")
        if not args.no_discord:
            lines = "\n".join(f"[{stage}] {instr}: {detail}"
                              for instr, stage, detail in failures[:25])
            more = f"\n…and {len(failures) - 25} more" if len(failures) > 25 else ""
            _send_discord(f"[PREFLIGHT] {len(failures)} execution-path failures "
                          f"({date.today().isoformat()})\n{lines}{more}")
        sys.exit(1)

    print(f"\n  ✓ ALL CLEAR — {n_checked} instruments, "
          f"{len(open_now)} live market-data checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
