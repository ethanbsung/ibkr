#!/usr/bin/env python3
"""
live_dynamic.py — Combined carry+trend (50/50) + dynamic portfolio optimisation, live on IBKR.

Replaces the IBS strategy (ibkr_fut/live_signals.py). Once per day this script:

  1. Connects to IBKR (paper Gateway, port 4002) and reads NetLiquidation (capital)
     and current futures positions.
  2. Builds the dynamic-optimisation universe over Carver's full Jumbo (ibkr_fut/
     jumbo.py) from the PST CSVs — the same validated pipeline the backtest uses
     (backtest_dynamic._build_universe → combined carry+trend forecasts, blended vol,
     weekly-EWMA covariance, handcraft weights, live-universe IDM).
  3. Runs ONE joint daily optimisation (dynamic_opt.optimise_positions) seeded with
     the *actual* held positions, restricted to a TRADABLE subset. The optimiser
     chooses which instruments to hold; everything else in the Jumbo is locked at
     its current position (0) so its risk transfers onto correlated tradables
     (Carver: optimise over ~150, trade ~100).
  4. Reconciles target vs held (handling contract rolls) and submits orders via
     Carver's passive-aggressive limit order algorithm, logging fills + a daily
     snapshot to paper/ledgers/ibkr_dynamic/.

TRADABLE SET (the menu the optimiser picks from) — computed each run as the Jumbo
instruments that pass instrument_selection filters (SR cost ≤ 0.01, annual vol ≥ 5%,
history ≥ 512 days, volume if cached) AND have fresh PST data (≤ FRESH_DAYS old).
This matches the filter applied in backtest_dynamic.py --filter, so live and backtest
use an identical eligible universe. Run volume_collector.py periodically to keep the
volume cache current and enable the liquidity filter.

Sizing/forecasts use PST daily closes (kept current by pst_updater.py — run it
before this script; see ibkr_fut/run_dynamic.sh).

MODES
-----
--mode compute  (6:00 PM ET)
    Connect to IBKR for capital + positions, run optimisation, save
    targets_snapshot.json. No orders placed.

--mode execute  (6:05 PM ET, one-shot)
    Load targets_snapshot.json, reconnect to IBKR for current positions,
    run pre-trade checks, place limit orders via passive-aggressive algo.

--mode daemon  (6:05 PM ET, long-running)
    Like execute but loops every DAEMON_SLEEP_SECS, checking market hours
    before each order. Defers instruments whose exchange is closed and retries
    on the next cycle. Picks up a new targets_snapshot.json automatically when
    the compute phase writes one (next day). Replaces the one-shot execute cron
    with a single persistent process that covers all time zones.

No --mode (default):
    Run compute then execute in a single session (manual testing / dry-runs).

ADDING STRATEGIES LATER: compute_targets() returns a {instrument: net_contracts}
dict. To add another strategy, produce a second such dict and merge (sum) the two
before reconcile_and_execute(). For a joint risk model you would instead extend the
optimisation universe; this netting hook is the simple path.

Dry-run (default) — print the plan, place nothing:
  python3 ibkr_fut/live_dynamic.py
Live execution (split cron):
  python3 ibkr_fut/live_dynamic.py --mode compute
  python3 ibkr_fut/live_dynamic.py --mode execute --execute
Daemon (started by run_execution.sh):
  python3 ibkr_fut/live_dynamic.py --mode daemon --execute
"""

import argparse
import json
import logging
import math
import asyncio
import os
import sys
import time
import traceback
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from ib_insync import IB, Future, Order, Bag, ComboLeg, LimitOrder, MarketOrder

# ib_insync streams portfolio price ticks every ~3 min at INFO level; filter them
# out while keeping fill confirmations (execDetails, commissionReport) and errors.
class _NoPortfolioTicks(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("updatePortfolio")

logging.getLogger("ib_insync").addFilter(_NoPortfolioTicks())

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ibkr_fut.pst_loader import PSTLoader
from ibkr_fut.instrument_universe import UNIVERSE
from ibkr_fut.backtest_dynamic import _build_universe, get_eligible_set
from ibkr_fut.backtest_ewmac import TARGET_RISK, IDM_CAP
from ibkr_fut.carry_trend_signals import (
    carry_trend_instrument_signals,
    TREND_WEIGHT as COMBINED_TREND_WEIGHT,
)
from ibkr_fut.dynamic_opt import optimise_positions
from ibkr_fut.algo_execution import pre_trade_checks, algo_exec, is_contract_okay_to_trade
from ibkr_fut.risk_check import (check_halt_file, check_gross_leverage,
                                 check_order_vol, check_daily_loss, raise_halt,
                                 _send_discord)
from ibkr_fut.trading_calendar import last_completed_session, sessions_behind
from paper.dyn_ledger import DynLedger

# ── Tradable-set filter ────────────────────────────────────────────────────────
FRESH_DAYS = 5   # PST data must be no more than this many calendar days old

# ── Daemon execution ───────────────────────────────────────────────────────────
DAEMON_SLEEP_SECS = 600   # seconds between daemon cycles (~10 min)
# Bound the live position re-request (fetch_positions). IB.RequestTimeout defaults
# to 0 = no timeout, so a half-open gateway (TCP up, API silent) would block the
# reqPositions await — and the whole daemon — forever. Cap it and fall back to the
# cache on timeout so a stuck connection degrades to "possibly stale", not "hung".
POSITIONS_TIMEOUT_SECS = 15
# Per-instrument churn circuit-breaker (BUG-7): if any one instrument is traded in
# more than this many cycles within a single daemon session, something is looping
# (stale positions, a roll that won't settle, …). Halt + alert instead of bleeding
# capital. A genuine roll touches an instrument a handful of times over a few days;
# 6 trades in one *session* is already pathological.
MAX_INSTR_TRADES_PER_SESSION = 6

# ── Contract rolling windows ───────────────────────────────────────────────────
PASSIVE_ROLL_DAYS = 10   # days before roll date: start routing rebalance orders to new month
SPREAD_ROLL_DAYS  = 3    # days before roll date: roll residual via BAG spread order
FORCE_ROLL_DAYS   = 1    # ≤ this many days to roll date: position MUST leave the
                         # expiring month — a failed spread limit escalates to MKT
SPREAD_PASSIVE_SECS = 60 # seconds to work the spread limit before cancelling
SPREAD_CANCEL_CONFIRM_SECS = 30  # max wait for cancel-ack before abandoning MKT escalation

# Mass-liquidation guard (reconcile_and_execute): if more than this fraction of the
# held book is absent from a FRESH snapshot, treat the snapshot as suspect — do not
# auto-close the absentees, alert and hold for manual review. ~1/3 ⇒ a 50%-absent
# event (the BUG-9 live incident) alerts rather than liquidating half the book.
ABSENT_HELD_MAX_FRAC = 0.34

# ── Strategy parameters (match the backtest you validated) ─────────────────────
DYN_TARGET_RISK = TARGET_RISK   # 0.25 — same as backtest_dynamic / backtest_ewmac


# Combined carry+trend forecast (Carver "Strategy 11"), the live forecast. Matches the
# signal_fn(spec, mp) contract _build_universe expects — identical to _combined_signal in
# backtest_carry_trend_dynamic.py, so live sizes off the same forecast the backtest validates.
def _combined_signal(spec, mp):
    return carry_trend_instrument_signals(spec, mp, COMBINED_TREND_WEIGHT)

# ── IBKR connection ────────────────────────────────────────────────────────────
IB_HOST         = "127.0.0.1"
IB_PORT         = 4002
CLIENT_ID       = 5    # daemon / execute mode
CLIENT_ID_COMPUTE = 6  # compute mode (runs concurrently with daemon)
CONNECT_TIMEOUT = 5
MAX_RETRIES     = 3
RETRY_DELAY     = 10
FALLBACK_CAPITAL = 250_000.0

# Repo-relative so the same checkout works on any host (laptop, VPS).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IB_CONFIG_PATH    = os.path.join(_REPO_ROOT, "Data/pst/ib_config/ib_config_futures.csv")
SNAPSHOT_PATH     = os.path.join(_REPO_ROOT, "ibkr_fut", "targets_snapshot.json")
LAST_TARGETS_PATH = os.path.join(_REPO_ROOT, "ibkr_fut", "last_targets.json")
HEARTBEAT_PATH    = os.path.join(_REPO_ROOT, "ibkr_fut", "daemon_heartbeat.txt")

pst = PSTLoader()

# conId → delivery month (YYYYMM). conId ↔ delivery month is immutable, so this
# cache persists for the daemon's lifetime — one reqContractDetails per contract, ever.
_CONID_MONTH_CACHE: dict[int, str] = {}


def delivery_month(ib, contract) -> str:
    """
    True delivery month (YYYYMM) for a futures contract.

    IB's contract.lastTradeDateOrContractMonth is the EXPIRY date, which for energy
    (NYMEX crude, Brent, NatGas, HeatOil, Gasoline) and some other contracts precedes
    the delivery month by one calendar month — e.g. QM Sep-delivery expires 20260819,
    so [:6] would mis-map it to 202608. ContractDetails.contractMonth is the canonical
    delivery month and matches the roll-calendar YYYYMM codes, so positions reconcile
    against the right roll-calendar month instead of triggering phantom rolls.
    Falls back to the expiry-date prefix only if contractMonth is unavailable.
    """
    cid = getattr(contract, "conId", 0)
    if cid and cid in _CONID_MONTH_CACHE:
        return _CONID_MONTH_CACHE[cid]
    month = ""
    try:
        cds = ib.reqContractDetails(contract)
        if cds and cds[0].contractMonth:
            month = str(cds[0].contractMonth)[:6]
    except Exception:
        month = ""
    if not month:                                   # safe fallback
        month = (contract.lastTradeDateOrContractMonth or "")[:6]
    if cid:
        _CONID_MONTH_CACHE[cid] = month
    return month


# ══════════════════════════════════════════════════════════════════════════════
# IBKR contract config
# ══════════════════════════════════════════════════════════════════════════════

def load_ib_config() -> pd.DataFrame:
    """IBKR contract mapping for every PST instrument (symbol/exchange/mult/…)."""
    return pd.read_csv(IB_CONFIG_PATH, index_col="Instrument")


def ib_spec(ibcfg: pd.DataFrame, instr: str) -> dict | None:
    """Resolved IBKR contract spec for one PST instrument, or None if unmapped."""
    if instr not in ibcfg.index:
        return None
    row = ibcfg.loc[instr]
    raw_curr = row.get("IBCurrency", "USD")
    currency = "USD" if (pd.isna(raw_curr) or str(raw_curr).strip().upper() in ("NA", "")) \
        else str(raw_curr).strip()
    raw_tc = row.get("IBTradingClass", "")
    trading_class = "" if pd.isna(raw_tc) else str(raw_tc).strip()
    mult = row.get("IBMultiplier", "")
    pricemag = row.get("priceMagnifier", 1)
    return {
        "symbol":        str(row["IBSymbol"]).strip(),
        "exchange":      str(row["IBExchange"]).strip(),
        "currency":      currency,
        "multiplier":    None if pd.isna(mult) else float(mult),
        "trading_class": trading_class,
        "price_magnifier": float(pricemag) if not pd.isna(pricemag) else 1.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tradable-set selection — the optimiser's menu
# ══════════════════════════════════════════════════════════════════════════════

def build_tradable_set(verbose: bool = True,
                       require_carry_fresh: bool = False) -> tuple[set, list]:
    """
    Decide which Jumbo instruments the optimiser may actually trade today.

    An instrument is tradable iff it passes instrument_selection filters (cost,
    annual vol floor, history) AND its PST data is fresh (≤ FRESH_DAYS old).
    Volume filters are skipped when no cached volume data exists.

    require_carry_fresh: when the live strategy uses a carry forecast (basic carry or
    combined carry+trend), the CARRY leg of multiple_prices can go stale INDEPENDENTLY of
    PRICE (pst_updater can refresh PRICE while the carry fetch fails). Pass True so the
    gate also ages the CARRY column — otherwise a frozen term structure passes the
    PRICE-only check and the optimiser sizes off stale carry. Default False = EWMAC trend
    (carry columns irrelevant), the current behaviour.

    Returns (tradable_set, rows) where rows is a per-instrument diagnostic table.
    """
    today = pd.Timestamp(date.today())

    # Step 1: instrument_selection filters (cost, too-safe, history; volume
    # when cached — get_eligible_set loads the volume cache itself).
    eligible = get_eligible_set(UNIVERSE)

    # Step 2: freshness check — PST data must be recent enough to trade on.
    tradable, rows = set(), []
    for instr in UNIVERSE:
        reason = ""
        last_date = None
        try:
            mp = pst.multiple_prices(instr)
            price = mp["PRICE"].dropna()
            last_date = price.index[-1]
            age = (today - last_date.normalize()).days
            if age > FRESH_DAYS:
                reason = f"stale ({last_date.date()}, {age}d old)"
            elif require_carry_fresh:
                # Carry leg can freeze independently of PRICE — age it too.
                carry = mp["CARRY"].dropna()
                if carry.empty:
                    reason = "no carry data"
                else:
                    c_last = carry.index[-1]
                    c_age = (today - c_last.normalize()).days
                    if c_age > FRESH_DAYS:
                        reason = f"stale carry ({c_last.date()}, {c_age}d old)"
        except Exception as e:
            reason = f"data error ({e})"

        if not reason and instr not in eligible:
            reason = "filtered (cost/vol/history)"

        if not reason:
            tradable.add(instr)
        rows.append({
            "instr":     instr,
            "class":     UNIVERSE[instr],
            "last_date": last_date.date() if last_date is not None else None,
            "tradable":  not reason,
            "reason":    reason,
        })

    if verbose:
        keep   = sorted(r["instr"] for r in rows if r["tradable"])
        filtered = [r for r in rows if not r["tradable"]]
        print(f"\n  TRADABLE SET — {len(tradable)} instruments "
              f"(instrument_selection filters · fresh PST data ≤ {FRESH_DAYS}d):")
        line, n = "    ", 0
        for instr in keep:
            line += f"{instr:<16}"
            n += 1
            if n % 5 == 0:
                print(line); line = "    "
        if n % 5:
            print(line)

        # Group filtered instruments by reason for a compact summary
        reason_groups: dict[str, list[str]] = {}
        for r in filtered:
            tag = r["reason"]
            if "cost" in tag or "vol" in tag or "history" in tag:
                tag = "filtered (cost/vol/history)"
            reason_groups.setdefault(tag, []).append(r["instr"])
        print(f"\n  Excluded ({len(filtered)}):")
        for reason, instrs in sorted(reason_groups.items()):
            print(f"    {reason}: {sorted(instrs)}")

    return tradable, rows


# ══════════════════════════════════════════════════════════════════════════════
# Daily optimisation — compute target net positions
# ══════════════════════════════════════════════════════════════════════════════

def compute_targets(uni: dict, capital: float, current_positions: dict,
                    target_risk: float = DYN_TARGET_RISK) -> tuple[dict, dict]:
    """
    Run the joint dynamic optimisation for the latest available date, seeded with
    the actual held positions. Returns:
      targets    {instrument: target_net_contracts}  (only instruments with a
                 nonzero target or a nonzero current holding)
      diag       {instrument: {forecast, n_ideal, raw_price, mult, sigma}} for
                 held/traded instruments, plus '_meta' with idm / n_live / date /
                 gross_lev.
    """
    names, idx = uni["names"], uni["idx"]
    price, raw, fx = uni["price"], uni["raw"], uni["fx"]
    sigma, forecast = uni["sigma"], uni["forecast"]
    mult, spread, commission = uni["mult"], uni["spread"], uni["commission"]
    W, C, est = uni["W"], uni["C"], uni["est"]
    tradable = uni["tradable"]
    t = len(idx) - 1                       # latest date
    as_of = pd.Timestamp(idx.values[t])

    p, r, f = price[t], raw[t], fx[t]
    s, fc = sigma[t], forecast[t]
    valid = (~np.isnan(p) & ~np.isnan(r) & (r > 0)
             & ~np.isnan(s) & (s > 0) & ~np.isnan(f) & ~np.isnan(fc))
    live_idx = np.flatnonzero(valid)

    prev_full = np.array([current_positions.get(nm, 0) for nm in names], dtype=float)

    targets, diag = {}, {}
    if live_idx.size == 0:
        # nothing live; hold everything we currently have (all frozen, no signal)
        for nm in names:
            if current_positions.get(nm, 0) != 0:
                targets[nm] = int(current_positions[nm])
        diag["_meta"] = {"idm": 1.0, "n_live": 0, "date": as_of.date(),
                         "gross_lev": 0.0,
                         "status": {nm: "frozen" for nm in targets}}
        return targets, diag

    # Live-universe renormalised weights + IDM.
    w_live = W[live_idx]
    ssum = float(w_live.sum())
    w_n = w_live / ssum if ssum > 0 else np.full(live_idx.size, 1.0 / live_idx.size)
    Csub = C[np.ix_(live_idx, live_idx)]
    var = float(w_n @ Csub @ w_n)
    idm_t = min(1.0 / np.sqrt(var), IDM_CAP) if var > 0 else 1.0

    ml, rl, fl, sl, fcl = mult[live_idx], r[live_idx], f[live_idx], s[live_idx], fc[live_idx]
    N_unrounded = (fcl * capital * idm_t * w_n * target_risk) / (10.0 * ml * rl * fl * sl)
    weight_per_contract = ml * rl * fl / capital
    cost_per_contract = (spread[live_idx] * ml + commission[live_idx] / 2.0) * fl

    cov = est.covariance_by_index(as_of, live_idx)
    prev_live = prev_full[live_idx]

    N_star = optimise_positions(
        covariance=cov,
        weight_per_contract=weight_per_contract,
        optimal_unrounded_positions=N_unrounded,
        previous_positions=prev_live,
        cost_per_contract=cost_per_contract,
        capital=capital,
        target_risk=target_risk,
        use_costs=True,
        use_buffering=True,
        tradable=tradable[live_idx],
    )

    gross_notional = float(np.sum(np.abs(N_star) * ml * rl * fl))
    for k, j in enumerate(live_idx):
        nm = names[j]
        tgt = int(round(N_star[k]))
        if tgt != 0 or current_positions.get(nm, 0) != 0:
            targets[nm] = tgt
            diag[nm] = {
                "forecast":  float(fcl[k]),
                "n_ideal":   float(N_unrounded[k]),
                "raw_price": float(rl[k]),
                "mult":      float(ml[k]),
                "fx":        float(fl[k]),    # local→USD; USD notional = mult*price*fx
                "sigma":     float(sl[k]),    # annualised vol fraction; used for price-divergence check
            }

    # Held instruments that aren't live today: hold them (don't blind-trade).
    for nm in names:
        if current_positions.get(nm, 0) != 0 and nm not in targets:
            targets[nm] = int(current_positions[nm])

    # Per-target status so the daily report can explain *why* a position is held:
    #   active      — in the tradable set; optimiser sized it freely.
    #   reduce_only — held but not tradable, with a valid signal today: the optimiser
    #                 may only unwind it toward 0 (never grow/flip). [dynamic_opt]
    #   frozen      — held but not tradable AND no valid signal today: held at current
    #                 (the no-blind-trade fallback above).
    tradable_by_name = {nm: bool(tradable[i]) for i, nm in enumerate(names)}
    valid_by_name = {nm: bool(valid[i]) for i, nm in enumerate(names)}
    status = {}
    for nm in targets:
        if tradable_by_name.get(nm, False):
            status[nm] = "active"
        elif valid_by_name.get(nm, False):
            status[nm] = "reduce_only"
        else:
            status[nm] = "frozen"

    diag["_meta"] = {
        "idm": round(idm_t, 3), "n_live": int(live_idx.size),
        "date": as_of.date(), "gross_lev": round(gross_notional / capital, 2),
        "n_held_target": int(np.count_nonzero(N_star)),
        "status": status,
    }
    return targets, diag


# ══════════════════════════════════════════════════════════════════════════════
# IBKR helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_equity(ib) -> float | None:
    for v in ib.accountValues():
        if v.tag == "NetLiquidation" and v.currency == "USD":
            return float(v.value)
    return None


def ib_symbol_to_instr(ibcfg: pd.DataFrame) -> dict:
    """
    {IBSymbol: instr} over UNIVERSE. Ambiguous symbols (carried by 2+ instruments)
    map to None — a bare symbol must not silently claim someone else's position.
    The single definition of the symbol→instrument map, reused by callers that
    only have an IB symbol (e.g. the daily report's positions.csv fallback).
    """
    sym_only: dict = {}
    for instr in UNIVERSE:
        spec = ib_spec(ibcfg, instr)
        if spec:
            sym = spec["symbol"]
            sym_only[sym] = None if sym in sym_only else instr
    return sym_only


def fetch_positions(ib, strict: bool = False) -> list:
    """
    Return a FRESH list of account positions, re-requested from IB.

    `ib.positions()` returns ib_insync's locally-cached position list, maintained
    only by the `reqPositions` subscription's `position` events. That cache can
    silently FREEZE — e.g. after a gateway disconnect/reconnect the subscription
    is not re-seeded, so the cache keeps returning the value it held at the moment
    the event stream stopped, while the real account moves on (BUG-7: the daemon
    re-rolled QM every cycle off a stale +1/+1 snapshot, never seeing its own
    fills, until the true position was a 27-lot phantom calendar spread).

    `reqPositionsAsync()` round-trips to IB and returns the authoritative current
    list, bypassing the cache. We always re-request rather than trusting
    `ib.positions()`. The request is bounded by POSITIONS_TIMEOUT_SECS — IB's own
    blocking `reqPositions()` inherits RequestTimeout=0 (no timeout), which on a
    half-open connection would hang the whole daemon waiting for a `positionEnd`
    that never arrives. On timeout OR any error we fall back to the cached
    `ib.positions()`, so a stuck/transient connection degrades to "possibly stale"
    rather than "hung" or "crash".
    When strict=True the caller is about to PERSIST the result (compute writes a
    snapshot the daemon then trades against). A silent fall back to the possibly-
    empty/frozen cache there is dangerous: a fresh reqPositions failure looks
    identical to "really flat", so the snapshot would be computed off a phantom-
    flat book, stranding every held position that isn't re-derived (BUG-8). In
    strict mode we raise PositionFetchError instead of guessing, so the caller can
    abort WITHOUT overwriting the last good snapshot.
    """
    try:
        return ib.run(asyncio.wait_for(ib.reqPositionsAsync(),
                                       POSITIONS_TIMEOUT_SECS))
    except Exception as e:
        kind = "timed out" if isinstance(e, asyncio.TimeoutError) else f"failed ({e})"
        if strict:
            raise PositionFetchError(
                f"live reqPositions {kind} — refusing to fall back to a "
                f"possibly-stale/empty cache while persisting a snapshot") from e
        print(f"[{_now()}] WARNING: live reqPositions {kind} — "
              f"falling back to cached ib.positions()")
        return ib.positions()


class PositionFetchError(RuntimeError):
    """reqPositions failed and the caller required a verified (non-cached) read."""


def get_positions_by_instr(ib, ibcfg: pd.DataFrame,
                           strict: bool = False) -> tuple[dict, list]:
    """
    Return ({instr: {YYYYMM: qty}}, unknown) for all held futures.
    Maps IBKR (symbol, exchange) back to a Jumbo PST instrument name.

    Positions are re-requested live each call via fetch_positions (NOT the
    passively-cached ib.positions()) — see BUG-7. strict propagates to
    fetch_positions: when True a failed re-request raises PositionFetchError
    rather than silently returning the cache (see BUG-8).
    """
    # (symbol, exchange) -> instr, restricted to Jumbo. Symbol-only fallback is
    # used only when exactly one universe instrument carries that IB symbol —
    # an ambiguous symbol must not silently claim someone else's position.
    rev = {}
    for instr in UNIVERSE:
        spec = ib_spec(ibcfg, instr)
        if spec:
            rev[(spec["symbol"], spec["exchange"])] = instr
    sym_only = ib_symbol_to_instr(ibcfg)

    held, unknown = {}, []
    for pos in fetch_positions(ib, strict=strict):
        c = pos.contract
        if c.secType != "FUT":
            continue
        qty = int(pos.position)
        if qty == 0:
            continue
        instr = rev.get((c.symbol, c.primaryExchange)) or rev.get((c.symbol, c.exchange)) \
            or sym_only.get(c.symbol)
        if instr is None:
            unknown.append((c.symbol, c.exchange, qty))
            continue
        month = delivery_month(ib, c)
        by_month = held.setdefault(instr, {})
        by_month[month] = by_month.get(month, 0) + qty
    return held, unknown


def get_roll_info(instr: str) -> tuple[str | None, str | None, int]:
    """
    Returns (current_month, next_month, days_to_roll).

    current_month : YYYYMM to hold today
    next_month    : YYYYMM to roll into at the upcoming roll date (None if no future roll)
    days_to_roll  : calendar days until the roll date (DATE_TIME row); 9999 if no upcoming roll

    Roll calendar row semantics: DATE_TIME is the last day to hold current_contract.
    The day after DATE_TIME, the following row's current_contract becomes active.
    """
    try:
        rc = pst.roll_calendar(instr)
    except FileNotFoundError:
        return None, None, 9999
    today = pd.Timestamp(date.today())
    fut = rc[rc.index.normalize() >= today]
    if fut.empty:
        row = rc.iloc[-1]
        return str(int(row["current_contract"]))[:6], None, 9999
    row = fut.iloc[0]
    current = str(int(row["current_contract"]))[:6]
    nxt_raw = row.get("next_contract")
    nxt     = str(int(nxt_raw))[:6] if pd.notna(nxt_raw) else None
    days    = (row.name.normalize() - today).days
    return current, nxt, days


def qualify(ib, spec: dict, month: str):
    """Qualify an IBKR future for (instrument spec, contract month). None on fail.

    Only symbol, exchange, month, and trading_class (if set) are sent — IB fills
    in currency, multiplier, conId, etc. from its contract database.
    """
    raw = Future(
        symbol=spec["symbol"],
        lastTradeDateOrContractMonth=month,
        exchange=spec["exchange"],
        tradingClass=spec["trading_class"] or "",
    )
    try:
        quals = ib.qualifyContracts(raw)
    except Exception:
        quals = []
    return quals[0] if quals else None


# ══════════════════════════════════════════════════════════════════════════════
# Snapshot helpers
# ══════════════════════════════════════════════════════════════════════════════

class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, date):
            return obj.isoformat()
        if hasattr(obj, "isoformat"):   # pd.Timestamp
            return obj.isoformat()
        return super().default(obj)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_snapshot_any(path: str) -> dict | None:
    """Load snapshot without enforcing today's date (daemon spans midnight)."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def save_snapshot(path: str, today: str, capital: float,
                  targets: dict, diag: dict) -> None:
    """Atomically write compute-phase results to disk."""
    payload = {
        "date":        today,
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capital":     capital,
        "targets":     targets,
        "diag":        diag,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, cls=_Encoder, indent=2)
    os.replace(tmp, path)


def load_snapshot(path: str, today: str) -> dict:
    """Load the snapshot written by the compute phase. Abort if missing or stale."""
    if not os.path.exists(path):
        print(f"ERROR: snapshot not found at {path}. "
              f"Run --mode compute first.")
        raise SystemExit(1)
    with open(path) as fh:
        snap = json.load(fh)
    if snap.get("date") != today:
        print(f"ERROR: snapshot is from {snap.get('date')}, expected {today}. "
              f"Re-run --mode compute.")
        raise SystemExit(1)
    return snap


def check_last_targets(path: str, current_positions: dict) -> None:
    """Warn about instruments where actual IBKR positions differ from last run's targets."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        last = json.load(fh)
    last_targets = last.get("targets", {})
    all_instrs = set(last_targets) | set(current_positions)
    mismatches = []
    for instr in sorted(all_instrs):
        expected = int(last_targets.get(instr, 0))
        actual   = int(current_positions.get(instr, 0))
        if expected != actual:
            mismatches.append(f"{instr}: expected {expected:+d}, actual {actual:+d}")
    if mismatches:
        print(f"\n  ⚠ RECONCILIATION vs last run ({last.get('date')}):")
        for m in mismatches:
            print(f"    {m}")


def save_last_targets(path: str, targets: dict, today: str) -> None:
    """Persist today's targets for reconciliation at the next execute run."""
    with open(path, "w") as fh:
        json.dump({"date": today, "targets": targets}, fh, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Spread roll execution
# ══════════════════════════════════════════════════════════════════════════════

def _spread_px_ok(px) -> bool:
    """True if px is a usable calendar-spread quote. Unlike outright prices,
    spread quotes are legitimately zero or negative; reject only missing values
    and IB's no-quote sentinels (-1, large negatives)."""
    if px is None:
        return False
    try:
        f = float(px)
    except (TypeError, ValueError):
        return False
    return not math.isnan(f) and f != -1.0 and f > -1e8


def spread_roll_exec(ib, spec: dict, from_month: str, to_month: str, qty: int,
                     is_long: bool, force: bool = False):
    """
    Roll qty contracts from from_month to to_month via a single IBKR BAG calendar
    spread order. Direction depends on position sign:
      is_long=True  → SELL from_month / BUY  to_month  (rolling a long)
      is_long=False → BUY  from_month / SELL to_month  (rolling a short)

    Works a passive limit at the spread's offside price (bid when buying the
    spread, ask when selling — never paying the spread) for SPREAD_PASSIVE_SECS,
    then cancels.

    force=False (early in the spread window): a failed/timed-out limit is simply
    cancelled — the remainder stays in the expiring month and is retried on the
    next cycle/day.
    force=True (≤ FORCE_ROLL_DAYS before the roll date): the position MUST leave
    the expiring month, so after the cancel is CONFIRMED the unfilled remainder
    is escalated to a market order. Without a confirmed cancel no market order
    is placed (a still-live limit could double-fill). If no spread quote is
    available, force goes straight to market; non-force skips.

    Returns (status, filled_qty, avg_price). filled_qty counts fills across the
    limit and any escalation market order.
    """
    sym      = spec["symbol"]
    exchange = spec["exchange"]

    # Direction: a long position is rolled by SELLing the spread (sell near, buy far);
    # a short position requires the opposite (buy near, sell far).
    bag_action  = "SELL" if is_long else "BUY"
    from_action = "SELL" if is_long else "BUY"
    to_action   = "BUY"  if is_long else "SELL"

    c_from = qualify(ib, spec, from_month)
    c_to   = qualify(ib, spec, to_month)
    if c_from is None or c_to is None:
        print(f"    SPREAD FAIL: could not qualify {sym} {from_month}/{to_month}")
        return "Failed", 0, 0.0

    spread_contract = Bag(
        symbol=sym,
        currency=c_from.currency,
        exchange=exchange,
        comboLegs=[
            ComboLeg(conId=c_from.conId, ratio=1, action=from_action, exchange=exchange),
            ComboLeg(conId=c_to.conId,   ratio=1, action=to_action,   exchange=exchange),
        ],
    )

    # Request spread market data (bid/ask on the calendar spread itself)
    ticker = ib.reqMktData(spread_contract, "", False, False)
    ib.sleep(5)
    bid, ask = ticker.bid, ticker.ask
    ib.cancelMktData(spread_contract)

    valid_quote = (_spread_px_ok(bid) and _spread_px_ok(ask) and bid <= ask)
    if valid_quote:
        # Offside limit, same as algo_exec: bid when buying the spread, ask when
        # selling — don't pay the spread. Unfilled non-force rolls retry next
        # daemon cycle; the force path guarantees completion at expiry.
        limit_px = round(bid if bag_action == "BUY" else ask, 4)
        order = LimitOrder(bag_action, qty, limit_px, tif="DAY")
        price_str = f"lmt {limit_px:.4f}  (bid {bid:.4f} / ask {ask:.4f})"
    elif force:
        order = MarketOrder(bag_action, qty, tif="DAY")
        price_str = "MKT (no spread quote — forced roll)"
    else:
        print(f"    SPREAD SKIP: no spread quote for {sym} {from_month}/{to_month} "
              f"— retry next cycle")
        return "Unfilled", 0, 0.0

    print(f"    SPREAD ROLL {qty} {sym} {from_month}→{to_month}  {price_str}")
    trade = ib.placeOrder(spread_contract, order)

    deadline = time.time() + SPREAD_PASSIVE_SECS
    while time.time() < deadline and not trade.isDone():
        ib.sleep(1)

    limit_filled = 0   # fills kept on the limit when we escalate to market
    if not trade.isDone():
        # Limit timed out — cancel, then (force only) escalate the remainder.
        ib.cancelOrder(trade.order)
        cancel_deadline = time.time() + SPREAD_CANCEL_CONFIRM_SECS
        while time.time() < cancel_deadline and not trade.isDone():
            ib.sleep(1)

        if force:
            if not trade.isDone():
                # Never place a market order while the limit may still be live.
                print(f"    SPREAD WARN: cancel unconfirmed for {sym} "
                      f"{from_month}/{to_month} — NOT escalating to MKT "
                      f"(double-fill risk); manual check required")
            else:
                # Use trade.filled() (sourced from execDetails) rather than
                # trade.orderStatus.filled (sourced from orderStatus messages) to
                # avoid over-sizing when a partial fill arrived before the cancel-ack.
                remaining = qty - int(trade.filled())
                if remaining > 0:
                    limit_filled = int(trade.filled())
                    print(f"    SPREAD ESCALATE → MKT {remaining} {sym} "
                          f"{from_month}→{to_month}  (forced roll, expiry imminent)")
                    mkt = MarketOrder(bag_action, remaining, tif="DAY")
                    trade = ib.placeOrder(spread_contract, mkt)
                    ib.sleep(10)

    status  = trade.orderStatus.status
    filled  = int(trade.orderStatus.filled) + limit_filled
    avg_px  = trade.orderStatus.avgFillPrice or 0.0
    print(f"    SPREAD → {status}  filled {filled}/{qty}  avg {avg_px:.4f}")
    return status, filled, avg_px


# ══════════════════════════════════════════════════════════════════════════════
# Reconcile + execute
# ══════════════════════════════════════════════════════════════════════════════

def reconcile_and_execute(ib, ibcfg, targets, held, diag, ledger, execute: bool,
                          skip_unchanged: bool = False, capital: float = 0.0,
                          snapshot_fresh: bool = True):
    """
    For each instrument with a target or current holding, roll out of old months and
    move the hold-month position to the target. Prints the plan; places orders via
    the passive-aggressive limit order algorithm when execute=True.

    snapshot_fresh: whether the snapshot driving `targets` passed the daemon's
        calendar-aware staleness gate (see run_daemon). A held instrument that is
        ABSENT from a fresh snapshot is unwound toward 0 (the snapshot is
        authoritative about what we should hold) — this is the fix for the
        stranding bug where `targets.get(instr, net_held)` defaulted to keep, so a
        position the optimiser wanted at 0 was never closed. When the snapshot is
        NOT fresh, absence is treated as "stale and silent" → hold (never infer a
        close off untrusted data).
    """
    pending = {t.contract.symbol for t in ib.openTrades()}
    placed, skipped, dry_run = [], [], []
    traded_instrs: set = set()   # instruments that placed a live order this cycle (BUG-7 churn cap)
    _mkt_open_cache: dict = {}   # conId → bool; one reqContractDetails per contract per cycle

    def _is_open(c) -> bool:
        if c.conId not in _mkt_open_cache:
            _mkt_open_cache[c.conId] = is_contract_okay_to_trade(ib, c)
        return _mkt_open_cache[c.conId]

    # ── Mass-liquidation guard ────────────────────────────────────────────────
    # The held-but-absent → unwind-toward-0 rule below makes the snapshot
    # authoritative about which positions to hold. That is correct for a normal
    # cleanup (a few instruments left the universe), but DANGEROUS if a fresh-but-
    # CORRUPT snapshot silently omits most of the book (e.g. compute sized off a
    # partial position read) — auto-closing then liquidates real exposure. If more
    # than ABSENT_HELD_MAX_FRAC of the held book is absent from a fresh snapshot,
    # treat the snapshot as suspect: do NOT auto-close, hold everything absent, and
    # alert. The clean recovery is a verified fresh compute. (Live BUG-9 incident:
    # 6 of 12 held = 50% absent → this guard would alert+hold, not liquidate.)
    held_instrs = {i for i in held if sum(held[i].values()) != 0}
    absent_held = held_instrs - set(targets)
    suspect_bad_snapshot = bool(
        snapshot_fresh and held_instrs
        and len(absent_held) / len(held_instrs) > ABSENT_HELD_MAX_FRAC)
    if suspect_bad_snapshot:
        msg = (f"{len(absent_held)}/{len(held_instrs)} held positions absent from a "
               f"fresh snapshot (> {ABSENT_HELD_MAX_FRAC:.0%}) — suspected bad "
               f"snapshot; NOT auto-closing absentees. Manual review. "
               f"Absent: {sorted(absent_held)}")
        print(f"[{_now()}] WARNING: [RECONCILE-ABSENT] {msg}")
        try:
            _send_discord(f"[RECONCILE-ABSENT] {msg}")
        except Exception as e:
            print(f"[{_now()}] WARNING: RECONCILE-ABSENT Discord alert failed: {e}")

    status_map = (diag.get("_meta") or {}).get("status", {})

    instruments = sorted(set(targets) | set(held))
    for instr in instruments:
        spec = ib_spec(ibcfg, instr)
        if spec is None:
            print(f"  {instr}: no IB config — skipping")
            continue
        sym = spec["symbol"]
        pm  = spec["price_magnifier"]
        ibmult = 1.0   # updated from qualified contract below

        # ── Determine roll phase ──────────────────────────────────────────────
        current_month, next_month, days_to_roll = get_roll_info(instr)
        if current_month is None:
            print(f"  {instr} ({sym}): no roll calendar — skipping")
            continue

        # Guard all order placement (spread roll AND rebalance) behind the pending
        # check. Placing a spread BAG order while a prior-cycle limit is still live
        # would create conflicting orders on the same leg.
        if execute and sym in pending:
            print(f"  {instr} ({sym}): SKIPPED — pending order exists for {sym}")
            skipped.append(sym)
            continue

        in_passive = (next_month is not None
                      and SPREAD_ROLL_DAYS < days_to_roll <= PASSIVE_ROLL_DAYS)
        in_spread  = (next_month is not None
                      and 0 <= days_to_roll <= SPREAD_ROLL_DAYS)

        # Months expected in the portfolio during a roll window (not "old")
        expected_months = {current_month}
        if in_passive or in_spread:
            expected_months.add(next_month)

        pos_by_month = held.get(instr, {})
        qty_current  = pos_by_month.get(current_month, 0)
        # Only track qty_next during a roll window; outside it next_month is not in
        # expected_months and would also appear in old_months, double-counting net_held.
        qty_next     = (pos_by_month.get(next_month, 0)
                        if (next_month and (in_passive or in_spread)) else 0)
        old_months   = {m: q for m, q in pos_by_month.items()
                        if m not in expected_months and q != 0}
        net_held     = qty_current + qty_next + sum(old_months.values())
        # Position held in the months we keep (current + incoming during a roll).
        # `desired` is compared against this for rebalancing; old months are always
        # closed separately via roll_closes, so they must NOT count toward "hold".
        expected_held = qty_current + qty_next

        # ── Determine desired net position ────────────────────────────────────
        # Normal case: the instrument is in the snapshot — honour its target.
        # Held-but-ABSENT case: the snapshot is authoritative about what to hold.
        #   • fresh snapshot → unwind toward 0 (reduce-only). This is the stranding
        #     fix: the old `targets.get(instr, net_held)` defaulted to KEEP, so a
        #     position the optimiser wanted closed was never traded out.
        #   • not fresh, or flagged by the mass-liquidation guard → hold (frozen);
        #     never infer a close off an untrusted/suspect snapshot.
        instr_status = status_map.get(instr)   # active | reduce_only | frozen | None
        if instr in targets:
            desired = int(targets[instr])
        elif (not snapshot_fresh) or (instr in absent_held and suspect_bad_snapshot):
            desired, instr_status = expected_held, "frozen"
        else:
            desired, instr_status = 0, "reduce_only"

        # Hard guard on status, independent of how `desired` was derived, so a bad
        # snapshot target can never grow or flip a position that should only shrink.
        # Mirrors dynamic_opt.optimise_positions' reduce_only step/overshoot logic.
        # Clamp against expected_held (the months we keep), so a stranded OLD month
        # is still closed by the roll path even when the live position is frozen.
        if instr_status == "frozen":
            desired = expected_held            # no directional change; old-month rolls still close
        elif instr_status == "reduce_only":
            if expected_held > 0:
                desired = max(0, min(desired, expected_held))
            elif expected_held < 0:
                desired = min(0, max(desired, expected_held))
            else:
                desired = 0

        d = diag.get(instr, {})
        fc      = d.get("forecast")
        n_ideal = d.get("n_ideal")
        sigma   = d.get("sigma", 0.20)
        raw_px  = d.get("raw_price")
        fx      = d.get("fx", 1.0)
        dmult   = d.get("mult")
        sig_px  = (raw_px / pm) if raw_px else None

        # ── SPREAD PHASE: roll only the portion that must survive into next month ──
        # Passive rolling (routing rebalance orders to the right leg) is always
        # preferred. The spread roll only moves contracts that would otherwise be
        # stranded in the expiring month. Contracts being closed can close directly
        # in current_month — that is cheaper (1 crossing vs 2).
        #
        # qty_to_roll = how much of qty_current actually needs to move:
        #   = desired position in next_month, minus what is already there,
        #     capped to what we actually hold in current_month.
        if in_spread and qty_current != 0:
            if qty_current > 0:
                qty_to_roll = max(0, min(qty_current, desired - qty_next))
            else:
                qty_to_roll = min(0, max(qty_current, desired - qty_next))

            if qty_to_roll != 0:
                if execute:
                    c_near = qualify(ib, spec, current_month)
                    if c_near is None:
                        print(f"    WARNING: could not qualify {sym} {current_month} "
                              f"— skip spread roll")
                    elif not _is_open(c_near):
                        print(f"    DEFERRED — {sym} spread roll, market closed")
                        skipped.append(f"{sym} roll (market closed)")
                    else:
                        sp_status, sp_filled, _ = spread_roll_exec(
                            ib, spec, current_month, next_month, abs(qty_to_roll),
                            is_long=qty_to_roll > 0,
                            force=days_to_roll <= FORCE_ROLL_DAYS)
                        # Credit fills whatever the final status — a cancelled
                        # limit can still carry partial fills, and ignoring them
                        # would mis-size this cycle's rebalance orders.
                        if sp_filled > 0:
                            direction = 1 if qty_to_roll > 0 else -1
                            qty_next    += direction * sp_filled
                            qty_current -= direction * sp_filled
                        # Any remainder stays in current_month; closed directly via algo_exec
                else:
                    force_tag = ("  [FORCE — MKT fallback]"
                                 if days_to_roll <= FORCE_ROLL_DAYS else "")
                    print(f"    [DRY-RUN] SPREAD ROLL {abs(qty_to_roll)} {sym} "
                          f"{current_month}→{next_month}{force_tag}")

        # ── Determine order routing and delta ─────────────────────────────────
        # rebalance_orders: [(month, signed_delta)]. During a roll window a
        # position-reducing delta closes the expiring leg first but never past
        # zero — the remainder comes out of the incoming month. Adds always go
        # to the incoming month.
        rebalance_orders: list[tuple[str, int]] = []
        if in_passive or in_spread:
            total_held   = qty_current + qty_next
            target_delta = desired - total_held
            if target_delta != 0:
                if qty_current != 0 and target_delta * qty_current < 0:
                    take = min(abs(target_delta), abs(qty_current))
                    from_current = take if target_delta > 0 else -take
                    rebalance_orders.append((current_month, from_current))
                    rest = target_delta - from_current
                    if rest != 0:
                        rebalance_orders.append((next_month, rest))
                else:
                    rebalance_orders.append((next_month, target_delta))
        else:
            target_delta = desired - qty_current
            if target_delta != 0:
                rebalance_orders.append((current_month, target_delta))

        roll_closes = [("SELL" if q > 0 else "BUY", abs(q), m)
                       for m, q in old_months.items()]

        if not rebalance_orders and not roll_closes:
            if not skip_unchanged:
                if in_passive or in_spread:
                    roll_tag = (f"PASSIVE d={days_to_roll}" if in_passive
                                else f"SPREAD d={days_to_roll}")
                    print(f"\n  {instr:<14} {sym:<6} [{roll_tag}] "
                          f"cur={current_month}:{qty_current:+d}  "
                          f"nxt={next_month}:{qty_next:+d}  (no change)")
                else:
                    fc_s = f"fcast {fc:+.1f}" if fc is not None else "fcast n/a"
                    ni_s = f"ideal {n_ideal:+.2f}" if n_ideal is not None else ""
                    print(f"\n  {instr:<14} {sym:<6} {current_month}  | held {net_held:+d} → "
                          f"target {desired:+d}  ({fc_s} {ni_s})")
                    print("    (no change)")
            continue

        fc_s = f"fcast {fc:+.1f}" if fc is not None else "fcast n/a"
        ni_s = f"ideal {n_ideal:+.2f}" if n_ideal is not None else ""
        if in_passive or in_spread:
            roll_tag = (f"PASSIVE d={days_to_roll}" if in_passive
                        else f"SPREAD d={days_to_roll}")
            print(f"\n  {instr:<14} {sym:<6} [{roll_tag}] "
                  f"cur={current_month}:{qty_current:+d}  nxt={next_month}:{qty_next:+d} "
                  f"→ target {desired:+d}  ({fc_s} {ni_s})")
        else:
            print(f"\n  {instr:<14} {sym:<6} {current_month}  | held {net_held:+d} → "
                  f"target {desired:+d}  ({fc_s} {ni_s})")
        if old_months:
            det = "  ".join(f"{m}:{q:+d}" for m, q in sorted(old_months.items()))
            print(f"    ⚠ ROLL — other months: {det}")

        for om, delta in rebalance_orders:
            act_label = "BUY " if delta > 0 else "SELL"
            print(f"    ACTION: {act_label} {abs(delta)} {sym} {om} [LMT ALGO]")
        for act, q, m in roll_closes:
            print(f"    ACTION: {act} {q} {sym} {m} [ROLL CLOSE LMT ALGO]")

        if not execute:
            qm = rebalance_orders[0][0] if rebalance_orders else current_month
            c = qualify(ib, spec, qm)
            if c:
                print(f"    qualify OK → {c.localSymbol}  conId={c.conId}  "
                      f"mult={c.multiplier}  {c.currency}")
            else:
                print(f"    WARNING: could not qualify {sym} {qm}")
            dry_run.append(sym)
            continue

        # ── Risk gate: reject pathologically-sized TARGET positions ───────────
        # Sanity check on the net target (not order delta), on the same USD basis
        # as sizing (mult*price*fx). A breach blocks only the rebalance orders —
        # old-month roll closes are risk-reducing and always execute — so a
        # misconfig (blown-up forecast, wrong multiplier) halts only that
        # instrument's new trading.
        risk_ok = True
        if desired != 0 and raw_px and dmult:
            ok, reason = check_order_vol(desired, dmult, raw_px, fx, sigma, capital)
            if not ok:
                print(f"    [RISK] SKIP {sym} rebalance: target {desired:+d} — {reason}")
                skipped.append(f"{sym} (risk: {reason})")
                risk_ok = False

        # ── Execute roll closes first ─────────────────────────────────────────
        for act, q, m in roll_closes:
            c = qualify(ib, spec, m)
            if c is None:
                print(f"    WARNING: could not qualify {sym} {m} — skip roll close")
                continue
            ibmult = float(c.multiplier) if c.multiplier else 1.0

            if not _is_open(c):
                print(f"    DEFERRED — {sym} {m} market closed")
                skipped.append(f"{sym} {m} (market closed)")
                continue

            ok, reason, ticker = pre_trade_checks(ib, c, sig_px, sigma, q)
            if not ok:
                print(f"    SKIP ROLL pre-trade [{sym} {m}]: {reason}")
                skipped.append(f"{sym} {m} (pre-trade)")
                continue

            result = algo_exec(ib, c, act, q, ticker)
            ib.cancelMktData(c)

            fp = result.avg_price or sig_px or 0.0
            aggr = "AGGRESSIVE" if result.was_aggressive else "passive"
            print(f"    ROLL {act} {q} {sym} {m} → {result.status}  "
                  f"fill {fp:.4f}  [{aggr}]")

            if result.status in ("Filled", "PartiallyFilled") and result.avg_price > 0:
                ledger.log_fill(symbol=sym, contract=m, action=act,
                                qty=result.filled_qty, multiplier=ibmult,
                                signal_price=(sig_px or result.avg_price),
                                fill_price=result.avg_price,
                                commission=result.commission, forecast=fc)
            if result.status == "PartiallyFilled":
                print(f"    WARN: partial roll fill {result.filled_qty}/{q} {sym} {m} — "
                      f"remainder stays in expiring month")
            if result.status in ("Unfilled", "Cancelled"):
                print(f"    WARN: {result.status} on ROLL {sym} {m} — no fill logged")

            placed.append((f"ROLL {act} {q} {sym} {m}", result.status, result.order_id))
            traded_instrs.add(instr)

        # ── Execute rebalance orders (blocked on a risk-gate breach) ──────────
        for om, delta in (rebalance_orders if risk_ok else []):
            act = "BUY" if delta > 0 else "SELL"
            q   = abs(delta)
            c   = qualify(ib, spec, om)
            if c is None:
                print(f"    WARNING: could not qualify {sym} {om} — skip")
                skipped.append(f"{sym} (qualify failed)")
            elif not _is_open(c):
                print(f"    DEFERRED — {sym} {om} market closed")
                skipped.append(f"{sym} (market closed)")
            else:
                ibmult = float(c.multiplier) if c.multiplier else 1.0
                ok, reason, ticker = pre_trade_checks(ib, c, sig_px, sigma, q)
                if not ok:
                    print(f"    SKIP pre-trade [{sym} {om}]: {reason}")
                    skipped.append(f"{sym} (pre-trade: {reason})")
                else:
                    result = algo_exec(ib, c, act, q, ticker)
                    ib.cancelMktData(c)

                    fp   = result.avg_price or sig_px or 0.0
                    aggr = "AGGRESSIVE" if result.was_aggressive else "passive"
                    print(f"    ORDER {act} {q} {sym} {om} → {result.status}  "
                          f"fill {fp:.4f}  comm ${result.commission:.2f}  [{aggr}]")

                    if result.status in ("Filled", "PartiallyFilled") and result.avg_price > 0:
                        ledger.log_fill(symbol=sym, contract=om, action=act,
                                        qty=result.filled_qty, multiplier=ibmult,
                                        signal_price=(sig_px or result.avg_price),
                                        fill_price=result.avg_price,
                                        commission=result.commission, forecast=fc)
                    if result.status == "PartiallyFilled":
                        print(f"    WARN: partial fill {result.filled_qty}/{q} — "
                              f"remainder will rebalance tomorrow")
                    if result.status in ("Unfilled", "Cancelled"):
                        print(f"    WARN: {result.status} — no fill logged for {sym}")

                    placed.append((f"{act} {q} {sym} {om}",
                                   result.status, result.order_id))
                    traded_instrs.add(instr)

    return placed, skipped, dry_run, traded_instrs


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _connect(client_id: int = CLIENT_ID) -> IB | None:
    """Connect to IBKR with retries. Returns IB instance or None on failure."""
    ib = IB()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ib.connect(IB_HOST, IB_PORT, clientId=client_id, timeout=CONNECT_TIMEOUT)
            ib.sleep(2)
            return ib
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"WARNING: IBKR connect {attempt}/{MAX_RETRIES} failed ({e}) — "
                      f"retry in {RETRY_DELAY}s…")
                time.sleep(RETRY_DELAY)
            else:
                print(f"ERROR: could not connect to IBKR after {MAX_RETRIES} attempts — {e}")
                print("Make sure IB Gateway / TWS is running on port 4002.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Daemon loop (Carver-style: long-running, market-hours-aware)
# ══════════════════════════════════════════════════════════════════════════════

def _touch_heartbeat():
    """
    Liveness signal for ibkr_fut/watchdog.py: write the current UTC timestamp
    to HEARTBEAT_PATH once per daemon cycle. Crash-proof by design — a failed
    heartbeat write must never kill the daemon.
    """
    try:
        with open(HEARTBEAT_PATH, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat() + "\n")
    except Exception as e:
        print(f"[{_now()}] WARNING: heartbeat write failed: {e}")


def run_daemon(args):
    """
    Long-running daemon that cycles every DAEMON_SLEEP_SECS.

    Each cycle:
      1. Reload targets_snapshot.json if the compute phase wrote a new one.
         Before switching, flush the daily ledger entry for the prior day.
      2. Reconnect to IBKR if the connection dropped.
      3. Get fresh positions and call reconcile_and_execute with skip_unchanged=True
         (suppresses "(no change)" noise) and market-open checks enabled.
      4. Sleep DAEMON_SLEEP_SECS; repeat.

    Instruments whose market is closed are printed as DEFERRED and retried next
    cycle. The daemon runs until killed; run_execution.sh manages the PID file.
    """
    ok, reason = check_halt_file()
    if not ok:
        print(f"[{_now()}] [RISK] halt file present — {reason}")
        print(f"[{_now()}] Remove ibkr_fut/risk_halt.txt to resume trading — daemon exiting")
        sys.exit(1)

    # Initial connect may land in the gateway's nightly restart window — don't
    # exit, fall through and let the loop's reconnect logic retry forever.
    ib = _connect()
    if ib is None:
        print(f"[{_now()}] WARNING: initial IBKR connect failed — "
              f"daemon will keep retrying every cycle")
    ibcfg = load_ib_config()

    ledger              = DynLedger()
    snapshot_computed_at = None
    targets = diag = capital = meta = None
    stale_alerted_for: str | None = None   # snapshot date we've already Discord-alerted as stale
    # BUG-7 churn cap: count cycles each instrument placed a live order this session.
    # Resets when a fresh snapshot loads (a new trading day's legitimate rebalances
    # should not be charged against yesterday's count).
    trade_counts: dict = {}

    print(f"[{_now()}] Daemon started (sleep={DAEMON_SLEEP_SECS}s, "
          f"execute={'YES' if args.execute else 'DRY-RUN'})")

    while True:
        # ── Heartbeat: every cycle, including the early-continue paths below
        #    (no snapshot yet / reconnect failure / stale PST) — the daemon is
        #    alive in all of those states and the watchdog must not restart it.
        _touch_heartbeat()

        # ── 0. Live kill switch: touch ibkr_fut/risk_halt.txt to stop trading ─
        ok, reason = check_halt_file()
        if not ok:
            print(f"[{_now()}] [RISK] halt file detected — {reason}")
            print(f"[{_now()}] Daemon exiting; remove ibkr_fut/risk_halt.txt "
                  f"and restart to resume trading")
            sys.exit(1)

        # ── 1. Reconnect if needed (must run before snapshot flush uses ib) ───
        if ib is None or not ib.isConnected():
            print(f"[{_now()}] IB disconnected — reconnecting…")
            ib = _connect()
            if ib is None:
                print(f"[{_now()}] Reconnect failed — sleeping 60s…")
                time.sleep(60)
                continue

        # ── 2. (Re)load snapshot if compute wrote a new one ───────────────────
        snap = _load_snapshot_any(SNAPSHOT_PATH)
        if snap is None:
            print(f"[{_now()}] No snapshot found — waiting for compute phase…")
            time.sleep(60)
            continue

        if snap.get("computed_at") != snapshot_computed_at:
            if snapshot_computed_at is not None:
                print(f"[{_now()}] New snapshot detected — flushing daily ledger…")
                try:
                    ledger.log_daily(ib,
                                     n_positions=meta.get("n_held_target") if meta else None,
                                     gross_leverage=meta.get("gross_lev") if meta else None)
                except Exception as e:
                    print(f"[{_now()}] WARNING: log_daily failed: {e}\n"
                          f"{traceback.format_exc()}")
                ledger = DynLedger()

            snapshot_computed_at = snap.get("computed_at")
            targets = snap["targets"]
            diag    = snap["diag"]
            capital = snap["capital"]
            meta    = diag.get("_meta", {})
            trade_counts = {}   # new snapshot = new target set; reset the churn cap
            # Staleness gate (calendar-aware). The snapshot's date is the last
            # PST bar it was built from, NOT wall-clock today. A CME session is
            # named by its 18:00 ET close, and the compute (run ~18:00 ET) sizes
            # off the just-settled close — so on a Sunday-evening run the freshest
            # legitimate data is the *previous Friday's* session. Comparing against
            # date.today() therefore false-alarmed every weekend/holiday run.
            #
            # Correct test: is the snapshot behind the most recent COMPLETED CME
            # session (sessions_behind > 0)? If so pst_updater/Gateway genuinely
            # failed → skip + Discord-alert (once per stale date). Otherwise trade.
            pst_date_str = meta.get("date", "unknown")
            try:
                pst_date = date.fromisoformat(pst_date_str)
                lag = sessions_behind(pst_date)
            except (ValueError, TypeError):
                pst_date = None
                lag = 1   # unparseable date → treat as stale, fail safe
            if lag > 0:
                expected = last_completed_session()
                msg = (f"snapshot PST data is from {pst_date_str}, "
                       f"{lag} session(s) behind last completed CME session "
                       f"({expected.isoformat()}) — pst_updater/Gateway may have "
                       f"failed. Skipping execution until fresh compute runs.")
                print(f"[{_now()}] WARNING: {msg}")
                if stale_alerted_for != pst_date_str:
                    try:
                        _send_discord(f"[DAEMON-STALE] {msg}")
                    except Exception as e:
                        print(f"[{_now()}] WARNING: stale Discord alert failed: {e}")
                    stale_alerted_for = pst_date_str
                snapshot_computed_at = None   # force reload next cycle
                time.sleep(60)
                continue
            stale_alerted_for = None   # snapshot is fresh; re-arm the alert
            print(f"[{_now()}] Snapshot loaded: {snapshot_computed_at}  "
                  f"capital ${capital:,.0f}  IDM {meta.get('idm')}  "
                  f"{meta.get('n_live')} live  target holds {meta.get('n_held_target')}")

        # ── 3. Fetch fresh positions ──────────────────────────────────────────
        try:
            held, unknown = get_positions_by_instr(ib, ibcfg)
        except Exception as e:
            print(f"[{_now()}] ERROR fetching positions: {e} — sleeping 60s")
            time.sleep(60)
            continue

        # ── 4. Risk gates ─────────────────────────────────────────────────────
        if args.execute:
            ok, reason = check_gross_leverage(meta.get("gross_lev", 0.0))
            if not ok:
                print(f"[{_now()}] [RISK] leverage breach — {reason} — skipping cycle")
                try:
                    ib.sleep(DAEMON_SLEEP_SECS)   # pump IB event loop, not plain time.sleep
                except Exception:
                    ib = None
                continue

            ok, reason = check_daily_loss(ib, capital)
            if not ok:
                print(f"[{_now()}] [RISK] CIRCUIT BREAKER — {reason} — halting daemon")
                sys.exit(1)

        # ── 5. Execute cycle ──────────────────────────────────────────────────
        print(f"\n[{_now()}] {'─'*60}")
        # The cycle only reaches here after the calendar-aware staleness gate
        # above (run_daemon `continue`s on lag > 0), so the snapshot is fresh.
        placed, skipped, dry_run, traded_instrs = reconcile_and_execute(
            ib, ibcfg, targets, held, diag, ledger,
            execute=args.execute, skip_unchanged=True, capital=capital,
            snapshot_fresh=True)

        market_closed = sum(1 for s in skipped if "market closed" in s)
        other_skips   = len(skipped) - market_closed
        if args.execute:
            print(f"[{_now()}] Cycle done: {len(placed)} placed  "
                  f"{market_closed} deferred (market closed)  "
                  f"{other_skips} skipped (other)")
        else:
            print(f"[{_now()}] Cycle done (DRY-RUN): {len(dry_run)} checked  "
                  f"{other_skips} skipped")

        if args.execute:
            save_last_targets(LAST_TARGETS_PATH, targets, snap["date"])

            # ── 5b. Churn circuit-breaker (BUG-7) ─────────────────────────────
            # Count cycles each instrument placed a live order this session. A
            # legitimate roll/rebalance touches an instrument a handful of times;
            # exceeding the cap means we're looping (stale positions, a roll that
            # never settles). Halt + alert rather than keep bleeding capital — the
            # exact failure that turned a clean +1 QM into a 27-lot phantom spread.
            for instr in traded_instrs:
                trade_counts[instr] = trade_counts.get(instr, 0) + 1
            runaway = {i: n for i, n in trade_counts.items()
                       if n > MAX_INSTR_TRADES_PER_SESSION}
            if runaway:
                detail = ", ".join(f"{i}×{n}" for i, n in sorted(runaway.items()))
                msg = (f"CHURN CIRCUIT-BREAKER (BUG-7): {detail} traded > "
                       f"{MAX_INSTR_TRADES_PER_SESSION}× this session — likely a "
                       f"stale-position / unsettled-roll loop. Halting daemon.")
                print(f"[{_now()}] [RISK] {msg}")
                raise_halt(msg, alert=(
                    f"[ibkr_fut] {msg}\nRemove ibkr_fut/risk_halt.txt after "
                    f"investigating + reconciling positions, then restart."))
                sys.exit(1)

        # ── 6. Sleep until next cycle ─────────────────────────────────────────
        print(f"[{_now()}] Next cycle in {DAEMON_SLEEP_SECS // 60}m…")
        try:
            ib.sleep(DAEMON_SLEEP_SECS)
        except Exception:
            ib = None   # reconnect at top of next cycle


def main():
    ap = argparse.ArgumentParser(
        description="Combined carry+trend dynamic-optimisation live executor")
    ap.add_argument("--execute", action="store_true",
                    help="Place orders (omit = dry-run)")
    ap.add_argument("--capital", type=float, default=None,
                    help="Override capital (default: live NetLiquidation)")
    ap.add_argument("--target-risk", type=float, default=DYN_TARGET_RISK,
                    help=f"Annual risk target (default {DYN_TARGET_RISK})")
    ap.add_argument("--mode", choices=["compute", "execute", "daemon"], default=None,
                    help="compute: optimise + save snapshot.  "
                         "execute: load snapshot + trade (one-shot).  "
                         "daemon: load snapshot + trade in a loop (market-hours aware).  "
                         "Default (omit): run both in sequence.")
    args = ap.parse_args()

    # Daemon mode: hand off entirely to run_daemon().
    if args.mode == "daemon":
        run_daemon(args)
        return

    today       = date.today().isoformat()
    mode_str    = "EXECUTE" if args.execute else "DRY-RUN"
    run_compute = args.mode in (None, "compute")
    run_execute = args.mode in (None, "execute")

    ib     = None   # shared connection for default (no --mode) runs
    ibcfg  = None
    targets = diag = meta = None

    # ══════════════════════════════════════════════════════════════════════════
    # COMPUTE PHASE — optimise and save snapshot
    # ══════════════════════════════════════════════════════════════════════════
    if run_compute:
        print("=" * 80)
        print(f"  COMBINED CARRY+TREND ({COMBINED_TREND_WEIGHT:.0%}/{1-COMBINED_TREND_WEIGHT:.0%}) "
              f"DYNAMIC OPT  [COMPUTE]  |  {today}  |  {mode_str}")
        print("=" * 80)

        ib = _connect(CLIENT_ID_COMPUTE)
        if ib is None:
            return
        ibcfg = load_ib_config()

        # Read held positions FIRST, on a fresh socket (BUG-8). The universe build
        # below takes ~15 min; the gateway frequently drops the API socket during
        # it, so a position fetch after the build was failing with "Socket
        # disconnect" and silently falling back to the empty cache → a snapshot
        # computed as if the book were flat → every held position stranded. Fetch
        # while the connection is new, and demand a VERIFIED read: if it fails we
        # abort WITHOUT overwriting the last good snapshot rather than trade off a
        # phantom-flat book.
        try:
            held_c, unknown_c = get_positions_by_instr(ib, ibcfg, strict=True)
        except PositionFetchError as e:
            msg = (f"compute aborting: {e}. Last good snapshot left untouched so "
                   f"the daemon keeps reconciling against a verified book.")
            print(f"[{_now()}] ERROR: {msg}")
            try:
                _send_discord(f"[COMPUTE-ABORT] {msg}")
            except Exception:
                pass
            ib.disconnect()
            return
        current_c = {instr: sum(m.values()) for instr, m in held_c.items()}

        capital = args.capital or get_equity(ib)
        if capital is None:
            print(f"WARNING: could not read equity — using ${FALLBACK_CAPITAL:,.0f}")
            capital = FALLBACK_CAPITAL
        capital = float(capital)

        print(f"\n  capital ${capital:,.0f}  |  target risk {args.target_risk:.0%}  |  "
              f"universe {len(UNIVERSE)} instr")
        if unknown_c:
            print(f"\n  ⚠ Held futures NOT in UNIVERSE: {unknown_c}")
        if current_c:
            print(f"\n  Current positions (net): "
                  + "  ".join(f"{k} {v:+d}" for k, v in sorted(current_c.items())))
        else:
            print(f"\n  Current positions: flat")

        # require_carry_fresh: the live forecast uses carry, whose term structure can go
        # stale independently of PRICE — age the CARRY leg too so a frozen multiple_prices
        # file fails the instrument out rather than sizing off months-old carry.
        tradable_set, _ = build_tradable_set(require_carry_fresh=True)
        if not tradable_set:
            print("ERROR: no tradable instruments at this capital — aborting.")
            ib.disconnect()
            return

        print(f"\n  Building universe from PST data…")
        # lookback_days=4000: blended_vol's long component is a 2520-TRADING-day rolling
        # average (≈3650 calendar days); 4000 calendar days fully covers it (with margin for
        # weekends/holidays) so live vol exactly matches the full-history backtest. The
        # EWMA covariance (25wk/32d) and the handcraft correlation (full history via pst)
        # don't depend on this window. Panel memory cost is modest — safe on the 1GB VPS.
        uni = _build_universe(list(UNIVERSE.keys()), tradable_set=tradable_set,
                              lookback_days=4000, signal_fn=_combined_signal)
        if uni is None:
            print("ERROR: universe build failed (no instruments with valid signals).")
            ib.disconnect()
            return

        targets, diag = compute_targets(uni, capital, current_c, args.target_risk)
        meta = diag.get("_meta", {})
        print(f"\n  Optimisation as of {meta.get('date')}  |  IDM {meta.get('idm')}  "
              f"|  {meta.get('n_live')} live  |  target holds {meta.get('n_held_target')}  "
              f"|  gross lev {meta.get('gross_lev')}x")

        save_snapshot(SNAPSHOT_PATH, today, capital, targets, diag)
        print(f"\n  Snapshot saved → {SNAPSHOT_PATH}")

        if args.mode == "compute":
            ib.disconnect()
            print(f"\n  Compute done. Run --mode execute --execute to place orders.")
            return

        # Default mode: keep connection open, fall through to execute phase.

    # ══════════════════════════════════════════════════════════════════════════
    # EXECUTE PHASE — pre-trade checks + passive-aggressive algo
    # ══════════════════════════════════════════════════════════════════════════
    if run_execute:
        if args.mode == "execute":
            # Separate cron run: load snapshot written by the compute job.
            snap    = load_snapshot(SNAPSHOT_PATH, today)
            targets = snap["targets"]
            diag    = snap["diag"]
            capital = snap["capital"]
            meta    = diag.get("_meta", {})

            ib = _connect()
            if ib is None:
                return
            ibcfg = load_ib_config()

        # Read current positions from IBKR (always fresh).
        held, unknown = get_positions_by_instr(ib, ibcfg)
        current = {instr: sum(m.values()) for instr, m in held.items()}

        if unknown:
            print(f"\n  ⚠ Held futures NOT in UNIVERSE (left untouched): {unknown}")
        if current:
            print(f"\n  Current positions (net): "
                  + "  ".join(f"{k} {v:+d}" for k, v in sorted(current.items())))
        else:
            print(f"\n  Current positions: flat")

        check_last_targets(LAST_TARGETS_PATH, current)

        print("\n" + "=" * 80)
        print(f"  COMBINED CARRY+TREND ({COMBINED_TREND_WEIGHT:.0%}/"
              f"{1-COMBINED_TREND_WEIGHT:.0%}) DYNAMIC OPT  [EXECUTE — {mode_str}]  |  {today}")
        print(f"  Snapshot as of {meta.get('date')}  |  capital ${capital:,.0f}  "
              f"|  IDM {meta.get('idm')}  |  {meta.get('n_live')} live  "
              f"|  target holds {meta.get('n_held_target')}  "
              f"|  gross lev {meta.get('gross_lev')}x")
        print("=" * 80)

        ledger = DynLedger()
        print("\n" + "-" * 80)
        # Snapshot is same-day fresh here: --mode execute loads via load_snapshot
        # (hard-exits if date != today); the fall-through default mode just wrote it.
        placed, skipped, dry_run, _ = reconcile_and_execute(
            ib, ibcfg, targets, held, diag, ledger, execute=args.execute,
            capital=capital, snapshot_fresh=True)

        save_last_targets(LAST_TARGETS_PATH, targets, today)

        # ── Summary ───────────────────────────────────────────────────────────
        print("\n" + "=" * 80)
        if args.execute:
            if placed:
                print("  ORDERS PLACED:")
                for desc, st, oid in placed:
                    print(f"    {desc:<32} status {st:<18} id {oid}")
            if skipped:
                print(f"  SKIPPED: {', '.join(str(s) for s in skipped)}")
            if not placed and not skipped:
                print("  NO ORDERS — already at target.")
            ib.sleep(2)
            ledger.log_daily(ib, n_positions=meta.get("n_held_target"),
                             gross_leverage=meta.get("gross_lev"))
        else:
            print("  DRY-RUN — no orders placed. Re-run with --execute to submit.")
        print("=" * 80 + "\n")

        ib.disconnect()


if __name__ == "__main__":
    main()
