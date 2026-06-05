#!/usr/bin/env python3
"""
live_dynamic.py — EWMAC + dynamic portfolio optimisation, live on IBKR.

Replaces the IBS strategy (ibkr_fut/live_signals.py). Once per day this script:

  1. Connects to IBKR (paper Gateway, port 4002) and reads NetLiquidation (capital)
     and current futures positions.
  2. Builds the dynamic-optimisation universe over Carver's full Jumbo (ibkr_fut/
     jumbo.py) from the PST CSVs — the same validated pipeline the backtest uses
     (backtest_dynamic._build_universe → ewmac forecasts, blended vol, weekly-EWMA
     covariance, handcraft weights, live-universe IDM).
  3. Runs ONE joint daily optimisation (dynamic_opt.optimise_positions) seeded with
     the *actual* held positions, restricted to a TRADABLE subset. The optimiser
     chooses which instruments to hold; everything else in the Jumbo is locked at
     its current position (0) so its risk transfers onto correlated tradables
     (Carver: optimise over ~150, trade ~100).
  4. Reconciles target vs held (handling contract rolls) and submits DAY market
     orders, logging fills + a daily snapshot to paper/ledgers/ibkr_dynamic/.

TRADABLE SET (the menu the optimiser picks from) — computed each run as the Jumbo
instruments that pass instrument_selection filters (SR cost ≤ 0.01, annual vol ≥ 5%,
history ≥ 512 days, volume if cached) AND have fresh PST data (≤ FRESH_DAYS old).
This matches the filter applied in backtest_dynamic.py --filter, so live and backtest
use an identical eligible universe. Run volume_collector.py periodically to keep the
volume cache current and enable the liquidity filter.

Sizing/forecasts use PST daily closes (kept current by pst_updater.py — run it
before this script; see ibkr_fut/run_dynamic.sh). No live quotes are needed: orders
are plain market orders.

ADDING STRATEGIES LATER: compute_targets() returns a {instrument: net_contracts}
dict. To add another strategy, produce a second such dict and merge (sum) the two
before reconcile_and_execute(). For a joint risk model you would instead extend the
optimisation universe; this netting hook is the simple path.

Dry-run (default) — print the plan, place nothing:
  python3 ibkr_fut/live_dynamic.py
Live execution:
  python3 ibkr_fut/live_dynamic.py --execute
"""

import argparse
import os
import sys
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
from ib_insync import IB, Future, Order

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pst_loader import PSTLoader
from ibkr_fut.jumbo import JUMBO
from ibkr_fut.instrument_universe import UNIVERSE
from ibkr_fut.backtest_dynamic import _build_universe, get_eligible_set
from ibkr_fut.backtest_ewmac import TARGET_RISK, IDM_CAP
from ibkr_fut.dynamic_opt import optimise_positions
from ibkr_fut.volume_collector import load_cache as load_volume_cache
from paper.dyn_ledger import DynLedger

# ── Tradable-set filter ────────────────────────────────────────────────────────
FRESH_DAYS = 5   # PST data must be no more than this many calendar days old

# ── Strategy parameters (match the backtest you validated) ─────────────────────
DYN_TARGET_RISK = TARGET_RISK   # 0.20 — same as backtest_dynamic / backtest_ewmac

# ── IBKR connection ────────────────────────────────────────────────────────────
IB_HOST         = "127.0.0.1"
IB_PORT         = 4002
CLIENT_ID       = 5
CONNECT_TIMEOUT = 5
MAX_RETRIES     = 3
RETRY_DELAY     = 10
FALLBACK_CAPITAL = 100_000.0

# Repo-relative so the same checkout works on any host (laptop, VPS).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IB_CONFIG_PATH = os.path.join(_REPO_ROOT, "Data/pst/ib_config/ib_config_futures.csv")

pst = PSTLoader()


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

def build_tradable_set(capital: float, ibcfg: pd.DataFrame,
                       verbose: bool = True) -> tuple[set, list]:
    """
    Decide which Jumbo instruments the optimiser may actually trade today.

    An instrument is tradable iff it passes instrument_selection filters (cost,
    annual vol floor, history) AND its PST data is fresh (≤ FRESH_DAYS old).
    Volume filters are skipped when no cached volume data exists.

    Returns (tradable_set, rows) where rows is a per-instrument diagnostic table.
    """
    today = pd.Timestamp(date.today())

    # Step 1: instrument_selection filters (cost, too-safe, history).
    volume_cache = load_volume_cache()
    eligible = get_eligible_set(UNIVERSE)

    # Step 2: freshness check — PST data must be recent enough to trade on.
    tradable, rows = set(), []
    for instr in UNIVERSE:
        reason = ""
        last_date = None
        try:
            mp = pst.multiple_prices(instr)["PRICE"].dropna()
            last_date = mp.index[-1]
            age = (today - last_date.normalize()).days
            if age > FRESH_DAYS:
                reason = f"stale ({last_date.date()}, {age}d old)"
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
      diag       {instrument: {forecast, n_ideal, raw_price, mult}} for held/traded
                 instruments, plus '_meta' with idm / n_live / date / gross_lev.
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
        diag["_meta"] = {"idm": 1.0, "n_live": 0, "date": as_of.date(),
                         "gross_lev": 0.0}
        # nothing live; hold everything we currently have
        for nm in names:
            if current_positions.get(nm, 0) != 0:
                targets[nm] = int(current_positions[nm])
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
                "forecast": float(fcl[k]),
                "n_ideal":  float(N_unrounded[k]),
                "raw_price": float(rl[k]),
                "mult":     float(ml[k]),
            }

    # Held instruments that aren't live today: hold them (don't blind-trade).
    for nm in names:
        if current_positions.get(nm, 0) != 0 and nm not in targets:
            targets[nm] = int(current_positions[nm])

    diag["_meta"] = {
        "idm": round(idm_t, 3), "n_live": int(live_idx.size),
        "date": as_of.date(), "gross_lev": round(gross_notional / capital, 2),
        "n_held_target": int(np.count_nonzero(N_star)),
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


def get_positions_by_instr(ib, ibcfg: pd.DataFrame) -> tuple[dict, list]:
    """
    Return ({instr: {YYYYMM: qty}}, unknown) for all held futures.
    Maps IBKR (symbol, exchange) back to a Jumbo PST instrument name.
    """
    # (symbol, exchange) -> instr, restricted to Jumbo
    rev = {}
    for instr in UNIVERSE:
        spec = ib_spec(ibcfg, instr)
        if spec:
            rev[(spec["symbol"], spec["exchange"])] = instr
            rev.setdefault((spec["symbol"],), instr)   # symbol-only fallback

    held, unknown = {}, []
    for pos in ib.positions():
        c = pos.contract
        if c.secType != "FUT":
            continue
        qty = int(pos.position)
        if qty == 0:
            continue
        instr = rev.get((c.symbol, c.primaryExchange)) or rev.get((c.symbol, c.exchange)) \
            or rev.get((c.symbol,))
        if instr is None:
            unknown.append((c.symbol, c.exchange, qty))
            continue
        month = (c.lastTradeDateOrContractMonth or "")[:6]
        held.setdefault(instr, {})[month] = held[instr].get(month, 0) + qty
    return held, unknown


def hold_contract_month(instr: str) -> str | None:
    """Contract month (YYYYMM) to hold today, from the maintained roll calendar."""
    try:
        rc = pst.roll_calendar(instr)
    except FileNotFoundError:
        return None
    today = pd.Timestamp(date.today())
    fut = rc[rc.index.normalize() >= today]
    row = fut.iloc[0] if not fut.empty else rc.iloc[-1]
    return str(int(row["current_contract"]))[:6]


def qualify(ib, spec: dict, month: str):
    """Qualify an IBKR future for (instrument spec, contract month). None on fail."""
    mult = spec.get("multiplier")
    raw = Future(
        symbol=spec["symbol"],
        lastTradeDateOrContractMonth=month,
        exchange=spec["exchange"],
        currency=spec["currency"],
        multiplier=str(int(mult)) if mult is not None else "",
        tradingClass=spec["trading_class"] or "",
    )
    try:
        quals = ib.qualifyContracts(raw)
    except Exception:
        quals = []
    return quals[0] if quals else None


def place_mkt(ib, contract, action: str, qty: int):
    order = Order(orderType="MKT", action=action, totalQuantity=qty, tif="DAY")
    return ib.placeOrder(contract, order)


# ══════════════════════════════════════════════════════════════════════════════
# Reconcile + execute
# ══════════════════════════════════════════════════════════════════════════════

def reconcile_and_execute(ib, ibcfg, targets, held, diag, ledger, execute: bool):
    """
    For each instrument with a target or current holding, roll out of old months and
    move the hold-month position to the target. Prints the plan; places orders and
    logs fills when execute=True.
    """
    pending = {t.contract.symbol for t in ib.openTrades()}
    placed, skipped = [], []

    instruments = sorted(set(targets) | set(held))
    for instr in instruments:
        spec = ib_spec(ibcfg, instr)
        if spec is None:
            print(f"  {instr}: no IB config — skipping")
            continue
        sym = spec["symbol"]
        pm  = spec["price_magnifier"]
        ibmult = spec["multiplier"] or 1.0

        target_month = hold_contract_month(instr)
        if target_month is None:
            print(f"  {instr} ({sym}): no roll calendar — skipping")
            continue

        pos_by_month = held.get(instr, {})
        qty_target   = pos_by_month.get(target_month, 0)
        old_months   = {m: q for m, q in pos_by_month.items()
                        if m != target_month and q != 0}
        net_held     = qty_target + sum(old_months.values())
        desired      = int(targets.get(instr, net_held))

        d = diag.get(instr, {})
        fc = d.get("forecast")
        n_ideal = d.get("n_ideal")
        # PST raw price → IBKR price units for slippage logging
        sig_px = (d.get("raw_price") / pm) if d.get("raw_price") else None

        fc_s = f"fcast {fc:+.1f}" if fc is not None else "fcast n/a"
        ni_s = f"ideal {n_ideal:+.2f}" if n_ideal is not None else ""
        print(f"\n  {instr:<14} {sym:<6} {target_month}  | held {net_held:+d} → "
              f"target {desired:+d}  ({fc_s} {ni_s})")
        if old_months:
            det = "  ".join(f"{m}:{q:+d}" for m, q in sorted(old_months.items()))
            print(f"    ⚠ ROLL — other months: {det}")

        # Build orders: roll-close old months, then move target month to `desired`.
        roll_closes = [("SELL" if q > 0 else "BUY", abs(q), m)
                       for m, q in old_months.items()]
        target_delta = desired - qty_target

        if target_delta == 0 and not roll_closes:
            print("    (no change)")
            continue

        if target_delta > 0:
            print(f"    ACTION: BUY  {target_delta} {sym} {target_month} [MKT]")
        elif target_delta < 0:
            print(f"    ACTION: SELL {-target_delta} {sym} {target_month} [MKT]")
        for act, q, m in roll_closes:
            print(f"    ACTION: {act} {q} {sym} {m} [ROLL CLOSE MKT]")

        if not execute:
            continue

        if sym in pending:
            print(f"    SKIPPED — open order already exists for {sym}")
            skipped.append(sym)
            continue

        # Execute roll closes first.
        for act, q, m in roll_closes:
            c = qualify(ib, spec, m)
            if c is None:
                print(f"    WARNING: could not qualify {sym} {m} — skip roll close")
                continue
            tr = place_mkt(ib, c, act, q)
            ib.sleep(2)
            st = tr.orderStatus.status
            fp = tr.orderStatus.avgFillPrice or (sig_px or 0.0)
            comm = sum(fl.commissionReport.commission for fl in tr.fills
                       if fl.commissionReport.commission == fl.commissionReport.commission)
            print(f"    ROLL {act} {q} {sym} {m} → {st}  fill {fp:.4f}")
            if st in ("Filled", "PartiallyFilled") and fp > 0:
                ledger.log_fill(symbol=sym, contract=m, action=act, qty=q,
                                multiplier=ibmult, signal_price=(sig_px or fp),
                                fill_price=fp, commission=comm, forecast=fc)
            placed.append((f"ROLL {act} {q} {sym} {m}", st, tr.order.orderId))

        # Execute target-month move.
        if target_delta != 0:
            act = "BUY" if target_delta > 0 else "SELL"
            q = abs(target_delta)
            c = qualify(ib, spec, target_month)
            if c is None:
                print(f"    WARNING: could not qualify {sym} {target_month} — skip")
            else:
                tr = place_mkt(ib, c, act, q)
                ib.sleep(2)
                st = tr.orderStatus.status
                fp = tr.orderStatus.avgFillPrice or (sig_px or 0.0)
                comm = sum(fl.commissionReport.commission for fl in tr.fills
                           if fl.commissionReport.commission == fl.commissionReport.commission)
                print(f"    ORDER {act} {q} {sym} {target_month} → {st}  "
                      f"fill {fp:.4f}  comm ${comm:.2f}")
                if st in ("Filled", "PartiallyFilled") and fp > 0:
                    ledger.log_fill(symbol=sym, contract=target_month, action=act,
                                    qty=q, multiplier=ibmult,
                                    signal_price=(sig_px or fp), fill_price=fp,
                                    commission=comm, forecast=fc)
                placed.append((f"{act} {q} {sym} {target_month}", st, tr.order.orderId))

    return placed, skipped


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="EWMAC dynamic-optimisation live executor")
    ap.add_argument("--execute", action="store_true", help="Place orders (omit = dry-run)")
    ap.add_argument("--capital", type=float, default=None,
                    help="Override capital (default: live NetLiquidation)")
    ap.add_argument("--target-risk", type=float, default=DYN_TARGET_RISK,
                    help=f"Annual risk target (default {DYN_TARGET_RISK})")
    args = ap.parse_args()

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    today = date.today().isoformat()

    # ── Connect ──────────────────────────────────────────────────────────────
    ib = IB()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=CONNECT_TIMEOUT)
            ib.sleep(2)
            break
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"WARNING: IBKR connect {attempt}/{MAX_RETRIES} failed ({e}) — "
                      f"retry in {RETRY_DELAY}s…")
                time.sleep(RETRY_DELAY)
            else:
                print(f"ERROR: could not connect to IBKR after {MAX_RETRIES} attempts — {e}")
                print("Make sure IB Gateway / TWS is running on port 4002.")
                return

    ibcfg = load_ib_config()

    equity = args.capital or get_equity(ib)
    if equity is None:
        print(f"WARNING: could not read equity — using ${FALLBACK_CAPITAL:,.0f}")
        equity = FALLBACK_CAPITAL
    capital = float(equity)

    print("=" * 80)
    print(f"  EWMAC DYNAMIC OPT  [{mode}]  |  {today}  |  capital ${capital:,.0f}")
    print(f"  target risk {args.target_risk:.0%}  |  optimise over UNIVERSE "
          f"({len(UNIVERSE)} instr), trade instrument-selection-eligible subset")
    print("=" * 80)

    # ── Tradable set + universe ──────────────────────────────────────────────
    tradable_set, _ = build_tradable_set(capital, ibcfg)
    if not tradable_set:
        print("ERROR: no tradable instruments at this capital — aborting.")
        ib.disconnect()
        return

    print(f"\n  Building universe from PST data…")
    uni = _build_universe(list(UNIVERSE.keys()), tradable_set=tradable_set,
                          lookback_days=3000)
    if uni is None:
        print("ERROR: universe build failed (no instruments with valid signals).")
        ib.disconnect()
        return

    # ── Current positions ────────────────────────────────────────────────────
    held, unknown = get_positions_by_instr(ib, ibcfg)
    current = {instr: sum(months.values()) for instr, months in held.items()}
    if unknown:
        print(f"\n  ⚠ Held futures NOT in UNIVERSE (left untouched): {unknown}")
    if current:
        print(f"\n  Current positions (net): "
              + "  ".join(f"{k} {v:+d}" for k, v in sorted(current.items())))
    else:
        print(f"\n  Current positions: flat")

    # ── Optimise ─────────────────────────────────────────────────────────────
    targets, diag = compute_targets(uni, capital, current, args.target_risk)
    meta = diag.get("_meta", {})
    print(f"\n  Optimisation as of {meta.get('date')}  |  IDM {meta.get('idm')}  "
          f"|  {meta.get('n_live')} live  |  target holds {meta.get('n_held_target')}  "
          f"|  gross lev {meta.get('gross_lev')}x")

    # ── Reconcile + execute ──────────────────────────────────────────────────
    ledger = DynLedger()
    print("\n" + "-" * 80)
    placed, skipped = reconcile_and_execute(
        ib, ibcfg, targets, held, diag, ledger, execute=args.execute)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    if args.execute:
        if placed:
            print("  ORDERS PLACED:")
            for desc, st, oid in placed:
                print(f"    {desc:<32} status {st:<18} id {oid}")
        if skipped:
            print(f"  SKIPPED (already pending): {', '.join(skipped)}")
        if not placed and not skipped:
            print("  NO ORDERS — already at target.")
        # Daily snapshot
        ib.sleep(2)
        ledger.log_daily(ib, n_positions=meta.get("n_held_target"),
                         gross_leverage=meta.get("gross_lev"))
    else:
        print("  DRY-RUN — no orders placed. Re-run with --execute to submit.")
    print("=" * 80 + "\n")

    ib.disconnect()


if __name__ == "__main__":
    main()
