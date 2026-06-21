"""
pst_updater.py

Updates pysystemtrade futures data from IBKR using correct Panama
backward adjustment. The last price in adjusted_prices_csv always
equals the actual current market close price.

Usage:
    python pst_updater.py                        # update all 252 instruments
    python pst_updater.py SP500 CRUDE_W BUND     # specific instruments
    python pst_updater.py --carry                # also fill carry prices
    python pst_updater.py --fx                   # also update FX rates
    python pst_updater.py --reset-to-pst         # truncate to original pst cutoff first

Requirements: IB Gateway running on port 4002 (paper) or 4001 (live).
"""

import os, sys, time, math, logging, argparse, glob
from datetime import date, timedelta

# Run as a top-level script (run_dynamic.sh does `python3 ibkr_fut/pst_updater.py`),
# so the repo root isn't on sys.path and `import ibkr_fut.*` would fail. Insert it
# like every other module here does, otherwise the lazy `from ibkr_fut.risk_check
# import _send_discord` in _alert_stale raises "No module named 'ibkr_fut'" and
# every PST-STALE Discord alert is silently swallowed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from ib_insync import IB, Future, Forex, util

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

_REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PST_BASE  = os.path.join(_REPO, "Data", "pst", "futures")
IB_CFG    = os.path.join(_REPO, "Data", "pst", "ib_config")
IB_HOST   = "127.0.0.1"
IB_PORT   = 4002    # IB Gateway paper; live = 4001
IB_CLIENT = 10

PST_CUTOFF = pd.Timestamp("2024-03-28")   # last date of original pst data

MONTH_MAP = dict(F=1,G=2,H=3,J=4,K=5,M=6,N=7,Q=8,U=9,V=10,X=11,Z=12)

# ── Volume-based roll (mirrors pysystemtrade's check_if_forward_liquid) ──────────
# Carver's production auto-roll defaults: roll the priced contract once the
# forward contract is "liquid" relative to it, rather than on the expiry calendar.
# Source: pst-group/pysystemtrade sysdata/config/defaults.yaml:79-88,
# sysproduction/data/volumes.py:102, sysproduction/reporting/data/rolls.py:75.
AUTO_ROLL_REL_VOL    = 1.0    # roll regardless if fwd/priced smoothed volume > this
MIN_REL_VOL          = 0.01   # else need rel > this AND abs below
MIN_ABS_VOL          = 100    # smoothed forward volume (contracts) must exceed this
VOL_IGNORE_DAYS      = 14     # ignore volume bars older than this (stale)
VOL_EWMA_SPAN        = 3      # exponential smoothing span
NOTIONALLY_ZERO_VOL  = 0.0001 # avoid div-by-zero when priced volume is ~0

# Log to the repo root (this is `~/ibkr/pst_updater.log` on the VPS, which is what
# scripts/daily_report.py reads for staleness). Resolve against _REPO so it works on
# any host without a pre-existing ~/ibkr dir; fall back to stdout-only if unopenable.
_LOG_PATH = os.path.join(_REPO, "pst_updater.log")
_handlers = [logging.StreamHandler(sys.stdout)]
try:
    _handlers.insert(0, logging.FileHandler(_LOG_PATH))
except OSError:
    pass
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=_handlers,
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Roll schedule helpers                                                #
# ------------------------------------------------------------------ #

def parse_cycle(cycle_str: str) -> list[int]:
    return sorted(MONTH_MAP[c] for c in str(cycle_str) if c in MONTH_MAP)

def next_in_cycle(year: int, month: int, cycle: list[int]) -> tuple[int, int]:
    later = [(year, m) for m in cycle if m > month]
    return later[0] if later else (year + 1, cycle[0])

def prev_in_cycle(year: int, month: int, cycle: list[int]) -> tuple[int, int]:
    earlier = [(year, m) for m in cycle if m < month]
    return earlier[-1] if earlier else (year - 1, cycle[-1])

def step_in_cycle(year: int, month: int, cycle: list[int], offset: int) -> tuple[int, int]:
    """Step offset positions forward (positive) or backward (negative) in cycle."""
    for _ in range(abs(offset)):
        if offset > 0:
            year, month = next_in_cycle(year, month, cycle)
        else:
            year, month = prev_in_cycle(year, month, cycle)
    return year, month

def expiry_date(year: int, month: int, expiry_offset: float) -> date:
    """First of contract month + expiry_offset days."""
    return date(year, month, 1) + timedelta(days=int(expiry_offset))

def roll_date(year: int, month: int, expiry_offset: float, roll_offset: float) -> date:
    return expiry_date(year, month, expiry_offset) + timedelta(days=int(roll_offset))

def build_schedule(
    last_contract: str,
    last_date: date,
    today: date,
    hold_cycle: list[int],
    expiry_offset: float,
    roll_offset: float,
) -> list[tuple[str, date, date]]:
    """Return list of (contract_yyyymm, seg_start, seg_end)."""
    year, month = int(last_contract[:4]), int(last_contract[4:6])
    segments, seg_start = [], last_date + timedelta(days=1)

    for _ in range(200):
        if seg_start > today:
            break
        r = roll_date(year, month, expiry_offset, roll_offset)
        seg_end = min(r, today)
        if seg_start <= seg_end:
            segments.append((f"{year:04d}{month:02d}", seg_start, seg_end))
            seg_start = seg_end + timedelta(days=1)
        year, month = next_in_cycle(year, month, hold_cycle)

    return segments

# ------------------------------------------------------------------ #
# IBKR helpers                                                         #
# ------------------------------------------------------------------ #

def _duration_str(start: date, end: date) -> str:
    days = (end - start).days + 5
    return f"{days} D" if days <= 365 else f"{math.ceil(days/365)} Y"

def fetch_bars(
    ib: IB,
    symbol: str,
    exchange: str,
    currency: str,
    contract_yyyymm: str,
    start: date,
    end: date,
    multiplier: str = "",
    trading_class: str = "",
    what_to_show: str = "TRADES",
    with_volume: bool = False,
):
    """Daily close bars for one contract. Returns empty Series on failure.

    By default returns just the close Series (unchanged contract). With
    ``with_volume=True`` returns ``(close, volume)`` — two index-aligned Series
    (volume in contracts, same date index as close) — used by the volume-based
    roll decision in ``update_prices``. The volume column is already present in
    the TRADES bars we fetch; we just stop discarding it.
    """
    empty = pd.Series(dtype=float)
    contract = Future(
        symbol=symbol,
        exchange=exchange,
        currency=currency or "",
        lastTradeDateOrContractMonth=contract_yyyymm,
        multiplier=multiplier,
        tradingClass=trading_class or "",
        includeExpired=True,
    )
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=f"{end:%Y%m%d} 23:59:59 UTC",
            durationStr=_duration_str(start, end),
            barSizeSetting="1 day",
            whatToShow=what_to_show,
            useRTH=False,
            formatDate=1,
            keepUpToDate=False,
        )
    except Exception as e:
        log.warning(f"    IB error ({symbol} {contract_yyyymm}): {e}")
        return (empty, empty.copy()) if with_volume else empty

    if not bars:
        return (empty, empty.copy()) if with_volume else empty

    df = util.df(bars)
    if df is None or df.empty:
        return (empty, empty.copy()) if with_volume else empty

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.set_index("date")
    df = df[pd.to_datetime(df.index).day_of_week < 5]   # drop Sat/Sun (Asian overnight sessions)
    df = df[(df.index >= start) & (df.index <= end)]

    s = df["close"].dropna()
    if not with_volume:
        return s
    # volume aligned to the (post-filter) bar index; absent or 0 → NaN
    vol = df["volume"] if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    vol = pd.to_numeric(vol, errors="coerce").replace(0, np.nan)
    return s, vol

# ------------------------------------------------------------------ #
# Volume-based roll decision (Carver's check_if_forward_liquid)        #
# ------------------------------------------------------------------ #

def smoothed_volume(vol: pd.Series, asof: date,
                    ignore_before_days: int = VOL_IGNORE_DAYS,
                    span: int = VOL_EWMA_SPAN) -> float:
    """EWMA-smoothed recent daily volume, ignoring bars older than
    ``ignore_before_days`` before ``asof``. 0.0 if no recent volume.

    Faithful port of pysystemtrade get_smoothed_volume_ignoring_old_data
    (sysproduction/data/volumes.py:102): drop stale bars, then ewm(span).mean()
    and take the last value.
    """
    if vol is None or len(vol) == 0:
        return 0.0
    v = pd.to_numeric(vol, errors="coerce").dropna()
    if v.empty:
        return 0.0
    idx = pd.to_datetime(v.index)
    cutoff = pd.Timestamp(asof) - pd.Timedelta(days=ignore_before_days)
    recent = v[idx >= cutoff]
    if recent.empty:
        return 0.0
    return float(recent.ewm(span=span).mean().iloc[-1])


def _alert_stale(instrument: str, contract: str, last_bar, seg_end: date, action: str) -> None:
    """Discord alert when the live front contract returns a stale series.

    Fires both when an early roll rescues it (informational) and when neither
    volume nor expiry could advance it (genuine data gap). Replaces the
    previously-invisible '0 failed'. Never aborts the price update on failure.
    """
    try:
        n = (seg_end - last_bar).days if last_bar is not None else None
        behind = f"{n}d behind" if n is not None else "no bars"
        msg = (f"[PST-STALE] {instrument} {contract}: last bar "
               f"{last_bar if last_bar is not None else 'none'} {behind} {seg_end} ({action})")
        log.warning("  " + msg)
        from ibkr_fut.risk_check import _send_discord
        _send_discord(msg)
    except Exception as e:
        log.warning(f"  {instrument}: staleness alert failed ({e})")


def forward_is_liquid(priced_vol: float, forward_vol: float) -> bool:
    """True when the forward contract is liquid enough to roll into, using
    Carver's rule (check_if_forward_liquid): roll if relative volume is very
    high, OR if it clears a smaller relative bar AND an absolute floor.

    relative_volume normalises the forward to the priced contract; when the
    priced contract has gone no-trade (smoothed ~0) the ratio explodes, which
    is exactly how Carver rolls out of a dying contract.
    """
    denom = priced_vol if priced_vol > 0 else NOTIONALLY_ZERO_VOL
    rel = forward_vol / denom
    if rel > AUTO_ROLL_REL_VOL:
        return True
    if rel > MIN_REL_VOL and forward_vol > MIN_ABS_VOL:
        return True
    return False


# ------------------------------------------------------------------ #
# Backward (Panama) adjustment                                         #
# ------------------------------------------------------------------ #

def compute_adjustments(
    raw_segments: list[tuple[str, str, pd.Series, pd.Series]],
) -> list[float]:
    """
    Same-day (Panama) backward adjustment.

    raw_segments: list of (contract, fwd_contract, front_bars, forward_bars).
    For each segment we hold the front contract and also carry the price of the
    forward contract (= the NEXT contract we roll into). The roll gap between
    segment k-1 and segment k is measured on a single date where BOTH legs
    trade:

        gap = forward_price - front_price     (on the latest common date)

    Measuring both legs on the *same* date isolates the pure inter-contract
    spread and removes the overnight market move that a cross-day
    (next_first - cur_last) difference would otherwise bake into the series.

    Returns a list of length len(raw_segments) + 1:
      index 0   → adjustment added to all historical (pre-new-data) prices
      index k+1 → adjustment added to raw_segments[k]'s front prices

    Method: suffix sums of roll gaps, anchored so the final segment is raw
    (adj = 0). gaps[0] = 0 because the first new segment continues the same
    contract that was the front at the cutoff (no roll there).
    """
    n = len(raw_segments)
    if n == 0:
        return [0.0]

    gaps = [0.0] * n   # gaps[k] = roll gap entering segment k (k >= 1)
    for k in range(1, n):
        front_prev, fwd_prev = raw_segments[k - 1][2], raw_segments[k - 1][3]
        gap = None
        if not front_prev.empty and not fwd_prev.empty:
            common = front_prev.index.intersection(fwd_prev.index)
            if len(common):
                d = max(common)
                gap = float(fwd_prev.loc[d]) - float(front_prev.loc[d])
        if gap is None:
            # Forward leg unavailable on a common date → fall back to the
            # cross-day difference (contaminated, but better than a jump).
            cur_front = raw_segments[k][2]
            if not front_prev.empty and not cur_front.empty:
                gap = float(cur_front.iloc[0]) - float(front_prev.iloc[-1])
                log.warning(f"    roll into seg {k}: no same-day forward, "
                            f"cross-day fallback gap {gap:+.4f}")
            else:
                gap = 0.0   # missing segment; accept price discontinuity
        gaps[k] = gap

    adjs = [0.0] * (n + 1)   # adjs[k+1] = adj for segment k; adjs[0] = historical
    for k in range(n - 1, -1, -1):
        adjs[k] = adjs[k + 1] + gaps[k]

    return adjs   # adjs[0] = historical adj, adjs[k+1] = adj for raw_segments[k]

# ------------------------------------------------------------------ #
# Per-instrument price update                                          #
# ------------------------------------------------------------------ #

def _ib_params(cfg: pd.Series) -> dict:
    mult = cfg.get("IBMultiplier", "")
    raw_curr = cfg.get("IBCurrency", "")
    # "NA" in config means no currency specified (treat as blank)
    currency = "" if (pd.isna(raw_curr) or str(raw_curr).strip().upper() == "NA") else str(raw_curr).strip()

    raw_tc = cfg.get("IBTradingClass", "")
    trading_class = "" if pd.isna(raw_tc) else str(raw_tc).strip()

    raw_hd = cfg.get("IBHistDataType", "")
    hist_data_type = "TRADES" if (pd.isna(raw_hd) or str(raw_hd).strip() == "") else str(raw_hd).strip()

    return {
        "symbol":        str(cfg["IBSymbol"]),
        "exchange":      str(cfg["IBExchange"]),
        "currency":      currency,
        "multiplier":    "" if pd.isna(mult) else str(int(float(mult))),
        "trading_class": trading_class,
        "hist_data_type": hist_data_type,
    }

def update_prices(
    ib: IB,
    instrument: str,
    ib_cfg: pd.Series,
    roll_cfg: pd.Series,
) -> None:
    """Update adjusted_prices_csv and multiple_prices_csv.

    Writes every multiple_prices column inline for each new bar — PRICE/PRICE_CONTRACT
    plus CARRY/CARRY_CONTRACT and FORWARD/FORWARD_CONTRACT (the carry contract is fetched
    per segment, see below). So carry stays fresh on every run without --carry; the
    --carry pass (update_carry) is only a legacy backstop that backfills rows left NaN.
    """

    adj_fp   = f"{PST_BASE}/adjusted_prices_csv/{instrument}.csv"
    multi_fp = f"{PST_BASE}/multiple_prices_csv/{instrument}.csv"

    if not os.path.exists(adj_fp) or not os.path.exists(multi_fp):
        log.warning(f"  {instrument}: missing CSV, skipping")
        return

    # --- Load ---
    adj = pd.read_csv(adj_fp, parse_dates=["DATETIME"], index_col="DATETIME")["price"]
    adj.index = pd.DatetimeIndex(adj.index)

    multi = pd.read_csv(multi_fp, parse_dates=["DATETIME"], index_col="DATETIME")
    multi.index = pd.DatetimeIndex(multi.index)

    adj_daily   = adj.resample("D").last().dropna()
    multi_daily = multi.resample("D").last()

    LOOKBACK_DAYS  = 3
    true_last_date = adj_daily.index[-1].date()
    today          = date.today()

    # Re-fetch the last LOOKBACK_DAYS so preliminary closes get replaced by
    # final settlement prices on subsequent runs.
    last_date = max(true_last_date - timedelta(days=LOOKBACK_DAYS), PST_CUTOFF.date())

    if last_date >= today:
        log.info(f"  {instrument}: up to date ({true_last_date})")
        return

    # --- Starting contract ---
    last_pc = multi_daily["PRICE_CONTRACT"].dropna()
    if last_pc.empty:
        log.warning(f"  {instrument}: no PRICE_CONTRACT, skipping")
        return
    last_contract = str(int(last_pc.iloc[-1]))[:6]

    # --- Roll schedule ---
    hold_cycle   = parse_cycle(str(roll_cfg.get("HoldRollCycle", "HMUZ")))
    priced_cycle = parse_cycle(str(roll_cfg.get("PricedRollCycle", "HMUZ")))
    carry_offset = int(float(roll_cfg.get("CarryOffset", 1)))
    expiry_off   = float(roll_cfg.get("ExpiryOffset", 14))
    roll_off     = float(roll_cfg.get("RollOffsetDays", -5))

    schedule = build_schedule(last_contract, last_date, today,
                               hold_cycle, expiry_off, roll_off)
    if not schedule:
        return

    log.info(f"  {instrument}: {last_date} → {today} ({len(schedule)} contracts)")

    # --- Fetch bars (front + forward + carry contract for each segment) ---
    # FORWARD = next hold-cycle contract (drives same-day roll gaps).
    # CARRY   = PricedRollCycle stepped by CarryOffset; when it coincides with
    #           the forward (offset +1, same cycle) we reuse those bars with no
    #           extra IB request, otherwise we fetch it.
    params = _ib_params(ib_cfg)
    raw_segs: list[tuple] = []
    # Mutable so an early roll on the final (live/open) segment can append a new
    # segment for the contract we roll into. early_roll records the transition so
    # the roll calendar can be advanced in lockstep after the loop.
    seg_list = list(schedule)
    early_roll: tuple[str, str, date] | None = None   # (old_contract, new_contract, roll_on)
    rolled_this_run = False
    i = 0
    while i < len(seg_list):
        contract, seg_start, seg_end = seg_list[i]
        is_final = (i == len(seg_list) - 1)
        cy_, cm_ = int(contract[:4]), int(contract[4:6])
        fy, fm = next_in_cycle(cy_, cm_, hold_cycle)
        fwd_contract = f"{fy:04d}{fm:02d}"
        ky, km = step_in_cycle(cy_, cm_, priced_cycle, carry_offset)
        carry_contract = f"{ky:04d}{km:02d}"

        # Front + forward with volume on the final segment (so the volume-roll
        # decision has both legs); historical segments need only closes.
        if is_final and not rolled_this_run:
            front, front_vol = fetch_bars(ib, params["symbol"], params["exchange"],
                                          params["currency"], contract, seg_start, seg_end,
                                          params["multiplier"], params["trading_class"],
                                          params["hist_data_type"], with_volume=True)
            time.sleep(0.6)
            forward, forward_vol = fetch_bars(ib, params["symbol"], params["exchange"],
                                              params["currency"], fwd_contract, seg_start, seg_end,
                                              params["multiplier"], params["trading_class"],
                                              params["hist_data_type"], with_volume=True)
        else:
            front = fetch_bars(ib, params["symbol"], params["exchange"],
                               params["currency"], contract, seg_start, seg_end,
                               params["multiplier"], params["trading_class"],
                               params["hist_data_type"])
            time.sleep(0.6)
            forward = fetch_bars(ib, params["symbol"], params["exchange"],
                                 params["currency"], fwd_contract, seg_start, seg_end,
                                 params["multiplier"], params["trading_class"],
                                 params["hist_data_type"])
            front_vol = forward_vol = None
        time.sleep(0.6)

        if carry_contract == fwd_contract:
            carry = forward
        elif carry_contract == contract:
            carry = front
        else:
            carry = fetch_bars(ib, params["symbol"], params["exchange"],
                               params["currency"], carry_contract, seg_start, seg_end,
                               params["multiplier"], params["trading_class"],
                               params["hist_data_type"])
            time.sleep(0.6)

        if front.empty:
            log.warning(f"    {instrument} {contract}: no front data")

        # ── Volume-based early roll (Carver) on the final/open segment ──────────
        # Roll the priced contract when the forward becomes liquid (volume rule),
        # or once the priced contract is at/past its expiry-based roll date
        # (expiry backstop, for the both-legs-no-volume case). Splice on the
        # latest common date so compute_adjustments' gap math is unchanged.
        if is_final and not rolled_this_run:
            priced_v = smoothed_volume(front_vol, today) if front_vol is not None else 0.0
            forward_v = smoothed_volume(forward_vol, today) if forward_vol is not None else 0.0
            vol_roll = (not forward.empty) and forward_is_liquid(priced_v, forward_v)
            past_roll = roll_date(cy_, cm_, expiry_off, roll_off) <= today
            front_stale = front.empty or (not front.empty
                          and (seg_end - front.index.max()).days > VOL_IGNORE_DAYS)

            if vol_roll or (past_roll and front_stale):
                common = front.index.intersection(forward.index)
                if len(common) and not forward.empty:
                    splice = max(common)
                    # Keep the front segment through the splice date; append the
                    # forward contract as a new final segment from splice+1.
                    seg_list[i] = (contract, seg_start, splice)
                    seg_end = splice
                    front_was_carry = (carry_contract == contract)
                    front_was_forward_carry = (carry_contract == fwd_contract)
                    front = front[front.index <= splice]
                    if front_was_carry:
                        carry = front          # carry leg == priced contract; re-slice
                    elif not front_was_forward_carry and carry is not None and not carry.empty:
                        carry = carry[carry.index <= splice]
                    seg_list.append((fwd_contract, splice + timedelta(days=1),
                                     min(roll_date(fy, fm, expiry_off, roll_off), today)))
                    early_roll = (contract, fwd_contract, splice)
                    rolled_this_run = True
                    reason = "volume-rolled" if vol_roll else "expiry-rolled"
                    log.info(f"    {instrument}: early roll {contract}→{fwd_contract} "
                             f"@ {splice} ({reason}; priced_v={priced_v:.0f} fwd_v={forward_v:.0f})")
                    _alert_stale(instrument, contract,
                                 front.index.max() if not front.empty else None,
                                 schedule[-1][2], reason)
                elif front_stale:
                    # Can't splice (forward has no overlapping/any data): genuine gap.
                    _alert_stale(instrument, contract,
                                 front.index.max() if not front.empty else None,
                                 seg_end, "no-roll: forward illiquid too")
            elif front_stale:
                _alert_stale(instrument, contract,
                             front.index.max() if not front.empty else None,
                             seg_end, "no-roll: forward not yet liquid")

        raw_segs.append((contract, fwd_contract, front, forward, carry_contract, carry))
        i += 1

    # Check if anything was fetched
    if all(seg[2].empty for seg in raw_segs):
        log.warning(f"  {instrument}: no data fetched")
        return

    # --- Compute backward adjustments (same-day roll gaps) ---
    adjs = compute_adjustments(raw_segs)
    hist_adj = adjs[0]

    # --- Build new adjusted and raw price series ---
    new_adj_rows   = {}
    new_multi_rows = {}

    for k, (contract, fwd_contract, front, forward, carry_contract, carry) in enumerate(raw_segs):
        seg_adj = adjs[k + 1]
        for dt, raw_price in front.items():
            fwd_price   = float(forward.loc[dt]) if dt in forward.index else np.nan
            carry_price = float(carry.loc[dt])   if dt in carry.index   else np.nan
            new_adj_rows[dt]   = float(raw_price) + seg_adj
            new_multi_rows[dt] = {
                "PRICE":          float(raw_price),
                "PRICE_CONTRACT": int(contract + "00"),
                "CARRY":          carry_price,
                "CARRY_CONTRACT": int(carry_contract + "00"),
                "FORWARD":        fwd_price,
                "FORWARD_CONTRACT": int(fwd_contract + "00"),
            }

    if not new_adj_rows:
        log.warning(f"  {instrument}: no new bars to save")
        return

    # --- Write adjusted prices ---
    # Trim the lookback window from existing data so re-fetched rows
    # (with final settlement prices) replace any preliminary closes.
    rewrite_history = abs(hist_adj) > 1e-9
    # build_schedule starts at last_date+1, so the first re-fetched row is last_date+1.
    # Trim existing data to strictly before that point so there are no gaps or overlaps.
    lookback_cutoff = pd.Timestamp(last_date + timedelta(days=1)).normalize()

    new_adj = pd.Series(new_adj_rows, name="price")
    new_adj.index = (pd.DatetimeIndex(pd.to_datetime(list(new_adj_rows.keys())))
                     .normalize() + pd.Timedelta(hours=23))
    new_adj.index.name = "DATETIME"

    adj_base = adj[adj.index.normalize() < lookback_cutoff]
    if rewrite_history:
        combined_adj = pd.concat([adj_base + hist_adj, new_adj]).sort_index()
    else:
        combined_adj = pd.concat([adj_base, new_adj]).sort_index()
    combined_adj.to_csv(adj_fp, header=True)

    # --- Write multiple prices ---
    new_multi = pd.DataFrame.from_dict(new_multi_rows, orient="index")
    new_multi.index = (pd.DatetimeIndex(pd.to_datetime(list(new_multi_rows.keys())))
                       .normalize() + pd.Timedelta(hours=23))
    new_multi.index.name = "DATETIME"
    new_multi = new_multi[list(multi.columns)]   # match existing column order
    multi_base = multi[multi.index.normalize() < lookback_cutoff]
    combined_multi = pd.concat([multi_base, new_multi]).sort_index()
    combined_multi.to_csv(multi_fp, header=True)

    log.info(f"  {instrument}: {last_date} → {today} | {len(new_adj_rows)} bars "
             f"| hist adj {hist_adj:+.4f} ({'roll rewrite' if rewrite_history else 'lookback rewrite'}) "
             f"| last price {new_adj.iloc[-1]:.4f}")

    # --- Advance roll calendar in lockstep with an early (volume) roll ---
    # Done after the price CSVs are written so the traded contract that
    # live_dynamic.get_roll_info reads matches the priced series. The normal
    # extend_roll_calendar pass (in main) then rebuilds the forward schedule
    # from this corrected pointer.
    if early_roll is not None:
        old_c, new_c, roll_on = early_roll
        advance_roll_calendar_to(instrument, old_c, new_c, roll_on, roll_cfg)

# ------------------------------------------------------------------ #
# Carry prices                                                         #
# ------------------------------------------------------------------ #

def update_carry(
    ib: IB,
    instrument: str,
    ib_cfg: pd.Series,
    roll_cfg: pd.Series,
) -> None:
    """Fill CARRY and CARRY_CONTRACT columns in multiple_prices_csv."""

    multi_fp = f"{PST_BASE}/multiple_prices_csv/{instrument}.csv"
    multi = pd.read_csv(multi_fp, parse_dates=["DATETIME"], index_col="DATETIME")
    multi.index = pd.DatetimeIndex(multi.index)

    # Find post-cutoff rows that have PRICE_CONTRACT but missing CARRY.
    # We only manage data we appended (> PST_CUTOFF); pre-existing gaps in the
    # original pst history belong to that source and often reference contracts
    # IBKR no longer serves, so we never try to backfill them. NOTE: carry is
    # now filled inline by update_prices; this pass is a legacy backstop.
    daily = multi.resample("D").last()
    need_carry = daily[daily["PRICE_CONTRACT"].notna() & daily["CARRY"].isna()]
    need_carry = need_carry[need_carry.index > PST_CUTOFF]

    if need_carry.empty:
        log.info(f"  {instrument} carry: already complete")
        return

    priced_cycle  = parse_cycle(str(roll_cfg.get("PricedRollCycle", "HMUZ")))
    carry_offset  = int(float(roll_cfg.get("CarryOffset", 1)))
    params        = _ib_params(ib_cfg)

    # Group by PRICE_CONTRACT and fetch carry contract for each period
    carry_updates: dict[pd.Timestamp, dict] = {}
    last_carry_contract = None
    last_carry_bars: pd.Series = pd.Series(dtype=float)

    for pc_raw, group in need_carry.groupby("PRICE_CONTRACT"):
        price_contract = str(int(pc_raw))[:6]
        py = int(price_contract[:4])
        pm = int(price_contract[4:6])

        cy, cm = step_in_cycle(py, pm, priced_cycle, carry_offset)
        carry_contract = f"{cy:04d}{cm:02d}"

        dates = group.index
        seg_start = dates[0].date()
        seg_end   = dates[-1].date()

        # Fetch carry contract bars (reuse if same contract)
        if carry_contract != last_carry_contract:
            last_carry_bars = fetch_bars(
                ib, params["symbol"], params["exchange"], params["currency"],
                carry_contract, seg_start, seg_end, params["multiplier"],
                params["trading_class"], params["hist_data_type"])
            time.sleep(0.6)
            last_carry_contract = carry_contract

        for dt_idx in dates:
            dt = dt_idx.date()
            carry_price = (float(last_carry_bars.loc[dt])
                           if dt in last_carry_bars.index else np.nan)
            carry_updates[dt_idx.normalize() + pd.Timedelta(hours=23)] = {
                "CARRY":          carry_price,
                "CARRY_CONTRACT": int(carry_contract + "00") if not np.isnan(carry_price) else np.nan,
            }

    if not carry_updates:
        return

    update_df = pd.DataFrame.from_dict(carry_updates, orient="index")
    update_df.index.name = "DATETIME"

    for col in ["CARRY", "CARRY_CONTRACT"]:
        if col in update_df:
            # Only overwrite rows that are currently NaN
            mask = multi[col].isna() & multi.index.isin(update_df.index)
            multi.loc[mask, col] = update_df.loc[multi.index[mask], col]

    multi.to_csv(multi_fp, header=True)
    filled = update_df["CARRY"].notna().sum()
    log.info(f"  {instrument} carry: filled {filled} bars with {carry_contract}")

# ------------------------------------------------------------------ #
# Roll calendar extension                                              #
# ------------------------------------------------------------------ #

def create_roll_calendar_from_history(instrument: str, roll_cfg: pd.Series) -> None:
    """
    Bootstrap a roll_calendars_csv for instruments that have multiple_prices data
    (with PRICE_CONTRACT transitions) but no existing roll calendar file.
    Reads the historical PRICE_CONTRACT transitions from multiple_prices_csv,
    derives the current/next/carry contracts, writes the initial CSV, then
    extend_roll_calendar will top it up to today+2yr.
    """
    rc_fp   = f"{PST_BASE}/roll_calendars_csv/{instrument}.csv"
    multi_fp = f"{PST_BASE}/multiple_prices_csv/{instrument}.csv"
    if not os.path.exists(multi_fp):
        log.warning(f"  {instrument}: no multiple_prices, cannot bootstrap roll calendar")
        return

    multi = pd.read_csv(multi_fp, parse_dates=["DATETIME"], index_col="DATETIME")
    multi.index = pd.DatetimeIndex(multi.index)
    daily = multi.resample("D").last()
    pc = daily["PRICE_CONTRACT"].dropna()

    if pc.empty:
        return

    priced_cycle = parse_cycle(str(roll_cfg.get("PricedRollCycle", "HMUZ")))
    hold_cycle   = parse_cycle(str(roll_cfg.get("HoldRollCycle", "HMUZ")))
    carry_offset = int(float(roll_cfg.get("CarryOffset", 1)))

    # Each row where PRICE_CONTRACT changes is a roll date
    transitions = pc[pc != pc.shift(1)].dropna()
    if transitions.empty:
        return

    rows = []
    prev_contract_raw = None
    for roll_dt, new_contract_raw in transitions.items():
        new_contract = str(int(new_contract_raw))[:6]
        ny, nm = int(new_contract[:4]), int(new_contract[4:6])

        if prev_contract_raw is not None:
            cur_contract = str(int(prev_contract_raw))[:6]
        else:
            # For the very first transition, the current contract is the one before next
            cur_y, cur_m = prev_in_cycle(ny, nm, hold_cycle)
            cur_contract = f"{cur_y:04d}{cur_m:02d}"

        cy, cm = step_in_cycle(ny, nm, priced_cycle, carry_offset)
        carry_contract = f"{cy:04d}{cm:02d}"

        rows.append({
            "DATE_TIME":        roll_dt.normalize() + pd.Timedelta(hours=20),
            "current_contract": int(cur_contract + "00"),
            "next_contract":    int(new_contract + "00"),
            "carry_contract":   int(carry_contract + "00"),
        })
        prev_contract_raw = new_contract_raw

    rc_df = pd.DataFrame(rows).set_index("DATE_TIME")
    rc_df.index.name = "DATE_TIME"
    rc_df = rc_df[~rc_df.index.duplicated(keep="last")].sort_index()
    rc_df.to_csv(rc_fp, header=True)
    log.info(f"  {instrument}: bootstrapped roll calendar from history ({len(rows)} rows)")


def advance_roll_calendar_to(instrument: str, old_current_yyyymm: str,
                             new_current_yyyymm: str, roll_on: date,
                             roll_cfg: pd.Series) -> None:
    """Record an early (volume-driven) roll in roll_calendars_csv so the traded
    contract advances in lockstep with the priced series.

    Calendar semantics (see live_dynamic.get_roll_info): a row's DATE_TIME is the
    last day to hold its ``current_contract``; the *following* row's
    ``current_contract`` becomes active the day after. So to roll *out of*
    ``old_current`` on ``roll_on`` and *into* ``new_current``, we:
      - set the row covering ``roll_on`` to (DATE_TIME=roll_on,
        current_contract=old_current) — last day on the old contract, and
      - ensure the next row has current_contract=new_current.
    Future rows are dropped so the normal extend_roll_calendar pass rebuilds the
    forward schedule from the corrected pointer.
    """
    rc_fp = f"{PST_BASE}/roll_calendars_csv/{instrument}.csv"
    if not os.path.exists(rc_fp):
        create_roll_calendar_from_history(instrument, roll_cfg)
        if not os.path.exists(rc_fp):
            log.warning(f"  {instrument}: no roll calendar to advance")
            return

    rc = pd.read_csv(rc_fp, parse_dates=["DATE_TIME"], index_col="DATE_TIME")
    rc.index = pd.DatetimeIndex(rc.index)

    hold_cycle   = parse_cycle(str(roll_cfg.get("HoldRollCycle", "HMUZ")))
    priced_cycle = parse_cycle(str(roll_cfg.get("PricedRollCycle", "HMUZ")))
    carry_offset = int(float(roll_cfg.get("CarryOffset", 1)))

    oy, om = int(old_current_yyyymm[:4]), int(old_current_yyyymm[4:6])
    ny, nm = int(new_current_yyyymm[:4]), int(new_current_yyyymm[4:6])
    # NB: pd.Timestamp(date_obj, hour=20) silently ignores hour=; build at 20:00
    # explicitly so the row matches the rest of the calendar's convention.
    roll_ts = pd.Timestamp(roll_on).replace(hour=20)

    # next/carry for the OLD contract's closing row (matches extend_roll_calendar)
    o_ny, o_nm = next_in_cycle(oy, om, hold_cycle)
    o_cy, o_cm = step_in_cycle(oy, om, priced_cycle, carry_offset)
    old_row = {
        "current_contract": int(f"{oy:04d}{om:02d}00"),
        "next_contract":    int(f"{o_ny:04d}{o_nm:02d}00"),
        "carry_contract":   int(f"{o_cy:04d}{o_cm:02d}00"),
    }

    # Drop any rows on/after the roll date, then append the corrected closing row.
    # extend_roll_calendar (run right after) seeds from this row's next_contract
    # (= new_current) and rebuilds the forward schedule.
    rc = rc[rc.index < roll_ts]
    new_df = pd.DataFrame([{"DATE_TIME": roll_ts, **old_row}]).set_index("DATE_TIME")
    new_df.index.name = "DATE_TIME"
    combined = pd.concat([rc, new_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined[list(rc.columns)] if len(rc.columns) else combined
    combined.to_csv(rc_fp, header=True)
    log.info(f"  {instrument} roll cal: early-roll {old_current_yyyymm}→{new_current_yyyymm} "
             f"on {roll_on} (last-hold row written)")


def extend_roll_calendar(instrument: str, roll_cfg: pd.Series) -> None:
    """
    Extend roll_calendars_csv forward from last entry to today+2yr.
    If the file does not exist yet, bootstrap it from multiple_prices history first.
    Pure computation — no IBKR needed.
    """
    rc_fp = f"{PST_BASE}/roll_calendars_csv/{instrument}.csv"
    if not os.path.exists(rc_fp):
        create_roll_calendar_from_history(instrument, roll_cfg)
        if not os.path.exists(rc_fp):
            return

    rc = pd.read_csv(rc_fp, parse_dates=["DATE_TIME"], index_col="DATE_TIME")
    rc.index = pd.DatetimeIndex(rc.index)

    if rc.empty:
        return

    last_row    = rc.iloc[-1]
    last_rc_date = rc.index[-1].date()
    horizon     = date.today() + timedelta(days=730)   # 2 years ahead

    hold_cycle  = parse_cycle(str(roll_cfg.get("HoldRollCycle", "HMUZ")))
    priced_cycle = parse_cycle(str(roll_cfg.get("PricedRollCycle", "HMUZ")))
    carry_offset = int(float(roll_cfg.get("CarryOffset", 1)))
    expiry_off  = float(roll_cfg.get("ExpiryOffset", 14))
    roll_off    = float(roll_cfg.get("RollOffsetDays", -5))

    current_contract_raw = str(int(last_row["next_contract"]))[:6]
    year, month = int(current_contract_raw[:4]), int(current_contract_raw[4:6])

    new_rows = []
    while True:
        r = roll_date(year, month, expiry_off, roll_off)
        if r > horizon:
            break
        if r > last_rc_date:
            ny, nm = next_in_cycle(year, month, hold_cycle)
            cy, cm = step_in_cycle(year, month, priced_cycle, carry_offset)
            new_rows.append({
                "DATE_TIME":        pd.Timestamp(r, hour=20),
                "current_contract": int(f"{year:04d}{month:02d}00"),
                "next_contract":    int(f"{ny:04d}{nm:02d}00"),
                "carry_contract":   int(f"{cy:04d}{cm:02d}00"),
            })
        year, month = next_in_cycle(year, month, hold_cycle)

    if not new_rows:
        return

    new_df = pd.DataFrame(new_rows).set_index("DATE_TIME")
    new_df.index.name = "DATE_TIME"
    combined = pd.concat([rc, new_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.to_csv(rc_fp, header=True)
    log.info(f"  {instrument} roll cal: +{len(new_rows)} rows → {new_rows[-1]['DATE_TIME'].date()}")

# ------------------------------------------------------------------ #
# FX prices                                                            #
# ------------------------------------------------------------------ #

def update_fx_prices(ib: IB) -> None:
    """
    Update FX rate CSVs from IBKR.

    Config (ib_config_spot_FX.csv) columns: CODE, CCY1, CCY2, INVERT.
      CODE   - file name / pst rate name, always "<CCY>USD" (USD value of 1 CCY)
      CCY1,2 - the IBKR forex pair to request (Forex(CCY1+CCY2))
      INVERT - "YES" when the pair is quoted the other way (e.g. USDJPY) and the
               stored rate must be 1/price to express it as CCY-per-USD inverted
               to USD-per-CCY.

    Files store the USD value of one unit of the foreign currency (e.g. EURUSD
    ~1.08, JPYUSD ~0.0066). Append-only: history is never re-serialised.
    """
    fx_cfg = pd.read_csv(f"{IB_CFG}/ib_config_spot_FX.csv")
    today  = date.today()

    for _, row in fx_cfg.iterrows():
        code   = str(row["CODE"]).strip()       # e.g. "JPYUSD" → JPYUSD.csv
        ccy1   = str(row["CCY1"]).strip()
        ccy2   = str(row["CCY2"]).strip()
        invert = str(row["INVERT"]).strip().upper() == "YES"
        fp = f"{PST_BASE}/fx_prices_csv/{code}.csv"

        if not os.path.exists(fp):
            log.warning(f"  FX {code}: no CSV, skipping")
            continue

        fx = pd.read_csv(fp, parse_dates=["DATETIME"], index_col="DATETIME")
        fx.index = pd.DatetimeIndex(fx.index)
        last_date = fx.index[-1].date()

        if last_date >= today:
            log.info(f"  FX {code}: up to date ({last_date})")
            continue

        contract = Forex(ccy1 + ccy2)
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=f"{today:%Y%m%d} 23:59:59 UTC",
                durationStr=_duration_str(last_date, today),
                barSizeSetting="1 day",
                whatToShow="MIDPOINT",
                useRTH=False,
                formatDate=1,
                keepUpToDate=False,
            )
        except Exception as e:
            log.warning(f"  FX {code}: {e}")
            time.sleep(0.6)
            continue

        time.sleep(0.6)
        if not bars:
            log.warning(f"  FX {code}: no data")
            continue

        df = util.df(bars)
        if df is None or df.empty:
            continue

        df["DATETIME"] = (pd.to_datetime(df["date"]).dt.normalize()
                          + pd.Timedelta(hours=23))
        close = df.set_index("DATETIME")["close"].astype(float)
        new_fx = (1.0 / close if invert else close).rename("PRICE")
        new_fx = new_fx[new_fx.index > fx.index[-1]].dropna()

        if new_fx.empty:
            continue

        new_fx.to_csv(fp, mode="a", header=False)
        log.info(f"  FX {code}: +{len(new_fx)} bars{' (inv)' if invert else ''} "
                 f"→ {new_fx.index[-1].date()} = {new_fx.iloc[-1]:.6g}")

# ------------------------------------------------------------------ #
# Reset to original pst cutoff                                         #
# ------------------------------------------------------------------ #

def reset_to_pst_cutoff(instruments: list[str]) -> None:
    """Truncate adjusted and multiple prices back to the original pst cutoff."""
    log.info(f"Resetting {len(instruments)} instruments to {PST_CUTOFF.date()} ...")
    for instrument in instruments:
        for subdir in ("adjusted_prices_csv", "multiple_prices_csv"):
            fp = f"{PST_BASE}/{subdir}/{instrument}.csv"
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp, parse_dates=[0], index_col=0)
            df.index = pd.DatetimeIndex(df.index)
            trimmed = df[df.index <= PST_CUTOFF + pd.Timedelta(hours=23)]
            trimmed.to_csv(fp, header=True)
        log.info(f"  {instrument}: reset to {PST_CUTOFF.date()}")

# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def load_configs():
    cfg     = f"{PST_BASE}/csvconfig"
    ib_cfg  = pd.read_csv(f"{IB_CFG}/ib_config_futures.csv", index_col="Instrument")
    roll_cfg = pd.read_csv(f"{cfg}/rollconfig.csv", index_col="Instrument")
    available = set(os.path.basename(f).replace(".csv", "")
                    for f in glob.glob(f"{PST_BASE}/adjusted_prices_csv/*.csv"))
    return ib_cfg, roll_cfg, available


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instruments", nargs="*")
    parser.add_argument("--carry",        action="store_true", help="Also fill carry prices")
    parser.add_argument("--fx",           action="store_true", help="Also update FX rates")
    parser.add_argument("--reset-to-pst", action="store_true", help="Truncate to original pst cutoff first")
    args = parser.parse_args()

    ib_cfg, roll_cfg, available = load_configs()
    target = args.instruments if args.instruments else sorted(available)
    target = [i for i in target if i in ib_cfg.index and i in roll_cfg.index]

    if args.reset_to_pst:
        reset_to_pst_cutoff(target)

    log.info(f"Connecting to {IB_HOST}:{IB_PORT} ...")
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT)
    log.info("Connected.")

    ok, failed = 0, []

    for i, instrument in enumerate(target):
        log.info(f"[{i+1}/{len(target)}] {instrument}")
        try:
            update_prices(ib, instrument, ib_cfg.loc[instrument], roll_cfg.loc[instrument])
            if args.carry:
                update_carry(ib, instrument, ib_cfg.loc[instrument], roll_cfg.loc[instrument])
            extend_roll_calendar(instrument, roll_cfg.loc[instrument])
            ok += 1
        except Exception as exc:
            log.error(f"  {instrument}: FAILED — {exc}")
            failed.append(instrument)

    if args.fx:
        log.info("Updating FX rates ...")
        update_fx_prices(ib)

    ib.disconnect()
    log.info(f"Done. {ok} updated, {len(failed)} failed.")
    if failed:
        log.warning(f"Failed: {failed}")


if __name__ == "__main__":
    main()
