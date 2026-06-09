# System State — Futures EWMAC Dynamic Optimisation (ibkr_fut)

*Reviewed 2026-06-09. Covers `live_dynamic.py`, `run_dynamic.sh`, `run_execution.sh`,
`algo_execution.py`, `risk_check.py`, and the `test_execution.py` suite (67 tests, all passing).*

---

## 1. What the system does

Once per day the system reads live capital and positions from IBKR (paper Gateway,
port 4002), builds Carver's full Jumbo universe from PST CSV data, runs one joint
dynamic portfolio optimisation (`dynamic_opt.optimise_positions`) seeded with the
actual held positions, and reconciles target vs held — placing orders via Carver's
passive-aggressive limit-order algorithm. Fills and a daily snapshot go to the
ledger at `paper/ledgers/ibkr_dynamic/` via `paper.dyn_ledger.DynLedger`.

The compute and execute phases are decoupled through a JSON snapshot file
(`ibkr_fut/targets_snapshot.json`), so a single nightly optimisation can feed a
long-running execution daemon that trades each instrument when its exchange opens
(CME ~6:05 PM ET, Asia ~8–9 PM ET, Eurex ~2–3 AM ET).

## 2. Daily schedule (cron, America/New_York)

| Time (ET) | Job | What it does |
|---|---|---|
| 6:00 PM weekdays | `run_dynamic.sh` | pst_updater (prices+FX) → volume_collector (weekly) → `live_dynamic.py --mode compute` |
| ~10:05 PM (per script header) | `run_execution.sh` | kills old daemon, starts `live_dynamic.py --mode daemon` in background |

**⚠ Current state:** `run_execution.sh:61` starts the daemon **without `--execute`**
("add --execute when market data active"), so the daemon is currently in dry-run —
it qualifies contracts and prints plans but places no orders.

## 3. Modes of `live_dynamic.py`

- **`--mode compute`** (clientId 6): connects to IBKR, reads NetLiquidation,
  builds the tradable set + universe, runs `compute_targets()`, atomically writes
  `targets_snapshot.json` (`date`, `computed_at`, `capital`, `targets`, `diag`).
  Places no orders.
- **`--mode execute`** (clientId 5, one-shot): loads today's snapshot (aborts if
  missing or stale-dated), reads fresh positions, runs `reconcile_and_execute()`
  once, writes `last_targets.json`, logs the daily ledger row.
- **`--mode daemon`** (clientId 5, long-running): loop every 600 s. Each cycle:
  reconnect if needed → reload snapshot if `computed_at` changed (flushing the
  prior day's ledger first) → refuse snapshots whose PST data date ≠ today →
  fetch positions → risk gates → `reconcile_and_execute(skip_unchanged=True)`.
  Deferred (market-closed) instruments retry next cycle.
- **No `--mode`**: compute then execute in one session (manual testing). Dry-run
  unless `--execute` is passed.

## 4. Compute phase pipeline

1. **Tradable set** (`build_tradable_set`): Jumbo instrument is tradable iff it
   passes `instrument_selection` filters (SR cost ≤ 0.01, annual vol ≥ 5%,
   history ≥ 512 days, volume if cached) **and** its PST data is ≤ `FRESH_DAYS=5`
   calendar days old. Matches the backtest's `--filter` so live and backtest use
   an identical eligible universe.
2. **Universe build**: `backtest_dynamic._build_universe` over the full Jumbo
   (3000-day lookback) — EWMAC forecasts, blended vol, weekly-EWMA covariance,
   handcraft weights. Non-tradable instruments stay in the risk model but are
   locked at their current position (Carver: optimise over ~150, trade ~100).
3. **Targets** (`compute_targets`): latest valid row; live-universe renormalised
   weights and IDM (capped at `IDM_CAP`); unrounded positions from
   `forecast · capital · IDM · w · τ / (10 · mult · price · fx · σ)`; then
   `optimise_positions` with costs + buffering, seeded with held positions.
   Target risk τ = 0.20 (same as the validated backtest). Held instruments that
   aren't live today are held as-is (no blind trading). Returns `targets`
   (net contracts) and `diag` (forecast, n_ideal, raw_price, mult, fx, sigma per
   instrument, plus `_meta`: idm / n_live / date / gross_lev / n_held_target).

## 5. Reconcile & execute (`reconcile_and_execute`)

For each instrument in `targets ∪ held`:

1. **Config / roll calendar**: skip if no IB config row or no roll calendar.
2. **Pending-order guard**: skip the instrument if any open IBKR trade exists on
   the same symbol (prevents conflicting orders across daemon cycles).
3. **Roll phases** from the PST roll calendar (`get_roll_info`):
   - **Passive window** (3 < days_to_roll ≤ 10): rebalance *adds* are routed to
     the next month; *reductions* come out of the expiring month first.
   - **Spread window** (0 ≤ days_to_roll ≤ 3): the portion of the expiring-month
     position that must survive is force-rolled via a BAG calendar-spread order
     (`spread_roll_exec`): passive limit at spread mid for 60 s, then market.
     Contracts being closed close directly in the expiring month (1 crossing, not 2).
   - **Old months** (anything outside the expected set): closed outright.
4. **Risk gate**: `check_order_vol` on the net *target* (|qty|·mult·price·fx·σ
   ≤ 50% of capital) — sized on the same USD basis as the optimiser.
5. **Market-hours check** (`is_contract_okay_to_trade`, cached per conId per
   cycle): closed markets are DEFERRED, retried next daemon cycle.
6. **Pre-trade checks** (`pre_trade_checks`): valid bid/ask within 10 s; live mid
   within 3 daily SDs (3σ/√256) of the PST close used for sizing.
7. **Execution** (`algo_exec`, Carver's algorithm): passive limit at the offside
   price (bid when buying); switch to aggressive (chase the inside quote) on
   passive timeout (300 s), adverse price move, or 5× book imbalance; cancel at
   600 s total. Fills (incl. partials) logged to the ledger with commission.

Dry-run mode prints the full plan and qualifies contracts but calls neither the
risk gate, pre-trade checks, nor the algo (verified by tests).

## 6. Risk controls (`risk_check.py`)

| Gate | Threshold | Action on breach |
|---|---|---|
| Halt file (`ibkr_fut/risk_halt.txt`) | exists | daemon and `run_execution.sh` refuse to start |
| `check_gross_leverage` | gross lev > 10× (from compute `_meta`) | skip the daemon cycle |
| `check_order_vol` | single-instrument annual $ vol > 50% of capital | skip that instrument |
| `check_daily_loss` | equity > 8% below compute-time capital | write halt file, Discord alert, `sys.exit(1)` |

Plus operational safeguards: atomic snapshot writes (`.tmp` + `os.replace`),
snapshot date validation, `last_targets.json` reconciliation warnings at the next
run, unknown held futures left untouched, stale-PST refusal in the daemon, and
PID-file management with force-kill in `run_execution.sh`.

## 7. Test coverage (`test_execution.py`, 67 passing)

- Pure helpers (`_valid`, `_sz`, adverse price/size detection).
- `pre_trade_checks`: bid/ask validity, crossed markets, price-divergence
  threshold, market-data cleanup on failure.
- `algo_exec`: passive fill, partial fill, all three aggressive triggers, total
  timeout + cancel wait, NaN commission filtering, limit re-pegging.
- Snapshot I/O: roundtrip, atomicity, numpy/date serialisation, stale-date abort.
- `reconcile_and_execute` guardrails: dry-run places nothing, pending-order skip,
  pre-trade failure skip, ledger logging on fill/partial, no logging on
  unfilled/cancelled, missing config/roll-calendar skips.

Not covered: roll-window routing logic (passive/spread phases), `spread_roll_exec`,
the risk gates inside `reconcile_and_execute`, the daemon loop, and
`is_contract_okay_to_trade`.

---

## 8. Issues found in review

### Bugs

1. **Roll-window reductions can over-sell the expiring month**
   (`live_dynamic.py:700-713`). During a passive/spread window, a negative
   `target_delta` is routed entirely to `current_month` whenever
   `qty_current != 0`, without capping at what that month holds. Example:
   `qty_current=+1`, `qty_next=+2`, target `0` → delta −3 is all sent to the
   expiring month, leaving it −2 and the next month +2. The reduction needs to be
   split across months (close the expiring leg up to `qty_current`, the rest from
   `next_month`).

2. **Partial spread-roll fills lost when the limit is cancelled**
   (`live_dynamic.py:688-694` + `spread_roll_exec`). ib_insync reports a
   partially-filled-then-cancelled order with status `"Cancelled"` and
   `filled > 0`. The caller only credits fills when status is
   `"Filled"`/`"PartiallyFilled"`, so the cycle's local `qty_current`/`qty_next`
   accounting ignores contracts that actually rolled, and the subsequent
   rebalance order in the *same* cycle is sized off stale numbers. (The next
   daemon cycle self-corrects from fresh positions, but the mis-sized order may
   already be live.) Fix: act on `filled > 0` regardless of status. Related: when
   the market-order fallback fires, `trade` is reassigned, so the returned
   `filled` counts only the market order's fills, dropping any limit-leg partials
   from the reported total.

3. **Spread roll bypasses the market-hours and pre-trade checks**
   (`live_dynamic.py:687`). `spread_roll_exec` is called before `_is_open` and
   `pre_trade_checks`; only outright orders get those gates. A BAG order can be
   submitted to a closed exchange (rejected or queued) and has no price-sanity
   check beyond its own quote-validity test.

4. **Risk-gate skip also blocks old-month roll closes**
   (`live_dynamic.py:771-776`). The comment claims "position-reducing
   rolls/closes are never blocked", but on a `check_order_vol` breach the
   `continue` skips the `roll_closes` loop too, stranding risk-*reducing* closes
   in expired months. The gate should only block the rebalance order.

5. **Zero-quantity market fallback possible in `spread_roll_exec`**
   (`live_dynamic.py:586`). If the limit fully filled but `isDone()` lags, the
   fallback places a `MarketOrder(..., qty - filled = 0)`, which IB rejects.
   Guard with `if qty - int(trade.filled()) > 0`.

### Weaknesses / design notes (not necessarily bugs)

6. **Symbol-only position fallback can mismap** (`live_dynamic.py:359`).
   `rev.setdefault((spec["symbol"],), instr)` means if two universe instruments
   share an IB symbol on different exchanges, the first in `UNIVERSE` iteration
   order silently claims an unmatched position.

7. **Spread quote validation is fragile** (`live_dynamic.py:559-561`).
   `bid != 0 and ask != 0` rejects legitimately zero-priced calendar spreads
   (spread prices near zero/negative are normal), forcing a market order; and
   `bid > -1e8` would admit IB's `-1` "no quote" sentinel if it ever arrives
   un-NaN'd, producing a nonsense mid.

8. **Daemon restarted after midnight refuses the snapshot**
   (`live_dynamic.py:957-965`). The `pst_date != today` guard is evaluated on
   (re)load, so a daemon restart at e.g. 2 AM ET skips all overnight (Eurex)
   execution until the next 6 PM compute. Deliberate safety tradeoff, but worth
   knowing during incident recovery.

9. **`run_execution.sh` currently runs the daemon in dry-run** — `--execute` is
   commented out pending market-data activation. No live orders are being placed
   by the daemon path today.

10. **`datetime.utcnow()` is deprecated** (`live_dynamic.py:444,460`) — emits
    DeprecationWarnings under Python 3.12+; use
    `datetime.now(timezone.utc)`.

11. **`build_tradable_set(capital, …)` never uses `capital`** — cosmetic; drop
    the parameter or apply the intended capital-based filter.
    

### Test gaps worth closing

- Passive/spread roll-window routing (would have caught issue 1).
- `spread_roll_exec` status handling (would have caught issues 2 and 5).
- The `check_order_vol` gate path inside `reconcile_and_execute` (issue 4).
