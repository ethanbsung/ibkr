# System State — Futures EWMAC Dynamic Optimisation (ibkr_fut)

*Reviewed 2026-06-09; updated same day after the execution-review session.
Covers `live_dynamic.py`, `run_dynamic.sh`, `run_execution.sh`,
`algo_execution.py`, `risk_check.py`, `preflight_check.py`, and the
`test_execution.py` suite (102 tests, all passing), plus `watchdog.py`.*

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

Deployment: code lives in git (laptop → push → pull on VPS at
`ethan@68.183.103.50:/home/ethan/ibkr`). **`Data/` is gitignored** — config and
PST fixes (e.g. `Data/pst/ib_config/ib_config_futures.csv`) must be applied
directly on the VPS.

## 2. Daily schedule (cron, America/New_York)

| Time (ET) | Job | What it does |
|---|---|---|
| 6:00 PM weekdays | `run_dynamic.sh` | pst_updater (prices+FX) → volume_collector (weekly) → **preflight_check** → `live_dynamic.py --mode compute` |
| 10:05 PM Sun–Thu | `run_execution.sh` | kills old daemon, starts `live_dynamic.py --mode daemon` in background |
| every 15 min | `watchdog.py` | restarts the daemon via `run_execution.sh` if the heartbeat is stale >25 min; never restarts while `risk_halt.txt` exists |
| 9:30 PM Sun–Thu / 3:30 AM Mon–Fri | `preflight_check.py` | regional preflight: live-quote checks while Asian / Eurex sessions are open |

**⚠ Current state:** `run_execution.sh` starts the daemon **without `--execute`**
(dry-run). The blocker is real and now enumerated: the paper account has **no
live futures market-data subscriptions** (IB error 354 on all 44 open-market
instruments in the 2026-06-09 preflight). Until subscribed, `pre_trade_checks`
would see nan/nan quotes and skip every order anyway.

## 3. Modes of `live_dynamic.py`

- **`--mode compute`** (clientId 6): connects to IBKR, reads NetLiquidation,
  builds the tradable set + universe, runs `compute_targets()`, atomically writes
  `targets_snapshot.json` (`date`, `computed_at`, `capital`, `targets`, `diag`).
  Places no orders.
- **`--mode execute`** (clientId 5, one-shot): loads today's snapshot (aborts if
  missing or stale-dated), reads fresh positions, runs `reconcile_and_execute()`
  once, writes `last_targets.json`, logs the daily ledger row.
- **`--mode daemon`** (clientId 5, long-running): loop every 600 s. Each cycle:
  **halt-file kill switch** → reconnect if needed → reload snapshot if
  `computed_at` changed (flushing the prior day's ledger first) → refuse
  snapshots whose PST data date ≠ today → fetch positions → risk gates →
  `reconcile_and_execute(skip_unchanged=True)`. Deferred (market-closed)
  instruments retry next cycle. A failed *initial* connect no longer exits —
  the daemon retries forever (added after the 2026-06-09 incident where a
  restart landed in the gateway's restart window and the daemon died for the
  night).
- **No `--mode`**: compute then execute in one session (manual testing). Dry-run
  unless `--execute` is passed.

Auxiliary clientIds: preflight=7, ad-hoc probes=77/78.

## 4. Compute phase pipeline

1. **Tradable set** (`build_tradable_set()`): Jumbo instrument is tradable iff it
   passes `instrument_selection` filters (SR cost ≤ 0.01, annual vol ≥ 5%,
   history ≥ 512 days, volume if cached) **and** its PST data is ≤ `FRESH_DAYS=5`
   calendar days old. Matches the backtest's `--filter` so live and backtest use
   an identical eligible universe.
2. **Universe build**: `backtest_dynamic._build_universe` over the full Jumbo
   (3000-day lookback) — EWMAC forecasts, blended vol, weekly-EWMA covariance,
   handcraft weights. Non-tradable instruments stay in the risk model but are
   locked at their current position (Carver: optimise over ~150, trade ~100).
   Note: `_build_universe` raises on a missing adjusted-prices CSV even for
   excluded instruments (crashed on BOBL 2026-06-08) — known robustness gap.
3. **Targets** (`compute_targets`): latest valid row; live-universe renormalised
   weights and IDM (capped at `IDM_CAP`); unrounded positions from
   `forecast · capital · IDM · w · τ / (10 · mult · price · fx · σ)`; then
   `optimise_positions` with costs + buffering, seeded with held positions.
   Target risk τ = 0.20. Held instruments that aren't live today are held as-is.
   Returns `targets` (net contracts) and `diag` (forecast, n_ideal, raw_price,
   mult, fx, sigma per instrument, plus `_meta`: idm / n_live / date / gross_lev
   / n_held_target).

## 5. Reconcile & execute (`reconcile_and_execute`)

For each instrument in `targets ∪ held`:

1. **Config / roll calendar**: skip if no IB config row or no roll calendar.
2. **Pending-order guard**: skip the instrument if any open IBKR trade exists on
   the same symbol (prevents conflicting orders across daemon cycles).
3. **Roll phases** from the PST roll calendar (`get_roll_info`):
   - **Passive window** (3 < days_to_roll ≤ 10): rebalance *adds* route to the
     next month; *reductions* close the expiring leg first but never past zero —
     the remainder comes out of the incoming month (split across both legs).
   - **Spread window** (0 ≤ days_to_roll ≤ 3): the portion of the expiring-month
     position that must survive is rolled via a BAG calendar-spread order
     (`spread_roll_exec`), gated behind the market-hours check. The passive
     limit sits at the spread's **offside price** (ask when selling the spread /
     bid when buying — never paying the spread) for 60 s per cycle, retried
     every daemon cycle. With ≤ `FORCE_ROLL_DAYS=1` day left (**force mode**),
     a timed-out limit escalates the remainder to a market order — but only
     after the cancel is *confirmed* (never stacks a market order on a
     possibly-live limit), and a missing spread quote goes straight to market.
     Outside force mode a failed limit just retries later, and a missing quote
     skips. Fills are credited to the legs whatever the final order status
     (cancelled-with-partials included). Contracts being closed close directly
     in the expiring month (1 crossing, not 2).
   - **Old months** (anything outside the expected set): closed outright.
4. **Risk gate**: `check_order_vol` on the net *target* (|qty|·mult·price·fx·σ
   ≤ 50% of capital). A breach blocks only the rebalance orders — risk-reducing
   old-month closes always execute.
5. **Market-hours check** (`is_contract_okay_to_trade`, cached per conId per
   cycle): parses the contract's own IB `tradingHours` (includes holidays) in
   the exchange's timezone; closed markets are DEFERRED and retried next cycle.
   Verified empirically 2026-06-09 across all 114 instruments: every timezone
   mapped, every open/closed verdict correct.
6. **Pre-trade checks** (`pre_trade_checks`): valid bid/ask within 10 s; live mid
   within 3 daily SDs (3σ/√256) of the PST close used for sizing.
7. **Execution** (`algo_exec`, Carver's algorithm): passive limit at the offside
   price (bid when buying); switch to aggressive (chase the inside quote) on
   passive timeout (300 s), adverse price move, or 5× book imbalance; cancel at
   600 s total. Fills (incl. partials) logged to the ledger with commission.

Dry-run mode prints the full plan and qualifies contracts but calls neither the
risk gate, pre-trade checks, market-hours check, nor the algo — which is why the
**preflight check exists** (§7): dry-run alone cannot detect execution-path
breakage.

## 6. Risk controls (`risk_check.py`)

| Gate | Threshold | Action on breach |
|---|---|---|
| Halt file (`ibkr_fut/risk_halt.txt`) | exists | **live kill switch**: daemon exits within one cycle (≤10 min); daemon and `run_execution.sh` also refuse to start |
| `check_gross_leverage` | gross lev > 10× (from compute `_meta`) | skip the daemon cycle |
| `check_order_vol` | single-instrument annual $ vol > 50% of capital | skip that instrument's rebalance orders |
| `check_daily_loss` | equity > 8% below compute-time capital | write halt file, Discord alert, `sys.exit(1)` |

Plus operational safeguards: atomic snapshot writes (`.tmp` + `os.replace`),
snapshot date validation, `last_targets.json` reconciliation warnings, unknown
held futures left untouched (including positions whose IB symbol is ambiguous
across exchanges), stale-PST refusal in the daemon, PID-file management with
force-kill in `run_execution.sh`, and retry-forever reconnects.

**Heartbeat + watchdog**: the daemon touches `ibkr_fut/daemon_heartbeat.txt`
(UTC timestamp) at the top of every cycle — including the no-snapshot /
reconnect-failure / stale-PST wait states — via a crash-proof `_touch_heartbeat()`.
`ibkr_fut/watchdog.py` (cron, every 15 min) checks the file's mtime: fresh
(< 25 min) → silent exit; stale/missing → Discord alert + restart via
`run_execution.sh` (which owns PID cleanup, log rotation, and its own halt
check). **If `risk_halt.txt` exists the watchdog alerts but never restarts** —
a daemon killed by the kill switch / circuit breaker stays down. Alerts are
rate-limited to one per 2 h (`watchdog_last_alert` marker); the restart is
still attempted on every run.

## 7. Preflight check (`preflight_check.py`)

Nightly execution-path health check — runs the same IB calls the executor makes,
for every UNIVERSE instrument, because dry-run exercises none of them. Checks per
instrument: config row → roll calendar → front-month qualification (plus next
month inside the passive-roll window) → trading-hours/timezone sanity → live
bid/ask (only evaluated while that exchange is open; chunked 30 at a time).
Failures print and alert via Discord; exit 1.

**Scheduling**: step 3 of `run_dynamic.sh` — after the PST refresh, before
compute (~6:20–6:45 PM ET: gateway verifiably up, CME reopened post-maintenance,
roll calendars fresh, alert lands before the daemon trades the new snapshot).
Non-blocking: broken instruments fail safe at order time. At that hour only US
sessions are open, so MKTDATA covers US instruments only. European/Asian
sessions are now covered by two extra cron runs of `preflight_check.py`
(~9:30 PM ET Sun–Thu for Asia, ~3:30 AM ET Mon–Fri for Eurex), logging to
`ibkr_fut/preflight_cron.log` — no wrapper needed, since the script naturally
checks bid/ask only for open exchanges and alerts Discord itself.

First real run (2026-06-09): structural checks clean except known-dead ATX/BTP5
(no roll calendar) and BRE (roll calendar requests a contract month IB doesn't
list — also why its PST data is stale); all 44 open-market instruments failed
MKTDATA with IB error 354 (**no live market-data subscriptions** — the go-live
gate).

## 8. Test coverage (`test_execution.py`, 102 passing)

- Pure helpers (`_valid`, `_sz`, adverse price/size detection).
- `pre_trade_checks`: bid/ask validity, crossed markets, price-divergence
  threshold, market-data cleanup on failure.
- `algo_exec`: passive fill, partial fill, all three aggressive triggers, total
  timeout + cancel wait, NaN commission filtering, limit re-pegging.
- Snapshot I/O: roundtrip, atomicity, numpy/date serialisation, stale-date abort.
- `reconcile_and_execute` guardrails: dry-run places nothing, pending-order skip,
  pre-trade failure skip, ledger logging on fill/partial, no logging on
  unfilled/cancelled, missing config/roll-calendar skips.
- Roll-window routing: long/short reduction splits across months, adds to the
  incoming month, non-roll-window routing, risk gate blocks rebalance but not
  roll closes.
- `spread_roll_exec`: offside pricing both directions on negative-priced spreads,
  `-1` sentinel rejected as no-quote, force vs non-force escalation, zero-qty
  and unconfirmed-cancel guards, fill accumulation across limit + market legs,
  spread-roll deferral when market closed, force-flag propagation from
  `days_to_roll`.
- `get_positions_by_instr`: unique symbol fallback, ambiguous symbol → unknown,
  exact exchange match wins.
- `preflight_check`: all-clear, qualify failure, next-month check inside roll
  window, unmapped timezone / empty hours, market-data failure.
- `watchdog`: fresh heartbeat no-op, stale/missing → alert + restart,
  halt-file present → alert but NO restart, alert suppression window (restart
  still retried), restart-failure reporting, `_touch_heartbeat` never raises.

Still uncovered: the daemon loop itself and `is_contract_okay_to_trade` parsing
(the latter validated empirically against live IB instead).

---

## 9. Issue log

### Fixed 2026-06-09 (all with regression tests)

1. **Roll-window reductions over-sold the expiring month** — reductions now split
   across legs, never past zero.
2. **Partial spread-roll fills lost** — fills credited regardless of final order
   status; limit + market-fallback fills summed.
3. **Spread roll bypassed the market-hours check** — now gated behind `_is_open`.
4. **Risk-gate breach stranded old-month roll closes** — gate now blocks only
   rebalance orders.
5. **Zero-quantity market fallback** in `spread_roll_exec` — guarded.
6. **`get_positions_by_instr` raised `KeyError` on any held position**
   (RHS `held[instr]` evaluated before `setdefault`) — position reconciliation
   could never have worked with a non-flat account.
7. **Spread-roll mid-price limit donated half the spread** — now quotes the
   spread's offside price, consistent with `algo_exec`.
8. **Market fallback semantics clarified**: escalation to market is a deliberate
   forced-roll path (≤1 day to roll date), not a generic timeout behavior.
9. **Symbol-only position fallback could mismap** shared IB symbols — now only
   resolves unambiguous symbols.
10. **Fragile spread-quote validation** — `_spread_px_ok()` accepts legit
    zero/negative spread prices, rejects NaN and IB's −1 sentinel, requires
    bid ≤ ask.
11. **Daemon died if started while the gateway was down** (happened live
    2026-06-09 22:15 UTC) — initial connect failure now falls into the
    retry-forever loop.
12. **Halt file was startup-only** — now a live kill switch checked every cycle.
13. **Seven instruments could never qualify at IBKR** (found by probing):
    VIX (weekly futures share the symbol → `tradingClass=VX`), DAX/AEX/CAC/
    SILVER (shared symbols across multipliers → FDXS/FTI/FCE/SIL), EUROSTX/SMI
    (empty currency defaulted to USD → EUR/CHF). Fixed in
    `ib_config_futures.csv` **on the VPS** (Data/ is not in git); all verified
    to qualify uniquely. DAX/EUROSTX/CAC/AEX/SMI were in the tradable set.
14. `datetime.utcnow()` deprecation; dead `build_tradable_set` params.

### Known open items

- **Market-data subscriptions** — the go-live gate. Subscribe per exchange
  (CME/CBOT/NYMEX/COMEX bundle, CFE for VIX, Eurex, OSE, SGX, KSE, Euronext),
  then rerun preflight during each region's session; flip `--execute` in
  `run_execution.sh` once ALL CLEAR.
- **BRE**: roll calendar misaligned with IB's listed contract months (6L lists
  Jul/Oct/…, calendar requests 202606) — also the cause of its stale PST data.
  Excluded from trading; fix the roll calendar when wanted.
- **MSCIASIA (M1MS)**: EUREX daily-vs-monthly expiry ambiguity is not fixable
  via trading class (all share `FMEA`); `pst_updater` would need to request the
  exact expiry date. Data dead since 2024-12; not in the current UNIVERSE.
- **ATX / BTP5**: no roll calendar / no price data; excluded.
- **Daemon restarted after midnight refuses the snapshot** (`pst_date != today`
  guard) — deliberate safety tradeoff; a correct fix needs trading-calendar
  awareness. Runbook: after a late-night restart, overnight (Eurex) execution
  is skipped until the next 6 PM compute.
- **`_build_universe` crashes on missing adjusted-prices CSVs** even for
  excluded instruments — harden eventually.
- **Pre-existing ambiguity errors in pst_updater logs** for VIX forward months
  should disappear now that the config carries `tradingClass=VX`; verify in the
  next nightly `dynamic_cron.log`.
