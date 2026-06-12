# Live System Issues

Running log of serious issues in the live futures trading system (`ibkr_fut/`).
Open issues are grouped by severity; resolved issues move to the **Resolved** section
at the bottom (kept, not deleted — they're the post-mortem record).

**Convention:** when a serious live-system issue is found, add it here with an ID
(`BUG-n` for defects, `OBS-n` for observations/improvements), severity, symptom, root
cause, and fix. When it's fixed, move the whole entry to **Resolved** with the commit
and the verification that proved it fixed.

First logged: 2026-06-11 live monitoring session.

---

## OPEN

### HIGH

#### [BUG-6] Daily report compares targets (PST name) against held (IB symbol) — phantom mismatches, no real reconciliation

The Discord daily report's FUTURES POSITIONS section (`scripts/daily_report.py:199` `section_futures_positions`) keys **held** positions by IB symbol (`held[c.symbol]`, line 224) but keys **targets** by PST instrument name (straight from `targets_snapshot.json`, line 211). The two dicts live in different namespaces and are never mapped, so the mismatch logic (lines 242-244) compares apples to oranges.

**Concrete failure (2026-06-12 report):** the snapshot target was `CRUDE_W_mini=+1, DAX=-1, JPY=-1, NZD=-1`; actually held was `QM=+1, DAX=-1, JPY=-1, NZD=-1` (`scripts/account_summary.py`). `CRUDE_W_mini` *is* IB symbol `QM` — the +1 crude position is **correct and matched**. But the report rendered:

```
Target:   CRUDE_W_mini=+1  DAX=-1  JPY=-1  NZD=-1  QM=+0
Held:     CRUDE_W_mini=+0  ← MISMATCH  DAX=-1  JPY=-1  NZD=-1  QM=+1  ← MISMATCH
```

One real, in-line position shown as **two false mismatches** plus two spurious warnings. DAX/JPY/NZD line up only because their PST name happens to equal their IB symbol; any instrument whose PST name ≠ IB symbol (energy, micros, anything with a magnifier/alias) will always read as a double-mismatch even when perfectly reconciled.

**Impact:** the report is useless for its one job — telling you whether held = target. It cries wolf on matched positions and would *hide* a genuine mismatch in the noise. (The DRY-RUN daemon means held never actually converges to target either — see context below — but that's a separate axis; even in execute mode this report would mislead.)

**Root cause:** `live_dynamic.get_positions_by_instr` already does this mapping correctly — it builds a reverse `(symbol, exchange) → PST instr` map from `ib_spec`/`UNIVERSE` and also handles delivery-month aggregation. The report reimplemented position-fetching with a bare `c.symbol` key and skipped the mapping.

**Fix needed:** in `section_futures_positions`, map each held contract's `(c.symbol, c.exchange)` back to its PST instrument name via the same `UNIVERSE`/`ib_spec` reverse map `get_positions_by_instr` uses (or just import and call it), so target and held are compared in the PST-name namespace. Net the multiple-month-per-instrument case while at it. Also note the snapshot it reads is stale relative to live (report dated 2026-06-12 but `Snapshot: 2026-06-11`) — surface the snapshot age so a one-day-late compute is visible, and reconcile against *today's* expected target, not a day-old one.

---

#### [BUG-2] No visibility into full optimization output

The snapshot `diag` only stores entries for instruments with `target != 0 or current_position != 0` (live_dynamic.py:346). All ~86 instruments the optimizer evaluated and assigned 0 are silently discarded.

**Impact**: Cannot audit why the optimizer chose the held positions over alternatives, cannot see "near-miss" instruments with ideal ≈ 0.3–0.4 contracts, cannot verify the carry/trend signals are correct.

**Notable anomaly**: DAX has ideal=-0.03 but target=-1; NZD has ideal=-0.03 but target=-1. The integer optimizer assigned 33× the unconstrained ideal. Cannot verify without seeing the full covariance/cost calculation.

**Fix needed**: Store full N_unrounded and N_star for all live instruments in snapshot diag. Add a viewer script.

---

#### [BUG-3] KOSDAQ / KOSPI perpetually deferred — KSE returns empty tradingHours

`is_contract_okay_to_trade` returns False when `tradingHours` is empty. The KSE exchange does not return `tradingHours`, `liquidHours`, or `timeZoneId` for any contract via IBKR's reqContractDetails API. Result: both KOSDAQ and KOSPI are deferred on every daemon cycle and can never execute.

**Confirmed**: `timeZoneId=''`, `tradingHours=''`, `liquidHours=''` for K200 on KSE. This is the inverse of an "ask IB" fix — we already ask IB and it has no answer, so a hardcoded fallback is required.

**Additional note on KOSPI**: K200 multiplier is 250,000 KRW ≈ $64k/contract at $250k capital. Ideal positions are ~0.1 → always rounds to 0 even if tradingHours were fixed. KOSPI is effectively un-tradeable at current capital.

**Fix needed**: Either hardcode KSE hours (09:00–15:30 KST = 00:00–06:30 UTC) in `is_contract_okay_to_trade` when tradingHours is empty, or add a per-exchange hours override map.

---

#### [BUG-4] BRE (Brazilian Real) — persistently stale, recurring preflight failures

BRE data has been stale since 2026-06-01. `pst_updater` fetches "1 bar" but the bar is already in the CSV — IBKR appears to have no data for BRE after June 1. The current-month contract also fails to qualify in preflight ("BRE 202606 on CME"). BRE is monthly, so this pattern recurs every roll cycle. (BRE also surfaced in the BUG-1 and BUG-5 cross-checks as the lone qualify-fail.)

**Impact**: BRE generates a Discord alert every day and never enters the tradable set. It is contributing nothing to the portfolio.

**Options**: Remove from UNIVERSE, or investigate why IBKR has no data (liquidity/subscription issue).

---

### MEDIUM

#### [BUG-5] Config multiplier / priceMagnifier trusted from CSV, never validated against IB

Sizing (`compute_targets`) and the pre-trade risk gate (`check_order_vol`, live_dynamic.py:911) read `IBMultiplier` from the `ib_config` CSV. The pre-trade divergence check derives its reference price as `sig_px = raw_px / priceMagnifier` (live_dynamic.py:783), also from the CSV. IB knows both values authoritatively (`Contract.multiplier`, `ContractDetails.priceMagnifier`) — in fact we already read IB's `c.multiplier` at execution time, but only for ledger logging, not for sizing.

This is the same *failure profile* as BUG-1: a per-instrument value re-derived/trusted locally that IB could confirm. If the CSV ever drifts (contract respec, bad manual edit), every position in that instrument is silently mis-sized and the risk gate is anchored to a wrong number — with no alert. Cannot simply swap IB's value into sizing, because the backtest uses the same CSV value and they must stay in parity; the right move is **validate, not replace**.

**Status**: Not currently triggered. Cross-check on 2026-06-11 (`/tmp/crosscheck_specs.py`) over 104 instruments found **0 multiplier mismatches, 0 priceMagnifier mismatches** — the CSV matches IB today. This is a preventive guard, not a live fix.

**Fix needed**: In `preflight_check.py` (which already calls `reqContractDetails` per instrument — zero extra IB round-trips), assert config multiplier == `c.multiplier` and config priceMagnifier == `ContractDetails.priceMagnifier`; alert on mismatch via the existing Discord path. Converts "silent mis-size someday" into "loud alert the night it drifts."

---

#### [SPEED] Nightly compute pipeline takes 15–20 min — breakdown + fixes

The 6 PM ET cron (`run_dynamic.sh`) chains pst_updater → volume_collector → preflight_check → live_dynamic compute, and the whole thing runs 15–20 min. Reviewed 2026-06-12. The cost is dominated by serial IBKR round-trips with mandatory throttle sleeps and a from-scratch covariance build over the full 105-instrument universe. Each sub-issue is logged separately below (OBS-6…OBS-9); this is the index.

| Stage | Where | Rough cost | Main lever |
|-------|-------|-----------|------------|
| PST data fetch | `pst_updater.py` | ~8–12 min | OBS-6 (serial fetch + 0.6s sleeps), OBS-7 (carry second pass) |
| Preflight | `preflight_check.py` | ~2–4 min | OBS-8 (qualify/hours over all 105, serial) |
| Compute (covariance/handcraft) | `_build_universe` | ~2–4 min | OBS-9 (4000-day model rebuilt nightly from scratch) |
| Execution (separate cron) | `algo_execution` | scales w/ #positions | OBS-1 |

These are all the *same* serial-IB-roundtrip-plus-sleep pattern. The single highest-leverage change is to stop re-fetching/recomputing data that hasn't changed since last night and to parallelise or de-throttle the IB requests that remain.

---

#### [OBS-6] pst_updater re-fetches every instrument serially with a hard 0.6s sleep per contract

`pst_updater.main` loops over all ~105 instruments on one IB connection (`pst_updater.py:757`). Per instrument `update_prices` fetches front + forward (+ carry when distinct) bars per roll segment, each followed by `time.sleep(0.6)` (`pst_updater.py:330/335/346`). At ≥2 fetches × 0.6s × 105 instruments that is **≥2 min of pure sleep** before counting IB latency (~0.3–1s/req), so realistically 8–12 min wall-clock.

The nightly incremental update only needs **1–2 new bars** per instrument (last close), yet `_duration_str(start, end)` and the segment loop are sized for the full gap-fill path. Most nights `start = last_date+1` so the request is tiny, but the round-trip + sleep cost is fixed regardless.

**Fix options** (in rough order of payoff):
1. **Drop the sleep / make it adaptive.** IBKR's pacing limit is ~60 hist requests / 10 min for *small* identical requests; 0.6s (1.7 req/s = 100/min) is already over a naive read of the limit but works because daily-bar requests are cheap. Test a smaller sleep (0.2–0.3s) or remove it and rely on ib_insync's own pacing/error backoff. Biggest single win.
2. **Parallelise.** ib_insync supports concurrent `reqHistoricalDataAsync`; fire N requests, gather, throttle as a batch. Even 4–8 in flight collapses the wall-clock.
3. **Skip up-to-date instruments early.** If `last_date >= today_close_date` (already fresh, e.g. holiday/weekend run), skip the IB call entirely. Cheap guard, big effect on no-op nights.

---

#### [OBS-7] `--carry` triggers a full second per-instrument fetch pass

`run_dynamic.sh` passes `--carry`, so after `update_prices` each instrument also runs `update_carry` (`pst_updater.py:760-762`), which does its *own* loop over PRICE_CONTRACT groups fetching the carry contract again with another `time.sleep(0.6)` (`pst_updater.py:469-475`). This roughly doubles the per-instrument IB cost for the carry-using live strategy.

`update_prices` already fetches the carry contract for each segment (`pst_updater.py:337-346`) and writes CARRY/CARRY_CONTRACT into `new_multi_rows` — `update_carry` is largely re-deriving what the price pass could emit. **Fix:** have `update_prices` persist the carry column it already computes and skip the separate `update_carry` pass (or restrict `update_carry` to backfilling rows the price pass left NaN), eliminating the second round-trip set entirely.

---

#### [OBS-8] preflight qualifies + checks hours for all 105 instruments serially every night

`preflight_check.check_contracts` loops all 105 instruments calling `qualify` (`qualifyContracts`) + `reqContractDetails` for hours, serially (`preflight_check.py:65`). Market-data probing already batches (`MKT_DATA_CHUNK=30`), but the qualify/hours structural pass does not.

conId↔(symbol,month,hours-template) is stable within a roll cycle, so most of this is recomputable-once, not nightly. **Fix:** cache qualified contracts + their tradingHours/timeZone by (instr, month) and only re-qualify when the held/target month changes or the cache is older than the roll window. This also feeds OBS-3 (compute finishing late) by shortening the preflight stage. (Note BUG-5's proposed multiplier/priceMagnifier validation rides on this same per-instrument `reqContractDetails` — fold them together.)

---

#### [OBS-9] Covariance + handcraft model rebuilt from scratch over a 4000-day panel every night

`_build_universe` (compute phase, `live_dynamic.py:1256`, `lookback_days=4000`) recomputes the full correlation matrix + handcraft weights (`compute_corr_matrix`, `handcraft_weights`) and constructs the `CovarianceEstimator` over a ~4000-calendar-day × 105-instrument panel **every night** (`backtest_dynamic.py:203-216`). The optimiser only ever uses `est.covariance_by_index(as_of, live_idx)` for the *latest* date (`live_dynamic.py:359`) — one day's covariance — yet the whole estimator is built across the entire history.

The handcraft weights/correlation are full-history and change negligibly night to night; the EWMA covariance only needs enough trailing window to seed the latest estimate (25wk corr / 32d vol ≈ a few hundred days, not 4000).

**Fix options:**
1. **Shrink the panel.** The covariance estimator needs only its EWMA warm-up window (~500–750 days covers 25wk/32d with margin), not 4000. The 4000-day window exists so *backtest* vol matches full history; live only needs *today's* number. Verify live `sigma`/cov match the full-history values at the warm-up length, then cut `lookback_days` for the live path. Big memory + CPU win.
2. **Cache the static handcraft model.** `compute_corr_matrix` + `handcraft_weights` are full-history and instrument-set-dependent only — cache to disk keyed by the tradable-set hash, recompute weekly (or when the set changes), not nightly.

---

#### [OBS-1] Sequential execution is inherently slow

`pre_trade_checks` sleeps 10s per instrument waiting for market data (`algo_execution.py:66`). Orders are placed serially — one fills (or times out) before the next starts. 4 instruments took 3.5 minutes on 2026-06-11. Scales linearly: 20 instruments ≈ 15 min, 50 ≈ hours.

Fine now at a handful of positions. Becomes a real problem as the portfolio grows toward 20–30 held positions. **Fix direction:** subscribe market data for all instruments-to-trade up front (one batched `reqMktData` wave, settle once), then place orders concurrently and poll fills, instead of the 10s-wait-then-place-then-wait serial chain. The 10s settle is the dominant per-instrument cost and is fully parallelisable across instruments.

---

#### [OBS-2] Most orders switch to aggressive quickly on FX/energy contracts

On 2026-06-11, 3/4 orders switched to aggressive (2 via adverse price at 30s, 1 via adverse size/imbalance at 60s). Fills were essentially at the passive price:

```
QM  BUY  passive=82.85  fill=83.075  slippage=$112 (1 contract × $500 × $0.225)
JPY SELL passive=0.0063 fill=0.0063  slippage=$0
NZD SELL passive=0.5848 fill=0.5847  slippage~$0
```

The adverse-price check fires when the market moves against the passive limit since placement. On liquid FX/energy in a moving session, 30s is enough to trigger it. Slippage is minimal so this is not a bug, but aggressive-as-the-norm suggests the 300s passive timeout only matters in quiet markets.

---

#### [OBS-3] Daemon restarts before compute finishes — stale snapshot warnings

`run_execution.sh` starts the daemon at 6:05 PM ET. `run_dynamic.sh` runs compute at 6:00 PM ET but pst_updater + preflight take ~23 minutes, so compute finishes ~6:23 PM. The daemon finds the prior day's snapshot and emits stale warnings for ~8 minutes until the fresh snapshot appears.

Not a correctness issue — the daemon waits correctly — but generates noise and Discord alerts.

---

#### [OBS-4] R1000 (RSV) illiquid; RUSSELL_mini too large to ever size at current capital

R1000 maps to symbol RSV (E-mini Russell 1000) on CME. Preflight confirms no bid/ask while market is open — the contract barely trades. Large-cap US is already covered by SP500_micro, NASDAQ_micro, DOW_mini.

**Note**: RUSSELL_mini (RTY, E-mini Russell 2000) at $50 multiplier sizes to an ideal of ~0.11 contracts at $250k capital — perpetually rounds to 0. M2K (E-micro Russell 2000, $5 mult) would size to ~1 contract. Kept in the universe deliberately (diversification in the optimizer even when never traded); no action for now.

---

#### [OBS-5] Spread-roll limit price rounded to hardcoded 4 decimals, not IB min tick

`spread_roll_exec` (live_dynamic.py:659) computes the calendar-spread limit as `round(bid/ask, 4)` — a hardcoded precision rather than the contract's real `minTick` from `ContractDetails`. An off-tick limit can be rejected by IB.

**Impact**: Low. The main `algo_exec` path is safe (it uses raw on-grid bid/ask). Only the spread BAG path rounds, and spread rolls are infrequent; 4 decimals is usually on-grid for these products. Another "ask IB" candidate (`ContractDetails.minTick`) but low priority.

---

## RESOLVED

#### [BUG-1] Energy contract month mismatch causes phantom spread rolls

**Resolved 2026-06-11** in `live_dynamic.py` (commit b35e577, deployed to VPS + daemon restarted 23:29Z). Added `delivery_month(ib, contract)` helper reading the canonical delivery month from `ContractDetails.contractMonth` (matches roll-calendar YYYYMM codes), with a persistent conId cache, used in `get_positions_by_instr` in place of `lastTradeDateOrContractMonth[:6]`.

**Verification**: held `+1 QMU6` now reads as `{'202609': 1}` (was `202608`); the post-restart daemon cycle placed ZERO QM orders (it had been phantom-rolling every 10-min cycle). Universe-wide cross-check (`/tmp/crosscheck_months.py`) confirmed **RED (round-trip mismatch) = 0** across all 105 instruments — the universal change is safe.

**Symptom**: On 2026-06-11 the daemon kept rolling QM (CRUDE_W_mini) every cycle. A clean `+1 QM Sep` was repeatedly turned into phantom legs (e.g. `-1 Aug / +2 Sep`), showing as "QM-COMB" (a BAG calendar spread) in IBKR.

**Root cause**: For energy futures IBKR's `lastTradeDateOrContractMonth` is the *expiry date*, not the *delivery month*. QM Sep-delivery expires 20260819, so `[:6]` mapped it to `202608` — which the roll calendar marks as the expiring current contract (days_to_roll=0, force) → forced spread roll every cycle.

```
qualify QM 202609 → lastTradeDateOrContractMonth = 20260819 → [:6] = 202608  ← BUG
                    ContractDetails.contractMonth = 202609                    ← used now
```

**Blast radius (cross-checked)**: the old `[:6]` was wrong in BOTH directions, not just "energy off by one" — SOFR −3 months (IMM quarterly), BRENT-LAST −2, CHEESE **+1** (expiry after contract month), and CRUDE_W_mini/GAS_US_mini/GAS-LAST/HEATOIL/GASOILINE/RUBBER/BRE −1. 191 contract-months were no-ops (FX/equity/bond/metal and most ags, incl. held legs JPY & NZD). The same bug still exists in the **retired** `live_signals.py:129` (IBS, off cron) — left as-is.

---

#### Stale snapshot on 2026-06-11 (operational, self-resolved)

Daemon restarted at 22:15 before the compute phase finished at 22:23, so it briefly saw the prior day's snapshot and emitted stale-PST warnings. Resolved automatically when it picked up the fresh snapshot at 22:23:52Z. (Underlying timing window tracked as OBS-3.)

#### Qualify failures at 22:23 cycle (operational, self-resolved)

IB connection dropped mid-qualify, so all four targets failed to qualify that cycle. The daemon's reconnect logic re-established the connection at 22:33 and all qualifications succeeded.
