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

#### [OBS-1] Sequential execution is inherently slow

`pre_trade_checks` sleeps 10s per instrument waiting for market data. Orders are placed serially — one fills (or times out) before the next starts. 4 instruments took 3.5 minutes on 2026-06-11. Scales linearly: 20 instruments ≈ 15 min, 50 ≈ hours.

Fine now at a handful of positions. Becomes a real problem as the portfolio grows toward 20–30 held positions.

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
