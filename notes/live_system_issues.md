# Live System Issues

Discovered during live monitoring session on 2026-06-11.
Listed roughly in order of severity.

---

## CRITICAL

### [BUG-1] Energy contract month mismatch causes phantom spread rolls — RESOLVED 2026-06-11

**Status**: FIXED in `live_dynamic.py` (commit b35e577, deployed to VPS + daemon restarted
2026-06-11 23:29Z). Added `delivery_month(ib, contract)` helper that reads the canonical
delivery month from `ContractDetails.contractMonth` (matches roll-calendar YYYYMM codes),
with a persistent conId cache, and use it in `get_positions_by_instr` in place of
`lastTradeDateOrContractMonth[:6]`. Verified: held `+1 QMU6` now reads as `{'202609': 1}`,
and the post-restart daemon cycle placed ZERO QM orders (was phantom-rolling every cycle).
Note: the same bug still exists in the retired `live_signals.py:129` (IBS, off cron) — left
as-is. Original analysis below.

**Affected instruments**: All NYMEX energy contracts — QM, CRUDE_W_mini, BRENT-LAST, GAS_US_mini, GAS-LAST, HEATOIL, GASOILINE (and likely others where delivery month ≠ expiry month).

**Root cause**: For WTI crude (QM) and other energy futures, IBKR's `lastTradeDateOrContractMonth` on the resolved contract is the *expiry date*, not the *delivery month*. These differ by one calendar month:

```
qualify QM 202607 → lastTradeDateOrContractMonth = 20260618  → [:6] = 202606
qualify QM 202608 → lastTradeDateOrContractMonth = 20260720  → [:6] = 202607
qualify QM 202609 → lastTradeDateOrContractMonth = 20260819  → [:6] = 202608  ← BUG
```

`get_positions_by_instr` maps each held contract to a month via `c.lastTradeDateOrContractMonth[:6]`. When we hold QM Sep delivery (202609), IBKR reports the position with expiry date 20260819, so the month maps to **202608**, not 202609.

**What happened on 2026-06-11**:
1. 22:33 cycle: bought +1 QM 202609 (Sep delivery) targeting +1. Correct.
2. 22:47 cycle: `get_positions_by_instr` read the position as `{202608: +1}` (wrong month).
3. Roll calendar says current_month=202608, days_to_roll=0, force=True.
4. System saw `qty_current(202608)=+1` and triggered a forced spread roll: SELL QM-202608 (Aug delivery, expires July 20) / BUY QM-202609 (Sep delivery).
5. Spread filled at -1.55 MKT. Result: **-1 QM Aug delivery, +2 QM Sep delivery** instead of the intended +1 Sep.
6. Account now holds a phantom short in the expiring August contract alongside a double-long in September.

**IBKR shows this as "QM-COMB"** (combination/BAG spread), which is why the ticker looked different.

**Fix needed**: `get_positions_by_instr` must not use `lastTradeDateOrContractMonth[:6]` for month classification. Instead either:
- Use the `tradingClass` or `localSymbol` (e.g. QMU6 = September) to infer delivery month, or
- Cross-reference the position's conId against qualified contracts for known roll-window months.

---

## HIGH

### [BUG-2] No visibility into full optimization output

The snapshot `diag` only stores entries for instruments with `target != 0 or current_position != 0` (live_dynamic.py:346). All 86 instruments the optimizer evaluated and assigned 0 are silently discarded.

**Impact**: Cannot audit why the optimizer chose the 4 held positions over alternatives, cannot see "near-miss" instruments with ideal ≈ 0.3–0.4 contracts, cannot verify the carry/trend signals are correct.

**What we can currently see**: 4 instruments (today: CRUDE_W_mini, DAX, JPY, NZD). What we cannot see: the other 86 live instruments' forecasts, ideals, and N_star=0 reasons.

**Notable anomaly**: DAX has ideal=-0.03 but target=-1; NZD has ideal=-0.03 but target=-1. The integer optimizer assigned 33× the unconstrained ideal. Cannot verify without seeing the full covariance/cost calculation.

**Fix needed**: Store full N_unrounded and N_star for all live instruments in snapshot diag. Add a viewer script.

---

### [BUG-3] KOSDAQ / KOSPI perpetually deferred — KSE returns empty tradingHours

`is_contract_okay_to_trade` returns False when `tradingHours` is empty. KSE exchange does not return `tradingHours`, `liquidHours`, or `timeZoneId` for any contract via IBKR's reqContractDetails API. Result: both KOSDAQ and KOSPI are deferred on every daemon cycle and can never execute.

**Confirmed**: `timeZoneId=''`, `tradingHours=''`, `liquidHours=''` for K200 on KSE.

**Additional note on KOSPI**: K200 multiplier is 250,000 KRW ≈ $64k/contract at $250k capital. Ideal positions are ~0.1 → always rounds to 0 even if tradingHours were fixed. KOSPI is effectively un-tradeable at current capital.

**Fix needed**: Either hardcode KSE hours (09:00–15:30 KST = 00:00–06:30 UTC) in `is_contract_okay_to_trade` when tradingHours is empty, or add a per-exchange hours override map.

---

### [BUG-4] BRE (Brazilian Real) — persistently stale, recurring preflight failures

BRE data has been stale since 2026-06-01 (10 days). `pst_updater` fetches "1 bar" but the bar is already in the CSV — IBKR appears to have no data for BRE after June 1. The June 2026 contract also failed to qualify in preflight ("BRE 202606 on CME"). BRE is monthly, so this pattern recurs every roll cycle.

**Impact**: BRE generates a Discord alert every day and never enters the tradable set. It is contributing nothing to the portfolio.

**Options**: Remove from UNIVERSE, or investigate why IBKR has no data (liquidity/subscription issue).

---

## MEDIUM

### [OBS-1] Sequential execution is inherently slow

Pre_trade_checks sleeps 10s per instrument waiting for market data. Orders are placed serially — one fills (or times out) before the next starts. 4 instruments took 3.5 minutes tonight. This scales linearly: 20 instruments ≈ 15 min, 50 instruments ≈ hours.

Fine now at 4 positions. Will become a real problem as portfolio grows toward 20–30 held positions.

---

### [OBS-2] Most orders switch to aggressive quickly on FX/energy contracts

Tonight 3/4 orders switched to aggressive (2 via adverse price at 30s, 1 via adverse size/imbalance at 60s). Fills were essentially at the passive price:

```
QM  BUY  passive=82.85  fill=83.075  slippage=$112 (1 contract × $500 × $0.225)
JPY SELL passive=0.0063 fill=0.0063  slippage=$0
NZD SELL passive=0.5848 fill=0.5847  slippage~$0
```

The adverse-price check fires when the market moves against the passive limit since order placement (ref_price = other side of spread at order time). On liquid FX/energy contracts in a moving session, 30 seconds is enough to trigger this. The slippage is minimal so this is not a bug, but the aggressive switch being the norm rather than the exception suggests the passive timeout (300s) is only relevant in quiet markets.

---

### [OBS-3] Daemon restarts before compute finishes — stale snapshot warnings

`run_execution.sh` starts the daemon at 6:05 PM ET. `run_dynamic.sh` runs compute at 6:00 PM ET but pst_updater + preflight take ~23 minutes, so compute doesn't finish until ~6:23 PM. The daemon finds the prior day's snapshot and emits stale warnings for ~8 minutes until the fresh snapshot appears.

Not a correctness issue — the daemon waits correctly. But generates noise and Discord alerts.

---

### [OBS-4] R1000 (RSV) — illiquid, no market data

R1000 maps to symbol RSV (E-mini Russell 1000) on CME. Preflight confirms no bid/ask while market is open. The contract barely trades. Large-cap US is already covered by SP500_micro, NASDAQ_micro, DOW_mini.

**Note**: RUSSELL_mini (RTY, E-mini Russell 2000) at $50 multiplier sizes to an ideal of ~0.11 contracts at $250k capital — it perpetually rounds to 0. M2K (E-micro Russell 2000, $5 mult) would correctly size to ~1 contract. Both issues noted but no action yet.

---

## RESOLVED

- **Stale snapshot on 2026-06-11**: Root cause was daemon restart at 22:15 before compute finished at 22:23. Resolved automatically when daemon picked up fresh snapshot at 22:23:52Z.
- **Qualify failures at 22:23 cycle**: IB connection dropped mid-qualify. Daemon auto-reconnected at 22:33 and all qualifications succeeded.
