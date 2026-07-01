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

### CRITICAL

#### [BUG-9] Compute writes the snapshot off a phantom-FLAT book when `reqPositions` fails → held positions never close

**Found 2026-06-24** investigating why CHF/JPY/NZD/KOSDQ150/DAX/MBT (held −1 each) weren't being closed even though they'd dropped out of the optimiser's daily targets. The reduce_only optimiser fix (commit f63a2b4) is correct but **never fired**, because compute never saw the held positions.

**Symptom.** `dynamic_cron.log` shows the COMPUTE job logging `WARNING: live reqPositions failed (Socket disconnect) — falling back to cached ib.positions()` then `Current positions: flat` on nearly every recent run (2026-06-16, 17, 21, 22, 23). The persisted `targets_snapshot.json` therefore lists only freshly-optimised positions, all tagged `active`, with **no entry** for the genuinely-held instruments and no `reduce_only`/`frozen` status. The daemon loads that snapshot and reconciles against it; for an instrument that's held but absent from `targets`, `reconcile_and_execute` (live_dynamic.py:858 `desired = int(targets.get(instr, net_held))`) defaults to **keep current** — so the stranded positions are carried indefinitely.

**Root cause.** In the compute path, `get_positions_by_instr` was called *after* the ~15-min `_build_universe` step. The gateway routinely drops the API socket during that long CPU-bound window, so by the time positions are requested the connection is half-open and `reqPositionsAsync` raises "Socket disconnect". `fetch_positions` then silently fell back to `ib.positions()`, whose cache was never seeded on this short-lived compute client → it returned **empty**, indistinguishable from "really flat". `compute_targets` ran with `current_positions = {}`, so even the no-blind-trade fallback (live_dynamic.py:407, gated on `current_positions.get(nm,0) != 0`) preserved nothing. A wrong, flat-book snapshot was persisted and traded against. Same *class* as BUG-7 (acting on an unverified position read), different trigger (one-shot compute client losing its socket mid-build, not a frozen long-lived daemon cache).

**Fix part 1 — compute-side (commit d411b39, deployed to VPS) in `live_dynamic.py`:**
1. **Fetch positions on a fresh socket, before the build.** Moved `get_positions_by_instr` to immediately after `_connect`/`load_ib_config`, so the request goes out while the connection is new (the early `get_equity` read already succeeds there).
2. **Strict (verified) read when persisting.** New `PositionFetchError` + `strict` flag threaded through `fetch_positions`/`get_positions_by_instr`. In strict mode a failed `reqPositions` **raises** instead of falling back to the cache. Compute catches it, alerts `[COMPUTE-ABORT]` to Discord, and **returns without overwriting the snapshot** — so the daemon keeps reconciling against the last *verified* book rather than a phantom-flat one. The daemon's own per-cycle fetch stays non-strict (it re-fetches every 10 min and degrades to "possibly stale", per BUG-7's design).

This stops *future* bad snapshots, but did **not** by itself release the already-stranded positions: when the gateway is down at compute time (e.g. 2026-06-24, `ConnectionRefused 4002`) compute correctly aborts, leaving the prior phantom-flat 06-23 snapshot in place — and the daemon's `targets.get(instr, net_held)` still defaulted those held-but-absent instruments to *keep*. Live book on 2026-06-25 held 12 positions while the snapshot listed only 6 targets; the 6 extras (DAX, JPY, MBT, QM, TSEMOTHR, XINA50) were stranded.

**Fix part 2 — reconciler-side (commit pending) in `live_dynamic.py:reconcile_and_execute`:** the companion change that actually unwinds stranded positions, and the durable safety net.
1. **Honour "close" for held-but-absent.** The `desired = int(targets.get(instr, net_held))` default-to-keep was replaced: when an instrument is held but absent from a **fresh** snapshot, the snapshot is authoritative → `desired = 0`, status `reduce_only` (unwind toward 0). When the snapshot is **not** fresh, absence means "stale and silent" → hold (`frozen`), never infer a close. This is the live mirror of what the backtest/`dynamic_opt.optimise_positions` already do (held + not-tradable → reduce_only to 0; no-data → carried forward / frozen). Backtest verified: `backtest_dynamic.py:_simulate` carries `pos` forward and only the `live_idx` slice is re-optimised, so no-data positions freeze and untradable-held positions unwind — matching the live rule exactly. No strategy re-validation needed.
2. **Wire the `status` flags into execution.** `diag["_meta"]["status"]` (active/reduce_only/frozen) is now read and enforced as a hard guard: `frozen` holds the live-month position unchanged (old-month rolls still close); `reduce_only` is clamped to `[0, expected_held]` (same sign, smaller magnitude) so a bad target can never grow or flip a position.
3. **Mass-liquidation guard (`ABSENT_HELD_MAX_FRAC = 0.34`).** If more than ~1/3 of the held book is absent from a *fresh* snapshot, treat it as a suspected bad snapshot: do **not** auto-close the absentees, hold them, and alert `[RECONCILE-ABSENT]` to Discord. The 06-25 incident (6 of 12 = 50% absent) would alert+hold under this guard rather than liquidate half the book; clean recovery is a verified fresh compute. Daemon passes `snapshot_fresh=True` (it only reaches the reconciler after the existing calendar-aware staleness gate); the one-shot `--mode execute` path is same-day-validated by `load_snapshot`.

Tests: `test_execution.py` Group 8 (10 new tests) — held-but-absent unwind (fresh) and hold (stale); reduce_only never-grows/never-flips/shrinks-partial; frozen holds + still closes a stranded old month; mass-liquidation guard trips at >34% and allows a single absentee; backward-compat with no `_meta`. 191 ibkr_fut tests pass.

**Verification owed before trusting live:** (a) on the next VPS compute run with the gateway up, confirm `Current positions (net): …` lists the real held book (not `flat`) and the snapshot's `status` map tags dropped instruments `reduce_only`/`frozen`; confirm a forced fetch failure produces `[COMPUTE-ABORT]` and leaves the prior `targets_snapshot.json` untouched. (b) After deploying fix part 2 + a successful compute, confirm the daemon's next cycle places closing orders for the stranded names (DAX/JPY/MBT/QM/TSEMOTHR/XINA50) during their market-open windows and the daily report shows held == target across instruments.

---

#### [BUG-10] systemd port-watchdog races IBC's silent restart → daily Gateway COLD-restart + full re-auth (the disconnect-storm root cause)

**Found 2026-06-26** investigating "why is the IB API so unreliable" — the through-line behind BUG-7 and BUG-9. Those two share a root cause (acting on an unverified position read), but the *trigger* — the connection dropping in the first place — is environmental, not in our Python.

**Symptom.** Chronic API instability: **52 `Peer closed connection` + 77 `IB disconnected` events** in `daemon_cron.log`; recurring `reqPositions failed (Socket disconnect)` → phantom-flat reads. The IBC log shows **7 `autorestart file not found: full authentication will be required`** events in a single day, and the Gateway cold-restarting **3× on 2026-06-25 (06:55, 22:19, 23:46)** — the 22:19/23:46 restarts landing squarely in the execution window (the 6/25 `[DAEMON-STALE]` alert + flat read came from the 23:46 one).

**Root cause.** The Gateway runs under IBC with `AutoRestartTime=03:00`, which is IBC's *clean, silent* daily restart — it writes an `autorestart` token so the restart re-authenticates with no 2FA. A separate **systemd `ibgateway-watchdog.timer` fires every 2 min** and runs `ss -tlnp | grep -q ":4002" || systemctl restart ibgateway.service`. During IBC's own AutoRestart the port is briefly down; the 2-min watchdog catches that window and **cold-restarts the whole service** via `start_ibgateway.sh`. A cold start has **no autorestart token** → full re-authentication. During that re-auth window the API port answers but positions/market data aren't loaded yet → exactly the half-open / phantom-flat state that BUG-7's frozen cache and BUG-9's empty-cache fallback then act on. The single-miss, 2-min watchdog turned IB's once-daily *silent* restart into repeated *cold* restarts that poisoned the trading window.

**Fix part 1 — environmental (VPS infra, gitignored; deployed 2026-06-26):** make the watchdog tolerant so it never races IBC's silent restart.
- New `/home/ethan/ibgateway_watchdog.sh`: requires **2 consecutive** port-4002-down checks (state in `~/.ibgateway_watchdog_misses`) before restarting; clears the counter the moment the port is back.
- Timer interval widened **2 min → 5 min** (`ibgateway-watchdog.timer`), so a restart only fires after ≥10 min of *sustained* downtime — IBC's brief silent-restart window passes untouched, but a genuinely dead Gateway is still recovered. `ibgateway-watchdog.service` now calls the script + logs to `~/logs/ibgateway_watchdog.log`.
- **Installed on VPS** (`sudo cp` of the staged unit files + `systemctl daemon-reload`); timer confirmed `active (waiting)` at the 5-min interval. `AutoRestartTime=03:00` left unchanged — already off the trading window (≈23:00 ET, between US close and Asia open).

**Fix part 2 — code, fail-safe so a bad read can never trade (commit `0ae957d`, deployed to VPS) in `live_dynamic.py` + `test_execution.py`:** even with the infra fixed, the daemon must never trade off an unverified read.
1. **Strict read in the daemon execute path.** The daemon cycle now calls `get_positions_by_instr(ib, ibcfg, strict=args.execute)`: a failed/timed-out `reqPositions` raises `PositionFetchError` → the cycle skips + alerts `[DAEMON-READFAIL]` and forces a clean reconnect, instead of silently falling back to the possibly-empty cache and reconciling to targets off a phantom-flat book. (Dry-run keeps the lenient fallback; it never trades.) Extends d411b39's compute-side strict read to execution.
2. **Halt-on-mismatch gate (Carver lock-on-mismatch).** New `expected_book_mismatch()`: before placing orders, compare the freshly-read broker book against the expected book persisted last execute run (`last_targets.json`). If > `EXPECTED_MISMATCH_MAX_FRAC = 0.5` of the union of {expected, actual} instruments disagree, the read is untrustworthy (e.g. the up-but-not-ready socket during a restart) → skip the cycle + alert `[DAEMON-MISMATCH]`, don't trade. A phantom-flat read → "everything mismatches → trade nothing" = safe failure. Complements the in-reconcile `ABSENT_HELD_MAX_FRAC` guard (snapshot-side) on the read-side.
3. **Compute/execute ordering race (also clears OBS-3).** New compute-in-progress marker (`computing.lock`, written/removed by `run_dynamic.sh` via an `EXIT` trap; `compute_in_progress()` ignores a marker older than 30 min). The daemon's staleness gate consults it: while compute is mid-run (starts 22:00 UTC, finishes ~22:17; daemon restarts 22:15) the daemon waits quietly for the imminent fresh snapshot instead of false-alarming on the prior-day snapshot and briefly reading flat during compute's connection churn.

Tests: `test_execution.py` Group 9 (13 new) — `expected_book_mismatch` (no-prior/flat/clean/one-differs/phantom-flat/most-of-book/threshold-strict/zero-entries), `compute_in_progress` (no-marker/fresh/stale-ignored), and strict propagation through `get_positions_by_instr`. **130 ibkr_fut/test_execution.py tests pass.**

**Why this is the durable answer to "IB is unreliable".** The investigation (codebase audit + issue-log synthesis + pysystemtrade research) showed these are not many bugs but a few connection-handling weaknesses plus one architectural mismatch: a resident daemon holding one socket for ~24h against a Gateway designed to be restarted daily and flaky right after. Phase 0 (this entry) stops a bad read from ever trading and stops the infra from manufacturing bad reads. **Phase 1 (planned, not yet built)** re-architects to Carver's model — short-lived workers (connection scoped to one run, can't rot), a clientId registry, `errorEvent`-driven health gating, and a SQLite single-source-of-truth — triggered by a supervisor (systemd timers) rather than blind cron. See the approved plan.

**Deployment (2026-06-26).** Code pushed + pulled to VPS (`git` HEAD `0ae957d`). The running daemon was still executing the *pre-pull* code in memory (a `git pull` does not restart it) — restarted via `run_execution.sh` (old PID 539036 → new PID 548216) during a markets-closed window (no orders in flight). New daemon confirmed: connected, loaded the fresh 2026-06-25 snapshot (`target holds 6`), ran a clean first cycle. A live connectivity blip during the handoff (`Error 1100` lost → `1102` restored, `ConnectionRefused 4002` on first connect) was recovered by the existing 3× connect-retry — the *old* failure mode handled gracefully; the new strict-read/mismatch gates didn't need to fire (markets closed, snapshot fresh). Tolerant watchdog timer active at 5-min interval.

**Verification still owed (watch over the next few sessions):** (a) at the next 03:00 UTC restart, confirm the IBC log no longer shows `autorestart file not found` and the Gateway is not cold-restarted in the trading window; (b) at the next compute/execute boundary (22:00–22:17 UTC) confirm no false `[DAEMON-STALE]` alert (the `computing.lock` marker); (c) confirm the `Peer closed connection` / `IB disconnected` counts drop sharply from the 52/77 baseline; (d) opportunistically, confirm that a genuine read failure in-window produces `[DAEMON-READFAIL]`/`[DAEMON-MISMATCH]` and the daemon resumes only after a verified read.

**Update 2026-06-29 — the watchdog was NOT the (only) trigger; the real cause is a broken IBC autorestart token + an unhonoured restart time.** Re-investigated after another `[DAEMON-READFAIL]` (2026-06-29 18:45, `Socket disconnect` → port 4002 `ConnectionRefused 111` for ~2 min → recovered 18:47). Direct VPS evidence corrects the Fix-part-1 theory:
- **The watchdog did not fire today.** `~/logs/ibgateway_watchdog.log` shows only `2026-06-29T18:46:07Z: port 4002 DOWN (1/2 consecutive)` for the 18:45 outage — it never reached 2/2. So the tolerant watchdog is working *as designed* and is not what restarted the Gateway. (It *does* still fire once daily at ~23:46–23:53 as a backstop, because the Gateway is genuinely down then — see below.)
- **The Gateway cold-restarts itself TWICE a day, at ~23:45 and ~00:00 UTC**, not at the configured `AutoRestartTime=03:00`. Proof: the per-session logs in the TWS settings dir (`~/Jts/cfbgcpdfjackilgfhipbjdeeblnlhenfdnhlemac/ibgateway.<date>.<time>.ibgzenc`) roll at `…221xxx → 2345xx → 000006` every single day. `AutoRestartTime=03:00` in `config.ini` is being **ignored** — TWS/Gateway is auto-restarting at local midnight in its display timezone instead.
- **The autorestart token does not exist on disk.** `find ~ -name autorestart` returns nothing, and every IBC restart logs `autorestart file not found: full authentication will be required`. So *every* daily restart is a **cold login** (full re-auth: the Monday IBC log shows the entire `Paper Log In → Authenticating → Login has completed` sequence at 18:46), which takes the API port fully offline for ~2 min — the `ConnectionRefused 111` window the daemon hits.
- **`ibcstart.sh: line 518: <pid> Killed`** appears in the IBC logs — the Gateway JVM is being `Killed` (SIGKILL) and then relaunched by systemd `Restart=on-failure`, i.e. the restart is not a clean IBC-managed cycle.

**Net corrected root cause:** the chronic port outages are IBC/Gateway-side, not Python-side and not (primarily) the watchdog. Two things must be fixed on the VPS (gitignored config, apply directly):
  1. **Make `AutoRestartTime` actually take effect and land off the trading window.** It currently restarts at ~23:45 + 00:00 UTC (inside the Asia execution window). Either get the configured `03:00` to stick, or set the restart to a genuinely idle hour and verify against the per-session log roll time.
  2. **Repair the autorestart token** so the daily restart is *silent* (no full re-auth, ~10–20 s port blip instead of ~2 min). Likely the empty `--tws-settings-path=` (passed blank in the running args) prevents IBC writing/finding the token; pin `IbDir`/`tws-settings-path` to the real settings dir so the token persists across restarts.

Until #1/#2 are fixed the existing Phase-0 code guards (strict read + mismatch gate) are what keep this **safe** — confirmed live today: the 18:45 outage produced `[DAEMON-READFAIL]`, **zero erroneous trades**, and the daemon resumed clean by 18:47. Markets were closed throughout (`0 placed (market closed)`), so no capital was at risk. The remaining work is availability, not safety.

**Fix applied 2026-07-01 (VPS config, gitignored) — jts.ini timezone.** Diagnosed the mechanism precisely: the Gateway reads its display timezone from `~/Jts/jts.ini` `[Logon] TimeZone` (the active `ibg.xml` has **no** TimeZone or restart-time field at all), and it was set to **`Africa/Abidjan`** — the UTC+0 "unknown zone" fallback. So the Gateway's own default auto-restart fired at *local midnight = 00:00 UTC* (inside the Asia window), and IBC's `AutoRestartTime=03:00` never persisted (evaporated each launch → Gateway used its midnight default). Changed `TimeZone=Africa/Abidjan → America/New_York` (CRLF line endings preserved; backup `jts.ini.bak-20260701-201755`). Applied with `risk_halt.txt` set + daemon stopped, then `sudo systemctl restart ibgateway.service` (16:20 ET, US-close window), Gateway back up in ~15s (port 4002 login OK), halt removed, daemon restarted (PID 755016) — clean first cycle: loaded the 2026-06-30 snapshot, read positions, deferred CORN (market closed), **no READFAIL/disconnect/phantom-flat**. Now `AutoRestartTime=03:00` means **03:00 ET** (a genuinely idle slot: US closed, Asia quiet, pre-Eurex). Note: `autorestart file not found` on *this manual* restart is expected — the token is only written by IBC's *own scheduled* restart, not a `systemctl` cold start.

**P1.3 verification still owed (first scheduled restart under the new TZ = 03:00 ET / 07:00 UTC on 2026-07-02):** (a) newest session log in `~/Jts/cfbgcpdfjackilgfhipbjdeeblnlhenfdnhlemac/ibgateway.*.ibgzenc` rolls at ~07:00 UTC, **not** 00:00 UTC — no more `2345xx→000006` pair in the trading window; (b) an `autorestart` token now exists on disk (`find ~/Jts -name autorestart`); (c) that restart's IBC log entry no longer says `autorestart file not found` (silent re-auth, ~15s blip not ~2 min); (d) `daemon_cron.log` READFAIL/Peer-closed/IB-disconnect counts drop to ~0. **When (a)–(d) confirm, move BUG-10 to Resolved** with this config change + the verification.

---

#### [BUG-7] Daemon's `ib.positions()` cache freezes → re-rolls QM every cycle off a stale position (live capital churn)

**Found 2026-06-16 live monitoring.** This is the QM "sell-Aug/buy-Sep every cycle" symptom that BUG-1 was thought to have fixed — but it is a **different bug**. BUG-1 (delivery_month mapping) is genuinely fixed and was *not* the cause here; `delivery_month` maps both QM legs correctly (`202609`/`202608`). Do **not** re-open BUG-1.

**Symptom.** From the first cycle of the 2026-06-15T22:15Z daemon run, every 10-min cycle logged the identical line and re-traded it:
```
CRUDE_W_mini  QM  202609  | held +1 → target +1   (fcast +14.6 ideal +0.19)
  ⚠ ROLL — other months: 202608:+1
  ACTION: BUY  1 QM 202609 [LMT ALGO]
  ACTION: SELL 1 QM 202608 [ROLL CLOSE LMT ALGO]
```
The daemon believed it held `{202609:+1, 202608:+1}` on **every** cycle and never saw its own fills. Over ~26 cycles the true position drifted to **+27 QM Sep / −27 QM Aug** (net +1, but **54 contracts gross** — a needless calendar spread) while the log kept printing `held +1`. Each cycle cost ~$2/contract commission + aggressive-fill slippage (most rolls switched to AGGRESSIVE).

**Root cause.** `get_positions_by_instr` (live_dynamic.py:447) reads `ib.positions()`, which in ib_insync returns a **cached** list maintained by the `reqPositions` subscription that `connectAsync` starts — it is *not* a fresh query. The daemon comment at live_dynamic.py:1154 ("Fetch **fresh** positions") is wrong. The cache froze at the run's first value `{202609:+1, 202608:+1}` and never updated, even across the 22:30Z reconnect (the log shows `Peer closed connection.` / `IB disconnected — reconnecting…` around the gateway's nightly-reset window). The daemon then:
- saw Aug as `+1` (long) → `old_months={202608:+1}` → "ROLL CLOSE **SELL** 1 Aug",
- but Aug was actually short, so the SELL **opened/extended** the short (−25→−26→−27),
- and the Sep BUY drove Sep +25→+26→+27.

So the reconcile logic was correct *given its inputs*; the inputs were a stale snapshot detached from reality. A fresh IB client always saw the true `+27/−27` (verified three times with throwaway clientIds) — proving the staleness is specific to the long-lived clientId-5 daemon session, not IB.

**Why no self-heal / no alert.** Nothing re-requests positions; `reqPositions` is never explicitly called anywhere in the futures path (only `ib.positions()` reads, live_dynamic.py:447). On reconnect the daemon builds a fresh `IB()` but never forces a position re-seed + settle before the first read, and there is no sanity check that "held" is consistent with the orders we just placed (e.g. we just BOUGHT Sep yet Sep didn't increase).

**Immediate remediation (done 2026-06-16, on VPS):**
- Wrote `ibkr_fut/risk_halt.txt` (manual halt) and killed daemon PID 306866. Watchdog respects the halt file (watchdog.py:107) and will not restart.
- Cancelled the orphaned working order left by the killed daemon (`orderId 220 BUY 1 QM Sep LMT 78.075`, clientId 5) — had to reconnect **as clientId 5** to cancel it; a cross-client `cancelOrder` was a no-op and `reqGlobalCancel` was (correctly) blocked.
- Flattened the spread with two market orders (SELL 26 QMU6, BUY 27 QMQ6) → **final position +1 QM Sep, 0 Aug = the true target**, 0 open orders. Other holdings (NZD −1, JPY −1, DAX −1) match the snapshot's `target holds 4`.
- **Daemon remains halted** pending the code fix below.

**Origin trace (how the Aug residue was born, 2026-06-11).** This was *not* a one-cycle accident — the stale-position root cause was already biting during 6/11's legitimate roll window (`[SPREAD d=0]`). In overlapping cycles 22:33–23:15Z the daemon fired an outright `BUY 1 QM 202609`, a `SPREAD ROLL 1 202608→202609` (= SELL Aug + BUY Sep), *and* a `SELL 1 202607` roll-close — because it couldn't see its own in-flight fills, it double-acted and left a residual short Aug leg. When the 6/15 daemon started it inherited that `{Sep:+1, Aug:+1}`-shaped residue, froze on it, and amplified it to +27/−27. So the frozen-cache bug both *created* and *fed* the spread.

**Fix implemented 2026-06-16 (commit pending) in `live_dynamic.py` + `test_execution.py`:**
1. **Live position re-request every cycle, bounded.** New `fetch_positions(ib)` re-requests positions authoritatively via `ib.run(asyncio.wait_for(ib.reqPositionsAsync(), POSITIONS_TIMEOUT_SECS))` instead of the passively-cached `ib.positions()`. `get_positions_by_instr` now reads through it, so the daemon can never act on a frozen snapshot again. **The `wait_for` bound is load-bearing:** `IB.RequestTimeout` defaults to 0 = *no* timeout, so a half-open gateway (TCP up, API silent — the very state that caused this bug) would hang the `positionEnd` await and block the whole daemon forever; the 15s cap raises `TimeoutError` and falls back to the cache, so a stuck/transient connection degrades to "possibly stale", never "hung". The reconnect path is covered for free: each cycle re-requests, and `_connect()`'s `ib.sleep(2)` settles the fresh connection first. (Code-review catch: the first cut used the unbounded `ib.run(ib.reqPositionsAsync())`, which had the indefinite-hang flaw.)
2. **Per-instrument churn circuit-breaker.** `MAX_INSTR_TRADES_PER_SESSION = 6`. `reconcile_and_execute` now returns the set of instruments it placed live orders for; the daemon counts cycles-with-a-trade per instrument and, if any exceeds the cap, halts + alerts via the shared `risk_check.raise_halt(reason, alert)` helper (extracted in this change — `check_daily_loss` now uses it too, replacing its inline halt-file-write + Discord), then `sys.exit(1)`. The counter resets when a fresh snapshot loads (a new day's legit rebalances aren't charged against yesterday). This would have stopped the 26-cycle bleed after ~6.
3. **Tests:** `test_fetch_positions_bypasses_stale_cache` / `..._falls_back_to_cache_on_error` / `..._falls_back_on_timeout`; all `get_positions_by_instr` + `reconcile_and_execute` tests updated for the live-request path and 4-tuple return (also fixed two *pre-existing* failures in those position tests where `reqContractDetails` wasn't stubbed). Full suite: 105 passed.

**Deferred (not blocking restart, lower-leverage):** a strict post-fill direction guard ("we BOUGHT Sep ⇒ Sep must not be unchanged/decreasing next read") — the live re-request + churn cap already break the loop; the direction guard is a tighter, single-cycle stop worth adding later.

**Verification owed before removing the halt + restart:** deploy to VPS (git pull), run the daemon, force a gateway disconnect/reconnect, place a fill, and confirm the next cycle's `held` reflects the fill (no repeated identical roll). Confirm the churn cap halts a deliberately-looped instrument. Then remove `risk_halt.txt` and restart via `run_execution.sh`. **Note:** the QM position was already manually flattened to the true target (+1 Sep) on 6/16, so the daemon will start from a clean book.

---

### HIGH

#### [BUG-8] `pst_updater` PST-STALE Discord alerts silently swallowed (`No module named 'ibkr_fut'`)

**Found 2026-06-20** investigating the daemon's repeating "snapshot PST data is from 2026-06-17, not today" warnings.

**Symptom.** Every `_alert_stale` call in `pst_updater.py` logged `staleness alert failed (No module named 'ibkr_fut')` and **no Discord alert was ever sent**. Visible all over `pst_updater.log` (e.g. `LEANHOG: staleness alert failed`, `BBCOMM: staleness alert failed`, `CHEESE`, `RUBBER`…). This is the exact early-warning channel meant to flag wrong-contract / data-gap problems on near-monthly instruments (the QM-class issue the volume-roll change addressed) — and it was dead.

**Root cause.** `_alert_stale` does a lazy `from ibkr_fut.risk_check import _send_discord`, but `pst_updater.py` is launched as a top-level script (`run_dynamic.sh` runs `python3 ibkr_fut/pst_updater.py …`), so the repo root was never on `sys.path` and the `ibkr_fut` package wasn't importable. Every other module in `ibkr_fut/` inserts the repo root at import time (`sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`); `pst_updater.py` was the one that didn't. Pure packaging oversight — unrelated to the volume-roll logic, but introduced/exposed when `_alert_stale` was added in the roll-updater commit (78d9345).

**Fix (commit pending).** Added the standard repo-root `sys.path.insert` at the top of `pst_updater.py` (right after the stdlib imports). Verified the lazy import resolves (in production on the 06-21 run: many `[PST-STALE]`/`[PST-ROLL]` alerts fired, zero `staleness alert failed`). The roll/splice logic itself was correct on 06-18 (LEANHOG/LIVECOW/FEEDCOW rolled as designed).

**Follow-up (alert-tag split, 2026-06-21).** Once the alert actually fired, it cried wolf: `_alert_stale` tagged *every* successful roll `[PST-STALE]` and Discord-alerted it, even though a roll ending the old contract's series is the desired routine behaviour (the "Nd behind" is just the weekend gap — a Friday last-bar shows "2d behind" on a Sunday). Split the function: a successful roll (`action` = `volume-rolled`/`expiry-rolled`) is now tagged `[PST-ROLL]` and **log-only**; only the genuine no-roll data gap (`action` = `no-roll: …`, front stale AND forward not liquid enough to splice) stays `[PST-STALE]` and escalates to Discord. So Discord now alerts only on real gaps, not on every roll cycle.

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

#### [BUG-4] `pst_updater` prices a dead (pre-roll, no-trade) contract → flatlined series on near-monthly instruments

**Re-diagnosed 2026-06-12** — the original "BRE has no IBKR data / remove it" framing was *wrong*. IBKR data is fine; the updater is asking for the wrong contract. NOT a subscription gap, NOT a symbol/mapping error.

**Real root cause.** `update_prices` seeds `build_schedule` from the CSV's last `PRICE_CONTRACT` and rolls forward by **expiry-based** date (`RollOffsetDays=-5`, i.e. roll 5 days before expiry). But thinly-traded near-monthly contracts go **no-trade ~2–3 weeks before expiry**. That opens a dead window where the schedule still prices the *expiring* contract (which now returns only stale `TRADES`) while the genuinely-liquid next contract is never fetched. The series flatlines until the calendar finally rolls.

Proven for BRE on 2026-06-12 (gateway up):
```
CSV last PRICE_CONTRACT = 20260600 (June)   ← schedule seeds from June
build_schedule → [('202606', 2026-05-30 → 2026-06-12)]   ← June roll_date is after the window, so it never advances
fetch_bars(202606, TRADES) → 1 bar, last 2026-06-01       ← June died after 6/01
fetch_bars(202607, TRADES) → 11 bars, last 2026-06-12     ← July (liquid, vol 27k) is RIGHT THERE, never requested
```
The nightly run logs `1 bars … last price 0.1978` and **`0 failed`** — the failure is invisible to the run's own accounting (no exception, the fetch "succeeded" with stale data).

**Blast radius (universe scan 2026-06-12, `multiple_prices` PRICE staleness over the 105-instr UNIVERSE):** 4 stale ≥3d, three of them this same bug, one a genuine data gap:
- **BRE** (CME 6L, monthly): dead-contract bug above. Self-heals once the calendar rolls to July, recurs every roll. Data exists.
- **FTSEINDO** (SGX WIIDN): same — schedule advanced to `202607` which has *no front data* yet; carry pass even queried an expired `202503`. Data exists on the liquid month.
- **ETHER-micro** (CME MET): *transient* HMDS backfill lag on 6/05; a manual re-run on 6/12 fetched 8 fresh bars and fully healed it. Not the dead-contract bug.
- **TWD-mini** (SGX TD, mini): **genuine gap** — `TDM26/TDN26/TDQ26@SGX` all return HMDS "no data" for both TRADES *and* MIDPOINT. Forward months simply aren't served. This is the real "illiquid/untradeable" case BUG-4 originally feared — but it's TWD-mini, not BRE. (The non-UNIVERSE Jumbo instruments showing 6/05 are *expected* stale — the nightly cron only fetches `UNIVERSE`, by design.)

**Impact**: any near-monthly UNIVERSE instrument silently flatlines for ~1–2 weeks every roll cycle. A flat price series → zero EWMAC signal + understated vol → mis-sizing and a stale carry leg, with no alert (`0 failed`). Recurs 12×/yr per affected instrument.

**Fix decision (settled 2026-06-12 by auditing pysystemtrade — `pst-group/pysystemtrade`, the source of all pre-Mar-2024 data):**

What pysystemtrade (Carver's production system) actually does:
- **`whatToShow`: TRADES for futures, MIDPOINT for FX** (`sysbrokers/IB/client/ib_price_client.py:41`, `ib_fx_client.py:89`) — *identical* to `pst_updater`'s convention. Our data collection is already in parity with the inherited history; a MIDPOINT switch (old option 2) is ruled out — it would *break* parity, and it also returns 0 bars for TWD-mini.
- **Rolls the priced contract on VOLUME, not the expiry calendar.** Production roll state comes from `update_roll_status` with auto-roll defaults `auto_roll_if_relative_volume_higher_than: 1.0` (roll when forward volume exceeds priced volume), `min_relative_volume: 0.01`, `min_absolute_volume: 100`, `auto_roll_expired: True` (`sysdata/config/defaults.yaml:79-88`). The `RollOffsetDays` CSV is only the *approximate/backtest* calendar; roll calendars built from prices are then `adjust_to_price_series`'d onto dates where both contracts actually have data.
- **Samples the whole contract chain daily** (`update_sampled_contracts` + `update_historical_prices`), so the next contract's history already exists whenever a roll registers.

⇒ The "backtest parity" worry was **inverted**: the pre-2024 multiple_prices embed Carver's *liquidity-driven* production rolls. Our expiry-offset `build_schedule` is the deviation, and the dead-window flatline is its direct consequence.

**Chosen fix (3 parts, in `pst_updater.py`):**
1. **Stale-front early roll in `update_prices`.** Front *and* forward are already fetched per segment (zero extra IB requests). If the front's last bar is materially older than `seg_end` while the forward has fresh bars beyond it, roll early: split the segment at the latest front/forward common date and continue with the forward as front. `compute_adjustments` already splices on the latest common date, so the Panama adjustment needs no change. Healthy instruments still roll on the calendar exactly as before — near-zero blast radius. (Full volume-crossover rolling à la Carver = future enhancement; `fetch_bars` would just need to keep the `volume` column.)
2. **Roll-calendar lockstep.** When an early roll fires, rewrite the affected `roll_calendars_csv` row to the actual roll date — `live_dynamic.get_roll_info` drives the *held* month and spread-roll timing from that file, so the traded contract must advance with the priced one or data and execution diverge.
3. **Loud staleness alert** (old option 4): Discord warning whenever a scheduled front returns a series whose last bar is materially older than `seg_end` — converts the invisible `0 failed` into a visible signal even when the early roll can't help (e.g. genuine gaps).

**Out of scope of the code fix:** TWD-mini — forward SGX data genuinely absent for TRADES *and* MIDPOINT; removal candidate pending a front-month tradability check. Paid data (Norgate/Databento) was considered and rejected as the fix: this is a roll-*pointer* bug, not a data-quality bug — the same schedule would have requested the same dead contract from any vendor, and Carver-style carry needs per-contract prices regardless. IB already returns volume in the bars we fetch, free.

**Verified working today**: re-running `pst_updater BRE ETHER-micro FTSEINDO TWD-mini --carry` healed ETHER-micro (8 bars→6/12); BRE/FTSEINDO still stuck (dead-contract bug persists until roll); TWD-mini still 1 bar (genuine gap).

---

### MEDIUM

#### [OBS-12] Daily report flagged weekday-only crons as failed over weekends (false ✗ → alert fatigue)

**Found 2026-06-29.** The daily monitoring report's CRON HEALTH section judged every job against a flat `max_age_hours` (26h). But `pst_updater`/`compute`/`etf_daily` are **Mon–Fri** jobs, so on a Monday morning the last legitimate run was the previous **Friday** (~64h earlier) — the report rendered three `✗ 64h ago` failures every weekend. This is the alarm that kicked off the whole 2026-06/07 reliability investigation, and it was itself a false positive: nothing had failed. Chronic false ✗ trains you to ignore the report, masking real signal.

**Fix (commit pending, `scripts/daily_report.py`).** Replaced the flat-age check with a **schedule-aware expected-last-run**: each cron carries `(days, hour_utc, grace_h, trading)` and the report computes the most recent datetime it was *scheduled* to run (honouring weekday-only schedules and, via the existing `ibkr_fut/trading_calendar.py` CME calendar, skipping holidays), flagging ✗ only when the log predates that expected run + grace. A Fri-only-last-run job is healthy all weekend; a genuinely missed Friday run still flags. The long-running `daemon` keeps its own 24h cycle-count check. Verified against live VPS logs: the report that previously showed 4 warnings/3 false ✗ now shows all-green with zero warnings, while a simulated genuine miss (log stuck at Thursday when Friday was due) correctly flags. Also handles the degenerate no-calendar case (falls back to a plain weekday check) so the section never crashes.

#### [OBS-13] No Gateway-uptime visibility; transient READFAIL blips paged Discord (alert fatigue)

**Added 2026-07-01** as part of the observability pass (Priority 2 of the reliability plan).

Two hygiene gaps: (1) there was **no at-a-glance view of Gateway restart frequency** — you only learned the Gateway had restarted when a `[DAEMON-READFAIL]` happened to land on an open market. (2) The daemon paged Discord on **every** single read failure / book mismatch, even a one-off blip that self-heals next cycle (the 2026-06-29 18:45 READFAIL recovered by 18:47 yet still pinged).

**Fix (commit pending).**
- **`scripts/daily_report.py` — new `section_gateway_health`.** Parses the IBC session-log roll files (`~/Jts/<settings>/ibgateway.<YYYYMMDD>.<HHMMSS>.ibgzenc`, one per Gateway start, UTC) to report restarts in the last 24h and flag any that landed **in the trading window** (22:00–07:00 UTC). A lightweight uptime SLO: after the 2026-07-01 TZ fix this should show a single off-window restart at 07:00 UTC; an in-window restart is now the alarm. On first run it correctly surfaced the pre-fix state (4 restarts, 3 in-window — the old midnight pair).
- **`ibkr_fut/live_dynamic.py` — streak-gated READFAIL/MISMATCH alerts.** A read failure / suspect-book mismatch is now **log-only** for the first cycle; Discord escalation fires only after `READFAIL_ALERT_AFTER = 2` **consecutive** failing cycles (a real sustained outage), with a matching `[DAEMON-RECOVERED]` when the read reconciles (mirrors the 1102 "connectivity restored" pattern and the `[PST-ROLL]`/`[PST-STALE]` split from BUG-8). **The safety behaviour is unchanged** — the daemon still skips the cycle and never trades off a bad read on the very first failure; only the *alert* is gated. 130 execution tests still pass. (Loop-level unit test deferred to the Phase-1 stateless-worker refactor, which makes the cycle body directly testable.)

---

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

Not a correctness issue — the daemon waits correctly — but generates noise and Discord alerts. (The *false-on-weekends/holidays* flavour of this warning is fixed by OBS-10; the ~8-min start-up race described here is still separate and open.)

---

#### [OBS-10] Daemon staleness gate compared snapshot date to `date.today()` — false "pst_updater may have failed" every weekend/holiday

**Found 2026-06-20.** The daemon gated execution on `pst_date != date.today()` (live_dynamic.py ~1197). But the snapshot's `date` is the **last PST bar** it was built from, and a CME Globex *session* is named by its 18:00 ET close: the compute runs at 18:00 ET and sizes off the just-settled close, so on a **Sunday-evening run the freshest legitimate data is the previous Friday's session**. Comparing that against `date.today()` (Sunday) made the daemon declare a perfectly fresh snapshot "failed" and refuse to trade — which is exactly what was filling `daemon_cron.log` on Sat/Sun 06-20/21 (snapshot legitimately at Fri's data, "today" ≠ Fri). CME also trades shortened sessions on some US holidays (e.g. Juneteenth 06-19), so "is the market closed today" is the wrong question too.

**Fix (commit pending).** New `ibkr_fut/trading_calendar.py` wraps the `pandas_market_calendars` `CMES` (CME Globex) calendar and exposes `last_completed_session()` / `sessions_behind()` / `is_snapshot_fresh()` (with a 20-min settlement grace so the gate never demands data inside the close→pst_updater window). The daemon now trades whenever `sessions_behind(pst_date) == 0` (snapshot == last completed session) and only skips + Discord-alerts (once per stale date, via the now-working `_send_discord`) when it genuinely lags. Verified: Fri-dated snapshot evaluated Sunday 18:00 → FRESH; Thu-dated → STALE(1); the real 06-17 snapshot → STALE(2). Dependency pinned in `requirements_live.txt`. Paired with the cron change (compute Mon–Fri, execute Sun–Thu) so Friday's compute feeds Sunday's open.

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

#### [BUG-6] Daily report compares targets (PST name) against held (IB symbol) — phantom mismatches, no real reconciliation

**Resolved 2026-06-12** in `scripts/daily_report.py` + `ibkr_fut/live_dynamic.py` (commits cfd0eb2 + d6b5599, deployed to VPS). Plus a one-time VPS ledger reseed (gitignored).

**Symptom**: the FUTURES POSITIONS section keyed **held** by IB symbol (`held[c.symbol]`) but **targets** by PST instrument name, two unmapped namespaces. The 2026-06-12 report rendered a correct, matched `+1 QM` (= `CRUDE_W_mini`) as *two* false `← MISMATCH` lines. Secondary symptoms: equity stuck at the pre-reset `$107,836` (stale `state.json`, `log_daily` not flushing since 6/10) and a hardcoded `(dry-run)` tag even though the daemon was live.

**Root cause**: the report reimplemented position-fetching with a bare `c.symbol` key, skipping the `(symbol, exchange) → PST instr` reverse map that `get_positions_by_instr` already builds. Equity read a frozen `last_equity`; the dry-run tag was a literal string.

**Fix**:
- Extracted `ib_symbol_to_instr(ibcfg)` in `live_dynamic.py` (shared symbol→instr map; ambiguous symbols dropped); `get_positions_by_instr` now uses it.
- `daily_report.py`: added cached `_fetch_ib_live()` (one IB connection, clientId 20) returning `(equity, held_by_instr, unknown)`. `section_futures_positions` now compares held vs target in the PST namespace and surfaces unmapped positions as warnings; CSV fallback translates IB symbols via `ib_symbol_to_instr`. `section_pnl_summary` prefers live NetLiquidation (`[live]` vs `[cached]`), warns on >2% drift. `_detect_daemon_mode()` derives the LIVE/DRY-RUN tag from the daemon log (shared with `section_daemon_summary`).
- Daemon `log_daily` failure now logs `traceback.format_exc()` (so the next missed flush is diagnosable).
- VPS-only: archived corrupt `state.json`/`daily.csv` (→ `.pre-reset-20260611`), reseeded inception `$250,088.90` @ 6/11 nav 1.0 + a 6/12 row at live `$248,908.68`. `trades.csv` preserved.

**Verification**: report now shows `CRUDE_W_mini`/`QM` collapsed to one matched line, zero false mismatches, `$248,941 [live]`, no `(dry-run)` suffix, `(LIVE)` daemon. Post-cleanup the equity-drift warning disappeared (4→3 warnings; the remaining 3 are legitimate: BRE stale, ETHER-micro stale, daemon errors).

---

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

Daemon restarted at 22:15 before the compute phase finished at 22:23, so it briefly saw the prior day's snapshot and emitted stale-PST warnings. Resolved automatically when it picked up the fresh snapshot at 22:23:52Z. (Underlying timing window tracked as OBS-3.) **Fixed 2026-06-26** by the compute-in-progress marker (`computing.lock`) in BUG-10 fix part 2.3: the daemon now waits quietly while compute is mid-run instead of false-alarming, so this window no longer emits a `[DAEMON-STALE]` alert or briefly reads flat.

#### Qualify failures at 22:23 cycle (operational, self-resolved)

IB connection dropped mid-qualify, so all four targets failed to qualify that cycle. The daemon's reconnect logic re-established the connection at 22:33 and all qualifications succeeded.
