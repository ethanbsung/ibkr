# archive/

Kept deliberately. Nothing here is on cron, imported by the live system, or maintained.

## `failed/`

Strategies that were built, backtested, and **rejected** — mean reversion, VWAP,
breakout, Donchian, genetic-algorithm parameter search, and various ES/NQ intraday
ideas. They are kept because the record of what was tested and discarded is part of
the research process: it is why the live system trades a slow carry + trend portfolio
and not one of these.

These are point-in-time scripts, not maintained code — the saved result files next to
them are stale or empty, so treat any numbers they print as unreproduced.

## `misc/`

One-off scripts, connection tests, and exploratory notebooks-as-scripts from early
work. No strategy claims attached.

## `retired/`

Superseded **live** modules — code that really did trade, then was replaced:

| File | What it was |
|------|-------------|
| `live_signals.py` | The IBS (internal bar strength) strategy, run per-instrument near the close. Replaced by `ibkr_fut/live_dynamic.py`. |
| `run_ibs.sh` | Cron launcher for the above. |
| `live_port.py` | Single-strategy portfolio sizing / order placement against IBKR. |
| `live_port_pc.py` | A per-contract variant of the same. |

Retired 2026-08-02 (see `OBS-21` in
[`ibkr_fut/notes/live_system_issues.md`](../ibkr_fut/notes/live_system_issues.md)).
`live_signals.py` still contains the known-wrong `[:6]` contract-month parse described
in `BUG-1`; it is archived rather than fixed, because the module no longer runs.
