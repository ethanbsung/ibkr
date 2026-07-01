# Systematic Futures Trading System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Broker](https://img.shields.io/badge/Broker-Interactive%20Brokers-green.svg)](https://www.interactivebrokers.com/)
[![Environment](https://img.shields.io/badge/Environment-IBKR%20Paper-orange.svg)](https://www.interactivebrokers.com/)

An autonomous, end-to-end systematic futures trading system that runs unattended
24/5 on a cloud VPS, connecting to the Interactive Brokers API to size, reconcile,
and execute a diversified multi-asset futures portfolio each session. The design
follows Robert Carver's *Advanced Futures Trading Strategies* framework — a
combined **carry + trend** forecast, volatility-targeted position sizing, and a
**joint portfolio optimiser** — reimplemented from scratch in NumPy/Pandas.

> **Trading environment.** The system runs live against IBKR's **paper-trading**
> Gateway (port 4002) — real broker infrastructure, real market data, real order
> routing and fills, but **simulated capital**. It executes autonomously against a
> ~$250K paper account; it has not been run against a real-money account. All
> performance-sensitive claims below refer to backtests or the paper deployment
> and are labelled as such.

---

## What it does, each session

1. **Reads state from the broker** — connects to IB Gateway and reads live account
   equity (NetLiquidation) and current futures positions.
2. **Refreshes data** — incrementally pulls the latest daily closes for the traded
   universe, applies Panama back-adjustment, and rolls contracts on a
   volume/liquidity basis (mirroring pysystemtrade's production roll logic).
3. **Generates forecasts** — a combined carry + trend forecast per instrument
   (EWMAC trend ensemble across multiple speeds + term-structure carry), with
   cost-based rule eligibility and forecast diversification multipliers.
4. **Sizes and optimises** — volatility-targets each position, then runs **one
   joint daily optimisation** that picks the integer position vector minimising
   tracking-error variance against the ideal risk-parity portfolio, subject to a
   trading-cost penalty and turnover buffering.
5. **Reconciles and executes** — diffs target vs. held positions (handling contract
   rolls), runs pre-trade risk checks, and submits orders through a
   passive-aggressive limit-order algorithm, one exchange session at a time across
   global time zones.
6. **Monitors** — logs every fill and a daily snapshot, reconciles the book, and
   pushes a daily P&L / position report to Discord.

The compute phase and the execution daemon are split: compute persists a target
snapshot; a long-running daemon executes it, picking up fresh targets automatically
and only trading instruments whose exchange is currently open.

---

## Highlights

- **Live, autonomous operation.** Runs unattended 24/5 on a VPS, scheduled around
  the CME close and the sequence of Asia/Europe/US session opens. Contract rolls,
  reconnects, and multi-time-zone execution are all handled without intervention.

- **Joint portfolio optimiser (from scratch).** A clean NumPy reimplementation of
  Carver's dynamic optimisation: a greedy integer optimiser that minimises
  tracking-error variance vs. the ideal unrounded portfolio, with a cost penalty
  and tracking-error-space buffering to suppress needless turnover. The objective
  is evaluated **incrementally in O(1) per one-contract move** (gradient update
  `V_new = V + 2·d·g[i] + d²·Σ_ii`), reducing each optimiser pass from O(n²) to
  O(n).

- **Optimise-over-many, trade-a-subset.** The optimiser reasons over the full
  ~100-instrument universe but only *trades* the eligible subset. Non-tradable
  instruments are handled with `locked` / `reduce_only` constraints so their risk
  transfers onto correlated, tradable neighbours through the off-diagonal
  covariance — and a position stranded by an instrument dropping out of the
  tradable set is unwound rather than frozen.

- **Research-to-production parity.** The backtester and the live system share the
  *exact same* universe-build, signal, sizing, and optimiser code path — the live
  eligible universe is filtered identically to the backtest, so there is no
  research-to-live gap.

- **Fail-safe reliability engineering.** The live system treats broker/network
  failures as adversarial. Guards include verified-position-read aborts (never
  trade off a stale or phantom-flat read), a halt-on-book-mismatch gate, a
  per-instrument churn circuit-breaker, a mass-liquidation limit, daily-loss and
  gross-leverage checks, a file-based kill switch, and a process watchdog. Every
  serious production incident is documented with root cause and fix in
  [`ibkr_fut/notes/live_system_issues.md`](ibkr_fut/notes/live_system_issues.md),
  and each becomes a regression test.

- **Tested.** ~200 automated tests in the futures system alone (≈130 in the
  execution suite), covering the optimiser, execution/reconciliation, contract
  rolls, the trading calendar, and each of the failure-mode guards above.

---

## Strategy & math

The framework implements Carver's *Advanced Futures Trading Strategies* end to end:

- **Forecasts** — EWMAC trend across multiple speeds (raw → risk-scaled → capped)
  blended with term-structure carry (Strategy 11: combined carry + trend). Trading
  rules are only activated when their risk-adjusted (SR-unit) cost clears a
  threshold; forecasts are combined with diversification multipliers and capped.

- **Position sizing** — volatility targeting to a portfolio risk target, using a
  blended volatility estimate (short-run EWMA + long-run average), an instrument
  diversification multiplier (IDM) derived from the actual correlation matrix, and
  correlation-clustering **handcrafted instrument weights**.

- **Portfolio construction** — a joint dynamic optimiser with time-varying
  covariance: weekly-EWMA correlations shrunk toward the identity (guaranteeing a
  positive-definite matrix) combined with daily-EWMA volatilities.

- **Execution** — Carver's passive-aggressive limit-order algorithm: rest passively
  at the offside price, then chase the spread aggressively on a timeout or on
  adverse price / order-book-imbalance triggers, with pre-trade sanity checks
  (reject on missing two-sided quotes or a >3σ divergence from the sizing close)
  and per-exchange trading-calendar awareness.

---

## Universe & data

- **~100 futures instruments across 8 asset classes** — equities (US / Europe /
  Asia, incl. sector indices), rates & bonds, FX (majors + crosses/EM), energy,
  metals, agriculturals, volatility, and crypto — mapped onto Carver's "Jumbo"
  universe.
- **Data warehouse** of daily OHLCV histories with **2–40 years** of data per
  instrument, kept current by a nightly incremental ingestion pipeline
  (Panama back-adjustment, volume-driven rolls, staleness / dead-contract
  detection, and data-health alerting).

---

## Architecture

The repository contains three independent trading pipelines, each with its own
ledger and data source. The **IBKR futures system is the primary focus.**

```
ibkr_fut/          # PRIMARY: live multi-strategy futures system (IBKR)
  live_dynamic.py    #   compute / execute / daemon entry point
  dynamic_opt.py     #   joint portfolio optimiser (tracking-error, cost, buffering)
  foundations.py     #   core Carver math: EWMAC, vol scaling, sizing, IDM, handcraft
  carry_trend_signals.py  # combined carry + trend forecast engine
  algo_execution.py  #   passive-aggressive limit-order algorithm
  pst_updater.py     #   nightly price ingestion + volume-driven rolls
  backtest_dynamic.py#   backtester sharing the live signal/sizing/optimiser path
  risk_check.py / watchdog.py / trading_calendar.py  # risk controls & ops
  notes/live_system_issues.md   # production incident log (root cause + fix)

etf/live/          # ETF EWMAC strategy executed via Alpaca
paper/             # crypto trend + carry paper-trading engine (Coinbase)
research/          # backtests & research (Carver framework, per-instrument studies)
Data/              # market-data warehouse (gitignored on the live host)
```

### Key technologies

- **Python** with **NumPy / Pandas / SciPy** for the quantitative core
- **ib_insync** for the Interactive Brokers API
- **pandas-market-calendars** for exchange session / holiday handling
- **pytest** for the test suite; cron + systemd on the VPS for scheduling/supervision

---

## Notes & disclosure

This is a personal research and engineering project. It trades **simulated capital
in IBKR's paper-trading environment** — no real money is at risk, and nothing here
is investment advice. Backtested and paper-trading results do not represent
real-money performance and are subject to the usual caveats (costs, slippage,
capacity, and regime change). The code is shared to demonstrate systematic-trading
research, portfolio construction, and production trading-systems engineering.
