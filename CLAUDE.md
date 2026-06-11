# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Always activate the venv before running any Python:
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements_live.txt       # live trading + backtesting stack
pip install -r requirements_scraper.txt    # x_scraper.py only
```

## Running Things

```bash
# Paper trading (crypto): run today's bar + report
python3 paper/run_paper.py
python3 paper/run_paper.py --report-only
python3 paper/run_paper.py --asof 2026-05-29

# Futures EWMAC live (dry-run by default, --execute to trade)
ibkr_fut/run_dynamic.sh

# ETF EWMAC live (dry-run by default, --execute to trade)
python3 etf/live/run_daily.py --execute

# ETF nightly data refresh
python3 etf/live/refresh_data.py

# Account summary (requires IB Gateway on port 4002)
python3 scripts/account_summary.py

# Rob_port backtests (Carver framework)
python3 research/rob_port/chapter4.py

# Crypto backtests
python3 research/crypto/crypto_trend.py
```

## Architecture Overview

There are three independent live-trading pipelines, each with its own ledger and data source:

### 1. Crypto Paper Trading (`paper/`)
- **Entry point:** `paper/run_paper.py` (cron: 6 AM ET daily)
- **Engine:** `paper/engine.py` — capital-agnostic, return/NAV-based. Each strategy tracks growth-of-$1 so the track record is unaffected by reallocation.
- **Strategies:** `paper/strategies.py` — `CryptoTrendStrategy` (EWMAC ensemble, long/flat, 10 coins via Coinbase public API) and `CryptoCarryStrategy` (funding-rate carry, long spot / short perp).
- **Config:** `paper/portfolio.json` — total capital + per-strategy weights. Editing this file is the only way to reallocate; the underlying ledgers are untouched.
- **Ledgers:** `paper/ledgers/<strategy_name>/` — `state.json` (resume point), `daily.csv`, `positions.csv`, `trades.csv`.

### 2. Futures Mutli-Strategy Dynamic - 250k starting balance (`ibkr_fut/`) - THIS IS THE MAIN TRADING SYSTEM AND MAIN FOCUS
- **Instrument universe** `ibkr_fut/instrument_universe.py` contains the instruments my system has currently determined are valid to trade. Jumbo universe is not the main universe.
- **Entry point:** `ibkr_fut/live_dynamic.py` (cron: 6 PM ET weekdays, run via `run_dynamic.sh`)
- Reads capital from IBKR (port 4002), builds a Carver "Jumbo" universe from PST CSV data, runs joint portfolio optimisation (`dynamic_opt.optimise_positions`), reconciles target vs held, and submits DAY market orders.
- **Data pipeline:** `ibkr_fut/pst_updater.py` must run before `live_dynamic.py` to pull fresh PST closes.

### 3. ETF EWMAC Live (`etf/live/`)
- **Entry point:** `etf/live/run_daily.py` (cron: 3:55 PM ET weekdays, executes via Alpaca)
- Fetches live prices from yfinance at 3:58 PM, computes EWMAC signals on CSV history, places market orders via Alpaca for essentially the closing price.
- **Shortability check:** `etf/live/check_shortability.py` runs at 3:50 PM before trading.
- **Nightly refresh:** `etf/live/refresh_data.py` keeps CSV history current (6:30 PM ET).
- **Ledger:** SQLite, exported to `paper/` book format.
- **Env:** Alpaca credentials in `.env`; loaded via `etf/live/_env.py`.

### Backtesting & Research (`research/`)
- **`research/rob_port/`** — Robert Carver "Systematic Trading" framework: `chapter1.py`–`chapter10.py`, handcrafted weights, dynamic optimisation, EWMAC signals across 75 instruments.
- **`research/crypto/`** — crypto strategy research (trend, carry, mean-reversion backtests). `paper/strategies.py` imports `crypto_trend.py` from here.
- **`research/etf/`** — ETF backtest scripts (EWMAC vol-adj, MR, greedy selection). `etf/live/` imports `ewmac_backtest.py` from `etf/live/`; research scripts reference it via sys.path.
- **`research/es/`, `research/nq/`, `research/cl/`, `research/orb/`** — standalone per-instrument backtests.
- **`archive/`** — failed experiments (`archive/failed/`) and miscellaneous one-offs (`archive/misc/`).
- **`Data/`** — market data warehouse: daily OHLCV CSVs, binance spot/perp/funding data, Coinbase candles.

### Data Flow
```
ibkr_fut/pst_updater.py → Data/<instrument>_daily_data.csv
                      ↓
ibkr_fut/live_dynamic.py  (futures, IBKR)
rob_port/backtest_dynamic.py  (backtests)

etf/live/refresh_data.py → etf/live/data/<etf>.csv
                               ↓
etf/live/run_daily.py  (ETF, Alpaca)

Coinbase API (real-time) → paper/engine.py → paper/ledgers/
```

## Key Files

| File | Purpose |
|------|---------|
| `paper/portfolio.json` | Capital and strategy weights for crypto paper trading |
| `ibkr_fut/jumbo.py` | Builds the full ~150-instrument Carver universe |
| `ibkr_fut/foundations.py` | Core Carver math: EWMAC, vol scaling, handcraft weights |
| `ibkr_fut/dynamic_opt.py` | Joint portfolio optimiser (IDM, SR-cost aware) |
| `ibkr_fut/pst_updater.py` / `ibkr_fut/pst_loader.py` | Price series table: fetch + load PST CSVs |
| `etf/live/_env.py` | Loads `.env` for Alpaca API keys |
| `crontab_vps.txt` | Canonical VPS cron schedule (America/New_York TZ) |

## IB Gateway Ports
- **4002** — paper trading
- **4001** — live trading
