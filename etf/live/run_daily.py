#!/usr/bin/env python3
"""
Daily execution script for the ETF live trading system.

Run at 3:58 PM ET (2 minutes before close).  Flow:
  1. Batch-fetch live prices from yfinance (~3:58 PM) — in memory only, never written to CSV
  2. Compute EWMAC signals on CSV history + today's live price
  3. Place market orders — fill at essentially the closing price
  4. Record fills, positions, and P&L in the SQLite ledger; export paper/ book

CSV history is kept current by the SEPARATE nightly job (etf/live/refresh_data.py),
so yfinance history downloads stay off this trade path.

Cron (America/New_York):  58 15 * * 1-5

  cd /home/ethanbsung/ibkr
  source venv/bin/activate
  python3 etf/live/run_daily.py

Flags:
  --capital-override FLOAT  Override Alpaca equity with a fixed amount (testing only)
  --execute                 Place real orders on Alpaca (default is dry-run)
  --strategy NAME           Run only this strategy (default: all)
  --no-report               Skip printing the ledger report

Adding a new strategy:
  1. Create etf/live/strategy_<name>.py with a class that has:
         name: str
         def get_signals(self, capital, today_prices) -> dict[str, float]
         def get_metadata(self, capital, today_prices) -> dict[str, dict]
  2. Import it below and add to CAPITAL_SPLIT and build_strategies().
"""

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from etf.live.strategy_ewmac import EWMACStrategy
from etf.live.executor import Executor
from etf.live.ledger import Ledger
from etf.live.prices import get_live_prices

DATA_DIR = "Data/etf"

# No single order may exceed this fraction of total capital — defense-in-depth
# against a mis-sized signal.  EWMAC single-name targets are normally a few %.
MAX_ORDER_FRACTION = 0.25


# ── Strategy registry ─────────────────────────────────────────────────────────
# Keys must match strategy.name.  Values are fractions of total capital.
# Must sum to 1.0.  To add a strategy: import it, add an entry here.
CAPITAL_SPLIT = {
    "ewmac": 1.0,
    # "carry": 0.0,   # future strategy
}

LEDGER_PATH       = "Data/live/trading.db"
PAPER_LEDGER_ROOT = "paper/ledgers"   # export a return/NAV book here for paper/


def build_strategies(capital: float) -> dict:
    """Instantiate all active strategies with their capital allocations."""
    return {
        "ewmac": EWMACStrategy(vol_target=0.25),
        # "carry": CarryStrategy(),
    }


def main():
    ap = argparse.ArgumentParser(description="ETF live trading — daily run")
    ap.add_argument("--capital-override", type=float, default=None,
                    help="Override Alpaca equity with a fixed capital amount (testing only)")
    ap.add_argument("--execute",     action="store_true",
                    help="Place real orders (default is dry-run)")
    ap.add_argument("--strategy",    default=None,
                    help="Run only this strategy name")
    ap.add_argument("--no-report",   action="store_true",
                    help="Skip ledger report at end")
    ap.add_argument("--force",       action="store_true",
                    help="Run even if the market is closed or a run already happened today")
    args = ap.parse_args()

    dry_run = not args.execute
    mode    = "dry_run" if dry_run else "live"
    today   = date.today().isoformat()

    ledger   = Ledger(LEDGER_PATH)
    executor = Executor(dry_run=dry_run)

    # Capital = Alpaca account equity (NAV), unless overridden for testing
    try:
        alpaca_equity = executor.get_account_equity()
    except Exception as e:
        print(f"\n  ERROR: Could not fetch Alpaca equity: {e}")
        raise SystemExit(1)

    total_capital = args.capital_override if args.capital_override else alpaca_equity
    executor.max_order_notional = MAX_ORDER_FRACTION * total_capital

    # Market-calendar guard: never queue live DAY orders into a closed session
    # (holidays / early closes).  Dry runs and --force bypass this.
    if not dry_run and not args.force and not executor.is_market_open():
        print("\n  Market is CLOSED — aborting (holiday/early-close/after-hours). "
              "Use --force to override.")
        raise SystemExit(0)

    print(f"\n{'='*68}")
    print(f"  ETF LIVE TRADING  {today}  "
          f"{'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Alpaca equity: ${alpaca_equity:,.2f}"
          + (f"  (override: ${total_capital:,.2f})" if args.capital_override else ""))
    print(f"  Strategies: {list(CAPITAL_SPLIT)}")
    print(f"{'='*68}")

    # NOTE: CSV history is refreshed by the separate nightly job
    # (etf/live/refresh_data.py), NOT here — yfinance history downloads must stay
    # off the trade path.  The execution run reads those CSVs (current through
    # last night) and appends today's yfinance *current* price in memory.
    strategies = build_strategies(total_capital)
    all_fills  = []

    for strat_name, strategy in strategies.items():
        if args.strategy and strat_name != args.strategy:
            continue

        alloc   = CAPITAL_SPLIT.get(strat_name, 0.0)
        capital = total_capital * alloc
        if capital <= 0:
            continue

        print(f"\n── {strat_name.upper()}  capital=${capital:,.0f} ──")

        # ── Idempotency guard: don't place live orders twice for the same
        # date.  Dry runs place nothing, so they're always repeatable.
        if not dry_run and ledger.has_run(strat_name, today, "live") and not args.force:
            print(f"  SKIP {strat_name}: a live run already completed today "
                  f"({today}). Use --force to re-run.")
            continue

        # ── Fetch live prices (used as today's close proxy) ───────────────
        # yfinance current price at ~3:58 PM ET (Alpaca's IEX feed is delayed /
        # incomplete for some ETFs).  In-memory only; CSVs are never modified.
        print(f"  Fetching live prices for {len(strategy.tickers)} tickers…")
        today_prices = get_live_prices(strategy.tickers)
        print(f"  Got prices for {len(today_prices)}/{len(strategy.tickers)} tickers")

        # ── Generate signals ──────────────────────────────────────────────
        print("  Generating signals…")
        signals  = strategy.get_signals(capital, today_prices=today_prices)
        metadata = strategy.get_metadata(capital, today_prices=today_prices)

        signal_prices = today_prices   # same prices signals were computed on

        # Log signals to ledger
        ledger.record_signals(strat_name, metadata, signal_prices, today)

        # Print signal summary
        nonzero = {k: v for k, v in signals.items() if abs(v) > 1}
        long_n  = sum(1 for v in nonzero.values() if v > 0)
        short_n = sum(1 for v in nonzero.values() if v < 0)
        gross_long  = sum(v for v in nonzero.values() if v > 0)
        gross_short = sum(v for v in nonzero.values() if v < 0)
        gross_total = gross_long - gross_short
        print(f"  Signals: {len(nonzero)} non-zero  "
              f"({long_n} long, {short_n} short)")
        print(f"  Gross long  ${gross_long:>+12,.2f}")
        print(f"  Gross short ${gross_short:>+12,.2f}")
        print(f"  Gross total ${gross_total:>12,.2f}  "
              f"({gross_total / capital:.2f}x equity)")
        scale = getattr(strategy, "_last_scale", 1.0)
        if scale < 1.0:
            print(f"  ⚠ Leverage cap applied: positions scaled to {scale:.1%} "
                  f"(cap {strategy.gross_leverage_cap:.1f}x)")

        # ── Determine which held names to fully close ─────────────────────
        # A ticker held on Alpaca but no longer in the universe must be
        # unwound.  A ticker that is in the universe but simply missing a live
        # price today is NOT closed (it's absent from `signals` → held).
        held      = executor.get_current_shares()
        universe  = set(strategy.tickers)
        close_set = {tk for tk in held if tk not in universe and held[tk] != 0}
        if close_set:
            print(f"  Closing {len(close_set)} names dropped from universe: "
                  f"{sorted(close_set)}")

        # ── Execute ───────────────────────────────────────────────────────
        print(f"\n  Executing orders {'(DRY RUN)' if dry_run else ''}…")
        fills = executor.execute_targets(
            signals, signal_prices, close_tickers=close_set, strategy=strat_name)
        print(f"  Placed {len(fills)} orders")

        # Record fills
        for f in fills:
            ledger.record_fill(f)
        all_fills.extend(fills)

        # ── Reconciliation: surface any order that did not cleanly fill ────
        if not dry_run:
            problem = [f for f in fills
                       if f["order_status"] not in ("filled", "dry_run")]
            if problem:
                print(f"\n  ⚠ RECONCILE: {len(problem)} order(s) need attention:")
                for f in problem:
                    print(f"      {f['ticker']:<6} {f['side']:<4} "
                          f"status={f['order_status']}  "
                          f"filled={f['filled_shares']:+.4f} sh  id={f['order_id']}")

        # ── Update position snapshot (Alpaca-authoritative cost basis) ────
        positions_detail = executor.get_positions_detail()
        close_prices     = signal_prices  # use signal prices as today's close proxy
        ledger.snapshot_positions(strat_name, positions_detail, close_prices, today)

        # ── Daily P&L entry (authoritative return from equity change) ─────
        ledger.record_daily_pnl(strat_name, alpaca_equity, fills, today)

        # ── Export return/NAV book for paper/ combined-portfolio views ────
        # Only on live runs — dry runs don't change real equity.
        if not dry_run:
            book = ledger.export_paper_book(strat_name, PAPER_LEDGER_ROOT)
            print(f"  Paper book updated: {book}")

        # ── Mark the run complete (duplicate-run protection) ──────────────
        ledger.mark_run(strat_name, today, mode, n_orders=len(fills))

    # ── Final report ──────────────────────────────────────────────────────────
    if not args.no_report:
        for strat_name in strategies:
            if args.strategy and strat_name != args.strategy:
                continue
            print(ledger.report(strat_name))

    print(f"\n  Done. Ledger: {LEDGER_PATH}")


if __name__ == "__main__":
    main()
