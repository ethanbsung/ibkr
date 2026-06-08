#!/usr/bin/env python3
"""
Paper-trading runner + report (cron entry point).

Runs each strategy (capital-agnostic; logs returns/NAV), then combines them into
a dollar portfolio using paper/portfolio.json. Per-strategy track records are
return-based and untouched by reallocation — edit portfolio.json to reallocate
or add a strategy's weight.

Usage:
  python3 paper/run_paper.py                 # run today's bar + report
  python3 paper/run_paper.py --report-only   # just the report
  python3 paper/run_paper.py --asof 2026-05-29
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
from engine import PaperEngine, Book
from strategies import CryptoTrendStrategy, CryptoCarryStrategy

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER_ROOT = os.path.join(HERE, 'ledgers')
CONFIG = os.path.join(HERE, 'portfolio.json')

# Deployed strategies. Capital/allocation is in portfolio.json, NOT here.
STRATEGIES = [
    CryptoTrendStrategy(band=0.004),
    CryptoCarryStrategy(band=0.01),
]


MIN_BARS = 20   # below this, annualized stats are noise — don't show them

def _pct(x):  return f"{x*100:7.2f}%" if x == x else "    n/a"
def _num(x):  return f"{x:6.2f}" if x == x else "   n/a"


def report():
    with open(CONFIG) as fh:
        cfg = json.load(fh)
    total, allocs = cfg['total_capital'], cfg['allocations']

    print("\n" + "=" * 80)
    print(f"  PAPER PORTFOLIO REPORT   total capital ${total:,.0f}")
    print("=" * 80)

    rets = {}
    for name, alloc in allocs.items():
        daily = Book(name, LEDGER_ROOT).daily_frame()
        if daily.empty:
            print(f"\n  [{name}]  alloc {alloc:.0%}  — no history yet")
            continue
        rets[name] = daily['ret']
        m = M.summarize(daily)
        cap = total * alloc
        print(f"\n  ── {name}   alloc {alloc:.0%}  (${cap:,.0f})  "
              f"value ${cap * m['nav']:,.2f}   {m['days']} bars")
        print(f"     NAV {m['nav']:.4f}   total {_pct(m['total_return'])}   "
              f"cum cost {_pct(m['cum_cost'])}")
        if m['days'] >= MIN_BARS:
            print(f"     CAGR {_pct(m['cagr'])}   Sharpe {_num(m['sharpe'])}   "
                  f"Calmar {_num(m['calmar'])}   vol {_pct(m['ann_vol'])}   "
                  f"maxDD {_pct(m['max_dd'])}   curDD {_pct(m['cur_dd'])}")
        else:
            print(f"     (accumulating — annualized stats shown after {MIN_BARS} bars)")

    # ── Combined portfolio ───────────────────────────────────────────────────
    if rets:
        port, aligned = M.combine_portfolio(rets, allocs)
        pm = M.return_metrics(port)
        nav = (1 + port).cumprod()
        print("\n  ── TOTAL PORTFOLIO " + "─" * 53)
        print(f"     value ${total * nav.iloc[-1]:,.2f}   total {_pct(nav.iloc[-1]-1)}")
        if len(port) >= MIN_BARS:
            print(f"     CAGR {_pct(pm['cagr'])}   Sharpe {_num(pm['sharpe'])}   "
                  f"Calmar {_num(pm['calmar'])}   vol {_pct(pm['ann_vol'])}   "
                  f"maxDD {_pct(pm['max_dd'])}   curDD {_pct(pm['cur_dd'])}")
        else:
            print(f"     (accumulating — annualized stats shown after {MIN_BARS} bars)")
        cash = 1 - sum(allocs.values())
        if cash > 1e-6:
            print(f"     (uninvested cash: {cash:.0%})")
        if aligned.shape[1] >= 2:
            corr = aligned.corr()
            print("\n     strategy return correlations:")
            print("       " + corr.round(2).to_string().replace("\n", "\n       "))
    print("\n" + "=" * 80)


def main():
    ap = argparse.ArgumentParser(description='Paper-trading runner + report')
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--asof', default=None)
    args = ap.parse_args()
    if not args.report_only:
        asof = pd.Timestamp(args.asof).date() if args.asof else None
        print(f"Running {len(STRATEGIES)} strategies{f' as of {asof}' if asof else ''}...")
        PaperEngine(LEDGER_ROOT).run(STRATEGIES, asof=asof)
    report()


if __name__ == '__main__':
    main()
