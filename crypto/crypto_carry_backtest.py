#!/usr/bin/env python3
"""
Funding-rate carry — STAGE 2 tradeable backtest (cash-and-carry)
──────────────────────────────────────────────────────────────────
The real trade: long spot + short perp, market-neutral, collect funding.
Daily return per coin (per unit notional):

    carry = funding_collected + (spot_return − perp_return)
                                 └── basis P&L: ~0 on average but adds the
                                     real vol/tail the stage-1 study omitted

This is what turns the fantasy "Sharpe 17" into a real number. Hold a coin only
when its (smoothed) funding is positive — don't short when you'd PAY. Two-leg
cost on every toggle: the SPOT leg's fee dominates (perp fee is ~0). Compares
the full 10-coin universe vs the quality subset (drop SOL/BNB/AVAX).

Data: funding + spot 1d (have) + perp 1d (Data/*_1d_perp_binance.csv).

Usage:  python3 crypto/crypto_carry_backtest.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto_trend as ct
from crypto_mr_backtest import FNAME_RE

QUALITY = ['BTC', 'ETH', 'XRP', 'ADA', 'LINK', 'DOGE', 'LTC']   # reliable funding
ALL_COINS = QUALITY + ['SOL', 'BNB', 'AVAX']

# ── Realistic two-leg transaction cost (per side) ────────────────────────────
# Cash-and-carry trades BOTH legs on entry and exit. Per-side cost components:
#   SPOT leg (the expensive one): 0.35% maker fee (limit orders that fill) + spread/slippage
#   PERP leg (cheap):  $0.10/contract ≈ ~0.04% of notional + spread/slippage
SPOT_FEE  = 0.0035     # Coinbase spot maker
SPOT_SLIP = 0.0003     # spot spread/slippage on a limit fill (~3 bps)
PERP_FEE  = 0.0004     # $0.10/contract ≈ 0.04% on a typical nano/alt notional
PERP_SLIP = 0.0003     # perp spread/slippage (~3 bps)
# One toggle (enter OR exit) trades both legs once → sum of both legs' per-side cost.
COST_TOGGLE = (SPOT_FEE + SPOT_SLIP) + (PERP_FEE + PERP_SLIP)
SIG_SPAN = 21          # days; smoothed funding signal to avoid over-toggling


def load_panel(coins):
    """Aligned daily funding, spot return, perp return per coin."""
    fund, spot, perp = {}, {}, {}
    for coin in coins:
        c = coin.lower()
        fp = f"Data/{c}usdt_funding_binance.csv"
        sp = f"Data/{c}usdt_1d_binance.csv"
        pp = f"Data/{c}usdt_1d_perp_binance.csv"
        if not (os.path.exists(fp) and os.path.exists(sp) and os.path.exists(pp)):
            continue
        f = pd.read_csv(fp, parse_dates=['time']).set_index('time')['rate']
        f_daily = f.groupby(f.index.normalize()).sum()                 # funding/day
        s = pd.read_csv(sp, parse_dates=['time']).set_index('time')['close']
        p = pd.read_csv(pp, parse_dates=['time']).set_index('time')['close']
        s.index = s.index.normalize(); p.index = p.index.normalize()
        fund[coin] = f_daily
        spot[coin] = s.pct_change()
        perp[coin] = p.pct_change()
    idx = sorted(set().union(*[set(v.index) for v in fund.values()]))
    idx = pd.DatetimeIndex(idx)
    Fund = pd.DataFrame({c: fund[c].reindex(idx) for c in fund})
    Sret = pd.DataFrame({c: spot[c].reindex(idx) for c in fund})
    Pret = pd.DataFrame({c: perp[c].reindex(idx) for c in fund})
    return Fund, Sret, Pret


def backtest(Fund, Sret, Pret, cost_side):
    """Equal-weight cash-and-carry, held per coin when smoothed funding > 0."""
    # carry per coin = funding + basis P&L
    carry = Fund.fillna(0.0) + (Sret - Pret).fillna(0.0)
    # hold signal: smoothed funding positive, lagged (no lookahead)
    sig = (Fund.ewm(span=SIG_SPAN, min_periods=5).mean().shift(1) > 0).astype(float)
    sig = sig.where(Fund.notna().shift(1), 0.0)
    n = sig.sum(axis=1).replace(0, np.nan)
    w = sig.div(n, axis=0).fillna(0.0)                 # equal weight among active
    gross = (w * carry).sum(axis=1)
    turnover = (w - w.shift(1)).abs().sum(axis=1)
    net = gross - turnover * cost_side
    return net, gross, turnover.mean() * 365, w


def metrics(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    return dict(
        sharpe=r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan,
        ann=(eq.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else np.nan,
        vol=r.std() * np.sqrt(365) * 100,
        maxdd=((eq - eq.cummax()) / eq.cummax()).min() * 100,
    )


def trend_daily():
    data = {}
    for p in sorted(glob.glob("Data/*_1d_binance.csv")):
        if 'perp' in p:
            continue
        m = FNAME_RE.search(os.path.basename(p))
        if m:
            data[m.group(1).upper()] = ct.load_clean(p, '1d', False)
    C, R, V = ct.build_panel(data, '1d')
    F = ct.forecast_panel(C, V, None)
    t, _, _ = ct.backtest(F, R, V, '1d', 'longflat', band=0.004)
    t.index = pd.DatetimeIndex(t.index).tz_localize(None).normalize()
    return t


def plot_equity(net, gross, trend, out='results/crypto_carry_equity.png'):
    import matplotlib.pyplot as plt
    df = pd.DataFrame({'net': net, 'gross': gross}).dropna()
    eq_n = (1 + df['net']).cumprod()
    eq_g = (1 + df['gross']).cumprod()
    dd = (eq_n - eq_n.cummax()) / eq_n.cummax() * 100
    m = metrics(df['net'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(eq_g.index, eq_g.values, color='lightgray', lw=1.0, label='gross (no cost)')
    ax1.plot(eq_n.index, eq_n.values, color='seagreen', lw=1.6, label='net (realistic cost)')
    ax1.set_yscale('log')
    ax1.set_ylabel('growth of $1 (log)')
    ax1.set_title(f"Funding Carry (quality-7, long spot / short perp)  |  net Sharpe "
                  f"{m['sharpe']:.2f}  |  {m['ann']:.1f}%/yr  |  maxDD {m['maxdd']:.1f}%  |  "
                  f"cost {COST_TOGGLE*100:.2f}%/toggle")
    ax1.legend(); ax1.grid(True, alpha=0.3, which='both')
    ax2.fill_between(dd.index, dd, 0, color='crimson', alpha=0.4)
    ax2.set_ylabel('drawdown (%)'); ax2.set_xlabel('date'); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\n  Equity curve saved to {out}")


def main():
    trend = trend_daily()
    cost = COST_TOGGLE             # realistic per toggle-side (both legs + slippage)
    print(f"Cost per toggle-side {cost*100:.2f}%  =  spot ({SPOT_FEE*100:.2f}% fee + "
          f"{SPOT_SLIP*100:.2f}% slip) + perp ({PERP_FEE*100:.2f}% fee + {PERP_SLIP*100:.2f}% slip)"
          f"\n  signal: {SIG_SPAN}d smoothed funding > 0  (cross-margined ⇒ liquidation tail mitigated)\n")

    print("=" * 90)
    print(f"  {'universe':<22}{'Sharpe':>8}{'AnnRet%':>9}{'Vol%':>7}{'MaxDD%':>8}"
          f"{'turn/yr':>9}{'corr→trend':>12}")
    print("=" * 90)
    results = {}
    for label, coins in [('quality (7)', QUALITY), ('all (10)', ALL_COINS)]:
        Fund, Sret, Pret = load_panel(coins)
        if Fund.empty:
            print(f"  {label:<22} no perp data yet")
            continue
        net, gross, turn, w = backtest(Fund, Sret, Pret, cost)
        m = metrics(net); mg = metrics(gross)
        d = pd.DataFrame({'c': net, 't': trend}).dropna()
        d = d[(d['c'] != 0) | (d['t'] != 0)]
        rho = d['c'].corr(d['t'])
        results[label] = (net, gross)
        print(f"  {label:<22}{m['sharpe']:>8.2f}{m['ann']:>9.1f}{m['vol']:>7.1f}"
              f"{m['maxdd']:>8.1f}{turn:>9.1f}{rho:>+12.3f}")
        print(f"  {'  (gross, no cost)':<22}{mg['sharpe']:>8.2f}{mg['ann']:>9.1f}")
    print("=" * 90)

    # By-year net Sharpe for quality universe
    if 'quality (7)' in results:
        net, gross = results['quality (7)']
        print("\n  quality net Sharpe by year:")
        row = "   "
        for y, r in net.groupby(net.index.year):
            row += f"{y}:{(r.mean()/r.std()*np.sqrt(365) if r.std()>0 else float('nan')):>5.1f} "
        print(row)
        print("\n  NOTE: daily-close basis understates intraday risk — see crypto_carry_hourly.py")
        print("  for the realistic ~2-2.5 net Sharpe. This curve shows the growth path / drawdowns.")
        plot_equity(net, gross, trend)


if __name__ == '__main__':
    main()
