#!/usr/bin/env python3
"""
Multi-asset trend — Coinbase-tradeable instruments only.

Universe:
  10 crypto (BTC/ETH/SOL/BNB/XRP/ADA/AVAX/LINK/DOGE/LTC)
    long/SHORT via Coinbase INTX perp futures      cost: 0.09%/side
  PAXG-USD spot (gold-backed token, Binance data)
    long/FLAT spot only                            cost: 0.40%/side

Same EWMAC ensemble as the validated crypto_trend (8/32 + 16/64 + 32/128, FDM 1.25).
Vol-targeted; risk budget divided equally across N live assets each bar.
No-trade band = 0.003.

Tests:
  1. Sub-portfolio stats (crypto L/S, PAXG L/F, combined) + OOS
  2. By-year returns — 2022 bear MUST be positive to justify adding this
  3. Regime (BTC above/below 200d MA)
  4. Correlation to existing 70/30 trend+carry book
  5. Portfolio blend — what allocation, if any, improves the existing book?

Usage:  python3 crypto/multiasset_trend.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto_trend as ct
from crypto_mr_backtest import load_clean
from crypto_carry_backtest import (load_panel as carry_load_panel,
                                   backtest as carry_bt,
                                   QUALITY, COST_TOGGLE)

CRYPTO = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'AVAX', 'LINK', 'DOGE', 'LTC']

# Per-side transaction costs
COST_PERP = 0.0009   # INTX perp: 0.04% maker + 0.05% slippage
COST_SPOT = 0.0040   # Coinbase spot: 0.35% maker + 0.05% slippage

BAND = 0.003         # no-trade band (fraction of capital)
BPY  = 365


# ── Data loading ─────────────────────────────────────────────────────────────

def _load(coin_or_path):
    """Load Binance daily spot CSV; strip tz so indices align cleanly."""
    path = (f"Data/{coin_or_path.lower()}usdt_1d_binance.csv"
            if not coin_or_path.endswith('.csv') else coin_or_path)
    df = load_clean(path, '1d', False)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df


# ── Panel + forecast helpers ──────────────────────────────────────────────────

def build_combined_panel(coin_data, paxg_df):
    """
    Merge 10-crypto data dict + PAXG DataFrame into a single aligned panel.
    PAXG starts later (2020-08); prior dates will be NaN.
    """
    data = dict(coin_data)
    data['PAXG'] = paxg_df
    return ct.build_panel(data, '1d')


def weights_mixed(F, V, paxg_col='PAXG'):
    """
    Vol-targeted weights:
      crypto columns → long/short (full ±)
      PAXG column    → long/flat  (clip to ≥0)
    """
    ann_vol = V * np.sqrt(BPY)
    n = F.notna().sum(axis=1).replace(0, np.nan)
    w = (F / ct.FORECAST_TARGET).div(n, axis=0).mul(ct.RISK_TARGET) / ann_vol
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if paxg_col in w.columns:
        w[paxg_col] = w[paxg_col].clip(lower=0)
    return w.clip(-2, 2)


def pnl_mixed(w, R, crypto_cols, paxg_col='PAXG'):
    """Net PnL applying per-asset costs (perp for crypto, spot for PAXG)."""
    ret = R.fillna(0.0)
    gross = (w.shift(1) * ret).sum(axis=1)
    delta = w.diff().abs()
    cost = delta[[c for c in crypto_cols if c in delta.columns]].sum(axis=1) * COST_PERP
    if paxg_col in delta.columns:
        cost = cost + delta[paxg_col] * COST_SPOT
    return gross - cost, gross


# ── Stats ─────────────────────────────────────────────────────────────────────

def stats(r, label='', print_row=True):
    r = r.dropna()
    if r.empty:
        return {}
    eq   = (1 + r).cumprod()
    yrs  = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd  = ((eq - eq.cummax()) / eq.cummax()).min()
    sh   = r.mean() / r.std() * np.sqrt(BPY) if r.std() > 0 else np.nan
    cal  = cagr / abs(mdd) if mdd < 0 else np.nan
    if print_row and label:
        print(f"  {label:<30}{sh:>7.2f}{cal:>7.2f}{cagr*100:>8.1f}"
              f"{r.std()*np.sqrt(BPY)*100:>7.1f}{mdd*100:>9.1f}  "
              f"{r.index[0].date()}→{r.index[-1].date()}")
    return dict(sharpe=sh, calmar=cal, cagr=cagr*100,
                vol=r.std()*np.sqrt(BPY)*100, maxdd=mdd*100)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load data ─────────────────────────────────────────────────────────
    coin_data = {}
    for c in CRYPTO:
        try:
            coin_data[c] = _load(c)
        except Exception as e:
            print(f"  WARN: {c} skipped: {e}")

    paxg_df = _load('Data/paxgusdt_1d_binance.csv')

    # ── 2. Combined panel: 10 crypto + PAXG ──────────────────────────────────
    C, R, V = build_combined_panel(coin_data, paxg_df)
    F       = ct.forecast_panel(C, V, None)
    w       = weights_mixed(F, V, 'PAXG')
    w       = ct.apply_buffer(w, BAND)
    crypto_cols = [c for c in CRYPTO if c in w.columns]
    net, gross  = pnl_mixed(w, R, crypto_cols, 'PAXG')

    # ── 3. Sub-portfolio: crypto only (L/S) ───────────────────────────────────
    C10, R10, V10 = ct.build_panel(coin_data, '1d')
    F10 = ct.forecast_panel(C10, V10, None)
    w10 = weights_mixed(F10, V10, paxg_col='__NONE__')
    w10 = ct.apply_buffer(w10, BAND)
    net10, _ = pnl_mixed(w10, R10, list(C10.columns), paxg_col='__NONE__')

    # ── 4. Sub-portfolio: PAXG only (L/F) ─────────────────────────────────────
    Cp = pd.DataFrame({'PAXG': paxg_df['close']})
    Rp = pd.DataFrame({'PAXG': paxg_df['ret']})
    Vp = pd.DataFrame({'PAXG': ct.per_bar_vol(paxg_df['ret'], ct.VOL_SPAN)})
    Fp = ct.forecast_panel(Cp, Vp, None)
    wp = weights_mixed(Fp, Vp, 'PAXG')
    wp = ct.apply_buffer(wp, BAND)
    netp, _ = pnl_mixed(wp, Rp, [], 'PAXG')

    # ── 5. Summary table ──────────────────────────────────────────────────────
    print("Multi-asset trend — Coinbase-tradeable only\n")
    print("=" * 90)
    print(f"  {'sub-portfolio':<30}{'Sharpe':>7}{'Calmar':>7}{'CAGR%':>8}{'Vol%':>7}{'MaxDD%':>9}  window")
    print("=" * 90)
    stats(net10, "crypto L/S (perp cost)")
    stats(netp,  "PAXG L/F  (spot cost)")
    stats(net,   "combined (11 assets)")
    oos_cut = net.dropna().index[int(len(net.dropna()) * 0.70)]
    stats(net.loc[oos_cut:], "  OOS — last 30%")
    print("=" * 90)

    # ── 6. By-year ────────────────────────────────────────────────────────────
    print("\n  Annual returns (net)  [2022 bear year is the key test]")
    print(f"  {'year':<6}{'crypto L/S':>12}{'PAXG L/F':>10}{'combined':>10}")
    all_years = sorted(set(net.dropna().index.year))
    for yr in all_years:
        def yr_sum(s): return s.dropna()[s.dropna().index.year == yr].sum() * 100
        print(f"  {yr:<6}{yr_sum(net10):>+11.1f}%{yr_sum(netp):>+9.1f}%{yr_sum(net):>+9.1f}%")

    # ── 7. Regime (BTC 200d MA) ───────────────────────────────────────────────
    btc  = C['BTC']
    bull = (btc > btc.rolling(200).mean()).reindex(net.dropna().index).ffill().fillna(False)
    print("\n  Regime (BTC vs 200d MA) — combined strategy:")
    for name, mask in [('BULL', bull), ('BEAR', ~bull)]:
        m   = mask.reindex(net.dropna().index).fillna(False)
        sub = net.dropna()[m]
        if len(sub) < 20:
            continue
        sh  = sub.mean() / sub.std() * np.sqrt(BPY) if sub.std() > 0 else float('nan')
        print(f"     {name}: {sub.mean()*BPY*100:>+7.1f}%/yr  "
              f"Sharpe {sh:>5.2f}  ({len(sub)} days)")

    # ── 8. Correlation to existing 70/30 book ────────────────────────────────
    # Build benchmark using ONLY the fixed CRYPTO list (excludes PAXG from trend)
    C_b = C[crypto_cols]; R_b = R[crypto_cols]; V_b = V[crypto_cols]
    F_b = ct.forecast_panel(C_b, V_b, None)
    trend_lf, _, _ = ct.backtest(F_b, R_b, V_b, '1d', 'longflat',
                                  cost=ct.COST_PER_SIDE, band=0.004)
    trend_lf.index = pd.DatetimeIndex(trend_lf.index).tz_localize(None).normalize()

    Fund, Sret, Pret = carry_load_panel(QUALITY)
    carry_r, _, _, _ = carry_bt(Fund, Sret, Pret, COST_TOGGLE)
    carry_r.index = pd.DatetimeIndex(carry_r.index).tz_localize(None).normalize()

    book = (0.70 * trend_lf + 0.30 * carry_r).dropna()
    ov   = pd.DataFrame({'book': book, 'cta': net}).dropna()
    print(f"\n  Correlation to 70/30 book:  {ov['book'].corr(ov['cta']):+.3f}  "
          f"({ov.index[0].date()} → {ov.index[-1].date()}, {len(ov)} days)")

    worst = ov['book'] <= ov['book'].quantile(0.10)
    hedge = 'HEDGES' if ov.loc[worst, 'cta'].mean() > 0 else 'bleeds with book'
    print(f"  Worst-decile hedge test:    book {ov.loc[worst,'book'].mean()*100:+.2f}%/day  "
          f"CTA {ov.loc[worst,'cta'].mean()*100:+.2f}%/day  ({hedge})")

    # ── 9. Portfolio blend ────────────────────────────────────────────────────
    print("\n  Portfolio blend (trend / carry / multiasset):")
    print(f"  {'allocation':<32}{'Sharpe':>7}{'Calmar':>7}{'CAGR%':>8}{'MaxDD%':>9}")
    blends = [
        ('70/30/0  current',       0.70, 0.30, 0.00),
        ('60/20/20',               0.60, 0.20, 0.20),
        ('55/20/25',               0.55, 0.20, 0.25),
        ('50/25/25',               0.50, 0.25, 0.25),
        ('50/15/35',               0.50, 0.15, 0.35),
        ('45/15/40',               0.45, 0.15, 0.40),
    ]
    for label, wt, wc, wcta in blends:
        b = (wt * trend_lf + wc * carry_r + wcta * net).dropna()
        s = stats(b, print_row=False)
        print(f"  {label:<32}{s['sharpe']:>7.2f}{s['calmar']:>7.2f}"
              f"{s['cagr']:>8.1f}{s['maxdd']:>9.1f}")

    # ── 10. Equity curve plot ─────────────────────────────────────────────────
    eq_comb  = (1 + net.dropna()).cumprod()
    eq_c10   = (1 + net10.dropna()).cumprod()
    eq_paxg  = (1 + netp.dropna()).cumprod()
    eq_book  = (1 + book.reindex(eq_comb.index).dropna()).cumprod()
    best_wt  = min(blends[1:], key=lambda x: -stats((x[1]*trend_lf + x[2]*carry_r + x[3]*net).dropna(), print_row=False).get('sharpe', -99))
    eq_blend = (1 + (best_wt[1]*trend_lf + best_wt[2]*carry_r + best_wt[3]*net).dropna()).cumprod()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(eq_c10,   color='steelblue', lw=1.0, label='crypto L/S (perp)')
    ax1.plot(eq_paxg,  color='goldenrod',  lw=1.0, label='PAXG L/F (spot)')
    ax1.plot(eq_comb,  color='purple',     lw=1.8, label='combined (11 assets)')
    ax1.plot(eq_book,  color='gray',       lw=1.0, ls='--', label='70/30 book (bench)')
    ax1.plot(eq_blend, color='black',      lw=1.2, ls=':', label=f'blend {best_wt[0]}')
    ax1.set_yscale('log')
    ax1.set_ylabel('growth of $1 (log)')
    ax1.set_title('Multi-asset trend: crypto L/S (perp) + PAXG gold L/F (spot)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')
    dd = (eq_comb - eq_comb.cummax()) / eq_comb.cummax() * 100
    ax2.fill_between(dd.index, dd, 0, color='crimson', alpha=0.4)
    ax2.set_ylabel('combined drawdown %')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/multiasset_trend_equity.png', dpi=150)
    print("\n  Chart saved to results/multiasset_trend_equity.png")


if __name__ == '__main__':
    main()
