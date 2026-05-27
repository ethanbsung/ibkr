#!/usr/bin/env python3
"""
Combined IBS + EWMAC Portfolio Backtest
3 instruments (MES/MNQ/MGC), equal-weight within each strategy.
$50,000 starting capital, split 50/50 between strategies.

IBS:   Mean-reversion. Enter long on IBS < 0.10, exit on IBS > 0.90.
EWMAC: Trend-following. Long/short on EMA(64) vs EMA(256) crossover.
       Position sized by vol target at each crossover.

Capital allocation: $25,000 to IBS sub-portfolio, $25,000 to EWMAC sub-portfolio.
Each sub-strategy gets 1/3 of its sub-portfolio capital.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from io import StringIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 50_000.0
IBS_ALLOC        = 0.75    # 75% of capital to IBS
EWMAC_ALLOC      = 0.25    # 25% of capital to EWMAC
COMMISSION_RT    = 5.0
START_DATE       = '2000-01-01'
END_DATE         = '2026-01-01'

# IBS parameters
IBS_ENTRY = 0.10
IBS_EXIT  = 0.90

# EWMAC parameters
RISK_TARGET = 0.20
FAST_SPAN   = 64
SLOW_SPAN   = 256
VOL_SPAN    = 32
VOL_FLOOR   = 0.05

# ── Contract specifications ────────────────────────────────────────────────────
CONTRACT_SPECS = {
    'ES': {'multiplier': 5,  'file': 'Data/mes_daily_data.csv'},
    'NQ': {'multiplier': 2,  'file': 'Data/mnq_daily_data.csv'},
    'GC': {'multiplier': 10, 'file': 'Data/mgc_daily_data.csv'},
}

N_INSTRUMENTS = len(CONTRACT_SPECS)
IBS_STRAT_ALLOC  = 1.0 / N_INSTRUMENTS   # within-strategy equal weight
EWMAC_STRAT_ALLOC = 1.0 / N_INSTRUMENTS


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(file_path):
    lines = []
    with open(file_path, 'r') as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith('"'):
                continue
            lines.append(stripped)
    if len(lines) < 2:
        return pd.DataFrame()
    df = pd.read_csv(StringIO('\n'.join(lines)), parse_dates=['Time'])
    if not {'High', 'Low', 'Last'}.issubset(df.columns):
        return pd.DataFrame()
    df.sort_values('Time', inplace=True)
    df = df[(df['Time'] >= START_DATE) & (df['Time'] <= END_DATE)]
    df = df.dropna(subset=['High', 'Low', 'Last']).reset_index(drop=True)
    return df


def load_all_data():
    data = {}
    for symbol, spec in CONTRACT_SPECS.items():
        df = load_data(spec['file'])
        if df.empty:
            logger.warning(f"No data loaded for {symbol}")
            continue
        data[symbol] = df.set_index('Time')
        logger.info(f"Loaded {len(data[symbol])} bars for {symbol}")
    return data


# ── Pre-computation ────────────────────────────────────────────────────────────

def precompute_indicators(instrument_data):
    indicators = {}
    for symbol, df in instrument_data.items():
        ind = df[['High', 'Low', 'Last']].copy()

        # IBS
        bar_range = ind['High'] - ind['Low']
        ind['IBS'] = np.where(bar_range > 0, (ind['Last'] - ind['Low']) / bar_range, 0.5)

        # EWMAC direction
        prices   = df['Last']
        fast_ema = prices.ewm(span=FAST_SPAN, min_periods=FAST_SPAN).mean()
        slow_ema = prices.ewm(span=SLOW_SPAN, min_periods=SLOW_SPAN).mean()
        direction = pd.Series(np.where(fast_ema > slow_ema, 1.0, -1.0), index=prices.index)
        direction[fast_ema.isna() | slow_ema.isna()] = np.nan
        ind['EWMAC_dir'] = direction

        # Annualised vol for EWMAC sizing
        pct_ret = prices.pct_change()
        ann_vol = pct_ret.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std() * np.sqrt(256)
        ind['ann_vol'] = ann_vol.clip(lower=VOL_FLOOR)

        indicators[symbol] = ind
    return indicators


# ── Backtests ──────────────────────────────────────────────────────────────────

def run_ibs_backtest(instrument_data, indicators):
    all_dates = sorted(set().union(*[set(df.index) for df in instrument_data.values()]))

    ibs_cap_per_strat = INITIAL_CAPITAL * IBS_ALLOC * IBS_STRAT_ALLOC

    state = {
        sym: {
            'capital':      ibs_cap_per_strat,
            'in_position':  False,
            'position':     None,
            'equity_curve': [],
        }
        for sym in CONTRACT_SPECS
    }

    total_ibs_equity = INITIAL_CAPITAL * IBS_ALLOC

    for date in all_dates:
        daily_sum = 0.0
        for sym in CONTRACT_SPECS:
            s   = state[sym]
            mul = CONTRACT_SPECS[sym]['multiplier']
            ind = indicators.get(sym)

            if ind is None or date not in ind.index:
                last_eq = s['equity_curve'][-1][1] if s['equity_curve'] else s['capital']
                s['equity_curve'].append((date, last_eq))
                daily_sum += last_eq
                continue

            row   = ind.loc[date]
            price = float(row['Last'])
            ibs   = float(row['IBS'])

            if s['in_position']:
                if ibs > IBS_EXIT:
                    pos = s['position']
                    pnl = ((price - pos['entry_price']) * mul * pos['contracts']
                           - COMMISSION_RT * pos['contracts'])
                    s['capital']    += pnl
                    s['in_position'] = False
                    s['position']    = None
            else:
                if ibs < IBS_ENTRY:
                    contracts = max(1, round(total_ibs_equity * IBS_STRAT_ALLOC / (price * mul)))
                    s['capital']    -= (COMMISSION_RT / 2) * contracts
                    s['in_position'] = True
                    s['position']    = {'entry_price': price, 'contracts': contracts}

            if s['in_position']:
                pos        = s['position']
                unrealized = (price - pos['entry_price']) * mul * pos['contracts']
                equity     = s['capital'] + unrealized
            else:
                equity = s['capital']

            s['equity_curve'].append((date, equity))
            daily_sum += equity

        total_ibs_equity = daily_sum

    # Force close
    for sym in CONTRACT_SPECS:
        s  = state[sym]
        df = instrument_data.get(sym)
        if s['in_position'] and df is not None:
            last_price = float(df.iloc[-1]['Last'])
            mul        = CONTRACT_SPECS[sym]['multiplier']
            pos        = s['position']
            pnl        = ((last_price - pos['entry_price']) * mul * pos['contracts']
                          - (COMMISSION_RT / 2) * pos['contracts'])
            s['capital'] += pnl
            if s['equity_curve']:
                s['equity_curve'][-1] = (s['equity_curve'][-1][0], s['capital'])

    return state


def run_ewmac_backtest(instrument_data, indicators):
    all_dates = sorted(set().union(*[set(df.index) for df in instrument_data.values()]))

    ewmac_cap_per_strat = INITIAL_CAPITAL * EWMAC_ALLOC * EWMAC_STRAT_ALLOC

    state = {
        sym: {
            'capital':      ewmac_cap_per_strat,
            'position':     0,
            'prev_price':   None,
            'prev_dir':     np.nan,
            'total_gross':  0.0,
            'total_comm':   0.0,
            'equity_curve': [],
        }
        for sym in CONTRACT_SPECS
    }

    for date in all_dates:
        for sym in CONTRACT_SPECS:
            s   = state[sym]
            mul = CONTRACT_SPECS[sym]['multiplier']
            df  = instrument_data.get(sym)
            ind = indicators.get(sym)

            if df is None or ind is None or date not in df.index:
                last_eq = s['equity_curve'][-1][1] if s['equity_curve'] else s['capital']
                s['equity_curve'].append((date, last_eq))
                continue

            price   = float(df.loc[date, 'Last'])
            new_dir = float(ind.loc[date, 'EWMAC_dir']) if date in ind.index and not pd.isna(ind.loc[date, 'EWMAC_dir']) else np.nan
            ann_vol = float(ind.loc[date, 'ann_vol'])   if date in ind.index else np.nan

            # Daily mark-to-market
            if s['prev_price'] is not None and s['position'] != 0:
                daily_pnl = (price - s['prev_price']) * mul * s['position']
                s['capital']    += daily_pnl
                s['total_gross'] += daily_pnl

            # Trade only on crossover
            signal_changed = (not pd.isna(new_dir)) and (new_dir != s['prev_dir'])

            if signal_changed and not (pd.isna(ann_vol) or ann_vol <= 0):
                avg_pos = (s['capital'] * RISK_TARGET) / (price * mul * ann_vol)
                new_pos = max(1, int(round(avg_pos))) * int(new_dir)

                trade = abs(new_pos - s['position'])
                if trade > 0:
                    comm = (COMMISSION_RT / 2) * trade
                    s['capital']    -= comm
                    s['total_comm'] += comm

                s['position'] = new_pos
                s['prev_dir'] = new_dir
            elif not pd.isna(new_dir) and pd.isna(s['prev_dir']):
                s['prev_dir'] = new_dir

            s['prev_price'] = price
            s['equity_curve'].append((date, s['capital']))

    # Close open positions
    for sym in CONTRACT_SPECS:
        s  = state[sym]
        df = instrument_data.get(sym)
        if s['position'] != 0 and df is not None and not df.empty:
            comm = (COMMISSION_RT / 2) * abs(s['position'])
            s['capital']    -= comm
            s['total_comm'] += comm
            if s['equity_curve']:
                s['equity_curve'][-1] = (s['equity_curve'][-1][0], s['capital'])

    return state


# ── Build equity curves ────────────────────────────────────────────────────────

def build_equity_curve(state_dict, common_dates):
    series = []
    for sym in CONTRACT_SPECS:
        curve = state_dict[sym]['equity_curve']
        df    = pd.DataFrame(curve, columns=['Time', 'Equity']).set_index('Time')
        df    = df[~df.index.duplicated(keep='last')].sort_index()
        df    = df.reindex(common_dates, method='ffill')
        series.append(df['Equity'])
    combined = pd.DataFrame({'Equity': sum(series)}, index=common_dates)
    combined.dropna(inplace=True)
    return combined


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(equity_df, label=''):
    eq   = equity_df['Equity']
    rets = eq.pct_change().dropna()

    final   = eq.iloc[-1]
    years   = (eq.index[-1] - eq.index[0]).days / 365.25
    ann_ret = ((final / eq.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else np.nan

    std     = rets.std()
    sharpe  = rets.mean() / std * np.sqrt(252) if std > 0 else np.nan
    d_std   = rets[rets < 0].std()
    sortino = rets.mean() / d_std * np.sqrt(252) if d_std > 0 else np.nan

    peak   = eq.cummax()
    dd     = (eq - peak) / peak
    max_dd = dd.min() * 100
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        'label':             label,
        'start':             eq.iloc[0],
        'Final Balance':     final,
        'Total Return %':    (final / eq.iloc[0] - 1) * 100,
        'Ann. Return %':     ann_ret,
        'Ann. Volatility %': std * np.sqrt(252) * 100,
        'Sharpe':            sharpe,
        'Sortino':           sortino,
        'Calmar':            calmar,
        'Max Drawdown %':    max_dd,
    }


def print_metrics(m):
    for k, v in m.items():
        if k in ('label', 'start'):
            continue
        if k == 'Final Balance':
            print(f"  {k:<26} ${v:>12,.2f}")
        elif isinstance(v, float) and not np.isnan(v):
            print(f"  {k:<26} {v:>12.2f}")
        else:
            print(f"  {k:<26} {'NaN':>12}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("Loading data...")
    instrument_data = load_all_data()
    if not instrument_data:
        logger.error("No data loaded.")
        return

    logger.info("Pre-computing indicators...")
    indicators = precompute_indicators(instrument_data)

    logger.info("Running IBS backtest...")
    ibs_state = run_ibs_backtest(instrument_data, indicators)

    logger.info("Running EWMAC backtest...")
    ewmac_state = run_ewmac_backtest(instrument_data, indicators)

    common_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')

    ibs_eq   = build_equity_curve(ibs_state, common_dates)
    ewmac_eq = build_equity_curve(ewmac_state, common_dates)

    # Align to common valid dates
    valid = ibs_eq.index.intersection(ewmac_eq.index)
    ibs_eq   = ibs_eq.loc[valid]
    ewmac_eq = ewmac_eq.loc[valid]

    combined_eq = pd.DataFrame({'Equity': ibs_eq['Equity'] + ewmac_eq['Equity']}, index=valid)

    # Normalise standalone strategy curves to full $50k for apples-to-apples comparison
    # (each was run with only half the capital)
    ibs_scaled   = pd.DataFrame({'Equity': ibs_eq['Equity']   * 2}, index=valid)
    ewmac_scaled = pd.DataFrame({'Equity': ewmac_eq['Equity'] * 2}, index=valid)

    ibs_m    = compute_metrics(ibs_scaled,   'IBS (standalone, 50k equiv)')
    ewmac_m  = compute_metrics(ewmac_scaled, 'EWMAC (standalone, 50k equiv)')
    combo_m  = compute_metrics(combined_eq,  'Combined IBS+EWMAC')

    # ── Correlation between strategy returns ───────────────────────────────────
    ibs_ret   = ibs_eq['Equity'].pct_change().dropna()
    ewmac_ret = ewmac_eq['Equity'].pct_change().dropna()
    aligned   = pd.concat([ibs_ret, ewmac_ret], axis=1, join='inner')
    aligned.columns = ['IBS', 'EWMAC']
    corr = aligned.corr().iloc[0, 1]

    # ── Print ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"COMBINED BACKTEST  |  IBS + EWMAC({FAST_SPAN},{SLOW_SPAN})  |  ${INITIAL_CAPITAL:,.0f} capital")
    print(f"50% IBS  |  50% EWMAC  |  3 instruments each")
    print("=" * 64)

    print(f"\n  Strategy return correlation (IBS vs EWMAC): {corr:+.3f}")
    print(f"  (near zero = genuine diversification, negative = ideal)")

    print(f"\n── IBS standalone (scaled to ${INITIAL_CAPITAL:,.0f}) ─────────────────────────")
    print_metrics(ibs_m)

    print(f"\n── EWMAC({FAST_SPAN},{SLOW_SPAN}) standalone (scaled to ${INITIAL_CAPITAL:,.0f}) ─────────────────")
    print_metrics(ewmac_m)

    print(f"\n── Combined IBS + EWMAC (${INITIAL_CAPITAL:,.0f}) ───────────────────────────────")
    print_metrics(combo_m)

    print(f"\n  Sharpe improvement: {combo_m['Sharpe'] - max(ibs_m['Sharpe'], ewmac_m['Sharpe']):+.3f}"
          f"  vs best standalone ({max(ibs_m['Sharpe'], ewmac_m['Sharpe']):.3f})")
    print(f"  Max DD improvement: {combo_m['Max Drawdown %'] - max(ibs_m['Max Drawdown %'], ewmac_m['Max Drawdown %']):+.1f}%"
          f"  vs best standalone ({max(ibs_m['Max Drawdown %'], ewmac_m['Max Drawdown %']):.1f}%)")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={'height_ratios': [3, 1, 1]})
    ax1, ax2, ax3 = axes

    ax1.plot(combined_eq.index, combined_eq['Equity'],
             color='seagreen', linewidth=2.0, label='Combined IBS+EWMAC', zorder=3)
    ax1.plot(ibs_scaled.index, ibs_scaled['Equity'],
             color='steelblue', linewidth=1.0, alpha=0.7, linestyle='--', label='IBS only (scaled)')
    ax1.plot(ewmac_scaled.index, ewmac_scaled['Equity'],
             color='darkorange', linewidth=1.0, alpha=0.7, linestyle='--', label=f'EWMAC({FAST_SPAN},{SLOW_SPAN}) only (scaled)')
    ax1.axhline(INITIAL_CAPITAL, color='gray', linestyle=':', linewidth=0.8, label='Starting capital')
    ax1.set_ylabel('Portfolio Equity ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.set_title(
        f'Combined IBS + EWMAC({FAST_SPAN},{SLOW_SPAN})  |  3 instruments  |  ${INITIAL_CAPITAL:,.0f} capital\n'
        f'Sharpe {combo_m["Sharpe"]:.2f}  |  Ann. Return {combo_m["Ann. Return %"]:.1f}%'
        f'  |  Max DD {combo_m["Max Drawdown %"]:.1f}%  |  IBS/EWMAC corr {corr:+.3f}',
        fontsize=11
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Combined drawdown
    peak_c = combined_eq['Equity'].cummax()
    dd_c   = (combined_eq['Equity'] - peak_c) / peak_c * 100
    ax2.fill_between(combined_eq.index, dd_c, 0, color='crimson', alpha=0.4)
    ax2.set_ylabel('Combined DD (%)')
    ax2.grid(True, alpha=0.3)

    # Rolling 252-day correlation
    roll_corr = (ibs_eq['Equity'].pct_change()
                 .rolling(252)
                 .corr(ewmac_eq['Equity'].pct_change()))
    ax3.plot(roll_corr.index, roll_corr, color='purple', linewidth=1.0)
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax3.set_ylabel('Rolling 1Y Corr')
    ax3.set_xlabel('Date')
    ax3.set_ylim(-1, 1)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('portfolio/combined_equity_curve.png', dpi=150)
    plt.show()
    logger.info("Chart saved to portfolio/combined_equity_curve.png")


if __name__ == '__main__':
    main()
