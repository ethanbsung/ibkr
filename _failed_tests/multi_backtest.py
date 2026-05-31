#!/usr/bin/env python3
"""
Combined IBS + EWMAC Portfolio Backtest — hybrid instrument split
  IBS   (75%): equity indices only — MNQ, M2K, ESTX50, CAC40
  EWMAC (25%): all 10 instruments — equity + bonds + FX + commodities

$250,000 starting capital
EUR instruments (ESTX50, CAC40, GBS): P&L converted to USD using daily EUR/USD rate.
EUR/USD data available from 2009-03-23; pre-2009 uses static 1.15 (historical avg).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from io import StringIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 250_000.0
IBS_ALLOC       = 0.75
EWMAC_ALLOC     = 0.25
COMMISSION_RT   = 5.0
START_DATE      = '2000-01-01'
END_DATE        = '2026-01-01'
DIV_YIELD       = 0.018

IBS_ENTRY   = 0.10
IBS_EXIT    = 0.90

RISK_TARGET = 0.20
FAST_SPAN   = 64
SLOW_SPAN   = 256
VOL_SPAN    = 32
VOL_FLOOR   = 0.05

# ── Contract specifications ────────────────────────────────────────────────────
# IBS: equity indices only (mean-reversion edge documented on equities)
IBS_SPECS = {
    'MNQ':   {'multiplier': 2,  'file': 'Data/mnq_daily_data.csv',    'currency': 'USD'},
    'M2K':   {'multiplier': 5,  'file': 'Data/m2k_daily_data.csv',    'currency': 'USD'},
    'ESTX50':{'multiplier': 10, 'file': 'Data/estx50_daily_data.csv', 'currency': 'EUR'},
    'CAC40': {'multiplier': 10, 'file': 'Data/cac40_daily_data.csv',  'currency': 'EUR'},
}

# EWMAC: all 10 instruments (trend following generalises across asset classes)
EWMAC_SPECS = {
    'MNQ':   {'multiplier': 2,          'file': 'Data/mnq_daily_data.csv',    'currency': 'USD'},
    'M2K':   {'multiplier': 5,          'file': 'Data/m2k_daily_data.csv',    'currency': 'USD'},
    'ESTX50':{'multiplier': 10,         'file': 'Data/estx50_daily_data.csv', 'currency': 'EUR'},
    'CAC40': {'multiplier': 10,         'file': 'Data/cac40_daily_data.csv',  'currency': 'EUR'},
    'ZF':    {'multiplier': 1_000,      'file': 'Data/zf_daily_data.csv',     'currency': 'USD'},
    'GBS':   {'multiplier': 1_000,      'file': 'Data/gbs_daily_data.csv',    'currency': 'EUR'},
    'JPY':   {'multiplier': 12_500_000, 'file': 'Data/jpy_daily_data.csv',    'currency': 'USD'},
    'ZM':    {'multiplier': 100,        'file': 'Data/zm_daily_data.csv',     'currency': 'USD'},
    'ZR':    {'multiplier': 2_000,      'file': 'Data/zr_daily_data.csv',     'currency': 'USD'},
    'QG':    {'multiplier': 2_500,      'file': 'Data/qg_daily_data.csv',     'currency': 'USD'},
}

# Combined unique set for data loading
CONTRACT_SPECS = {**EWMAC_SPECS}

N_IBS_INSTR   = len(IBS_SPECS)
N_EWMAC_INSTR = len(EWMAC_SPECS)
IBS_STRAT_ALLOC   = 1.0 / N_IBS_INSTR
EWMAC_STRAT_ALLOC = 1.0 / N_EWMAC_INSTR


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(file_path):
    lines = []
    with open(file_path, 'r') as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('"'):
                continue
            lines.append(s)
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
            logger.warning(f"No data for {symbol}")
            continue
        data[symbol] = df.set_index('Time')
        logger.info(f"Loaded {len(data[symbol])} bars for {symbol}")
    return data


# ── FX rates ─────────────────────────────────────────────────────────────────

EUR_STATIC_PRE_DATA = 1.15   # rough 2000-2009 EUR/USD average used before file starts

def load_fx_rates():
    """
    Load EUR/USD daily rate from Data/eur_daily_data.csv.
    Pre-2009 dates fall back to EUR_STATIC_PRE_DATA (1.15).
    Returns a dict: {'EUR': pd.Series(index=DatetimeIndex, data=rates)}
    """
    fx = {}
    path = 'Data/eur_daily_data.csv'
    try:
        lines = []
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith('"'):
                    continue
                lines.append(s)
        df = pd.read_csv(StringIO('\n'.join(lines)), parse_dates=['Time'])
        df = df.sort_values('Time').dropna(subset=['Last'])
        fx['EUR'] = df.set_index('Time')['Last']
        logger.info(f"Loaded EUR/USD: {len(fx['EUR'])} bars "
                    f"({df['Time'].min().date()} → {df['Time'].max().date()})")
    except Exception as e:
        logger.warning(f"Could not load EUR/USD data: {e} — using static {EUR_STATIC_PRE_DATA}")
    return fx


def get_fx_rate(date, currency, fx_data):
    """Return the FX rate (foreign per USD, i.e. multiply EUR P&L by this to get USD)."""
    if currency == 'USD':
        return 1.0
    series = fx_data.get(currency)
    if series is None or series.empty:
        return EUR_STATIC_PRE_DATA if currency == 'EUR' else 1.0
    if date in series.index:
        return float(series.loc[date])
    if date < series.index.min():
        return EUR_STATIC_PRE_DATA if currency == 'EUR' else float(series.iloc[0])
    if date > series.index.max():
        return float(series.iloc[-1])
    # nearest available
    idx = series.index.get_indexer([date], method='nearest')[0]
    return float(series.iloc[idx])


# ── Indicators ────────────────────────────────────────────────────────────────

def precompute_indicators(instrument_data):
    indicators = {}
    for symbol, df in instrument_data.items():
        ind = df[['High', 'Low', 'Last']].copy()

        rng        = ind['High'] - ind['Low']
        ind['IBS'] = np.where(rng > 0, (ind['Last'] - ind['Low']) / rng, 0.5)

        prices    = df['Last']
        fast_ema  = prices.ewm(span=FAST_SPAN, min_periods=FAST_SPAN).mean()
        slow_ema  = prices.ewm(span=SLOW_SPAN, min_periods=SLOW_SPAN).mean()
        direction = pd.Series(np.where(fast_ema > slow_ema, 1.0, -1.0), index=prices.index)
        direction[fast_ema.isna() | slow_ema.isna()] = np.nan
        ind['EWMAC_dir'] = direction

        vol            = prices.pct_change().ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std()
        ind['ann_vol'] = (vol * np.sqrt(256)).clip(lower=VOL_FLOOR)

        indicators[symbol] = ind
    return indicators


# ── IBS backtest ──────────────────────────────────────────────────────────────

def run_ibs_backtest(instrument_data, indicators, fx_data):
    all_dates = sorted(set().union(*[set(df.index) for df in instrument_data.values()]))

    state = {
        sym: {
            'capital':      INITIAL_CAPITAL * IBS_ALLOC * IBS_STRAT_ALLOC,
            'in_position':  False,
            'position':     None,
            'equity_curve': [],
        }
        for sym in IBS_SPECS
    }
    total_ibs_equity = INITIAL_CAPITAL * IBS_ALLOC

    for date in all_dates:
        daily_sum = 0.0
        for sym in IBS_SPECS:
            s    = state[sym]
            spec = IBS_SPECS[sym]
            mul  = spec['multiplier']
            fx   = get_fx_rate(date, spec['currency'], fx_data)
            ind  = indicators.get(sym)

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
                    pnl = ((price - pos['entry_price']) * mul * pos['fx_entry'] * pos['contracts']
                           - COMMISSION_RT * pos['contracts'])
                    s['capital']    += pnl
                    s['in_position'] = False
                    s['position']    = None
            else:
                if ibs < IBS_ENTRY:
                    notional_usd = price * mul * fx
                    contracts    = max(1, round(total_ibs_equity * IBS_STRAT_ALLOC / notional_usd))
                    s['capital']    -= (COMMISSION_RT / 2) * contracts
                    s['in_position'] = True
                    s['position']    = {'entry_price': price, 'fx_entry': fx,
                                        'contracts': contracts}

            if s['in_position']:
                pos    = s['position']
                equity = s['capital'] + (price - pos['entry_price']) * mul * pos['fx_entry'] * pos['contracts']
            else:
                equity = s['capital']

            s['equity_curve'].append((date, equity))
            daily_sum += equity

        total_ibs_equity = daily_sum

    # Force close
    for sym in IBS_SPECS:
        s    = state[sym]
        spec = IBS_SPECS[sym]
        df   = instrument_data.get(sym)
        if s['in_position'] and df is not None:
            last_price = float(df.iloc[-1]['Last'])
            mul        = spec['multiplier']
            pos        = s['position']
            pnl        = ((last_price - pos['entry_price']) * mul * pos['fx_entry'] * pos['contracts']
                          - (COMMISSION_RT / 2) * pos['contracts'])
            s['capital'] += pnl
            if s['equity_curve']:
                s['equity_curve'][-1] = (s['equity_curve'][-1][0], s['capital'])

    return state


# ── EWMAC backtest ────────────────────────────────────────────────────────────

def run_ewmac_backtest(instrument_data, indicators, fx_data):
    all_dates = sorted(set().union(*[set(df.index) for df in instrument_data.values()]))

    state = {
        sym: {
            'capital':      INITIAL_CAPITAL * EWMAC_ALLOC * EWMAC_STRAT_ALLOC,
            'position':     0,
            'prev_price':   None,
            'prev_fx':      1.0,
            'prev_dir':     np.nan,
            'total_gross':  0.0,
            'total_comm':   0.0,
            'equity_curve': [],
        }
        for sym in EWMAC_SPECS
    }

    for date in all_dates:
        for sym in EWMAC_SPECS:
            s    = state[sym]
            spec = EWMAC_SPECS[sym]
            mul  = spec['multiplier']
            fx   = get_fx_rate(date, spec['currency'], fx_data)
            df   = instrument_data.get(sym)
            ind  = indicators.get(sym)

            if df is None or ind is None or date not in df.index:
                last_eq = s['equity_curve'][-1][1] if s['equity_curve'] else s['capital']
                s['equity_curve'].append((date, last_eq))
                continue

            price   = float(df.loc[date, 'Last'])
            new_dir = float(ind.loc[date, 'EWMAC_dir']) if not pd.isna(ind.loc[date, 'EWMAC_dir']) else np.nan
            ann_vol = float(ind.loc[date, 'ann_vol'])

            # Daily mark-to-market using entry-day FX rate (locks in currency at trade time)
            if s['prev_price'] is not None and s['position'] != 0:
                daily_pnl        = (price - s['prev_price']) * mul * s['prev_fx'] * s['position']
                s['capital']    += daily_pnl
                s['total_gross'] += daily_pnl

            signal_changed = (not pd.isna(new_dir)) and (new_dir != s['prev_dir'])

            if signal_changed and ann_vol > 0:
                notional_usd = price * mul * fx
                avg_pos      = (s['capital'] * RISK_TARGET) / (notional_usd * ann_vol)
                new_pos      = max(1, int(round(avg_pos))) * int(new_dir)
                trade        = abs(new_pos - s['position'])
                if trade > 0:
                    comm             = (COMMISSION_RT / 2) * trade
                    s['capital']    -= comm
                    s['total_comm'] += comm
                s['position'] = new_pos
                s['prev_dir'] = new_dir
                s['prev_fx']  = fx
            elif not pd.isna(new_dir) and pd.isna(s['prev_dir']):
                s['prev_dir'] = new_dir
                s['prev_fx']  = fx

            s['prev_price'] = price
            s['equity_curve'].append((date, s['capital']))

    # Close open positions
    for sym in EWMAC_SPECS:
        s  = state[sym]
        df = instrument_data.get(sym)
        if s['position'] != 0 and df is not None and not df.empty:
            comm             = (COMMISSION_RT / 2) * abs(s['position'])
            s['capital']    -= comm
            s['total_comm'] += comm
            if s['equity_curve']:
                s['equity_curve'][-1] = (s['equity_curve'][-1][0], s['capital'])

    return state


# ── Equity curves ─────────────────────────────────────────────────────────────

def build_equity_curve(state_dict, common_dates, specs):
    series = []
    for sym in specs:
        curve = state_dict[sym]['equity_curve']
        df    = pd.DataFrame(curve, columns=['Time', 'Equity']).set_index('Time')
        df    = df[~df.index.duplicated(keep='last')].sort_index()
        df    = df.reindex(common_dates, method='ffill')
        series.append(df['Equity'])
    combined = pd.DataFrame({'Equity': sum(series)}, index=common_dates)
    combined.dropna(inplace=True)
    return combined


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(equity_df, label=''):
    eq   = equity_df['Equity']
    rets = eq.pct_change().dropna()
    if rets.empty or len(rets) < 5:
        return None

    final   = eq.iloc[-1]
    years   = (eq.index[-1] - eq.index[0]).days / 365.25
    ann_ret = ((final / eq.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else np.nan
    std     = rets.std()
    sharpe  = rets.mean() / std * np.sqrt(252) if std > 0 else np.nan
    d_std   = rets[rets < 0].std()
    sortino = rets.mean() / d_std * np.sqrt(252) if d_std > 0 else np.nan
    peak    = eq.cummax()
    dd      = (eq - peak) / peak
    max_dd  = dd.min() * 100
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        'label':             label,
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
        if k == 'label':
            continue
        if k == 'Final Balance':
            print(f"  {k:<26} ${v:>12,.2f}")
        elif isinstance(v, float) and not np.isnan(v):
            print(f"  {k:<26} {v:>12.2f}")
        else:
            print(f"  {k:<26} {'NaN':>12}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("Loading data...")
    instrument_data = load_all_data()
    if not instrument_data:
        logger.error("No data loaded.")
        return

    logger.info("Loading FX rates...")
    fx_data = load_fx_rates()

    logger.info("Pre-computing indicators...")
    indicators = precompute_indicators(instrument_data)

    logger.info("Running IBS backtest...")
    ibs_state = run_ibs_backtest(instrument_data, indicators, fx_data)

    logger.info("Running EWMAC backtest...")
    ewmac_state = run_ewmac_backtest(instrument_data, indicators, fx_data)

    common_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    ibs_eq   = build_equity_curve(ibs_state,   common_dates, IBS_SPECS)
    ewmac_eq = build_equity_curve(ewmac_state, common_dates, EWMAC_SPECS)

    valid        = ibs_eq.index.intersection(ewmac_eq.index)
    ibs_eq       = ibs_eq.loc[valid]
    ewmac_eq     = ewmac_eq.loc[valid]
    combined_eq  = pd.DataFrame({'Equity': ibs_eq['Equity'] + ewmac_eq['Equity']}, index=valid)

    ibs_scaled   = pd.DataFrame({'Equity': ibs_eq['Equity']   * (1.0 / IBS_ALLOC)},   index=valid)
    ewmac_scaled = pd.DataFrame({'Equity': ewmac_eq['Equity'] * (1.0 / EWMAC_ALLOC)}, index=valid)

    ibs_m   = compute_metrics(ibs_scaled,   'IBS standalone')
    ewmac_m = compute_metrics(ewmac_scaled, 'EWMAC standalone')
    combo_m = compute_metrics(combined_eq,  'Combined')

    ibs_ret   = ibs_eq['Equity'].pct_change().dropna()
    ewmac_ret = ewmac_eq['Equity'].pct_change().dropna()
    corr      = ibs_ret.corr(ewmac_ret.reindex(ibs_ret.index))

    # MNQ B&H benchmark (equity proxy)
    bnh_eq = None
    mnq_prices = instrument_data['MNQ']['Last'].reindex(valid, method='ffill').dropna() \
                 if 'MNQ' in instrument_data else None
    if mnq_prices is not None and not mnq_prices.empty:
        years = (mnq_prices.index - mnq_prices.index[0]).days / 365.25
        bnh_eq = pd.DataFrame(
            {'Equity': INITIAL_CAPITAL
                       * (mnq_prices / float(mnq_prices.iloc[0]))
                       * (1 + DIV_YIELD) ** years},
            index=mnq_prices.index,
        )

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"HYBRID BACKTEST  |  IBS + EWMAC({FAST_SPAN},{SLOW_SPAN})  |  ${INITIAL_CAPITAL:,.0f}")
    print(f"IBS  (75%): {', '.join(IBS_SPECS)}  [{N_IBS_INSTR} instruments]")
    print(f"EWMAC(25%): {', '.join(EWMAC_SPECS)}  [{N_EWMAC_INSTR} instruments]")
    eur_start = fx_data['EUR'].index.min().date() if 'EUR' in fx_data else 'N/A'
    print(f"EUR/USD: daily rates from {eur_start}, static {EUR_STATIC_PRE_DATA} before that")
    print("=" * 64)

    print(f"\n  IBS/EWMAC return correlation: {corr:+.3f}")

    print(f"\n── IBS standalone (scaled to ${INITIAL_CAPITAL:,.0f}) ─────────────────────────")
    print_metrics(ibs_m)

    print(f"\n── EWMAC({FAST_SPAN},{SLOW_SPAN}) standalone (scaled to ${INITIAL_CAPITAL:,.0f}) ─────────────────")
    print_metrics(ewmac_m)

    print(f"\n── Combined IBS + EWMAC (${INITIAL_CAPITAL:,.0f}) ───────────────────────────────")
    print_metrics(combo_m)

    if ibs_m and ewmac_m and combo_m:
        best_sharpe  = max(ibs_m['Sharpe'], ewmac_m['Sharpe'])
        best_max_dd  = max(ibs_m['Max Drawdown %'], ewmac_m['Max Drawdown %'])
        print(f"\n  vs best standalone  Sharpe: {best_sharpe:.3f}  →  {combo_m['Sharpe']:.3f}"
              f"  ({combo_m['Sharpe'] - best_sharpe:+.3f})")
        print(f"  vs best standalone  Max DD: {best_max_dd:.1f}%  →  {combo_m['Max Drawdown %']:.1f}%"
              f"  ({combo_m['Max Drawdown %'] - best_max_dd:+.1f}%)")

    if bnh_eq is not None:
        bnh_m = compute_metrics(bnh_eq, 'MNQ B&H')
        if bnh_m:
            print(f"\n── MNQ B&H benchmark (price + ~1.8% div, ${INITIAL_CAPITAL:,.0f}) ────────────")
            print_metrics(bnh_m)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={'height_ratios': [3, 1, 1]})
    ax1, ax2, ax3 = axes

    ax1.plot(combined_eq.index, combined_eq['Equity'],
             color='seagreen', linewidth=2.0, label='Combined IBS+EWMAC', zorder=3)
    ax1.plot(ibs_scaled.index, ibs_scaled['Equity'],
             color='steelblue', linewidth=1.0, alpha=0.7, linestyle='--', label='IBS only (scaled)')
    ax1.plot(ewmac_scaled.index, ewmac_scaled['Equity'],
             color='darkorange', linewidth=1.0, alpha=0.7, linestyle='--',
             label=f'EWMAC({FAST_SPAN},{SLOW_SPAN}) only (scaled)')
    if bnh_eq is not None:
        ax1.plot(bnh_eq.index, bnh_eq['Equity'],
                 color='goldenrod', linewidth=1.0, alpha=0.7, linestyle=':',
                 label='MNQ B&H (price + div, est.)')
    ax1.axhline(INITIAL_CAPITAL, color='gray', linestyle=':', linewidth=0.8,
                label='Starting capital')
    ax1.set_ylabel('Portfolio Equity ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    if combo_m:
        ax1.set_title(
            f'IBS + EWMAC({FAST_SPAN},{SLOW_SPAN})  |  {len(CONTRACT_SPECS)} instruments  |  ${INITIAL_CAPITAL:,.0f}\n'
            f'Sharpe {combo_m["Sharpe"]:.2f}  |  Ann. Return {combo_m["Ann. Return %"]:.1f}%'
            f'  |  Max DD {combo_m["Max Drawdown %"]:.1f}%  |  IBS/EWMAC corr {corr:+.3f}',
            fontsize=11,
        )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    peak_c = combined_eq['Equity'].cummax()
    dd_c   = (combined_eq['Equity'] - peak_c) / peak_c * 100
    ax2.fill_between(combined_eq.index, dd_c, 0, color='crimson', alpha=0.4)
    ax2.set_ylabel('Combined DD (%)')
    ax2.grid(True, alpha=0.3)

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
    plt.savefig('portfolio/multi_equity_curve.png', dpi=150)
    plt.show()
    logger.info("Chart saved to portfolio/multi_equity_curve.png")


if __name__ == '__main__':
    main()
