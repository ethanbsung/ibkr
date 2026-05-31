#!/usr/bin/env python3
"""
EWMAC Ensemble Backtest — continuous forecasting (Carver method)

Replaces binary +1/-1 crossover signal with a scaled continuous forecast:
  raw_norm = (EMA_fast - EMA_slow) / (price × daily_vol)
  forecast = clip(raw_norm × scalar, -20, 20)   mean |forecast| ≈ 10

Ensemble forecast = mean(N forecasts) × FDM, re-capped at ±20.

Position sizing:
  target = (forecast / 10) × risk_target × capital / (price × mult × ann_vol)
  Positions can be 0 when forecast is weak — no forced minimum contract.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from io import StringIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Parameters ───────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 250_000.0
COMMISSION_RT   = 5.0
START_DATE      = '2000-01-01'
END_DATE        = '2026-01-01'
DIV_YIELD       = 0.018

RISK_TARGET     = 0.20
VOL_SPAN        = 32
VOL_FLOOR       = 0.05
FORECAST_TARGET = 10.0    # target mean absolute forecast
FORECAST_CAP    = 20.0    # hard cap on forecast magnitude

# EWMAC speeds to test and ensemble
EWMAC_SPEEDS = [(16, 64), (32, 128), (64, 256)]

# Forecast scalars from Carver "Systematic Trading"
# raw_norm = (EMA_fast - EMA_slow) / (price × daily_vol_pct)
# multiply by scalar so mean |forecast| ≈ 10 across typical instruments
FORECAST_SCALARS = {
    (16,  64): 4.10,
    (32, 128): 2.79,
    (64, 256): 1.91,
}

# Forecast Diversification Multiplier — restores mean |ensemble| ≈ 10
# after averaging correlated signals; ~1.25 for 3 adjacent EWMAC speeds
ENSEMBLE_FDM = 1.25

# ── Contract specifications ───────────────────────────────────────────────────
SPECS = {
    'MNQ':   {'multiplier': 2,          'file': 'Data/mnq_daily_data.csv',    'currency': 'USD'},
    'M2K':   {'multiplier': 5,          'file': 'Data/m2k_daily_data.csv',    'currency': 'USD'},
    'ESTX50':{'multiplier': 10,         'file': 'Data/estx50_daily_data.csv', 'currency': 'EUR'},
    'CAC40': {'multiplier': 10,         'file': 'Data/cac40_daily_data.csv',  'currency': 'EUR'},
    'ZF':    {'multiplier': 1_000,      'file': 'Data/zf_daily_data.csv',     'currency': 'USD'},
    'GBS':   {'multiplier': 1_000,      'file': 'Data/gbs_daily_data.csv',    'currency': 'EUR'},
    'JPY':   {'multiplier': 12_500_000, 'file': 'Data/jpy_daily_data.csv',    'currency': 'USD'},
}

N_INSTR = len(SPECS)


# ── Data loading ──────────────────────────────────────────────────────────────

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
    return df.dropna(subset=['High', 'Low', 'Last']).reset_index(drop=True)


def load_all_data():
    data = {}
    for symbol, spec in SPECS.items():
        df = load_data(spec['file'])
        if df.empty:
            logger.warning(f"No data for {symbol}")
            continue
        data[symbol] = df.set_index('Time')
        logger.info(f"Loaded {len(data[symbol])} bars for {symbol}")
    return data


# ── FX rates ──────────────────────────────────────────────────────────────────

EUR_STATIC = 1.15

def load_fx_rates():
    fx = {}
    try:
        lines = []
        with open('Data/eur_daily_data.csv') as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith('"'):
                    continue
                lines.append(s)
        df = pd.read_csv(StringIO('\n'.join(lines)), parse_dates=['Time'])
        df = df.sort_values('Time').dropna(subset=['Last'])
        fx['EUR'] = df.set_index('Time')['Last']
        logger.info(f"Loaded EUR/USD: {len(fx['EUR'])} bars")
    except Exception as e:
        logger.warning(f"Could not load EUR/USD: {e}")
    return fx


def get_fx_rate(date, currency, fx_data):
    if currency == 'USD':
        return 1.0
    series = fx_data.get(currency)
    if series is None or series.empty:
        return EUR_STATIC if currency == 'EUR' else 1.0
    if date in series.index:
        return float(series.loc[date])
    if date < series.index.min():
        return EUR_STATIC if currency == 'EUR' else float(series.iloc[0])
    if date > series.index.max():
        return float(series.iloc[-1])
    return float(series.iloc[series.index.get_indexer([date], method='nearest')[0]])


# ── Indicators and forecasts ──────────────────────────────────────────────────

def precompute_indicators(instrument_data):
    """
    Per instrument:
      - ann_vol: annualised daily volatility (floored at VOL_FLOOR)
      - forecast_{fast}_{slow}: continuous EWMAC forecast in [-20, 20]
    """
    indicators = {}
    for symbol, df in instrument_data.items():
        prices = df['Last']
        ind    = pd.DataFrame(index=prices.index)
        ind['Last'] = prices

        daily_vol_pct  = prices.pct_change().ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std()
        daily_vol_pct  = daily_vol_pct.clip(lower=VOL_FLOOR / np.sqrt(256))
        inst_vol_price = prices * daily_vol_pct   # daily vol in price units

        ind['ann_vol'] = (daily_vol_pct * np.sqrt(256)).clip(lower=VOL_FLOOR)

        for fast, slow in EWMAC_SPEEDS:
            fast_ema = prices.ewm(span=fast, min_periods=fast).mean()
            slow_ema = prices.ewm(span=slow, min_periods=slow).mean()
            raw      = fast_ema - slow_ema
            raw_norm = raw / inst_vol_price
            forecast = (raw_norm * FORECAST_SCALARS[(fast, slow)]).clip(-FORECAST_CAP, FORECAST_CAP)
            forecast[fast_ema.isna() | slow_ema.isna() | inst_vol_price.isna()] = np.nan
            ind[f'forecast_{fast}_{slow}'] = forecast

        indicators[symbol] = ind
    return indicators


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(instrument_data, indicators, fx_data, speeds=None):
    """
    Continuous-forecast EWMAC backtest.
    speeds=None  → ensemble over all EWMAC_SPEEDS (FDM applied when all present)
    speeds=[...] → single speed or custom subset (no FDM)

    Capital is fixed at INITIAL_CAPITAL / N_INSTR per instrument for sizing.
    Positions recomputed daily; trade only when the integer target changes.
    """
    if speeds is None:
        speeds = EWMAC_SPEEDS
    n_speeds     = len(speeds)
    use_fdm      = (n_speeds > 1)
    all_dates    = sorted(set().union(*[set(df.index) for df in instrument_data.values()]))
    cap_per_instr = INITIAL_CAPITAL / N_INSTR

    state = {
        sym: {'capital': cap_per_instr, 'position': 0, 'prev_price': None,
              'prev_fx': 1.0, 'equity': []}
        for sym in SPECS
    }

    for date in all_dates:
        for sym in SPECS:
            s    = state[sym]
            spec = SPECS[sym]
            mul  = spec['multiplier']
            fx   = get_fx_rate(date, spec['currency'], fx_data)
            ind  = indicators.get(sym)

            if ind is None or date not in ind.index:
                last_eq = s['equity'][-1][1] if s['equity'] else s['capital']
                s['equity'].append((date, last_eq))
                continue

            price   = float(ind.loc[date, 'Last'])
            ann_vol = float(ind.loc[date, 'ann_vol'])

            # Daily mark-to-market
            if s['prev_price'] is not None and s['position'] != 0:
                s['capital'] += (price - s['prev_price']) * mul * s['prev_fx'] * s['position']

            # Gather valid forecasts
            forecast_vals = []
            for fast, slow in speeds:
                col = f'forecast_{fast}_{slow}'
                if col in ind.columns:
                    val = float(ind.loc[date, col])
                    if not np.isnan(val):
                        forecast_vals.append(val)

            if forecast_vals and ann_vol > 0 and price > 0:
                avg_fc = float(np.mean(forecast_vals))
                if use_fdm and len(forecast_vals) == n_speeds:
                    fc = float(np.clip(avg_fc * ENSEMBLE_FDM, -FORECAST_CAP, FORECAST_CAP))
                else:
                    fc = float(np.clip(avg_fc, -FORECAST_CAP, FORECAST_CAP))

                notional_usd = price * mul * fx
                target_pos   = (fc / FORECAST_TARGET) * (s['capital'] * RISK_TARGET) / (notional_usd * ann_vol)
                new_pos      = int(round(target_pos))

                if new_pos != s['position']:
                    trade = abs(new_pos - s['position'])
                    s['capital'] -= (COMMISSION_RT / 2) * trade
                    s['position'] = new_pos
                    s['prev_fx']  = fx

            s['prev_price'] = price
            s['equity'].append((date, s['capital']))

    # Close remaining positions
    for sym in SPECS:
        s  = state[sym]
        df = instrument_data.get(sym)
        if s['position'] != 0 and df is not None and not df.empty:
            s['capital'] -= (COMMISSION_RT / 2) * abs(s['position'])
            if s['equity']:
                s['equity'][-1] = (s['equity'][-1][0], s['capital'])

    return {sym: pd.Series(dict(state[sym]['equity']), name=sym).sort_index() for sym in SPECS}


# ── Equity curve helper ───────────────────────────────────────────────────────

def sum_equity(instrument_equities, common_dates):
    series = [
        ser[~ser.index.duplicated(keep='last')].sort_index().reindex(common_dates, method='ffill')
        for ser in instrument_equities.values()
    ]
    return pd.DataFrame({'Equity': sum(series)}, index=common_dates)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(equity_df, label=''):
    eq   = equity_df['Equity'].dropna()
    eq   = eq[eq > 0]
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

    logger.info("Pre-computing indicators and forecasts...")
    indicators = precompute_indicators(instrument_data)

    common_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')

    # ── Single-speed standalone (full capital each, for comparison) ───────────
    speed_results = {}
    for fast, slow in EWMAC_SPEEDS:
        logger.info(f"Running EWMAC({fast},{slow}) standalone...")
        equities = run_backtest(instrument_data, indicators, fx_data, speeds=[(fast, slow)])
        speed_results[(fast, slow)] = sum_equity(equities, common_dates)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    logger.info("Running ensemble (all speeds, FDM applied)...")
    ensemble_eq = sum_equity(
        run_backtest(instrument_data, indicators, fx_data, speeds=None),
        common_dates,
    )

    # ── MNQ B&H benchmark ─────────────────────────────────────────────────────
    bnh_eq = None
    if 'MNQ' in instrument_data:
        mnq_p = instrument_data['MNQ']['Last'].reindex(common_dates, method='ffill').dropna()
        if not mnq_p.empty:
            years  = (mnq_p.index - mnq_p.index[0]).days / 365.25
            bnh_eq = pd.DataFrame(
                {'Equity': INITIAL_CAPITAL * (mnq_p / float(mnq_p.iloc[0])) * (1 + DIV_YIELD) ** years},
                index=mnq_p.index,
            )

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"EWMAC ENSEMBLE  (continuous forecast)  |  ${INITIAL_CAPITAL:,.0f}  |  {N_INSTR} instr.")
    print(f"Speeds: {', '.join(f'({f},{s})' for f,s in EWMAC_SPEEDS)}")
    print(f"Scalars: {', '.join(f'({f},{s})={v}' for (f,s),v in FORECAST_SCALARS.items())}"
          f"  |  FDM={ENSEMBLE_FDM}")
    print("=" * 68)

    standalone_metrics = {}
    for (fast, slow), eq_df in speed_results.items():
        label = f'EWMAC({fast},{slow})'
        m = compute_metrics(eq_df, label)
        standalone_metrics[(fast, slow)] = m
        print(f"\n── {label} standalone ──────────────────────────────────")
        if m:
            print_metrics(m)

    ens_m = compute_metrics(ensemble_eq, 'Ensemble')
    print(f"\n── Ensemble [{', '.join(f'({f},{s})' for f,s in EWMAC_SPEEDS)}]  FDM={ENSEMBLE_FDM} ───────────")
    if ens_m:
        print_metrics(ens_m)

    sharpes = {k: m['Sharpe'] for k, m in standalone_metrics.items() if m}
    if sharpes and ens_m:
        best_key = max(sharpes, key=sharpes.get)
        best_m   = standalone_metrics[best_key]
        print(f"\n  vs best standalone EWMAC({best_key[0]},{best_key[1]}):")
        print(f"  Sharpe:  {best_m['Sharpe']:.3f}  →  {ens_m['Sharpe']:.3f}"
              f"  ({ens_m['Sharpe'] - best_m['Sharpe']:+.3f})")
        print(f"  Max DD:  {best_m['Max Drawdown %']:.1f}%  →  {ens_m['Max Drawdown %']:.1f}%"
              f"  ({ens_m['Max Drawdown %'] - best_m['Max Drawdown %']:+.1f}%)")

    if bnh_eq is not None:
        bnh_m = compute_metrics(bnh_eq, 'MNQ B&H')
        if bnh_m:
            print(f"\n── MNQ B&H benchmark (price + ~1.8% div) ─────────────────────")
            print_metrics(bnh_m)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={'height_ratios': [3, 1]})

    colors = ['steelblue', 'darkorange', 'mediumseagreen']
    for i, ((fast, slow), eq_df) in enumerate(speed_results.items()):
        eq_df = eq_df.dropna()
        ax1.plot(eq_df.index, eq_df['Equity'],
                 color=colors[i], linewidth=1.0, alpha=0.55, linestyle='--',
                 label=f'EWMAC({fast},{slow})')

    ens_clean = ensemble_eq.dropna()
    ax1.plot(ens_clean.index, ens_clean['Equity'],
             color='crimson', linewidth=2.0, zorder=3,
             label=f'Ensemble (FDM={ENSEMBLE_FDM})')

    if bnh_eq is not None:
        ax1.plot(bnh_eq.index, bnh_eq['Equity'],
                 color='goldenrod', linewidth=1.0, alpha=0.7, linestyle=':',
                 label='MNQ B&H (price + div)')

    ax1.axhline(INITIAL_CAPITAL, color='gray', linestyle=':', linewidth=0.8,
                label='Starting capital')
    ax1.set_ylabel('Portfolio Equity ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    title_line2 = (
        f"Sharpe {ens_m['Sharpe']:.2f}  |  Ann. Return {ens_m['Ann. Return %']:.1f}%"
        f"  |  Max DD {ens_m['Max Drawdown %']:.1f}%"
    ) if ens_m else ''
    ax1.set_title(
        f"EWMAC Ensemble (Continuous Forecast)  |  {N_INSTR} instruments  |  ${INITIAL_CAPITAL:,.0f}\n"
        + title_line2,
        fontsize=11,
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    peak = ens_clean['Equity'].cummax()
    dd   = (ens_clean['Equity'] - peak) / peak * 100
    ax2.fill_between(ens_clean.index, dd, 0, color='crimson', alpha=0.4)
    ax2.set_ylabel('Ensemble DD (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = 'portfolio/ewmac_ensemble_equity_curve.png'
    plt.savefig(out, dpi=150)
    plt.show()
    logger.info(f"Chart saved to {out}")


if __name__ == '__main__':
    main()
