#!/usr/bin/env python3
"""
IBS Portfolio Backtest
3 instruments, equal-weight, $50,000 starting capital.
ES (MES), NQ (MNQ), GC (MGC)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from io import StringIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 50_000.0
COMMISSION_RT   = 5.0        # round-trip commission per contract ($)
START_DATE      = '2000-01-01'
END_DATE        = '2026-01-01'

IBS_ENTRY = 0.10
IBS_EXIT  = 0.90

# ── Contract specifications ────────────────────────────────────────────────────
CONTRACT_SPECS = {
    'ES': {'multiplier': 5,  'file': 'Data/mes_daily_data.csv'},  # MES micro
    'NQ': {'multiplier': 2,  'file': 'Data/mnq_daily_data.csv'},  # MNQ micro
    'GC': {'multiplier': 10, 'file': 'Data/mgc_daily_data.csv'},  # MGC micro
}

STRATEGIES = [
    ('IBS_ES', 'ES'),
    ('IBS_NQ', 'NQ'),
    ('IBS_GC', 'GC'),
]

N_STRATS = len(STRATEGIES)
ALLOC    = 1.0 / N_STRATS


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(file_path):
    """Load Barchart-format daily CSV, stripping footer and empty lines."""
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
    """Load and date-index data for every instrument in CONTRACT_SPECS."""
    data = {}
    for symbol, spec in CONTRACT_SPECS.items():
        df = load_data(spec['file'])
        if df.empty:
            logger.warning(f"No data loaded for {symbol}")
            continue
        data[symbol] = df.set_index('Time')
        logger.info(f"Loaded {len(data[symbol])} bars for {symbol}")
    return data


# ── Indicator pre-computation ─────────────────────────────────────────────────

def precompute_ibs(instrument_data):
    """Pre-compute IBS for every instrument."""
    indicators = {}
    for symbol, df in instrument_data.items():
        ind = df[['High', 'Low', 'Last']].copy()
        bar_range = ind['High'] - ind['Low']
        ind['IBS'] = np.where(bar_range > 0,
                              (ind['Last'] - ind['Low']) / bar_range,
                              0.5)
        indicators[symbol] = ind
    return indicators


# ── Position sizing ────────────────────────────────────────────────────────────

def position_size(total_equity, price, multiplier):
    """
    Contracts = (portfolio_equity × per-strategy allocation) / contract_value.
    Always at least 1. Scales up naturally as equity grows.
    At $50k starting capital this returns 1 for all instruments.
    MES reaches 2 contracts around $495k total equity.
    """
    contract_value = price * multiplier
    if contract_value <= 0:
        return 1
    return max(1, round(total_equity * ALLOC / contract_value))


# ── Backtest engine ────────────────────────────────────────────────────────────

def run_backtest(instrument_data, indicators):
    """
    Date-aligned IBS backtest. Iterates over the union of all trading dates
    so instruments on different exchange calendars (CME vs Eurex) stay aligned.
    """
    all_dates = sorted(set().union(*[set(df.index) for df in instrument_data.values()]))

    state = {
        name: {
            'capital':      INITIAL_CAPITAL * ALLOC,
            'in_position':  False,
            'position':     None,   # {entry_price, entry_date, contracts}
            'equity_curve': [],
        }
        for name, _ in STRATEGIES
    }

    total_equity = INITIAL_CAPITAL

    for date in all_dates:
        daily_equity_sum = 0.0

        for name, symbol in STRATEGIES:
            s   = state[name]
            mul = CONTRACT_SPECS[symbol]['multiplier']
            ind = indicators.get(symbol)

            if ind is None or date not in ind.index:
                last_eq = s['equity_curve'][-1][1] if s['equity_curve'] else s['capital']
                s['equity_curve'].append((date, last_eq))
                daily_equity_sum += last_eq
                continue

            row   = ind.loc[date]
            price = row['Last']
            ibs   = row['IBS']

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
                    contracts = position_size(total_equity, price, mul)
                    s['capital']    -= (COMMISSION_RT / 2) * contracts
                    s['in_position'] = True
                    s['position']    = {
                        'entry_price': price,
                        'entry_date':  date,
                        'contracts':   contracts,
                    }

            if s['in_position']:
                pos       = s['position']
                unrealized = (price - pos['entry_price']) * mul * pos['contracts']
                equity     = s['capital'] + unrealized
            else:
                equity = s['capital']

            s['equity_curve'].append((date, equity))
            daily_equity_sum += equity

        total_equity = daily_equity_sum

    # Force-close any open positions at end of backtest
    for name, symbol in STRATEGIES:
        s  = state[name]
        df = instrument_data.get(symbol)
        if s['in_position'] and df is not None and not df.empty:
            last_price = df.iloc[-1]['Last']
            mul        = CONTRACT_SPECS[symbol]['multiplier']
            pos        = s['position']
            pnl        = ((last_price - pos['entry_price']) * mul * pos['contracts']
                          - (COMMISSION_RT / 2) * pos['contracts'])
            s['capital'] += pnl
            if s['equity_curve']:
                s['equity_curve'][-1] = (s['equity_curve'][-1][0], s['capital'])
            s['in_position'] = False

    return state


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(equity_df):
    """Standard performance metrics from a daily equity DataFrame."""
    eq   = equity_df['Equity']
    rets = eq.pct_change().dropna()

    final   = eq.iloc[-1]
    years   = (eq.index[-1] - eq.index[0]).days / 365.25
    ann_ret = ((final / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else np.nan

    std     = rets.std()
    sharpe  = rets.mean() / std * np.sqrt(252) if std > 0 else np.nan
    d_std   = rets[rets < 0].std()
    sortino = rets.mean() / d_std * np.sqrt(252) if d_std > 0 else np.nan

    peak   = eq.cummax()
    dd     = (eq - peak) / peak
    max_dd = dd.min() * 100
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        'Final Balance':     final,
        'Total Return %':    (final / INITIAL_CAPITAL - 1) * 100,
        'Ann. Return %':     ann_ret,
        'Ann. Volatility %': std * np.sqrt(252) * 100,
        'Sharpe':            sharpe,
        'Sortino':           sortino,
        'Calmar':            calmar,
        'Max Drawdown %':    max_dd,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("Loading data...")
    instrument_data = load_all_data()
    if not instrument_data:
        logger.error("No data loaded — check DATA directory.")
        return

    logger.info("Pre-computing IBS...")
    indicators = precompute_ibs(instrument_data)

    logger.info("Running backtest...")
    state = run_backtest(instrument_data, indicators)

    # ── Build combined equity curve ────────────────────────────────────────────
    common_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    series = []
    for name, _ in STRATEGIES:
        curve = state[name]['equity_curve']
        df    = pd.DataFrame(curve, columns=['Time', 'Equity']).set_index('Time')
        df    = df[~df.index.duplicated(keep='last')].sort_index()
        df    = df.reindex(common_dates, method='ffill')
        series.append(df['Equity'])
    combined_df = pd.DataFrame({'Equity': sum(series)}, index=common_dates)
    combined_df.dropna(inplace=True)

    # ── Benchmark (MES price return, no dividends) ─────────────────────────────
    es_data = instrument_data.get('ES')
    if es_data is not None:
        bm_start = es_data['Last'].iloc[0]
        bm_end   = es_data['Last'].iloc[-1]
        bm_years = (es_data.index[-1] - es_data.index[0]).days / 365.25
        bm_ann   = ((bm_end / bm_start) ** (1 / bm_years) - 1) * 100
        bm_total = (bm_end / bm_start - 1) * 100
    else:
        bm_ann = bm_total = float('nan')

    # ── Print results ──────────────────────────────────────────────────────────
    metrics = compute_metrics(combined_df)

    print("\n" + "=" * 60)
    print(f"IBS PORTFOLIO  |  {N_STRATS} instruments  |  ${INITIAL_CAPITAL:,.0f} capital")
    print(f"IBS entry < {IBS_ENTRY}  |  IBS exit > {IBS_EXIT}  |  Equal weight")
    print("=" * 60)

    print(f"\n  Benchmark note: MES price return only (no dividends).")
    print(f"  Starting Jan 2000 = dot-com peak; S&P total return with")
    print(f"  dividends reinvested would be ~{bm_ann + 1.8:.1f}% annualised.\n")

    for k, v in metrics.items():
        if k == 'Final Balance':
            print(f"  {k:<26} ${v:>12,.2f}")
        elif not np.isnan(v):
            print(f"  {k:<26} {v:>12.2f}")
        else:
            print(f"  {k:<26} {'NaN':>12}")

    print(f"\n  {'MES B&H Total Return':26} {bm_total:>11.2f}%")
    print(f"  {'MES B&H Ann. Return':26} {bm_ann:>11.2f}%  (price only)")

    print(f"\n  {'Strategy':<12} {'Final $':>10}  {'Return %':>9}")
    print(f"  {'-'*34}")
    for name, _ in STRATEGIES:
        fc  = state[name]['capital']
        ret = (fc / (INITIAL_CAPITAL * ALLOC) - 1) * 100
        print(f"  {name:<12} ${fc:>9,.0f}  {ret:>8.1f}%")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(combined_df.index, combined_df['Equity'],
             color='steelblue', linewidth=1.5, label='IBS Portfolio')
    ax1.axhline(INITIAL_CAPITAL, color='gray', linestyle='--',
                linewidth=0.8, label='Starting capital')
    ax1.set_ylabel('Portfolio Equity ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.set_title(
        f'IBS Portfolio  |  {N_STRATS} instruments  |  ${INITIAL_CAPITAL:,.0f} capital\n'
        f'Sharpe {metrics["Sharpe"]:.2f}  |  Ann. Return {metrics["Ann. Return %"]:.1f}%'
        f'  |  Max DD {metrics["Max Drawdown %"]:.1f}%',
        fontsize=11
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    peak = combined_df['Equity'].cummax()
    dd   = (combined_df['Equity'] - peak) / peak * 100
    ax2.fill_between(combined_df.index, dd, 0, color='crimson', alpha=0.4)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('portfolio/aggregate_port_equity_curve.png', dpi=150)
    plt.show()
    logger.info("Chart saved to portfolio/aggregate_port_equity_curve.png")


if __name__ == '__main__':
    main()
