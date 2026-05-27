#!/usr/bin/env python3
"""
EWMAC Trend Following Backtest
3 instruments (MES/MNQ/MGC), equal-weight, $50,000 starting capital.

Signal:   EWMAC(16,64) — go long when 16-day EMA > 64-day EMA, short when below.
          Trades only on crossovers (~4-8 per year per instrument).
          No continuous rebalancing — that creates commission drag that kills returns.

Sizing:   Vol-targeted at each trade entry:
          N = max(1, round((capital × ALLOC × RISK_TARGET) / (price × mult × ann_vol)))
          Position is held at fixed size until the next crossover.

Dates and commission match aggregate_port.py for direct comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from io import StringIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Parameters (match aggregate_port.py) ──────────────────────────────────────
INITIAL_CAPITAL = 50_000.0
COMMISSION_RT   = 5.0
START_DATE      = '2000-01-01'
END_DATE        = '2026-01-01'

# ── EWMAC parameters ───────────────────────────────────────────────────────────
RISK_TARGET  = 0.20    # 20% annualised vol target per sub-strategy
FAST_SPAN    = 64
SLOW_SPAN    = 256
VOL_SPAN     = 32      # EWMA span for vol estimate
VOL_FLOOR    = 0.05    # 5% minimum annual vol (avoids blow-up in low-vol regimes)

# ── Contract specifications ────────────────────────────────────────────────────
CONTRACT_SPECS = {
    'ES': {'multiplier': 5,  'file': 'Data/mes_daily_data.csv'},
    'NQ': {'multiplier': 2,  'file': 'Data/mnq_daily_data.csv'},
    'GC': {'multiplier': 10, 'file': 'Data/mgc_daily_data.csv'},
}

N_STRATS = len(CONTRACT_SPECS)
ALLOC    = 1.0 / N_STRATS


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
    if 'Last' not in df.columns:
        return pd.DataFrame()
    df.sort_values('Time', inplace=True)
    df = df[(df['Time'] >= START_DATE) & (df['Time'] <= END_DATE)]
    df = df.dropna(subset=['Last']).reset_index(drop=True)
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


# ── Signal pre-computation ─────────────────────────────────────────────────────

def precompute_signals(instrument_data):
    """
    Compute EWMAC crossover signal and vol for each instrument.

    signal:  +1 (long) when fast EMA > slow EMA, -1 (short) when below.
             NaN during the warmup period (first SLOW_SPAN bars).
    ann_vol: annualised % volatility from EWMA(VOL_SPAN) of daily returns.
    """
    signals = {}
    for symbol, df in instrument_data.items():
        prices = df['Last']

        fast_ema = prices.ewm(span=FAST_SPAN, min_periods=FAST_SPAN).mean()
        slow_ema = prices.ewm(span=SLOW_SPAN, min_periods=SLOW_SPAN).mean()

        # +1 long / -1 short; NaN during warmup
        direction = pd.Series(np.where(fast_ema > slow_ema, 1.0, -1.0), index=prices.index)
        direction[fast_ema.isna() | slow_ema.isna()] = np.nan

        # Annualised vol for position sizing
        pct_ret = prices.pct_change()
        ann_vol = pct_ret.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std() * np.sqrt(256)
        ann_vol = ann_vol.clip(lower=VOL_FLOOR)

        signals[symbol] = {'direction': direction, 'ann_vol': ann_vol}
    return signals


# ── Backtest engine ────────────────────────────────────────────────────────────

def run_backtest(instrument_data, signals):
    """
    Daily mark-to-market P&L.  Positions only change on EMA crossovers.

    Per-strategy state:
      capital      — cash balance (mark-to-market daily)
      position     — contracts held (positive = long, negative = short)
      prev_price   — previous trading day's close (for daily P&L)
      prev_dir     — previous signal direction (+1 / -1 / nan)
      total_gross  — cumulative gross P&L (before commission)
      total_comm   — cumulative commission paid
    """
    all_dates = sorted(set().union(*[set(df.index) for df in instrument_data.values()]))

    state = {
        sym: {
            'capital':     INITIAL_CAPITAL * ALLOC,
            'position':    0,
            'prev_price':  None,
            'prev_dir':    np.nan,
            'total_gross': 0.0,
            'total_comm':  0.0,
            'equity_curve': [],
        }
        for sym in CONTRACT_SPECS
    }

    for date in all_dates:
        for sym in CONTRACT_SPECS:
            s   = state[sym]
            mul = CONTRACT_SPECS[sym]['multiplier']
            df  = instrument_data.get(sym)
            sig = signals.get(sym)

            if df is None or sig is None or date not in df.index:
                last_eq = s['equity_curve'][-1][1] if s['equity_curve'] else s['capital']
                s['equity_curve'].append((date, last_eq))
                continue

            price   = float(df.loc[date, 'Last'])
            new_dir = sig['direction'].loc[date] if date in sig['direction'].index else np.nan
            ann_vol = sig['ann_vol'].loc[date]   if date in sig['ann_vol'].index   else np.nan

            # ── Step 1: daily mark-to-market P&L ──────────────────────────────
            if s['prev_price'] is not None and s['position'] != 0:
                daily_pnl = (price - s['prev_price']) * mul * s['position']
                s['capital']     += daily_pnl
                s['total_gross'] += daily_pnl

            # ── Step 2: trade only on signal crossover ─────────────────────────
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

            # initialise prev_dir on the first valid signal day (no trade)
            elif not pd.isna(new_dir) and pd.isna(s['prev_dir']):
                s['prev_dir'] = new_dir

            s['prev_price'] = price
            s['equity_curve'].append((date, s['capital']))

    # Close all open positions at end of period
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


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(equity_df):
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
        logger.error("No data loaded — check Data/ directory.")
        return

    logger.info("Pre-computing EWMAC signals...")
    signals = precompute_signals(instrument_data)

    logger.info("Running backtest...")
    state = run_backtest(instrument_data, signals)

    # Build combined equity curve
    common_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    series = []
    for sym in CONTRACT_SPECS:
        curve = state[sym]['equity_curve']
        df    = pd.DataFrame(curve, columns=['Time', 'Equity']).set_index('Time')
        df    = df[~df.index.duplicated(keep='last')].sort_index()
        df    = df.reindex(common_dates, method='ffill')
        series.append(df['Equity'])
    combined_df = pd.DataFrame({'Equity': sum(series)}, index=common_dates)
    combined_df.dropna(inplace=True)

    metrics = compute_metrics(combined_df)

    print("\n" + "=" * 60)
    print(f"EWMAC({FAST_SPAN},{SLOW_SPAN}) TREND  |  3 instruments  |  ${INITIAL_CAPITAL:,.0f} capital")
    print(f"Risk target {RISK_TARGET:.0%}  |  Equal weight  |  Long + Short")
    print("=" * 60)

    for k, v in metrics.items():
        if k == 'Final Balance':
            print(f"  {k:<26} ${v:>12,.2f}")
        elif not np.isnan(v):
            print(f"  {k:<26} {v:>12.2f}")
        else:
            print(f"  {k:<26} {'NaN':>12}")

    print(f"\n  {'Strategy':<14} {'Final $':>10}  {'Return %':>9}  {'Gross P&L':>11}  {'Commission':>11}")
    print(f"  {'-'*60}")
    for sym in CONTRACT_SPECS:
        s   = state[sym]
        fc  = s['capital']
        ret = (fc / (INITIAL_CAPITAL * ALLOC) - 1) * 100
        print(f"  {'EWMAC_'+sym:<14} ${fc:>9,.0f}  {ret:>8.1f}%"
              f"  ${s['total_gross']:>10,.0f}  ${s['total_comm']:>10,.0f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(combined_df.index, combined_df['Equity'],
             color='darkorange', linewidth=1.5, label='EWMAC Trend Portfolio')
    ax1.axhline(INITIAL_CAPITAL, color='gray', linestyle='--',
                linewidth=0.8, label='Starting capital')
    ax1.set_ylabel('Portfolio Equity ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.set_title(
        f'EWMAC({FAST_SPAN},{SLOW_SPAN}) Trend Portfolio  |  3 instruments  |  ${INITIAL_CAPITAL:,.0f} capital\n'
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
    plt.savefig('portfolio/ewmac_equity_curve.png', dpi=150)
    plt.show()
    logger.info("Chart saved to portfolio/ewmac_equity_curve.png")


if __name__ == '__main__':
    main()
