#!/usr/bin/env python3
"""
IBS Portfolio — In-Sample + Out-of-Sample Backtest
Part 1: local CSV data (2000-01-01 to 2025-03-12)    — in-sample
Part 2: IBKR live data (2025-03-13 to today)          — out-of-sample

Backtest engine is identical to aggregate_port.py (date-based iteration,
same position sizing, same commission). Results split at the transition date
so the two periods can be compared directly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import calendar
from io import StringIO
from ib_insync import IB, ContFuture, util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Parameters (must match aggregate_port.py) ─────────────────────────────────
INITIAL_CAPITAL = 50_000.0
COMMISSION_RT   = 4.0
IBS_ENTRY       = 0.10
IBS_EXIT        = 0.90

LOCAL_START = '2000-01-01'
LOCAL_END   = '2025-03-12'   # last date of local CSV data (inclusive)
OOS_START   = '2025-03-13'   # first date fetched from IBKR
OOS_END     = datetime.now().strftime('%Y-%m-%d')

IB_HOST   = '127.0.0.1'
IB_PORT   = 4001
CLIENT_ID = 3

# ── Contract specifications ────────────────────────────────────────────────────
CONTRACT_SPECS = {
    'ES': {'multiplier': 5,  'file': 'Data/mes_daily_data.csv', 'ibkr_symbol': 'MES', 'exchange': 'CME'},
    'NQ': {'multiplier': 2,  'file': 'Data/mnq_daily_data.csv', 'ibkr_symbol': 'MNQ', 'exchange': 'CME'},
    'GC': {'multiplier': 10, 'file': 'Data/mgc_daily_data.csv', 'ibkr_symbol': 'MGC', 'exchange': 'COMEX'},
}

STRATEGIES = [('IBS_ES', 'ES'), ('IBS_NQ', 'NQ'), ('IBS_GC', 'GC')]
N_STRATS   = len(STRATEGIES)
ALLOC      = 1.0 / N_STRATS


# ── Data loading ───────────────────────────────────────────────────────────────

def load_local_data():
    data = {}
    for symbol, spec in CONTRACT_SPECS.items():
        lines = []
        with open(spec['file'], 'r') as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith('"'):
                    continue
                lines.append(s)
        if len(lines) < 2:
            logger.warning(f"No data in {spec['file']}")
            continue
        df = pd.read_csv(StringIO('\n'.join(lines)), parse_dates=['Time'])
        df.sort_values('Time', inplace=True)
        df = df[(df['Time'] >= LOCAL_START) & (df['Time'] <= LOCAL_END)]
        df = df.dropna(subset=['High', 'Low', 'Last']).reset_index(drop=True)
        if not df.empty:
            data[symbol] = df.set_index('Time')
            logger.info(f"Loaded {len(data[symbol])} local bars for {symbol}")
    return data


def fetch_ibkr_data(ib):
    """Fetch OOS data via IBKR continuous futures (ADJUSTED_LAST)."""
    oos_data = {}
    end_dt_str = datetime.strptime(OOS_END, '%Y-%m-%d').strftime('%Y%m%d 23:59:59')
    start_dt      = datetime.strptime(OOS_START, '%Y-%m-%d')
    duration_days = (datetime.strptime(OOS_END, '%Y-%m-%d') - start_dt).days + 5
    # IBKR requires year format for requests > 365 days
    if duration_days > 365:
        duration_years = -(-duration_days // 365)  # ceiling division
        duration_str = f"{duration_years} Y"
    else:
        duration_str = f"{duration_days} D"

    for symbol, spec in CONTRACT_SPECS.items():
        try:
            contract = ContFuture(
                symbol=spec['ibkr_symbol'],
                exchange=spec['exchange'],
                currency='USD',
            )
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                logger.warning(f"Could not qualify continuous contract for {symbol}")
                continue
            contract = qualified[0]

            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',          # empty = up to now (required for ContFuture)
                durationStr=duration_str,
                barSizeSetting='1 day',
                whatToShow='ADJUSTED_LAST',
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
            )
            if not bars:
                logger.warning(f"No IBKR bars returned for {symbol}")
                continue

            df = util.df(bars).rename(columns={
                'date': 'Time', 'open': 'Open',
                'high': 'High', 'low': 'Low', 'close': 'Last',
            })
            df['Time'] = pd.to_datetime(df['Time'])
            df = df[(df['Time'] >= OOS_START) & (df['Time'] <= OOS_END)]
            df = df.dropna(subset=['High', 'Low', 'Last']).reset_index(drop=True)
            if not df.empty:
                oos_data[symbol] = df.set_index('Time')
                logger.info(f"Fetched {len(oos_data[symbol])} IBKR bars for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching IBKR data for {symbol}: {e}")

    return oos_data


# ── Indicator & sizing ─────────────────────────────────────────────────────────

def precompute_ibs(instrument_data):
    indicators = {}
    for symbol, df in instrument_data.items():
        ind = df[['High', 'Low', 'Last']].copy()
        bar_range = ind['High'] - ind['Low']
        ind['IBS'] = np.where(bar_range > 0,
                              (ind['Last'] - ind['Low']) / bar_range,
                              0.5)
        indicators[symbol] = ind
    return indicators


def position_size(total_equity, price, multiplier):
    contract_value = price * multiplier
    if contract_value <= 0:
        return 1
    return max(1, round(total_equity * ALLOC / contract_value))


# ── Backtest engine ────────────────────────────────────────────────────────────

def run_backtest(instrument_data, indicators):
    """Date-aligned IBS backtest — identical logic to aggregate_port.py."""
    all_dates = sorted(set().union(*[set(df.index) for df in instrument_data.values()]))

    state = {
        name: {
            'capital':      INITIAL_CAPITAL * ALLOC,
            'in_position':  False,
            'position':     None,
            'equity_curve': [],
        }
        for name, _ in STRATEGIES
    }
    total_equity = INITIAL_CAPITAL

    for date in all_dates:
        daily_eq = 0.0
        for name, symbol in STRATEGIES:
            s   = state[name]
            mul = CONTRACT_SPECS[symbol]['multiplier']
            ind = indicators.get(symbol)

            if ind is None or date not in ind.index:
                last = s['equity_curve'][-1][1] if s['equity_curve'] else s['capital']
                s['equity_curve'].append((date, last))
                daily_eq += last
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
                    s['position']    = {'entry_price': price, 'entry_date': date,
                                        'contracts': contracts}

            if s['in_position']:
                pos    = s['position']
                equity = s['capital'] + (price - pos['entry_price']) * mul * pos['contracts']
            else:
                equity = s['capital']

            s['equity_curve'].append((date, equity))
            daily_eq += equity

        total_equity = daily_eq

    # Force-close open positions
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


def build_equity_curve(state, start, end):
    dates = pd.date_range(start=start, end=end, freq='D')
    series = []
    for name, _ in STRATEGIES:
        curve = state[name]['equity_curve']
        df    = pd.DataFrame(curve, columns=['Time', 'Equity']).set_index('Time')
        df    = df[~df.index.duplicated(keep='last')].sort_index()
        df    = df.reindex(dates, method='ffill')
        series.append(df['Equity'])
    combined = pd.DataFrame({'Equity': sum(series)}, index=dates)
    return combined.dropna()


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(equity_df, start_capital):
    eq   = equity_df['Equity']
    rets = eq.pct_change().dropna()
    if rets.empty:
        return None

    final  = eq.iloc[-1]
    years  = (eq.index[-1] - eq.index[0]).days / 365.25
    ann_r  = ((final / start_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan
    std    = rets.std()
    sharpe = rets.mean() / std * np.sqrt(252) if std > 0 else np.nan
    d_std  = rets[rets < 0].std()
    sortino = rets.mean() / d_std * np.sqrt(252) if d_std > 0 else np.nan
    peak   = eq.cummax()
    dd     = (eq - peak) / peak
    max_dd = dd.min() * 100
    calmar = ann_r / abs(max_dd) if max_dd != 0 else np.nan

    return {
        'Final Balance':     final,
        'Total Return %':    (final / start_capital - 1) * 100,
        'Ann. Return %':     ann_r,
        'Ann. Volatility %': std * np.sqrt(252) * 100,
        'Sharpe':            sharpe,
        'Sortino':           sortino,
        'Calmar':            calmar,
        'Max Drawdown %':    max_dd,
    }


def print_metrics(metrics, label):
    print(f"\n  {label}")
    print(f"  {'-'*40}")
    for k, v in metrics.items():
        if k == 'Final Balance':
            print(f"  {k:<26} ${v:>12,.2f}")
        elif not np.isnan(v):
            print(f"  {k:<26} {v:>12.2f}")
        else:
            print(f"  {k:<26} {'NaN':>12}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Part 1: in-sample backtest on local CSV data ───────────────────────────
    logger.info("Loading local CSV data...")
    local_data = load_local_data()
    if not local_data:
        logger.error("No local data loaded.")
        return

    logger.info("Running in-sample backtest...")
    is_indicators = precompute_ibs(local_data)
    is_state      = run_backtest(local_data, is_indicators)
    is_equity     = build_equity_curve(is_state, LOCAL_START, LOCAL_END)

    is_end_capital = sum(is_state[name]['capital'] for name, _ in STRATEGIES)
    logger.info(f"In-sample final capital: ${is_end_capital:,.2f}")

    # ── Part 2: out-of-sample data from IBKR ──────────────────────────────────
    logger.info("Connecting to IBKR for out-of-sample data...")
    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
        logger.info("Connected to IBKR.")
    except Exception as e:
        logger.error(f"IBKR connection failed: {e}")
        logger.info("Plotting in-sample results only.")
        _plot_and_print(is_equity, None, is_end_capital)
        return

    oos_data = fetch_ibkr_data(ib)
    ib.disconnect()
    logger.info("Disconnected from IBKR.")

    if not oos_data:
        logger.warning("No OOS data fetched. Plotting in-sample only.")
        _plot_and_print(is_equity, None, is_end_capital)
        return

    # ── Continue backtest on OOS data, starting from in-sample end state ──────
    logger.info("Running out-of-sample backtest continuation...")

    # Carry forward capital and open positions from in-sample state
    oos_state = {}
    for name, symbol in STRATEGIES:
        s = is_state[name]
        oos_state[name] = {
            'capital':      s['capital'],
            'in_position':  s['in_position'],
            'position':     s['position'].copy() if s['position'] else None,
            'equity_curve': [],
        }

    oos_indicators = precompute_ibs(oos_data)

    all_dates    = sorted(set().union(*[set(df.index) for df in oos_data.values()]))
    total_equity = is_end_capital

    for date in all_dates:
        daily_eq = 0.0
        for name, symbol in STRATEGIES:
            s   = oos_state[name]
            mul = CONTRACT_SPECS[symbol]['multiplier']
            ind = oos_indicators.get(symbol)

            if ind is None or date not in ind.index:
                last = s['equity_curve'][-1][1] if s['equity_curve'] else s['capital']
                s['equity_curve'].append((date, last))
                daily_eq += last
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
                    s['position']    = {'entry_price': price, 'entry_date': date,
                                        'contracts': contracts}

            if s['in_position']:
                pos    = s['position']
                equity = s['capital'] + (price - pos['entry_price']) * mul * pos['contracts']
            else:
                equity = s['capital']

            s['equity_curve'].append((date, equity))
            daily_eq += equity

        total_equity = daily_eq

    # Force-close OOS open positions
    for name, symbol in STRATEGIES:
        s  = oos_state[name]
        df = oos_data.get(symbol)
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

    oos_equity = build_equity_curve(oos_state, OOS_START, OOS_END)

    _plot_and_print(is_equity, oos_equity, is_end_capital)


def _plot_and_print(is_equity, oos_equity, is_end_capital):
    is_metrics  = compute_metrics(is_equity, INITIAL_CAPITAL)
    oos_metrics = compute_metrics(oos_equity, is_end_capital) if oos_equity is not None else None

    print("\n" + "=" * 60)
    print(f"IBS PORTFOLIO  |  {N_STRATS} instruments  |  ${INITIAL_CAPITAL:,.0f} capital")
    print(f"IBS entry < {IBS_ENTRY}  |  IBS exit > {IBS_EXIT}  |  Equal weight")
    print("=" * 60)

    if is_metrics:
        print_metrics(is_metrics, f"IN-SAMPLE  ({LOCAL_START} to {LOCAL_END})")

    if oos_metrics:
        print_metrics(oos_metrics, f"OUT-OF-SAMPLE  ({OOS_START} to {OOS_END})")

        # Combined
        combined = pd.concat([is_equity, oos_equity])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        comb_metrics = compute_metrics(pd.DataFrame({'Equity': combined['Equity']}), INITIAL_CAPITAL)
        if comb_metrics:
            print_metrics(comb_metrics, "COMBINED")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(is_equity.index, is_equity['Equity'],
             color='steelblue', linewidth=1.5, label='In-sample (local CSV)')

    if oos_equity is not None and not oos_equity.empty:
        # Bridge the gap between in-sample end and OOS start
        bridge = pd.concat([is_equity.iloc[[-1]], oos_equity.iloc[[0]]])
        ax1.plot(bridge.index, bridge['Equity'], color='crimson', linewidth=1.5)
        ax1.plot(oos_equity.index, oos_equity['Equity'],
                 color='crimson', linewidth=1.5, label=f'Out-of-sample (IBKR {OOS_START}→)')
        ax1.axvline(pd.Timestamp(OOS_START), color='gray', linestyle='--',
                    linewidth=0.8, label='OOS start')

    ax1.axhline(INITIAL_CAPITAL, color='gray', linestyle=':', linewidth=0.8)
    ax1.set_ylabel('Portfolio Equity ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    sharpe_str = f"{is_metrics['Sharpe']:.2f}" if is_metrics else 'N/A'
    ax1.set_title(
        f'IBS Portfolio  |  {N_STRATS} instruments  |  In-sample Sharpe {sharpe_str}',
        fontsize=11
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown — combined or just in-sample
    if oos_equity is not None and not oos_equity.empty:
        full_eq = pd.concat([is_equity, oos_equity])
        full_eq = full_eq[~full_eq.index.duplicated(keep='last')].sort_index()
    else:
        full_eq = is_equity

    peak = full_eq['Equity'].cummax()
    dd   = (full_eq['Equity'] - peak) / peak * 100
    ax2.fill_between(full_eq.index, dd, 0, color='crimson', alpha=0.4)
    if oos_equity is not None:
        ax2.axvline(pd.Timestamp(OOS_START), color='gray', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('portfolio/aggregate_port_ib_equity_curve.png', dpi=150)
    plt.show()
    logger.info("Chart saved to portfolio/aggregate_port_ib_equity_curve.png")


if __name__ == '__main__':
    main()
