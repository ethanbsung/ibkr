#!/usr/bin/env python3
"""
Combined IBS + EWMAC Portfolio — In-Sample + Out-of-Sample Backtest
75% IBS mean-reversion  |  25% EWMAC(64,256) trend-following
3 instruments: MES / MNQ / MGC  |  $50,000 starting capital

Part 1: local CSV data (LOCAL_START → LOCAL_END)   — in-sample
Part 2: IBKR live data (OOS_START → today)         — out-of-sample

The OOS section continues directly from the IS end state:
  IBS  — capital + any open position carried forward
  EWMAC — capital + position + last price carried forward
           EMA direction re-derived from 2Y of IBKR history for crossover detection
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

# ── Parameters ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 50_000.0
COMMISSION_RT   = 4.0

IBS_ALLOC   = 0.75
EWMAC_ALLOC = 0.25
N_INSTR     = 3
IBS_PER_INSTR   = IBS_ALLOC  / N_INSTR   # 0.2500
EWMAC_PER_INSTR = EWMAC_ALLOC / N_INSTR  # 0.0833

IBS_ENTRY = 0.10
IBS_EXIT  = 0.90

RISK_TARGET = 0.20
DIV_YIELD   = 0.018   # approx S&P 500 annual dividend yield for total-return benchmark
FAST_SPAN   = 64
SLOW_SPAN   = 256
VOL_SPAN    = 32
VOL_FLOOR   = 0.05

LOCAL_START = '2000-01-01'
LOCAL_END   = '2025-03-12'
OOS_START   = '2025-03-13'
OOS_END     = datetime.now().strftime('%Y-%m-%d')

IB_HOST   = '127.0.0.1'
IB_PORT   = 7497
CLIENT_ID = 3

# ── Contract specifications ────────────────────────────────────────────────────
CONTRACT_SPECS = {
    'ES': {'multiplier': 5,  'file': 'Data/mes_daily_data.csv', 'ibkr_symbol': 'MES', 'exchange': 'CME'},
    'NQ': {'multiplier': 2,  'file': 'Data/mnq_daily_data.csv', 'ibkr_symbol': 'MNQ', 'exchange': 'CME'},
    'GC': {'multiplier': 10, 'file': 'Data/mgc_daily_data.csv', 'ibkr_symbol': 'MGC', 'exchange': 'COMEX'},
}


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
    """
    Fetch OOS data from IBKR.  Requests 2 years of ADJUSTED_LAST so that
    EWMAC EMA(256) has enough warmup before OOS_START.  Only OOS_START→today
    is used for trading; the earlier portion is warmup only.
    """
    oos_data = {}
    for symbol, spec in CONTRACT_SPECS.items():
        try:
            contract  = ContFuture(symbol=spec['ibkr_symbol'],
                                   exchange=spec['exchange'], currency='USD')
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                logger.warning(f"Could not qualify continuous contract for {symbol}")
                continue
            contract = qualified[0]

            bars = ib.reqHistoricalData(
                contract, endDateTime='', durationStr='2 Y',
                barSizeSetting='1 day', whatToShow='ADJUSTED_LAST',
                useRTH=True, formatDate=1, keepUpToDate=False,
            )
            if not bars:
                logger.warning(f"No IBKR bars for {symbol}")
                continue

            df = util.df(bars).rename(columns={
                'date': 'Time', 'open': 'Open',
                'high': 'High', 'low': 'Low', 'close': 'Last',
            })
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.dropna(subset=['High', 'Low', 'Last']).reset_index(drop=True)
            if not df.empty:
                oos_data[symbol] = df.set_index('Time')
                logger.info(f"Fetched {len(oos_data[symbol])} IBKR bars for {symbol} "
                            f"({df['Time'].min().date()} → {df['Time'].max().date()})")
        except Exception as e:
            logger.error(f"Error fetching IBKR data for {symbol}: {e}")
    return oos_data


# ── Indicator pre-computation ──────────────────────────────────────────────────

def precompute_indicators(data):
    """Compute IBS, EWMAC direction, and ann_vol for every bar in data."""
    indicators = {}
    for symbol, df in data.items():
        ind = df[['High', 'Low', 'Last']].copy()

        # IBS
        rng      = ind['High'] - ind['Low']
        ind['IBS'] = np.where(rng > 0, (ind['Last'] - ind['Low']) / rng, 0.5)

        # EWMAC direction (+1 / -1 / NaN during warmup)
        prices   = df['Last']
        fast_ema = prices.ewm(span=FAST_SPAN, min_periods=FAST_SPAN).mean()
        slow_ema = prices.ewm(span=SLOW_SPAN, min_periods=SLOW_SPAN).mean()
        direction = pd.Series(np.where(fast_ema > slow_ema, 1.0, -1.0), index=prices.index)
        direction[fast_ema.isna() | slow_ema.isna()] = np.nan
        ind['EWMAC_dir'] = direction

        # Annualised vol for EWMAC sizing
        vol          = prices.pct_change().ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std()
        ind['ann_vol'] = (vol * np.sqrt(256)).clip(lower=VOL_FLOOR)

        indicators[symbol] = ind
    return indicators


# ── IBS backtest engine ────────────────────────────────────────────────────────

def run_ibs(instrument_data, indicators, init_state=None, init_total_equity=None):
    """
    IBS backtest.  If init_state is provided, continue from that state
    (used for the OOS continuation).
    """
    all_dates    = sorted(set().union(*[set(df.index) for df in instrument_data.values()]))
    start_cap    = INITIAL_CAPITAL * IBS_PER_INSTR

    if init_state is None:
        state = {
            sym: {
                'capital':      start_cap,
                'in_position':  False,
                'position':     None,
                'equity_curve': [],
            }
            for sym in CONTRACT_SPECS
        }
        total_equity = INITIAL_CAPITAL * IBS_ALLOC
    else:
        state        = init_state
        total_equity = init_total_equity

    for date in all_dates:
        daily_sum = 0.0
        for sym in CONTRACT_SPECS:
            s   = state[sym]
            mul = CONTRACT_SPECS[sym]['multiplier']
            ind = indicators.get(sym)

            if ind is None or date not in ind.index:
                last = s['equity_curve'][-1][1] if s['equity_curve'] else s['capital']
                s['equity_curve'].append((date, last))
                daily_sum += last
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
                    contracts = max(1, round(total_equity / N_INSTR / (price * mul)))
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
            daily_sum += equity

        total_equity = daily_sum

    # Force-close open positions at end of period
    for sym in CONTRACT_SPECS:
        s  = state[sym]
        df = instrument_data.get(sym)
        if s['in_position'] and df is not None and not df.empty:
            last_price = float(df.iloc[-1]['Last'])
            mul        = CONTRACT_SPECS[sym]['multiplier']
            pos        = s['position']
            pnl        = ((last_price - pos['entry_price']) * mul * pos['contracts']
                          - (COMMISSION_RT / 2) * pos['contracts'])
            s['capital'] += pnl
            if s['equity_curve']:
                s['equity_curve'][-1] = (s['equity_curve'][-1][0], s['capital'])
            s['in_position'] = False

    return state, total_equity


# ── EWMAC backtest engine ──────────────────────────────────────────────────────

def run_ewmac(instrument_data, indicators, init_state=None):
    """
    EWMAC backtest.  If init_state is provided, continue from that state
    (used for the OOS continuation).
    """
    all_dates = sorted(set().union(*[set(df.index) for df in instrument_data.values()]))
    start_cap = INITIAL_CAPITAL * EWMAC_PER_INSTR

    if init_state is None:
        state = {
            sym: {
                'capital':      start_cap,
                'position':     0,
                'prev_price':   None,
                'prev_dir':     np.nan,
                'total_gross':  0.0,
                'total_comm':   0.0,
                'equity_curve': [],
            }
            for sym in CONTRACT_SPECS
        }
    else:
        state = init_state

    for date in all_dates:
        for sym in CONTRACT_SPECS:
            s   = state[sym]
            mul = CONTRACT_SPECS[sym]['multiplier']
            df  = instrument_data.get(sym)
            ind = indicators.get(sym)

            if df is None or ind is None or date not in df.index:
                last = s['equity_curve'][-1][1] if s['equity_curve'] else s['capital']
                s['equity_curve'].append((date, last))
                continue

            price   = float(df.loc[date, 'Last'])
            row     = ind.loc[date]
            new_dir = float(row['EWMAC_dir']) if not pd.isna(row['EWMAC_dir']) else np.nan
            ann_vol = float(row['ann_vol'])

            # Daily mark-to-market P&L
            if s['prev_price'] is not None and s['position'] != 0:
                daily_pnl        = (price - s['prev_price']) * mul * s['position']
                s['capital']    += daily_pnl
                s['total_gross'] += daily_pnl

            # Trade only on EMA crossover
            signal_changed = (not pd.isna(new_dir)) and (new_dir != s['prev_dir'])

            if signal_changed and not (pd.isna(ann_vol) or ann_vol <= 0):
                avg_pos = (s['capital'] * RISK_TARGET) / (price * mul * ann_vol)
                new_pos = max(1, int(round(avg_pos))) * int(new_dir)
                trade   = abs(new_pos - s['position'])
                if trade > 0:
                    comm             = (COMMISSION_RT / 2) * trade
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
            comm             = (COMMISSION_RT / 2) * abs(s['position'])
            s['capital']    -= comm
            s['total_comm'] += comm
            if s['equity_curve']:
                s['equity_curve'][-1] = (s['equity_curve'][-1][0], s['capital'])

    return state


# ── Equity curve assembly ──────────────────────────────────────────────────────

def build_curve(state_dict, start, end):
    dates  = pd.date_range(start=start, end=end, freq='D')
    series = []
    for sym in CONTRACT_SPECS:
        curve = state_dict[sym]['equity_curve']
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
    if rets.empty or len(rets) < 5:
        return None

    final   = eq.iloc[-1]
    years   = (eq.index[-1] - eq.index[0]).days / 365.25
    ann_ret = ((final / start_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan
    std     = rets.std()
    sharpe  = rets.mean() / std * np.sqrt(252) if std > 0 else np.nan
    d_std   = rets[rets < 0].std()
    sortino = rets.mean() / d_std * np.sqrt(252) if d_std > 0 else np.nan
    peak    = eq.cummax()
    dd      = (eq - peak) / peak
    max_dd  = dd.min() * 100
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        'Final Balance':     final,
        'Total Return %':    (final / start_capital - 1) * 100,
        'Ann. Return %':     ann_ret,
        'Ann. Volatility %': std * np.sqrt(252) * 100,
        'Sharpe':            sharpe,
        'Sortino':           sortino,
        'Calmar':            calmar,
        'Max Drawdown %':    max_dd,
    }


def print_metrics(metrics, label):
    print(f"\n  {label}")
    print(f"  {'─'*44}")
    for k, v in metrics.items():
        if k == 'Final Balance':
            print(f"  {k:<26} ${v:>12,.2f}")
        elif isinstance(v, float) and not np.isnan(v):
            print(f"  {k:<26} {v:>12.2f}")
        else:
            print(f"  {k:<26} {'NaN':>12}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Part 1: in-sample on local CSV ────────────────────────────────────────
    logger.info("Loading local CSV data...")
    local_data = load_local_data()
    if not local_data:
        logger.error("No local data loaded.")
        return

    logger.info("Pre-computing in-sample indicators...")
    is_indicators = precompute_indicators(local_data)

    logger.info("Running in-sample IBS backtest...")
    is_ibs_state, is_ibs_total_eq = run_ibs(local_data, is_indicators)

    logger.info("Running in-sample EWMAC backtest...")
    is_ewmac_state = run_ewmac(local_data, is_indicators)

    is_ibs_eq   = build_curve(is_ibs_state,   LOCAL_START, LOCAL_END)
    is_ewmac_eq = build_curve(is_ewmac_state, LOCAL_START, LOCAL_END)

    # Align and combine IS equity curves
    valid_is     = is_ibs_eq.index.intersection(is_ewmac_eq.index)
    is_combined  = pd.DataFrame(
        {'Equity': is_ibs_eq.loc[valid_is, 'Equity'] + is_ewmac_eq.loc[valid_is, 'Equity']},
        index=valid_is
    )

    is_end_ibs_capital   = sum(is_ibs_state[sym]['capital']   for sym in CONTRACT_SPECS)
    is_end_ewmac_capital = sum(is_ewmac_state[sym]['capital'] for sym in CONTRACT_SPECS)
    is_end_total         = is_end_ibs_capital + is_end_ewmac_capital
    logger.info(f"IS end — IBS: ${is_end_ibs_capital:,.0f}  EWMAC: ${is_end_ewmac_capital:,.0f}"
                f"  Total: ${is_end_total:,.0f}")

    # IS benchmark (computed here so it's available for early-return fallback paths)
    is_bnh_eq = None
    es_local  = local_data.get('ES')
    if es_local is not None:
        is_bnh_p = es_local['Last'].reindex(is_combined.index, method='ffill').dropna()
        if not is_bnh_p.empty:
            years = (is_bnh_p.index - is_bnh_p.index[0]).days / 365.25
            is_bnh_eq = pd.DataFrame(
                {'Equity': INITIAL_CAPITAL
                           * (is_bnh_p / float(is_bnh_p.iloc[0]))
                           * (1 + DIV_YIELD) ** years},
                index=is_bnh_p.index,
            )

    # ── Part 2: OOS data from IBKR ────────────────────────────────────────────
    logger.info("Connecting to IBKR for out-of-sample data...")
    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
        logger.info("Connected to IBKR.")
    except Exception as e:
        logger.error(f"IBKR connection failed: {e}")
        logger.info("Plotting in-sample results only.")
        _plot_and_print(is_combined, None, INITIAL_CAPITAL, is_end_total, is_bnh_eq)
        return

    oos_raw = fetch_ibkr_data(ib)
    ib.disconnect()
    logger.info("Disconnected from IBKR.")

    if not oos_raw:
        logger.warning("No OOS data fetched.")
        _plot_and_print(is_combined, None, INITIAL_CAPITAL, is_end_total, is_bnh_eq)
        return

    # ── Precompute OOS indicators (full 2Y fetch, for EMA warmup) ─────────────
    logger.info("Pre-computing OOS indicators...")
    oos_indicators_full = precompute_indicators(oos_raw)

    # Filter OOS data to trading period only (OOS_START → today)
    oos_data = {}
    for sym, df in oos_raw.items():
        filtered = df[df.index >= OOS_START]
        if not filtered.empty:
            oos_data[sym] = filtered

    if not oos_data:
        logger.warning("No OOS bars after filtering to OOS_START.")
        _plot_and_print(is_combined, None, INITIAL_CAPITAL, is_end_total, is_bnh_eq)
        return

    # Filter indicators to match filtered oos_data
    oos_indicators = {}
    for sym, ind in oos_indicators_full.items():
        filtered = ind[ind.index >= OOS_START]
        if not filtered.empty:
            oos_indicators[sym] = filtered

    # ── Build OOS IBS init state ───────────────────────────────────────────────
    oos_ibs_init = {}
    for sym in CONTRACT_SPECS:
        s = is_ibs_state[sym]
        oos_ibs_init[sym] = {
            'capital':      s['capital'],
            'in_position':  s['in_position'],
            'position':     s['position'].copy() if s['position'] else None,
            'equity_curve': [],
        }

    # ── Build OOS EWMAC init state ─────────────────────────────────────────────
    # EMA direction at OOS_START is taken from IBKR-computed indicators so it's
    # consistent with the live data going forward.  Position (contracts) and
    # capital carry over from IS end.
    oos_ewmac_init = {}
    for sym in CONTRACT_SPECS:
        s = is_ewmac_state[sym]

        # Derive starting EMA direction from IBKR data at OOS_START
        ind_full = oos_indicators_full.get(sym)
        if ind_full is not None:
            oos_start_ts = pd.Timestamp(OOS_START)
            # Find the last bar before or at OOS_START in the full indicator set
            pre_oos = ind_full[ind_full.index <= oos_start_ts]
            if not pre_oos.empty and not pd.isna(pre_oos.iloc[-1]['EWMAC_dir']):
                oos_start_dir = float(pre_oos.iloc[-1]['EWMAC_dir'])
            else:
                oos_start_dir = s['prev_dir']   # fall back to IS end direction
        else:
            oos_start_dir = s['prev_dir']

        # Last IS price for mark-to-market continuity
        local_df = local_data.get(sym)
        last_is_price = float(local_df.iloc[-1]['Last']) if local_df is not None else None

        oos_ewmac_init[sym] = {
            'capital':      s['capital'],
            'position':     s['position'],
            'prev_price':   last_is_price,
            'prev_dir':     oos_start_dir,
            'total_gross':  0.0,
            'total_comm':   0.0,
            'equity_curve': [],
        }

    # ── Run OOS backtests ──────────────────────────────────────────────────────
    logger.info("Running out-of-sample IBS backtest...")
    oos_ibs_state, _ = run_ibs(oos_data, oos_indicators,
                                init_state=oos_ibs_init,
                                init_total_equity=is_end_ibs_capital)

    logger.info("Running out-of-sample EWMAC backtest...")
    oos_ewmac_state = run_ewmac(oos_data, oos_indicators, init_state=oos_ewmac_init)

    oos_ibs_eq   = build_curve(oos_ibs_state,   OOS_START, OOS_END)
    oos_ewmac_eq = build_curve(oos_ewmac_state, OOS_START, OOS_END)

    valid_oos   = oos_ibs_eq.index.intersection(oos_ewmac_eq.index)
    oos_combined = pd.DataFrame(
        {'Equity': oos_ibs_eq.loc[valid_oos, 'Equity'] + oos_ewmac_eq.loc[valid_oos, 'Equity']},
        index=valid_oos
    )

    # ── OOS benchmark (IS benchmark already computed above) ───────────────────
    oos_bnh_eq = None
    es_oos = oos_data.get('ES')
    if es_oos is not None and is_bnh_eq is not None:
        bnh_is_end = float(is_bnh_eq.iloc[-1]['Equity'])
        oos_bnh_p  = es_oos['Last'].reindex(oos_combined.index, method='ffill').dropna()
        if not oos_bnh_p.empty:
            years = (oos_bnh_p.index - oos_bnh_p.index[0]).days / 365.25
            oos_bnh_eq = pd.DataFrame(
                {'Equity': bnh_is_end
                           * (oos_bnh_p / float(oos_bnh_p.iloc[0]))
                           * (1 + DIV_YIELD) ** years},
                index=oos_bnh_p.index,
            )

    _plot_and_print(is_combined, oos_combined, INITIAL_CAPITAL, is_end_total,
                    is_bnh_eq, oos_bnh_eq)


def _plot_and_print(is_equity, oos_equity, is_start_capital, is_end_capital,
                    is_bnh_eq=None, oos_bnh_eq=None):
    is_metrics  = compute_metrics(is_equity, is_start_capital)
    oos_metrics = compute_metrics(oos_equity, is_end_capital) if oos_equity is not None else None

    print("\n" + "=" * 64)
    print(f"COMBINED IBS+EWMAC  |  3 instruments  |  ${INITIAL_CAPITAL:,.0f} capital")
    print(f"75% IBS  |  25% EWMAC({FAST_SPAN},{SLOW_SPAN})  |  Long+Short")
    print("=" * 64)

    if is_metrics:
        print_metrics(is_metrics, f"IN-SAMPLE  ({LOCAL_START} → {LOCAL_END})")
    if is_bnh_eq is not None:
        bnh_is_m = compute_metrics(is_bnh_eq, INITIAL_CAPITAL)
        if bnh_is_m:
            print_metrics(bnh_is_m, f"  VOO equiv. benchmark  ({LOCAL_START} → {LOCAL_END}, price + ~1.8% div)")

    if oos_metrics:
        print_metrics(oos_metrics, f"OUT-OF-SAMPLE  ({OOS_START} → {OOS_END})")
        if oos_bnh_eq is not None:
            bnh_oos_m = compute_metrics(oos_bnh_eq, float(is_bnh_eq.iloc[-1]['Equity']))
            if bnh_oos_m:
                print_metrics(bnh_oos_m, f"  VOO equiv. benchmark  ({OOS_START} → {OOS_END}, price + ~1.8% div)")

        combined = pd.concat([is_equity, oos_equity])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        full_metrics = compute_metrics(
            pd.DataFrame({'Equity': combined['Equity']}), is_start_capital
        )
        if full_metrics:
            print_metrics(full_metrics, f"FULL PERIOD  ({LOCAL_START} → {OOS_END})")
        if is_bnh_eq is not None and oos_bnh_eq is not None:
            full_bnh = pd.concat([is_bnh_eq, oos_bnh_eq])
            full_bnh = full_bnh[~full_bnh.index.duplicated(keep='last')].sort_index()
            full_bnh_m = compute_metrics(full_bnh, INITIAL_CAPITAL)
            if full_bnh_m:
                print_metrics(full_bnh_m, f"  VOO equiv. benchmark  ({LOCAL_START} → {OOS_END}, price + ~1.8% div)")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(is_equity.index, is_equity['Equity'],
             color='seagreen', linewidth=1.5, label='In-sample (local CSV)')

    if oos_equity is not None and not oos_equity.empty:
        bridge = pd.concat([is_equity.iloc[[-1]], oos_equity.iloc[[0]]])
        ax1.plot(bridge.index, bridge['Equity'], color='crimson', linewidth=1.5)
        ax1.plot(oos_equity.index, oos_equity['Equity'],
                 color='crimson', linewidth=1.5,
                 label=f'Out-of-sample (IBKR {OOS_START}→)')
        ax1.axvline(pd.Timestamp(OOS_START), color='gray', linestyle='--',
                    linewidth=0.8, label='OOS start')

    ax1.axhline(INITIAL_CAPITAL, color='gray', linestyle=':', linewidth=0.8)

    if is_bnh_eq is not None and not is_bnh_eq.empty:
        full_bnh = pd.concat([is_bnh_eq, oos_bnh_eq]) if oos_bnh_eq is not None else is_bnh_eq
        full_bnh = full_bnh[~full_bnh.index.duplicated(keep='last')].sort_index()
        ax1.plot(full_bnh.index, full_bnh['Equity'],
                 color='goldenrod', linewidth=1.0, alpha=0.7, linestyle=':',
                 label='VOO equiv. (price + div, est.)')

    ax1.set_ylabel('Portfolio Equity ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    sharpe_str = f"{is_metrics['Sharpe']:.2f}" if is_metrics else 'N/A'
    ax1.set_title(
        f'Combined IBS + EWMAC({FAST_SPAN},{SLOW_SPAN})  |  3 instruments  |  '
        f'${INITIAL_CAPITAL:,.0f} capital\n'
        f'In-sample Sharpe {sharpe_str}  |  75% IBS / 25% EWMAC',
        fontsize=11,
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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
    plt.savefig('portfolio/combined_port_ib_equity_curve.png', dpi=150)
    plt.show()
    logger.info("Chart saved to portfolio/combined_port_ib_equity_curve.png")


if __name__ == '__main__':
    main()
