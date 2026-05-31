import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

try:
    from arch import arch_model
    GARCH_AVAILABLE = True
except ImportError:
    print("`arch` library not found. GARCH simulation will be skipped.")
    GARCH_AVAILABLE = False

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# Parameters & User Settings
# -------------------------------
data_file = "Data/mes_daily_data.csv"  # Must include: Time, High, Low, Last, Volume (if available)
initial_capital = 10000.0
commission_per_order = 1.24
num_contracts = 1
multiplier = 5
start_date = '2000-01-01'
end_date   = '2024-12-31'

# IBS thresholds
ibs_buy_threshold = 0.1
ibs_sell_threshold = 0.9

# Monte Carlo parameters
num_iterations = 1000
block_size = 5  # for block bootstrap

# ================================================================
# 1) Data Preparation
# ================================================================
data = pd.read_csv(data_file, parse_dates=['Time'])
data.sort_values('Time', inplace=True)
data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)

# Calculate IBS
data['IBS'] = (data['Last'] - data['Low']) / (data['High'] - data['Low'])
data.dropna(subset=['IBS'], inplace=True)
data.reset_index(drop=True, inplace=True)

# We'll also compute log returns for GARCH on the underlying price
data['LogReturn'] = np.log(data['Last'] / data['Last'].shift(1))
data.dropna(subset=['LogReturn'], inplace=True)
data.reset_index(drop=True, inplace=True)

# ================================================================
# 1A) Build a Distribution of Intraday Range Ratios
#    (High - Low) / Last from Historical Data
# ================================================================
# Because we dropped some rows above, let's store them first:
data['range_ratio'] = (data['High'] - data['Low']) / data['Last']
# Filter out any zero or negative (should be rare)
data = data[data['range_ratio'] > 0]
range_ratios = data['range_ratio'].values

# ================================================================
# 2) IBS Strategy on Actual Historical Data
# ================================================================
def run_ibs_strategy_on_data(df):
    """
    Runs the IBS strategy on a DataFrame with columns: Time, Last, IBS.
    Returns final_capital, list_of_trades, equity_df
    """
    capital = initial_capital
    in_position = False
    position = None
    trade_results = []
    equity_curve = []
    
    for i, row in df.iterrows():
        current_time = row['Time']
        current_price = row['Last']
        current_ibs = row['IBS']
        
        # If in position, check exit
        if in_position:
            if current_ibs > ibs_sell_threshold:
                exit_price = current_price
                profit = (exit_price - position['entry_price']) * multiplier * num_contracts
                profit -= commission_per_order * num_contracts
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'contracts': num_contracts
                })
                capital += profit
                in_position = False
                position = None
        else:
            # Entry condition
            if current_ibs < ibs_buy_threshold:
                entry_price = current_price
                in_position = True
                capital -= commission_per_order * num_contracts
                position = {
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'contracts': num_contracts
                }
        
        # Mark-to-market equity
        if in_position:
            unrealized = (current_price - position['entry_price']) * multiplier * num_contracts
            equity = capital + unrealized
        else:
            equity = capital
        
        equity_curve.append((current_time, equity))
    
    # Close any open position at the end
    if in_position:
        row = df.iloc[-1]
        current_time = row['Time']
        current_price = row['Last']
        exit_price = current_price
        profit = (exit_price - position['entry_price']) * multiplier * num_contracts
        profit -= commission_per_order * num_contracts
        trade_results.append({
            'entry_time': position['entry_time'],
            'exit_time': current_time,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'profit': profit,
            'contracts': num_contracts
        })
        capital += profit
        equity_curve[-1] = (current_time, capital)
        in_position = False
        position = None
    
    eq_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
    eq_df.set_index('Time', inplace=True)
    
    return capital, trade_results, eq_df

# Run IBS on actual historical data
final_capital, trade_results, equity_df = run_ibs_strategy_on_data(data)

# Compute performance metrics for the historical backtest
initial_close = data['Last'].iloc[0]
benchmark_equity = (data.set_index('Time')['Last'] / initial_close) * initial_capital
benchmark_equity = benchmark_equity.reindex(equity_df.index, method='ffill').fillna(method='ffill')

def calc_performance(equity_df, trade_results):
    final_balance = equity_df['Equity'].iloc[-1]
    total_return_pct = (final_balance / initial_capital - 1) * 100
    
    equity_df['returns'] = equity_df['Equity'].pct_change()
    years = (end_dt - start_dt).days / 365.25 if (end_dt - start_dt).days > 0 else 1
    annual_ret_pct = ((final_balance / initial_capital) ** (1/years) - 1) * 100 if years > 0 else np.nan
    
    vol_annual = equity_df['returns'].std() * np.sqrt(252) * 100
    total_trades = len(trade_results)
    winning_trades = [t for t in trade_results if t['profit'] > 0]
    losing_trades = [t for t in trade_results if t['profit'] <= 0]
    win_rate = (len(winning_trades)/total_trades*100) if total_trades>0 else 0
    pf = np.nan
    if losing_trades and sum(t['profit'] for t in losing_trades) != 0:
        pf = sum(t['profit'] for t in winning_trades)/abs(sum(t['profit'] for t in losing_trades))
    
    avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else np.nan
    avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else np.nan
    
    sharpe = np.nan
    if equity_df['returns'].std() != 0:
        sharpe = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252)
    downside_std = equity_df[equity_df['returns']<0]['returns'].std()
    sortino = np.nan
    if downside_std != 0 and not np.isnan(downside_std):
        sortino = equity_df['returns'].mean() / downside_std * np.sqrt(252)
    
    equity_df['EquityPeak'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['EquityPeak']) / equity_df['EquityPeak']
    max_dd_pct = equity_df['Drawdown'].min() * 100
    equity_df['DrawdownDollar'] = equity_df['EquityPeak'] - equity_df['Equity']
    max_dd_dollar = equity_df['DrawdownDollar'].max()
    avg_dd_dollar = equity_df.loc[equity_df['DrawdownDollar']>0,'DrawdownDollar'].mean()
    avg_dd_pct = equity_df['Drawdown'].mean() * 100
    
    calmar = np.nan
    if max_dd_pct != 0:
        calmar = annual_ret_pct / abs(max_dd_pct)
    
    results_dict = {
        "Final Account Balance": final_balance,
        "Total Return (%)": total_return_pct,
        "Annualized Return (%)": annual_ret_pct,
        "Volatility (Annual %)": vol_annual,
        "Total Trades": total_trades,
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win Rate (%)": win_rate,
        "Profit Factor": pf,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown (%)": max_dd_pct,
        "Average Drawdown (%)": avg_dd_pct,
        "Max Drawdown ($)": max_dd_dollar,
        "Average Drawdown ($)": avg_dd_dollar,
        "Average Win ($)": avg_win,
        "Average Loss ($)": avg_loss
    }
    return results_dict

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
hist_results = calc_performance(equity_df, trade_results)

print("\n--- Historical IBS Backtest Results ---")
for k,v in hist_results.items():
    if isinstance(v, float):
        print(f"{k:30}: {v:,.2f}")
    else:
        print(f"{k:30}: {v}")

plt.figure(figsize=(14,7))
plt.plot(equity_df.index, equity_df['Equity'], label='Strategy Equity')
plt.title('Equity Curve: IBS Strategy (Historical)')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# We'll do naive & block bootstrap on the strategy's daily returns
strategy_daily_returns = equity_df['returns'].dropna().values
num_days = len(strategy_daily_returns)

# Helper for final metrics from an equity curve
def compute_metrics(equity_curve):
    final_equity = equity_curve[-1]
    final_return_pct = (final_equity / initial_capital - 1) * 100
    daily_rets = pd.Series(equity_curve).pct_change().dropna()
    
    if daily_rets.std() != 0:
        sr = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
    else:
        sr = np.nan
    
    curve_series = pd.Series(equity_curve)
    running_max = curve_series.cummax()
    drawdowns = (curve_series - running_max)/running_max
    mdd = drawdowns.min()
    
    daily_pnl = np.diff(equity_curve)
    gross_profit = daily_pnl[daily_pnl>0].sum()
    gross_loss = abs(daily_pnl[daily_pnl<0].sum())
    pf = np.nan
    if gross_loss>0:
        pf = gross_profit/gross_loss
    
    return final_return_pct, sr, mdd, pf, final_equity

# ================================================================
# 3) Naive (Daily) Bootstrap on Strategy Returns
# ================================================================
mc_iterations = num_iterations
sim_final_returns = []
sim_sharpes = []
sim_max_dds = []
sim_pfs = []
all_equity_curves = []

for _ in range(mc_iterations):
    sampled_rets = np.random.choice(strategy_daily_returns, size=num_days, replace=True)
    sim_curve = np.empty(num_days + 1)
    sim_curve[0] = initial_capital
    sim_curve[1:] = initial_capital * np.cumprod(1 + sampled_rets)
    all_equity_curves.append(sim_curve)
    
    fr, sr, mdd, pf, fe = compute_metrics(sim_curve)
    sim_final_returns.append(fr)
    sim_sharps = sim_sharpes.append(sr)
    sim_max_dds.append(mdd)
    sim_pfs.append(pf)

sim_final_returns = np.array(sim_final_returns)
sim_sharpes = np.array(sim_sharpes)
sim_max_dds = np.array(sim_max_dds)
sim_pfs = np.array(sim_pfs)

print("\n--- Monte Carlo (Naive Bootstrap on Strategy Returns) ---")
mean_fr = np.nanmean(sim_final_returns)
median_fr = np.nanmedian(sim_final_returns)
perc5_fr = np.nanpercentile(sim_final_returns, 5)
perc95_fr = np.nanpercentile(sim_final_returns, 95)

mean_sr = np.nanmean(sim_sharpes)
mean_mdd = np.nanmean(sim_max_dds)
mean_pf = np.nanmean(sim_pfs)

worst_fr = np.nanmin(sim_final_returns)
best_fr = np.nanmax(sim_final_returns)
worst_sr = np.nanmin(sim_sharpes)
best_sr = np.nanmax(sim_sharpes)
worst_mdd = np.nanmin(sim_max_dds)
best_mdd = np.nanmax(sim_max_dds)
worst_pf = np.nanmin(sim_pfs)
best_pf = np.nanmax(sim_pfs)

print(f"Mean Final Return (%): {mean_fr:.2f}")
print(f"Median Final Return (%): {median_fr:.2f}")
print(f"5th Percentile Return (%): {perc5_fr:.2f}")
print(f"95th Percentile Return (%): {perc95_fr:.2f}")
print(f"\nMean Sharpe Ratio: {mean_sr:.2f}")
print(f"Mean Max Drawdown: {mean_mdd*100:.2f}%")
print(f"Mean Profit Factor: {mean_pf:.2f}")

print("\nWorst / Best Cases (Naive Bootstrap):")
print(f"Worst Final Return (%): {worst_fr:.2f}")
print(f"Best Final Return (%): {best_fr:.2f}")
print(f"Worst Sharpe Ratio: {worst_sr:.2f}")
print(f"Best Sharpe Ratio: {best_sr:.2f}")
print(f"Worst Max Drawdown: {worst_mdd*100:.2f}%")
print(f"Best Max Drawdown: {best_mdd*100:.2f}%")
print(f"Worst Profit Factor: {worst_pf:.2f}")
print(f"Best Profit Factor: {best_pf:.2f}")

plt.figure(figsize=(10,6))
plt.hist(sim_final_returns, bins=50, edgecolor='k', alpha=0.7)
plt.title('Naive Bootstrap: Distribution of Final Returns (%)')
plt.xlabel('Final Return (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,7))
x_axis = range(num_days+1)
for sim_curve in all_equity_curves:
    plt.plot(x_axis, sim_curve, color='gray', alpha=0.1)
plt.plot(range(len(equity_df)), equity_df['Equity'].values, label='Actual Equity Curve', color='blue', linewidth=2)
plt.title('Naive Bootstrap: Equity Curves')
plt.xlabel('Simulated Days')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================================================
# 4) Block Bootstrapping (Strategy Returns)
# ================================================================
def block_bootstrap_returns(returns, block_size, n_days):
    num_blocks = int(np.ceil(n_days / block_size))
    start_indices = np.random.randint(low=0, high=len(returns) - block_size + 1, size=num_blocks)
    blocks = []
    for idx in start_indices:
        block = returns[idx:idx+block_size]
        blocks.append(block)
    sampled = np.concatenate(blocks)
    sampled = sampled[:n_days]
    return sampled

sim_final_returns_bb = []
sim_sharpes_bb = []
sim_max_dds_bb = []
sim_pfs_bb = []
all_equity_curves_bb = []

for _ in range(num_iterations):
    sampled_rets = block_bootstrap_returns(strategy_daily_returns, block_size, num_days)
    sim_curve = np.empty(num_days + 1)
    sim_curve[0] = initial_capital
    sim_curve[1:] = initial_capital * np.cumprod(1 + sampled_rets)
    all_equity_curves_bb.append(sim_curve)
    
    fr, sr, mdd, pf, fe = compute_metrics(sim_curve)
    sim_final_returns_bb.append(fr)
    sim_sharpes_bb.append(sr)
    sim_max_dds_bb.append(mdd)
    sim_pfs_bb.append(pf)

sim_final_returns_bb = np.array(sim_final_returns_bb)
sim_sharpes_bb = np.array(sim_sharpes_bb)
sim_max_dds_bb = np.array(sim_max_dds_bb)
sim_pfs_bb = np.array(sim_pfs_bb)

print("\n--- Block Bootstrap Monte Carlo (Strategy Returns) ---")
mean_fr_bb = np.nanmean(sim_final_returns_bb)
median_fr_bb = np.nanmedian(sim_final_returns_bb)
perc5_fr_bb = np.nanpercentile(sim_final_returns_bb, 5)
perc95_fr_bb = np.nanpercentile(sim_final_returns_bb, 95)

mean_sr_bb = np.nanmean(sim_sharpes_bb)
mean_mdd_bb = np.nanmean(sim_max_dds_bb)
mean_pf_bb = np.nanmean(sim_pfs_bb)

worst_fr_bb = np.nanmin(sim_final_returns_bb)
best_fr_bb = np.nanmax(sim_final_returns_bb)
worst_sr_bb = np.nanmin(sim_sharpes_bb)
best_sr_bb = np.nanmax(sim_sharpes_bb)
worst_mdd_bb = np.nanmin(sim_max_dds_bb)
best_mdd_bb = np.nanmax(sim_max_dds_bb)
worst_pf_bb = np.nanmin(sim_pfs_bb)
best_pf_bb = np.nanmax(sim_pfs_bb)

print(f"Mean Final Return (%): {mean_fr_bb:.2f}")
print(f"Median Final Return (%): {median_fr_bb:.2f}")
print(f"5th Percentile Return (%): {perc5_fr_bb:.2f}")
print(f"95th Percentile Return (%): {perc95_fr_bb:.2f}")
print(f"\nMean Sharpe Ratio: {mean_sr_bb:.2f}")
print(f"Mean Max Drawdown: {mean_mdd_bb*100:.2f}%")
print(f"Mean Profit Factor: {mean_pf_bb:.2f}")

print("\nWorst / Best Cases (Block Bootstrap):")
print(f"Worst Final Return (%): {worst_fr_bb:.2f}")
print(f"Best Final Return (%): {best_fr_bb:.2f}")
print(f"Worst Sharpe Ratio: {worst_sr_bb:.2f}")
print(f"Best Sharpe Ratio: {best_sr_bb:.2f}")
print(f"Worst Max Drawdown: {worst_mdd_bb*100:.2f}%")
print(f"Best Max Drawdown: {best_mdd_bb*100:.2f}%")
print(f"Worst Profit Factor: {worst_pf_bb:.2f}")
print(f"Best Profit Factor: {best_pf_bb:.2f}")

plt.figure(figsize=(10,6))
plt.hist(sim_final_returns_bb, bins=50, edgecolor='k', alpha=0.7)
plt.title('Block Bootstrap: Distribution of Final Returns (%)')
plt.xlabel('Final Return (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,7))
for sim_curve in all_equity_curves_bb:
    plt.plot(x_axis, sim_curve, color='gray', alpha=0.1)
plt.plot(range(len(equity_df)), equity_df['Equity'].values, label='Actual Equity Curve', color='blue', linewidth=2)
plt.title('Block Bootstrap: Equity Curves')
plt.xlabel('Simulated Days')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================================================
# 5) GARCH(1,1) Simulation on Underlying Price with Realistic Ranges
# ================================================================
def run_ibs_on_price_series(prices, times, range_ratios):
    """
    Given:
      prices (np.array) of length N: synthetic daily close prices
      times (np.array)  of length N: corresponding timestamps
      range_ratios (np.array): distribution of (High - Low)/Close from historical data
    Returns:
      final_cap, trades, eq_df after running IBS
    """
    import random
    
    high_vals = []
    low_vals = []
    last_vals = []
    time_vals = []
    
    for i in range(len(prices)):
        last_price = prices[i]
        chosen_ratio = random.choice(range_ratios)  # sample from historical distribution
        intraday_range = last_price * chosen_ratio
        half_range = intraday_range / 2.0
        
        synthetic_high = last_price + half_range
        synthetic_low  = last_price - half_range
        
        high_vals.append(synthetic_high)
        low_vals.append(synthetic_low)
        last_vals.append(last_price)
        time_vals.append(times[i])
    
    df = pd.DataFrame({
        'Time': time_vals,
        'High': high_vals,
        'Low': low_vals,
        'Last': last_vals
    })
    # IBS
    df['IBS'] = (df['Last'] - df['Low']) / (df['High'] - df['Low'])
    df['IBS'].fillna(0.5, inplace=True)
    
    final_cap, trades, eq_df = run_ibs_strategy_on_data(df)
    return final_cap, trades, eq_df

if GARCH_AVAILABLE:
    log_returns_pct = data['LogReturn'] * 100.0
    am = arch_model(log_returns_pct, p=1, q=1, vol='GARCH', dist='normal')
    res = am.fit(disp='off')
    
    sim_final_returns_garch = []
    sim_sharpes_garch = []
    sim_max_dds_garch = []
    sim_pfs_garch = []
    all_equity_curves_garch = []
    
    # We'll keep the same number of days as log_returns_pct
    n_days_garch = len(log_returns_pct)
    times_array = data['Time'].values  # must match length n_days_garch
    
    for _ in range(num_iterations):
        sim_data = am.simulate(res.params, nobs=n_days_garch)
        sim_log_rets = sim_data['data'] / 100.0  # decimal
        
        # Build synthetic close prices
        start_price = data['Last'].iloc[0]
        synthetic_prices = np.empty(n_days_garch)
        synthetic_prices[0] = start_price
        for i in range(1, n_days_garch):
            synthetic_prices[i] = synthetic_prices[i-1] * np.exp(sim_log_rets.iloc[i])
        
        # Run IBS with realistic High/Low
        final_cap_g, trades_g, eq_df_g = run_ibs_on_price_series(
            synthetic_prices,
            times_array[:n_days_garch],  # ensure same length
            range_ratios
        )
        # Compute metrics
        fr, sr, mdd, pf, fe = compute_metrics(eq_df_g['Equity'].values)
        sim_final_returns_garch.append(fr)
        sim_sharpes_garch.append(sr)
        sim_max_dds_garch.append(mdd)
        sim_pfs_garch.append(pf)
        all_equity_curves_garch.append(eq_df_g['Equity'].values)
    
    sim_final_returns_garch = np.array(sim_final_returns_garch)
    sim_sharpes_garch = np.array(sim_sharpes_garch)
    sim_max_dds_garch = np.array(sim_max_dds_garch)
    sim_pfs_garch = np.array(sim_pfs_garch)
    
    mean_fr_g = np.nanmean(sim_final_returns_garch)
    median_fr_g = np.nanmedian(sim_final_returns_garch)
    perc5_fr_g = np.nanpercentile(sim_final_returns_garch, 5)
    perc95_fr_g = np.nanpercentile(sim_final_returns_garch, 95)
    
    mean_sr_g = np.nanmean(sim_sharpes_garch)
    mean_mdd_g = np.nanmean(sim_max_dds_garch)
    mean_pf_g = np.nanmean(sim_pfs_garch)
    
    worst_fr_g = np.nanmin(sim_final_returns_garch)
    best_fr_g = np.nanmax(sim_final_returns_garch)
    worst_sr_g = np.nanmin(sim_sharpes_garch)
    best_sr_g = np.nanmax(sim_sharpes_garch)
    worst_mdd_g = np.nanmin(sim_max_dds_garch)
    best_mdd_g = np.nanmax(sim_max_dds_garch)
    worst_pf_g = np.nanmin(sim_pfs_garch)
    best_pf_g = np.nanmax(sim_pfs_garch)
    
    print("\n--- GARCH(1,1) Monte Carlo (Synthetic Price with Intraday Ranges) ---")
    print(f"Mean Final Return (%): {mean_fr_g:.2f}")
    print(f"Median Final Return (%): {median_fr_g:.2f}")
    print(f"5th Percentile Return (%): {perc5_fr_g:.2f}")
    print(f"95th Percentile Return (%): {perc95_fr_g:.2f}")
    print(f"\nMean Sharpe Ratio: {mean_sr_g:.2f}")
    print(f"Mean Max Drawdown: {mean_mdd_g*100:.2f}%")
    print(f"Mean Profit Factor: {mean_pf_g:.2f}")
    
    print("\nWorst / Best Cases (GARCH on Price):")
    print(f"Worst Final Return (%): {worst_fr_g:.2f}")
    print(f"Best Final Return (%): {best_fr_g:.2f}")
    print(f"Worst Sharpe Ratio: {worst_sr_g:.2f}")
    print(f"Best Sharpe Ratio: {best_sr_g:.2f}")
    print(f"Worst Max Drawdown: {worst_mdd_g*100:.2f}%")
    print(f"Best Max Drawdown: {best_mdd_g*100:.2f}%")
    print(f"Worst Profit Factor: {worst_pf_g:.2f}")
    print(f"Best Profit Factor: {best_pf_g:.2f}")
    
    plt.figure(figsize=(10,6))
    plt.hist(sim_final_returns_garch, bins=50, edgecolor='k', alpha=0.7)
    plt.title('GARCH(1,1) Monte Carlo: Distribution of Final Returns (%)')
    plt.xlabel('Final Return (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot the GARCH equity curves
    plt.figure(figsize=(14,7))
    for curve in all_equity_curves_garch:
        plt.plot(range(len(curve)), curve, color='gray', alpha=0.1)
    plt.plot(range(len(equity_df)), equity_df['Equity'].values, label='Actual Equity Curve', color='blue', linewidth=2)
    plt.title('GARCH(1,1) Monte Carlo: Equity Curves (Synthetic Price)')
    plt.xlabel('Simulated Days')
    plt.ylabel('Account Balance ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("\nSkipping GARCH simulation because `arch` library not installed.")