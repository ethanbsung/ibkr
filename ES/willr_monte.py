import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# For GARCH
try:
    from arch import arch_model
except ImportError:
    print("Please install the 'arch' package for the GARCH model: pip install arch")

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# Parameters & User Settings
# -------------------------------

# Input file path for ES futures daily data
data_file = "Data/es_daily_data.csv"  # Must include: Time, High, Low, Last, Volume (if available)

# Backtest parameters
initial_capital = 10000.0         # starting account balance in dollars
commission_per_order = 1.24       # commission per order (per contract)
num_contracts = 1                 # number of contracts to trade
multiplier = 5                    # each point move is worth $5 per contract

# Custom start and end date (format: 'YYYY-MM-DD')
start_date = '2000-01-01'
end_date   = '2024-12-31'

# Williams %R parameters
williams_period = 2  # 2-day lookback
buy_threshold = -90
sell_threshold = -30

# Monte Carlo parameters
num_iterations = 5000  # number of Monte Carlo runs

# For Block Bootstrapping
block_size = 10  # number of consecutive days in each block

# -------------------------------
# 1) Data Preparation
# -------------------------------
data = pd.read_csv(data_file, parse_dates=['Time'])
data.sort_values('Time', inplace=True)

# Filter data based on custom date range
data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)

# -------------------------------
# Williams %R Calculation
# -------------------------------
# Williams %R = -100 * (HighestHigh(n) - Close) / (HighestHigh(n) - LowestLow(n))
data['HighestHigh'] = data['High'].rolling(window=williams_period).max()
data['LowestLow'] = data['Low'].rolling(window=williams_period).min()

# Drop rows that have NaNs (first n-1 rows won't have valid HighestHigh/LowestLow)
data.dropna(subset=['HighestHigh', 'LowestLow'], inplace=True)
data.reset_index(drop=True, inplace=True)

data['WilliamsR'] = -100 * (data['HighestHigh'] - data['Last']) / (data['HighestHigh'] - data['LowestLow'])

# -------------------------------
# 2) Strategy Backtest
# -------------------------------
capital = initial_capital
in_position = False
position = None
trade_results = []
equity_curve = []

initial_close = data['Last'].iloc[0]
benchmark_equity = (data.set_index('Time')['Last'] / initial_close) * initial_capital
benchmark_equity = benchmark_equity.reindex(data['Time'], method='ffill').fillna(method='ffill')

for i in range(len(data)):
    row = data.iloc[i]
    current_time = row['Time']
    current_price = row['Last']
    current_wr = row['WilliamsR']
    
    # Exit condition if in position
    if in_position:
        # Exit if today's close > yesterday's high OR Williams %R > -30
        if i > 0:
            yesterdays_high = data['High'].iloc[i-1]
            if (current_price > yesterdays_high) or (current_wr > sell_threshold):
                exit_price = current_price
                profit = (exit_price - position['entry_price']) * multiplier * num_contracts
                # Subtract exit commission
                profit -= commission_per_order * num_contracts
                trade_results.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit': profit,
                    'contracts': num_contracts
                })
                logger.info(f"SELL signal at {current_time} | Exit Price: {exit_price:.2f} | Profit: {profit:.2f}")
                capital += profit
                in_position = False
                position = None
    else:
        # Entry condition: Williams %R < -90
        if current_wr < buy_threshold:
            entry_price = current_price
            in_position = True
            # Subtract entry commission
            capital -= commission_per_order * num_contracts
            position = {
                'entry_price': entry_price,
                'entry_time': current_time,
                'contracts': num_contracts
            }
            logger.info(f"BUY signal at {current_time} | Entry Price: {entry_price:.2f} | Contracts: {num_contracts}")

    # Mark-to-market equity
    if in_position:
        unrealized = (current_price - position['entry_price']) * multiplier * num_contracts
        equity = capital + unrealized
    else:
        equity = capital
    equity_curve.append((current_time, equity))

# Close any open position at end of backtest
if in_position:
    row = data.iloc[-1]
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
    logger.info(f"Closing open position at end {current_time} | Exit Price: {exit_price:.2f} | Profit: {profit:.2f}")
    capital += profit
    equity = capital
    in_position = False
    position = None
    equity_curve[-1] = (current_time, equity)

# Convert equity curve to DataFrame
equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
equity_df.set_index('Time', inplace=True)

# -------------------------------
# Performance Metrics Calculation
# -------------------------------
final_account_balance = capital
total_return_percentage = ((final_account_balance / initial_capital) - 1) * 100

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
years = (end_dt - start_dt).days / 365.25
annualized_return_percentage = ((final_account_balance / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan

equity_df['returns'] = equity_df['Equity'].pct_change()
volatility_annual = equity_df['returns'].std() * np.sqrt(252) * 100

total_trades = len(trade_results)
winning_trades = [t for t in trade_results if t['profit'] > 0]
losing_trades = [t for t in trade_results if t['profit'] <= 0]
win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
profit_factor = (sum(t['profit'] for t in winning_trades) / abs(sum(t['profit'] for t in losing_trades))
                 if losing_trades and sum(t['profit'] for t in losing_trades) != 0 else np.nan)

avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else np.nan
avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else np.nan

sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252)
                if equity_df['returns'].std() != 0 else np.nan)
downside_std = equity_df[equity_df['returns'] < 0]['returns'].std()
sortino_ratio = (equity_df['returns'].mean() / downside_std * np.sqrt(252)
                 if downside_std != 0 else np.nan)

equity_df['EquityPeak'] = equity_df['Equity'].cummax()
equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['EquityPeak']) / equity_df['EquityPeak']
max_drawdown_percentage = equity_df['Drawdown'].min() * 100
equity_df['DrawdownDollar'] = equity_df['EquityPeak'] - equity_df['Equity']
max_drawdown_dollar = equity_df['DrawdownDollar'].max()
average_drawdown_dollar = equity_df.loc[equity_df['DrawdownDollar'] > 0, 'DrawdownDollar'].mean()
average_drawdown_percentage = equity_df['Drawdown'].mean() * 100
calmar_ratio = (annualized_return_percentage / abs(max_drawdown_percentage)
                if max_drawdown_percentage != 0 else np.nan)

results = {
    "Start Date": start_date,
    "End Date": end_date,
    "Final Account Balance": f"${final_account_balance:,.2f}",
    "Total Return": f"{total_return_percentage:.2f}%",
    "Annualized Return": f"{annualized_return_percentage:.2f}%",
    "Volatility (Annual)": f"{volatility_annual:.2f}%",
    "Total Trades": total_trades,
    "Winning Trades": len(winning_trades),
    "Losing Trades": len(losing_trades),
    "Win Rate": f"{win_rate:.2f}%",
    "Profit Factor": f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "NaN",
    "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "NaN",
    "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "NaN",
    "Calmar Ratio": f"{calmar_ratio:.2f}" if not np.isnan(calmar_ratio) else "NaN",
    "Max Drawdown (%)": f"{max_drawdown_percentage:.2f}%",
    "Average Drawdown (%)": f"{average_drawdown_percentage:.2f}%",
    "Max Drawdown ($)": f"${max_drawdown_dollar:,.2f}",
    "Average Drawdown ($)": f"${average_drawdown_dollar:,.2f}",
    "Average Win ($)": f"${avg_win:,.2f}",
    "Average Loss ($)": f"${avg_loss:,.2f}",
}

print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key:30}: {value:>15}")

# Plot Equity Curve
plt.figure(figsize=(14, 7))
plt.plot(equity_df.index, equity_df['Equity'], label='Strategy Equity')
plt.title('Equity Curve: Williams %R Strategy')
plt.xlabel('Time')
plt.ylabel('Account Balance ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# Monte Carlo Section: ALTERNATIVE METHODS
# --------------------------------------------------------------------
daily_returns = equity_df['returns'].dropna().values
num_days = len(daily_returns)

# ================================================================
# 1) Block Bootstrapping (Preserving Short-Term Correlations)
# ================================================================
def block_bootstrap_returns(returns, block_size, n_days):
    """
    Perform block bootstrapping of daily returns.
    - returns: original daily returns array
    - block_size: number of consecutive days in each block
    - n_days: total number of days to sample
    """
    # Number of blocks needed
    num_blocks = int(np.ceil(n_days / block_size))
    
    # Indices for starting each block
    start_indices = np.random.randint(low=0, high=len(returns) - block_size + 1, size=num_blocks)
    
    # Collect blocks
    blocks = []
    for idx in start_indices:
        block = returns[idx:idx+block_size]
        blocks.append(block)
    
    # Concatenate blocks
    sampled = np.concatenate(blocks)
    # Trim to desired length
    sampled = sampled[:n_days]
    return sampled

# ================================================================
# 2) Synthetic Price Generation (GARCH(1,1))
# ================================================================
def generate_garch_prices(returns, start_price, n_days):
    """
    Fit a GARCH(1,1) model to the returns, then simulate n_days of new returns,
    and build a synthetic price series from start_price.
    """
    # Fit GARCH(1,1) to historical returns
    # We'll convert daily_returns to percentage or log-returns if needed
    # For simplicity, let's assume 'returns' is already close to a daily percent return
    am = arch_model(returns * 100, p=1, q=1, vol='GARCH', dist='normal')
    res = am.fit(disp='off')
    
    # Simulate new data
    sim = res.simulate(res.params, nobs=n_days)
    
    # Convert from percentage back to decimal
    simulated_returns = sim['data'] / 100.0
    
    # Build price series from start_price
    prices = np.empty(n_days + 1)
    prices[0] = start_price
    for i in range(n_days):
        prices[i+1] = prices[i] * (1.0 + simulated_returns[i])
    return prices

# ================================================================
# Helper Functions to Compute Metrics on a Simulated Equity Curve
# ================================================================
def compute_metrics(equity_curve):
    """
    Given a simulated equity curve (numpy array), compute:
    - Final Equity
    - Sharpe
    - Max Drawdown
    - Profit Factor (based on daily PnL)
    """
    final_equity = equity_curve[-1]
    # daily returns
    daily_rets = pd.Series(equity_curve).pct_change().dropna()
    
    # Sharpe
    if daily_rets.std() != 0:
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
    else:
        sharpe = np.nan
    
    # Max Drawdown
    running_max = pd.Series(equity_curve).cummax()
    drawdowns = (pd.Series(equity_curve) - running_max) / running_max
    max_dd = drawdowns.min()  # negative
    
    # Profit Factor
    daily_pnl = np.diff(equity_curve)
    gross_profit = daily_pnl[daily_pnl > 0].sum()
    gross_loss = abs(daily_pnl[daily_pnl < 0].sum())
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    else:
        pf = np.nan
    
    return final_equity, sharpe, max_dd, pf

# --------------------------------------------------------------------
# MONTE CARLO: BLOCK BOOTSTRAP
# --------------------------------------------------------------------
mc_iterations = 1000
sim_final_equities = []
sim_sharpes = []
sim_max_dds = []
sim_pfs = []

for _ in range(mc_iterations):
    # block bootstrap the daily returns
    sampled_rets = block_bootstrap_returns(daily_returns, block_size, num_days)
    
    # build the equity curve
    sim_curve = np.empty(num_days + 1)
    sim_curve[0] = initial_capital
    sim_curve[1:] = initial_capital * np.cumprod(1 + sampled_rets)
    
    # compute metrics
    fe, sr, mdd, pf = compute_metrics(sim_curve)
    sim_final_equities.append(fe)
    sim_sharpes.append(sr)
    sim_max_dds.append(mdd)
    sim_pfs.append(pf)

# Summaries
sim_final_equities = np.array(sim_final_equities)
mean_fe = np.mean(sim_final_equities)
median_fe = np.median(sim_final_equities)
perc5_fe = np.percentile(sim_final_equities, 5)
perc95_fe = np.percentile(sim_final_equities, 95)

mean_sr = np.nanmean(sim_sharpes)
mean_mdd = np.nanmean(sim_max_dds)
mean_pf = np.nanmean(sim_pfs)

print("\nBlock Bootstrap Monte Carlo (Preserving Short-Term Correlation):")
print(f"Mean Final Equity: ${mean_fe:,.2f}")
print(f"Median Final Equity: ${median_fe:,.2f}")
print(f"5th Percentile: ${perc5_fe:,.2f}")
print(f"95th Percentile: ${perc95_fe:,.2f}")
print(f"Mean Sharpe Ratio: {mean_sr:.2f}")
print(f"Mean Max Drawdown: {mean_mdd*100:.2f}%")
print(f"Mean Profit Factor: {mean_pf:.2f}")

plt.figure(figsize=(10, 6))
plt.hist(sim_final_equities, bins=50, edgecolor='k', alpha=0.7)
plt.title('Block Bootstrap Monte Carlo: Final Equity Distribution')
plt.xlabel('Final Equity ($)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------
# MONTE CARLO: SYNTHETIC PRICE GENERATION (GARCH)
# --------------------------------------------------------------------
# We can simulate new prices, then derive daily returns from those synthetic prices,
# and assume the strategy's distribution of daily returns is the same or re-apply the same logic.
# For simplicity, let's just replicate the final equity approach with random draws from the GARCH model.

try:
    # Fit GARCH(1,1) to historical daily_returns
    # We'll treat daily_returns as decimal returns, so multiply by 100 for the arch library
    am = arch_model(daily_returns * 100, p=1, q=1, vol='GARCH', dist='normal')
    res = am.fit(disp='off')

    sim_final_equities_garch = []
    sim_sharpes_garch = []
    sim_max_dds_garch = []
    sim_pfs_garch = []

    for _ in range(mc_iterations):
        # Simulate new daily returns from the GARCH model object 'am'
        # using the fitted parameters 'res.params'
        sim_data = am.simulate(res.params, nobs=num_days)
        sim_rets = sim_data['data'] / 100.0  # convert back to decimal returns

        # Build equity curve
        sim_curve = np.empty(num_days + 1)
        sim_curve[0] = initial_capital
        sim_curve[1:] = initial_capital * np.cumprod(1 + sim_rets.values)
        
        fe, sr, mdd, pf = compute_metrics(sim_curve)
        sim_final_equities_garch.append(fe)
        sim_sharpes_garch.append(sr)
        sim_max_dds_garch.append(mdd)
        sim_pfs_garch.append(pf)

    sim_final_equities_garch = np.array(sim_final_equities_garch)
    mean_fe_g = np.mean(sim_final_equities_garch)
    median_fe_g = np.median(sim_final_equities_garch)
    perc5_fe_g = np.percentile(sim_final_equities_garch, 5)
    perc95_fe_g = np.percentile(sim_final_equities_garch, 95)

    mean_sr_g = np.nanmean(sim_sharpes_garch)
    mean_mdd_g = np.nanmean(sim_max_dds_garch)
    mean_pf_g = np.nanmean(sim_pfs_garch)

    print("\nGARCH(1,1) Monte Carlo (Synthetic Market Regimes):")
    print(f"Mean Final Equity: ${mean_fe_g:,.2f}")
    print(f"Median Final Equity: ${median_fe_g:,.2f}")
    print(f"5th Percentile: ${perc5_fe_g:,.2f}")
    print(f"95th Percentile: ${perc95_fe_g:,.2f}")
    print(f"Mean Sharpe Ratio: {mean_sr_g:.2f}")
    print(f"Mean Max Drawdown: {mean_mdd_g*100:.2f}%")
    print(f"Mean Profit Factor: {mean_pf_g:.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(sim_final_equities_garch, bins=50, edgecolor='k', alpha=0.7)
    plt.title('GARCH(1,1) Monte Carlo: Final Equity Distribution')
    plt.xlabel('Final Equity ($)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except NameError:
    print("\nThe 'arch' library is required for GARCH simulations. Please install it with: pip install arch")
except:
    import traceback
    traceback.print_exc()
    print("\nAn error occurred with GARCH simulation.")