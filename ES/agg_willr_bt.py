# es, gc, ym, sq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# Parameters & User Settings
# -------------------------------

# Backtest parameters
initial_capital = 30000.0         # starting account balance in dollars
commission_per_order = 1.24       # commission per order (per contract)

# Dynamic Position Sizing Parameters (matching live trading)
risk_multiplier = 3.0             # 3x larger positions for higher risk/reward
allocation_per_instrument = 0.25  # 25% allocation per instrument (4 instruments = 100%)
min_contracts = 1                 # minimum number of contracts

# Custom start and end date (format: 'YYYY-MM-DD')
start_date = '2000-01-01'
end_date   = '2020-01-01'

# Williams %R Strategy parameters
williams_period = 2   # 2-day lookback period
buy_threshold = -90   # Buy when Williams %R is below -90
sell_threshold = -30  # Sell when Williams %R is above -30

# Portfolio instruments (matching live trading setup)
portfolio_instruments = {
    'MES': {'file': 'Data/mes_daily_data.csv', 'multiplier': 5},
    'MNQ': {'file': 'Data/mnq_daily_data.csv', 'multiplier': 2},
    'MGC': {'file': 'Data/mgc_daily_data.csv', 'multiplier': 10},
    'MYM': {'file': 'Data/mym_daily_data.csv', 'multiplier': 0.50}
}

# -------------------------------
# Dynamic Position Sizing Function
# -------------------------------
def calculate_position_size(current_equity, target_allocation_pct, price, multiplier, min_contracts=1):
    """
    Calculate number of contracts based on current equity and target allocation with enhanced risk.
    
    Args:
        current_equity: Current account equity
        target_allocation_pct: Target percentage allocation (0.0 to 1.0)
        price: Current price of the instrument
        multiplier: Contract multiplier
        min_contracts: Minimum number of contracts (default 1)
    
    Returns:
        Number of contracts to trade
    """
    target_dollar_amount = current_equity * target_allocation_pct * risk_multiplier
    contract_value = price * multiplier
    
    if contract_value <= 0:
        return min_contracts
    
    calculated_contracts = target_dollar_amount / contract_value
    
    # Round to nearest integer, minimum specified contracts
    contracts = max(min_contracts, round(calculated_contracts))
    
    return int(contracts)

# -------------------------------
# Portfolio Williams %R Backtest
# -------------------------------
def run_portfolio_williams_backtest():
    """
    Runs the Williams %R strategy backtest on a portfolio of instruments.
    Returns performance metrics for the combined portfolio.
    """
    
    # Load and prepare data for all instruments
    instrument_data = {}
    
    for symbol, config in portfolio_instruments.items():
        try:
            data = pd.read_csv(config['file'], parse_dates=['Time'])
            data.sort_values('Time', inplace=True)
            # Filter data based on the custom date range
            data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)].reset_index(drop=True)
            
            if data.empty:
                logger.warning(f"No data for {symbol} in the given date range.")
                continue
                
            # Calculate Williams %R
            data['HighestHigh'] = data['High'].rolling(window=williams_period).max()
            data['LowestLow'] = data['Low'].rolling(window=williams_period).min()
            data.dropna(subset=['HighestHigh', 'LowestLow'], inplace=True)
            data.reset_index(drop=True, inplace=True)
            
            # Williams %R formula: -100 * (HighestHigh - Last) / (HighestHigh - LowestLow)
            data['WilliamsR'] = -100 * (data['HighestHigh'] - data['Last']) / (data['HighestHigh'] - data['LowestLow'])
            
            instrument_data[symbol] = {
                'data': data,
                'multiplier': config['multiplier'],
                'in_position': False,
                'position': None
            }
            
            logger.info(f"Loaded {len(data)} bars for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            continue
    
    if not instrument_data:
        logger.error("No instrument data loaded successfully")
        return None
    
    # Get all unique dates and sort them
    all_dates = set()
    for symbol, info in instrument_data.items():
        all_dates.update(info['data']['Time'].tolist())
    
    all_dates = sorted(list(all_dates))
    logger.info(f"Portfolio backtest from {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} trading days)")
    
    # -------------------------------
    # Portfolio Backtest Simulation (Fixed Logic)
    # -------------------------------
    portfolio_capital = initial_capital  # Realized capital (cash)
    trade_results = []
    equity_curve = []
    
    for current_date in all_dates:
        # Process each instrument for entry/exit signals
        for symbol, info in instrument_data.items():
            data = info['data']
            multiplier = info['multiplier']
            
            # Find current day's data
            current_row = data[data['Time'] == current_date]
            if current_row.empty:
                continue
                
            current_row = current_row.iloc[0]
            current_price = current_row['Last']
            current_wr = current_row['WilliamsR']
            
            # Find current index for yesterday's high check
            current_idx = data[data['Time'] == current_date].index[0]
            
            # Process position logic (matching willr_bt.py exactly)
            if info['in_position']:
                # Check exit conditions (same as willr_bt.py)
                if current_idx > 0:
                    yesterdays_high = data.iloc[current_idx - 1]['High']
                    if (current_price > yesterdays_high) or (current_wr > sell_threshold):
                        # Exit position
                        position = info['position']
                        exit_price = current_price
                        profit = (exit_price - position['entry_price']) * multiplier * position['contracts']
                        profit -= commission_per_order * position['contracts']  # Exit commission
                        
                        trade_results.append({
                            'symbol': symbol,
                            'entry_time': position['entry_time'],
                            'exit_time': current_date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'profit': profit,
                            'contracts': position['contracts']
                        })
                        
                        # Add realized profit to capital
                        portfolio_capital += profit
                        info['in_position'] = False
                        info['position'] = None
                        
                        logger.info(f"{symbol} EXIT at {current_date} | Price: {exit_price:.2f} | Contracts: {position['contracts']} | Profit: ${profit:.2f}")
            
            else:
                # Check entry condition (same as willr_bt.py)
                if current_wr < buy_threshold:
                    # Calculate position size based on current portfolio capital
                    num_contracts = calculate_position_size(
                        portfolio_capital,
                        allocation_per_instrument,
                        current_price,
                        multiplier,
                        min_contracts
                    )
                    
                    # Enter position
                    entry_cost = commission_per_order * num_contracts
                    portfolio_capital -= entry_cost  # Deduct entry commission from capital
                    
                    info['in_position'] = True
                    info['position'] = {
                        'entry_price': current_price,
                        'entry_time': current_date,
                        'contracts': num_contracts
                    }
                    
                    # Enhanced logging
                    target_dollar = portfolio_capital * allocation_per_instrument * risk_multiplier
                    logger.info(f"{symbol} ENTRY at {current_date} | Capital: ${portfolio_capital:,.0f} | Target: ${target_dollar:,.0f}")
                    logger.info(f"{symbol} ENTRY | Price: {current_price:.2f} | Contracts: {num_contracts} | Williams %R: {current_wr:.2f}")
        
        # Calculate mark-to-market equity (realized capital + unrealized P&L)
        total_unrealized = 0.0
        for symbol, info in instrument_data.items():
            if info['in_position']:
                data = info['data']
                current_row = data[data['Time'] == current_date]
                if not current_row.empty:
                    current_price = current_row.iloc[0]['Last']
                    position = info['position']
                    unrealized_pnl = (current_price - position['entry_price']) * info['multiplier'] * position['contracts']
                    total_unrealized += unrealized_pnl
        
        portfolio_equity = portfolio_capital + total_unrealized
        equity_curve.append((current_date, portfolio_equity))
    
    # Close any remaining positions at the end (same as willr_bt.py)
    for symbol, info in instrument_data.items():
        if info['in_position']:
            data = info['data']
            last_row = data.iloc[-1]
            position = info['position']
            
            exit_price = last_row['Last']
            profit = (exit_price - position['entry_price']) * info['multiplier'] * position['contracts']
            profit -= commission_per_order * position['contracts']
            
            trade_results.append({
                'symbol': symbol,
                'entry_time': position['entry_time'],
                'exit_time': last_row['Time'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'profit': profit,
                'contracts': position['contracts']
            })
            
            portfolio_capital += profit
            logger.info(f"{symbol} FINAL EXIT | Price: {exit_price:.2f} | Contracts: {position['contracts']} | Profit: ${profit:.2f}")
    
    # Update final equity curve point
    if equity_curve:
        equity_curve[-1] = (equity_curve[-1][0], portfolio_capital)
    
    # Convert equity curve to DataFrame
    equity_df = pd.DataFrame(equity_curve, columns=['Time', 'Equity'])
    equity_df.set_index('Time', inplace=True)
    
    # -------------------------------
    # Performance Metrics Calculation (matching willr_bt.py)
    # -------------------------------
    final_balance = portfolio_capital
    total_return_pct = ((final_balance / initial_capital) - 1) * 100
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    years = (end_dt - start_dt).days / 365.25
    annualized_return_pct = ((final_balance / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan
    
    equity_df['returns'] = equity_df['Equity'].pct_change()
    volatility_annual = equity_df['returns'].std() * np.sqrt(252) * 100
    
    # Calculate Sharpe Ratio
    sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252)
                    if equity_df['returns'].std() != 0 else np.nan)
    
    # Calculate maximum drawdown (matching willr_bt.py)
    equity_df['EquityPeak'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['EquityPeak']) / equity_df['EquityPeak']
    max_drawdown = equity_df['Drawdown'].min() * 100
    
    # Trade statistics by instrument
    trade_stats = {}
    for symbol in portfolio_instruments.keys():
        symbol_trades = [t for t in trade_results if t['symbol'] == symbol]
        if symbol_trades:
            profits = [t['profit'] for t in symbol_trades]
            winning_trades = [p for p in profits if p > 0]
            trade_stats[symbol] = {
                'trades': len(symbol_trades),
                'total_profit': sum(profits),
                'avg_profit': np.mean(profits),
                'win_rate': len(winning_trades) / len(profits) * 100 if profits else 0
            }
    
    # Overall trade statistics
    all_profits = [t['profit'] for t in trade_results]
    winning_trades = [p for p in all_profits if p > 0]
    losing_trades = [p for p in all_profits if p <= 0]
    
    win_rate = (len(winning_trades) / len(all_profits) * 100) if all_profits else 0
    profit_factor = (sum(winning_trades) / abs(sum(losing_trades))
                     if losing_trades and sum(losing_trades) != 0 else np.nan)
    
    performance = {
        'Initial Capital': initial_capital,
        'Final Balance': final_balance,
        'Total Return (%)': total_return_pct,
        'Annualized Return (%)': annualized_return_pct,
        'Volatility (Annual %)': volatility_annual,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Total Trades': len(trade_results),
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor,
        'Trade Stats': trade_stats,
        'Equity Curve': equity_df
    }
    
    return performance

# -------------------------------
# Run Portfolio Backtest
# -------------------------------
logger.info(f"Running Williams %R Portfolio Backtest")
logger.info(f"Risk Multiplier: {risk_multiplier}x | Allocation per instrument: {allocation_per_instrument*100:.0f}%")
logger.info(f"Date Range: {start_date} to {end_date}")
logger.info(f"Portfolio: {list(portfolio_instruments.keys())}")

results = run_portfolio_williams_backtest()

if results:
    print(f"\n{'='*80}")
    print("WILLIAMS %R PORTFOLIO BACKTEST RESULTS")
    print(f"Risk Multiplier: {risk_multiplier}x | 25% allocation per instrument")
    print(f"{'='*80}")
    
    # Strategy overview
    print(f"\n📊 STRATEGY OVERVIEW")
    print("-" * 60)
    print(f"Williams %R Portfolio Strategy with Enhanced Risk:")
    print(f"  • Risk Multiplier: {risk_multiplier}x (LARGER POSITION SIZES)")
    print(f"  • Allocation per instrument: {allocation_per_instrument*100:.0f}%")
    print(f"  • Dynamic position sizing scales with portfolio equity")
    print(f"  • Portfolio of 4 instruments: {list(portfolio_instruments.keys())}")
    
    # Overall performance
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("-" * 60)
    print(f"Initial Capital:        ${results['Initial Capital']:,.2f}")
    print(f"Final Balance:          ${results['Final Balance']:,.2f}")
    print(f"Total Return:           {results['Total Return (%)']:,.2f}%")
    print(f"Annualized Return:      {results['Annualized Return (%)']:,.2f}%")
    print(f"Volatility (Annual):    {results['Volatility (Annual %)']:,.2f}%")
    print(f"Sharpe Ratio:           {results['Sharpe Ratio']:,.3f}")
    print(f"Max Drawdown:           {results['Max Drawdown (%)']:,.2f}%")
    print(f"Total Trades:           {results['Total Trades']}")
    print(f"Overall Win Rate:       {results['Win Rate (%)']:,.1f}%")
    print(f"Profit Factor:          {results['Profit Factor']:,.2f}" if not np.isnan(results['Profit Factor']) else "Profit Factor:          N/A")
    
    # Per-instrument statistics
    print(f"\n{'='*60}")
    print("PER-INSTRUMENT STATISTICS:")
    print(f"{'Symbol':<8} {'Trades':<8} {'Total P&L':<12} {'Avg P&L':<10} {'Win Rate':<10}")
    print("-" * 60)
    
    for symbol, stats in results['Trade Stats'].items():
        print(f"{symbol:<8} {stats['trades']:<8} ${stats['total_profit']:<11,.0f} ${stats['avg_profit']:<9,.0f} {stats['win_rate']:<9.1f}%")
    
    # Portfolio summary
    print(f"\n{'='*60}")
    print("PORTFOLIO SUMMARY:")
    total_profit = sum([stats['total_profit'] for stats in results['Trade Stats'].values()])
    total_trades = sum([stats['trades'] for stats in results['Trade Stats'].values()])
    
    print(f"Total Portfolio P&L:    ${total_profit:,.0f}")
    print(f"Profit per Trade:       ${total_profit/total_trades:.0f}" if total_trades > 0 else "Profit per Trade:       N/A")
    
    # Plot equity curve
    print(f"\n📊 Generating equity curve plot...")
    
    equity_df = results['Equity Curve']
    plt.figure(figsize=(14, 8))
    plt.plot(equity_df.index, equity_df['Equity'], label='Williams %R Portfolio Strategy', color='steelblue', linewidth=2)
    
    plt.title(f'Williams %R Portfolio Strategy Performance ({risk_multiplier}x Risk Multiplier)\n'
              f'Dynamic Position Sizing - 25% Allocation per Instrument')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    
    # Format y-axis to show dollar amounts clearly
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add horizontal grid lines
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
else:
    print("Portfolio backtest failed - check data files and date range.")