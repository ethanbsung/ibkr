#!/usr/bin/env python3
"""
Bollinger Band Spread Trading Strategy - Out-of-Sample Testing
ES vs NQ Futures

Manual parameter testing for out-of-sample validation.
Easily modify parameters below to test different combinations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BollingerSpreadOOSTest:
    """
    Bollinger Band spread trading strategy for out-of-sample testing
    """
    
    def __init__(self, es_file='Data/es_1h_data.csv', 
                 nq_file='Data/nq_1h_data.csv',
                 lookback=20, std_mult=2.0, commission=5.0,
                 max_hold_hours=48):
        """
        Initialize the backtest parameters for OOS testing
        """
        self.es_file = es_file
        self.nq_file = nq_file
        self.lookback = lookback
        self.std_mult = std_mult
        self.commission = commission
        self.max_hold_hours = max_hold_hours
        
        # Data containers
        self.data = None
        self.trades = []
        self.performance_metrics = {}
        
    def load_and_preprocess_data(self, start_date=None, end_date=None):
        """
        Load and preprocess hourly ES and NQ data with optional date filtering
        """
        print("Loading and preprocessing hourly data...")
        
        # Load ES data
        es_data = pd.read_csv(self.es_file)
        es_data['Time'] = pd.to_datetime(es_data['Time'])
        es_data = es_data[['Time', 'Last']].rename(columns={'Last': 'ES_Close'})
        es_data.set_index('Time', inplace=True)
        
        # Load NQ data  
        nq_data = pd.read_csv(self.nq_file)
        nq_data['Time'] = pd.to_datetime(nq_data['Time'])
        nq_data = nq_data[['Time', 'Last']].rename(columns={'Last': 'NQ_Close'})
        nq_data.set_index('Time', inplace=True)
        
        # Merge data on timestamps
        self.data = pd.merge(es_data, nq_data, left_index=True, right_index=True, how='inner')
        self.data.dropna(inplace=True)
        
        # Apply date filtering if specified
        if start_date:
            self.data = self.data[self.data.index >= pd.to_datetime(start_date)]
        if end_date:
            self.data = self.data[self.data.index <= pd.to_datetime(end_date)]
        
        print(f"Hourly data loaded: {len(self.data)} observations")
        print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        
        return self.data
    
    def calculate_spread(self):
        """Calculate the log spread"""
        print("Calculating spread...")
        self.data['ES_Log'] = np.log(self.data['ES_Close'])
        self.data['NQ_Log'] = np.log(self.data['NQ_Close'])
        self.data['Spread'] = self.data['ES_Log'] - self.data['NQ_Log']
        print(f"Spread calculated. Mean: {self.data['Spread'].mean():.4f}, Std: {self.data['Spread'].std():.4f}")
        
    def calculate_bollinger_bands(self):
        """Calculate Bollinger Bands"""
        print("Calculating Bollinger Bands...")
        self.data['Spread_Mean'] = self.data['Spread'].rolling(window=self.lookback).mean()
        self.data['Spread_Std'] = self.data['Spread'].rolling(window=self.lookback).std()
        self.data['Upper_Band'] = self.data['Spread_Mean'] + (self.std_mult * self.data['Spread_Std'])
        self.data['Lower_Band'] = self.data['Spread_Mean'] - (self.std_mult * self.data['Spread_Std'])
        
    def generate_signals(self):
        """Generate trading signals"""
        print("Generating trading signals...")
        self.data['Signal'] = 0
        self.data['Position'] = 0
        
        position = 0
        entry_time = None
        
        for i in range(self.lookback, len(self.data)):
            current_spread = self.data.iloc[i]['Spread']
            lower_band = self.data.iloc[i]['Lower_Band']
            upper_band = self.data.iloc[i]['Upper_Band']
            spread_mean = self.data.iloc[i]['Spread_Mean']
            current_time = self.data.index[i]
            
            # Check for maximum hold time
            if position != 0 and entry_time is not None:
                hours_held = (current_time - entry_time).total_seconds() / 3600
                if hours_held >= self.max_hold_hours:
                    self.data.iloc[i, self.data.columns.get_loc('Signal')] = 0
                    position = 0
                    entry_time = None
                    continue
            
            # Entry signals
            if position == 0:
                if current_spread < lower_band:
                    position = 1
                    entry_time = current_time
                    self.data.iloc[i, self.data.columns.get_loc('Signal')] = 1
                elif current_spread > upper_band:
                    position = -1
                    entry_time = current_time
                    self.data.iloc[i, self.data.columns.get_loc('Signal')] = -1
            
            # Exit signals
            elif position != 0:
                prev_spread = self.data.iloc[i-1]['Spread']
                if ((position == 1 and current_spread >= spread_mean and prev_spread < spread_mean) or
                    (position == -1 and current_spread <= spread_mean and prev_spread > spread_mean)):
                    self.data.iloc[i, self.data.columns.get_loc('Signal')] = 0
                    position = 0
                    entry_time = None
            
            self.data.iloc[i, self.data.columns.get_loc('Position')] = position
        
        long_entries = (self.data['Signal'] == 1).sum()
        short_entries = (self.data['Signal'] == -1).sum()
        exits = (self.data['Signal'] == 0).sum()
        print(f"Signals generated - Long: {long_entries}, Short: {short_entries}, Exits: {exits}")
    
    def calculate_pnl(self):
        """Calculate P&L"""
        print("Calculating P&L...")
        self.data['Trade_PnL'] = 0.0
        self.data['Cumulative_PnL'] = 0.0
        
        trades = []
        in_trade = False
        entry_idx = None
        trade_direction = 0
        
        for i in range(len(self.data)):
            signal = self.data.iloc[i]['Signal']
            es_price = self.data.iloc[i]['ES_Close']
            nq_price = self.data.iloc[i]['NQ_Close']
            timestamp = self.data.index[i]
            
            if signal in [1, -1] and not in_trade:
                in_trade = True
                entry_idx = i
                trade_direction = signal
                
            elif signal == 0 and in_trade:
                entry_es = self.data.iloc[entry_idx]['ES_Close']
                entry_nq = self.data.iloc[entry_idx]['NQ_Close']
                entry_time = self.data.index[entry_idx]
                
                es_multiplier = 50
                nq_multiplier = 20
                
                if trade_direction == 1:  # Long spread
                    es_pnl = es_multiplier * (es_price - entry_es)
                    nq_pnl = -nq_multiplier * (nq_price - entry_nq)
                else:  # Short spread
                    es_pnl = -es_multiplier * (es_price - entry_es)
                    nq_pnl = nq_multiplier * (nq_price - entry_nq)
                
                gross_pnl = es_pnl + nq_pnl
                net_pnl = gross_pnl - (2 * self.commission)
                duration_hours = (timestamp - entry_time).total_seconds() / 3600
                
                self.data.iloc[i, self.data.columns.get_loc('Trade_PnL')] = net_pnl
                
                trade = {
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'direction': 'Long Spread' if trade_direction == 1 else 'Short Spread',
                    'entry_es': entry_es,
                    'exit_es': es_price,
                    'entry_nq': entry_nq,
                    'exit_nq': nq_price,
                    'net_pnl': net_pnl,
                    'duration_hours': duration_hours
                }
                trades.append(trade)
                
                in_trade = False
                entry_idx = None
                trade_direction = 0
        
        self.data['Cumulative_PnL'] = self.data['Trade_PnL'].cumsum()
        self.trades = trades
        
        if trades:
            total_pnl = sum(trade['net_pnl'] for trade in trades)
            avg_duration = sum(trade['duration_hours'] for trade in trades) / len(trades)
            print(f"Total trades: {len(trades)}, Total P&L: ${total_pnl:,.2f}, Avg duration: {avg_duration:.1f}h")
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            self.performance_metrics = {}
            return
        
        trade_pnls = [trade['net_pnl'] for trade in self.trades]
        durations = [trade['duration_hours'] for trade in self.trades]
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        # Calculate Sharpe ratio
        returns = self.data['Trade_PnL'].copy()
        returns = returns[returns != 0]
        sharpe_ratio = 0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(24 * 252) * np.mean(returns) / np.std(returns)
        
        # Calculate max drawdown
        cumulative = self.data['Cumulative_PnL']
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        self.performance_metrics = {
            'Total Return ($)': sum(trade_pnls),
            'Number of Trades': len(trade_pnls),
            'Win Rate (%)': len(winning_trades) / len(trade_pnls) * 100 if trade_pnls else 0,
            'Average Trade ($)': np.mean(trade_pnls) if trade_pnls else 0,
            'Average Duration (hours)': np.mean(durations) if durations else 0,
            'Largest Winner ($)': max(trade_pnls) if trade_pnls else 0,
            'Largest Loser ($)': min(trade_pnls) if trade_pnls else 0,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown ($)': max_drawdown,
            'Profit Factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else np.inf
        }
        
        # Print metrics
        print("\nOUT-OF-SAMPLE PERFORMANCE METRICS")
        print("="*45)
        for key, value in self.performance_metrics.items():
            if '$' in key:
                print(f"{key:<25}: ${value:>10,.2f}")
            elif '%' in key or 'Rate' in key:
                print(f"{key:<25}: {value:>10.2f}%")
            elif 'Ratio' in key or 'Factor' in key:
                print(f"{key:<25}: {value:>10.3f}")
            elif 'hours' in key:
                print(f"{key:<25}: {value:>10.1f}")
            else:
                print(f"{key:<25}: {value:>10.0f}")
    
    def run_backtest(self, start_date=None, end_date=None):
        """
        Run the complete backtest
        """
        print("Starting OUT-OF-SAMPLE Bollinger Band Spread Trading Backtest")
        print("="*65)
        
        self.load_and_preprocess_data(start_date, end_date)
        self.calculate_spread()
        self.calculate_bollinger_bands()
        self.generate_signals()
        self.calculate_pnl()
        self.calculate_performance_metrics()
        
        print("\nOut-of-sample backtest completed!")
        return self.data, self.trades, self.performance_metrics

    def plot_results(self):
        """
        Create plots for backtest results
        """
        if self.data is None:
            print("No data to plot. Run backtest first.")
            return
        
        # Subsample data for plotting performance
        sample_freq = max(1, len(self.data) // 2000)
        plot_data = self.data.iloc[::sample_freq]
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Spread with Bollinger Bands
        ax1 = axes[0]
        ax1.plot(plot_data.index, plot_data['Spread'], label='Spread', linewidth=1, alpha=0.8)
        ax1.plot(plot_data.index, plot_data['Spread_Mean'], label='Mean', color='orange', linewidth=1)
        ax1.fill_between(plot_data.index, plot_data['Lower_Band'], plot_data['Upper_Band'], 
                        alpha=0.2, color='gray', label='Bollinger Bands')
        
        # Mark signals
        signal_data = self.data[self.data['Signal'] != 0]
        if len(signal_data) > 100:
            signal_sample = signal_data.iloc[::max(1, len(signal_data) // 100)]
        else:
            signal_sample = signal_data
            
        long_entries = signal_sample[signal_sample['Signal'] == 1]
        short_entries = signal_sample[signal_sample['Signal'] == -1]
        exits = signal_sample[signal_sample['Signal'] == 0]
        
        if not long_entries.empty:
            ax1.scatter(long_entries.index, long_entries['Spread'], color='green', 
                       marker='^', s=20, label='Long Entry', zorder=5)
        if not short_entries.empty:
            ax1.scatter(short_entries.index, short_entries['Spread'], color='red', 
                       marker='v', s=20, label='Short Entry', zorder=5)
        if not exits.empty:
            ax1.scatter(exits.index, exits['Spread'], color='blue', 
                       marker='x', s=15, label='Exit', zorder=5)
        
        ax1.set_title(f'OUT-OF-SAMPLE: ES-NQ Spread with Bollinger Bands\n(Lookback: {self.lookback}h, Std: {self.std_mult}, Max Hold: {self.max_hold_hours}h)')
        ax1.set_ylabel('Log Price Spread')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative P&L
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['Cumulative_PnL'], linewidth=1, color='blue')
        ax2.set_title('Cumulative P&L (Out-of-Sample)')
        ax2.set_ylabel('P&L ($)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function for out-of-sample testing
    
    *** MODIFY THESE PARAMETERS TO TEST DIFFERENT COMBINATIONS ***
    """
    
    # =============================================================================
    # PARAMETERS TO MODIFY FOR TESTING
    # =============================================================================
    
    # Strategy parameters (modify these based on optimization results)
    LOOKBACK = 20           # Bollinger Band lookback period in hours
    STD_MULTIPLIER = 2.0    # Standard deviation multiplier for bands
    MAX_HOLD_HOURS = 48     # Maximum position hold time in hours
    COMMISSION = 5.0        # Round-trip commission per contract
    
    # Date range for out-of-sample testing
    OOS_START_DATE = '2020-01-01'  # Start of out-of-sample period
    OOS_END_DATE = None            # End of OOS period (None = use all available data)
    
    # =============================================================================
    # END OF PARAMETERS TO MODIFY
    # =============================================================================
    
    print("Bollinger Band Spread Trading - OUT-OF-SAMPLE TESTING")
    print("="*60)
    print(f"Testing Parameters:")
    print(f"  Lookback: {LOOKBACK} hours")
    print(f"  Std Multiplier: {STD_MULTIPLIER}")
    print(f"  Max Hold Time: {MAX_HOLD_HOURS} hours")
    print(f"  Commission: ${COMMISSION} per round-trip")
    print(f"  OOS Period: {OOS_START_DATE} to {'present' if OOS_END_DATE is None else OOS_END_DATE}")
    print("="*60)
    
    # Initialize backtest with specified parameters
    oos_backtest = BollingerSpreadOOSTest(
        lookback=LOOKBACK,
        std_mult=STD_MULTIPLIER,
        max_hold_hours=MAX_HOLD_HOURS,
        commission=COMMISSION
    )
    
    # Run the out-of-sample backtest
    data, trades, metrics = oos_backtest.run_backtest(
        start_date=OOS_START_DATE,
        end_date=OOS_END_DATE
    )
    
    # Create plots
    oos_backtest.plot_results()
    
    # Display summary
    print(f"\nOUT-OF-SAMPLE SUMMARY:")
    print(f"Period: {OOS_START_DATE} to {'present' if OOS_END_DATE is None else OOS_END_DATE}")
    print(f"Parameters: Lookback={LOOKBACK}h, Std={STD_MULTIPLIER}, MaxHold={MAX_HOLD_HOURS}h")
    
    if metrics:
        print(f"Results:")
        print(f"  Total Return: ${metrics['Total Return ($)']:,.2f}")
        print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
        print(f"  Win Rate: {metrics['Win Rate (%)']:.1f}%")
        print(f"  Number of Trades: {metrics['Number of Trades']:.0f}")
        print(f"  Max Drawdown: ${metrics['Max Drawdown ($)']:,.2f}")
    
    return oos_backtest


if __name__ == "__main__":
    backtest = main() 