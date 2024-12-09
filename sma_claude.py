from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MovingAverageStrategy:
    def __init__(self, fast_period=20, slow_period=50, take_profit=5, stop_loss=3):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.position = 0
        self.position_price = 0
        self.trades = []

    def calculate_signals(self, data):
        data['FastMA'] = data['Close'].rolling(window=self.fast_period).mean()
        data['SlowMA'] = data['Close'].rolling(window=self.slow_period).mean()
        data['Signal'] = np.where(data['FastMA'] > data['SlowMA'], 1, -1)
        return data

    def backtest(self, data):
        data = self.calculate_signals(data)
        
        for i in range(len(data)):
            if self.position == 0:
                if data['Signal'].iloc[i] == 1:
                    self.position = 1
                    self.position_price = data['Close'].iloc[i]
                    self.trades.append(('BUY', data.index[i], self.position_price))
                elif data['Signal'].iloc[i] == -1:
                    self.position = -1
                    self.position_price = data['Close'].iloc[i]
                    self.trades.append(('SELL', data.index[i], self.position_price))
            elif self.position == 1:
                if data['Signal'].iloc[i] == -1 or \
                   data['Close'].iloc[i] >= self.position_price + self.take_profit or \
                   data['Close'].iloc[i] <= self.position_price - self.stop_loss:
                    self.trades.append(('SELL', data.index[i], data['Close'].iloc[i]))
                    self.position = 0
            elif self.position == -1:
                if data['Signal'].iloc[i] == 1 or \
                   data['Close'].iloc[i] <= self.position_price - self.take_profit or \
                   data['Close'].iloc[i] >= self.position_price + self.stop_loss:
                    self.trades.append(('BUY', data.index[i], data['Close'].iloc[i]))
                    self.position = 0

        return pd.DataFrame(self.trades, columns=['Action', 'Date', 'Price'])

def get_historical_data(ib, contract, duration):
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting='15 mins',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    df = util.df(bars)
    df.set_index('date', inplace=True)
    return df

def calculate_metrics(trades):
    if len(trades) < 2:
        return "Not enough trades to calculate metrics"

    total_trades = len(trades) // 2
    pnl = []
    for i in range(0, len(trades), 2):
        if i + 1 < len(trades):
            if trades['Action'].iloc[i] == 'BUY':
                pnl.append(trades['Price'].iloc[i+1] - trades['Price'].iloc[i])
            else:
                pnl.append(trades['Price'].iloc[i] - trades['Price'].iloc[i+1])

    winners = sum(1 for p in pnl if p > 0)
    losers = sum(1 for p in pnl if p < 0)
    avg_winner = np.mean([p for p in pnl if p > 0]) if winners > 0 else 0
    avg_loser = np.mean([p for p in pnl if p < 0]) if losers > 0 else 0
    win_rate = winners / total_trades if total_trades > 0 else 0
    profit_factor = abs(sum(p for p in pnl if p > 0)) / abs(sum(p for p in pnl if p < 0)) if sum(p for p in pnl if p < 0) != 0 else float('inf')
    
    cumulative_pnl = np.cumsum(pnl)
    max_drawdown = np.max(np.maximum.accumulate(cumulative_pnl) - cumulative_pnl)
    
    total_return = sum(pnl)
    
    return {
        'Total Trades': total_trades,
        'Winners': winners,
        'Losers': losers,
        'Average Winner': avg_winner,
        'Average Loser': avg_loser,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Max Drawdown': max_drawdown,
        'Total Return': total_return
    }

# Main execution
if __name__ == "__main__":
    # Connect to Interactive Brokers TWS or Gateway
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)

    # Define the MES futures contract
    mes_contract = Future(
        symbol='MES',
        exchange='CME',
        currency='USD',
        lastTradeDateOrContractMonth='202409'
    )

    # Qualify the contract
    ib.qualifyContracts(mes_contract)

    # Get historical data (e.g., last 30 days)
    data = get_historical_data(ib, mes_contract, '30 D')

    # Run backtest
    strategy = MovingAverageStrategy()
    trades = strategy.backtest(data)

    # Calculate and print results
    results = calculate_metrics(trades)
    for key, value in results.items():
        print(f"{key}: {value}")

    # Disconnect from IB
    ib.disconnect()