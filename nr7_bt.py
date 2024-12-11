from ib_insync import *
import pandas as pd
import numpy as np

# Connect to IBKR TWS or Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define the ES futures contract (E-mini S&P 500)
es_contract = Future(
    symbol='ES',
    exchange='CME',
    currency='USD',
    lastTradeDateOrContractMonth='202412'
)

# Qualify the contract
ib.qualifyContracts(es_contract)

# Retrieve historical daily data for NR7 calculation
daily_bars = ib.reqHistoricalData(
    es_contract,
    endDateTime='20241209 23:59:59',
    durationStr='7 M',
    barSizeSetting='1 day',
    whatToShow='TRADES',
    useRTH=False,
    formatDate=1
)

# Convert to DataFrame
df_daily = util.df(daily_bars)

# Ensure date format and indexing
df_daily['date'] = pd.to_datetime(df_daily['date'])
df_daily.set_index('date', inplace=True)

# Extract correct start and end dates
start_date = df_daily.index.min()
end_date = df_daily.index.max()

# Identify NR7 Days
df_daily['Range'] = df_daily['high'] - df_daily['low']
df_daily['NR7'] = df_daily['Range'].rolling(7).min() == df_daily['Range']

# Backtesting Parameters
initial_cash = 5000
cash = initial_cash
ES_MULTIPLIER = 50  # 1 ES point = $50 profit/loss per contract
stop_loss_points = 10  # Stop Loss in points
take_profit_points = 30  # Take Profit in points
position_size = 1  # Always trade 1 contract for simplicity
trade_results = []
equity_curve = []

# Iterate through NR7 days
for nr7_date in df_daily[df_daily['NR7']].index:
    next_trading_day = nr7_date + pd.Timedelta(days=1)
    
    # Request 5-min data for the next trading day
    intraday_bars = ib.reqHistoricalData(
        es_contract,
        endDateTime=f'{next_trading_day.strftime("%Y%m%d")} 23:59:59',
        durationStr='1 D',
        barSizeSetting='5 mins',
        whatToShow='TRADES',
        useRTH=False,
        formatDate=1
    )
    
    # Check if data is None before converting to DataFrame
    if not intraday_bars:
        print(f"No data returned for {next_trading_day.strftime('%Y-%m-%d')}")
        continue

    # Convert to DataFrame
    df_intraday = util.df(intraday_bars)

    # Ensure datetime format
    df_intraday['date'] = pd.to_datetime(df_intraday['date'])
    df_intraday.set_index('date', inplace=True)

    # Calculate 5-bar volume average
    df_intraday['5_bar_vol_avg'] = df_intraday['volume'].rolling(window=5).mean()

    prev_high = df_daily.loc[nr7_date, 'high']
    prev_low = df_daily.loc[nr7_date, 'low']

    # Check for breakouts
    entered_trade = False
    for i, bar in df_intraday.iterrows():
        if bar['volume'] > bar['5_bar_vol_avg']:
            if bar['high'] > prev_high and not entered_trade:
                # Long entry
                entry_price = bar['close']
                stop_loss_price = entry_price - stop_loss_points
                take_profit_price = entry_price + take_profit_points
                entered_trade = True
                direction = "long"
                print(f"[ENTRY] Long entered at {entry_price} on {i}")

            elif bar['low'] < prev_low and not entered_trade:
                # Short entry
                entry_price = bar['close']
                stop_loss_price = entry_price + stop_loss_points
                take_profit_price = entry_price - take_profit_points
                entered_trade = True
                direction = "short"
                print(f"[ENTRY] Short entered at {entry_price} on {i}")
        
        if entered_trade:
            # Check Stop Loss and Take Profit
            if direction == "long":
                if bar['low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    pnl = (exit_price - entry_price) * position_size * ES_MULTIPLIER
                    cash += pnl
                    trade_results.append(pnl)
                    print(f"[STOP LOSS] Long exit at {exit_price} on {i}, PnL: ${pnl:,.2f}")
                    entered_trade = False
                    break

                elif bar['high'] >= take_profit_price:
                    exit_price = take_profit_price
                    pnl = (exit_price - entry_price) * position_size * ES_MULTIPLIER
                    cash += pnl
                    trade_results.append(pnl)
                    print(f"[TAKE PROFIT] Long exit at {exit_price} on {i}, PnL: ${pnl:,.2f}")
                    entered_trade = False
                    break

            elif direction == "short":
                if bar['high'] >= stop_loss_price:
                    exit_price = stop_loss_price
                    pnl = (entry_price - exit_price) * position_size * ES_MULTIPLIER
                    cash += pnl
                    trade_results.append(pnl)
                    print(f"[STOP LOSS] Short exit at {exit_price} on {i}, PnL: ${pnl:,.2f}")
                    entered_trade = False
                    break

                elif bar['low'] <= take_profit_price:
                    exit_price = take_profit_price
                    pnl = (entry_price - exit_price) * position_size * ES_MULTIPLIER
                    cash += pnl
                    trade_results.append(pnl)
                    print(f"[TAKE PROFIT] Short exit at {exit_price} on {i}, PnL: ${pnl:,.2f}")
                    entered_trade = False
                    break

        # Exit at the end of the trading day if neither stop nor take profit was hit
        if entered_trade and i == df_intraday.index[-1]:
            exit_price = bar['close']
            if direction == "long":
                pnl = (exit_price - entry_price) * position_size * ES_MULTIPLIER
            else:  # Short trade exit
                pnl = (entry_price - exit_price) * position_size * ES_MULTIPLIER

            cash += pnl
            trade_results.append(pnl)
            print(f"[END OF DAY] Exit {direction} trade at {exit_price} on {i}, PnL: ${pnl:,.2f}")
            entered_trade = False

# Calculate Metrics
equity_series = pd.Series([initial_cash] + [initial_cash + sum(trade_results[:i]) for i in range(1, len(trade_results)+1)])
total_return = (cash - initial_cash) / initial_cash * 100

# Performance Summary
results = {
    "Start Date": start_date.strftime("%Y-%m-%d"),
    "End Date": end_date.strftime("%Y-%m-%d"),
    "Total Trades": len(trade_results),
    "Final Account Balance": f"${cash:,.2f}",
    "Total Return": f"{total_return:.2f}%",
    "Winning Trades": sum(1 for tr in trade_results if tr > 0),
    "Losing Trades": sum(1 for tr in trade_results if tr < 0),
    "Win Rate": f"{(sum(1 for tr in trade_results if tr > 0) / len(trade_results) * 100):.2f}%" if trade_results else "0.00%",
    "Profit Factor": f"{sum(tr for tr in trade_results if tr > 0) / abs(sum(tr for tr in trade_results if tr < 0)):.2f}" if sum(tr for tr in trade_results if tr < 0) != 0 else "Inf"
}

# Print Results
print("\nPerformance Summary:")
for key, value in results.items():
    print(f"{key}: {value}")

# Disconnect from IB
ib.disconnect()