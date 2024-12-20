import pandas as pd
import numpy as np
import random
import os
import multiprocessing as mp
from functools import partial

# Define the list of timeframes you want to test
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', 'daily']

# Load data function with correct columns
def load_data(timeframes):
    data_dict = {}
    for timeframe in timeframes:
        file_name = f'es_{timeframe}_data.csv'
        if os.path.exists(file_name):
            data = pd.read_csv(file_name, parse_dates=['date'])
            data.rename(columns={'date': 'datetime'}, inplace=True)
            data['returns'] = data['close'].pct_change().fillna(0)
            data_dict[timeframe] = data[['datetime', 'open', 'high', 'low', 'close', 'volume', 'returns']]
        else:
            raise FileNotFoundError(f"File {file_name} not found.")
    return data_dict

# Generate a random chromosome ensuring logical parameter constraints
def generate_chromosome():
    short_ma = random.randint(5, 50)
    long_ma = random.randint(short_ma + 1, 200)  # Ensure long_ma > short_ma
    return {
        'short_ma': short_ma,
        'long_ma': long_ma,
        'rsi_period': random.randint(5, 30),
        'rsi_overbought': random.randint(70, 90),
        'rsi_oversold': random.randint(10, 30),
        'stop_loss': round(random.uniform(0.01, 0.05), 4),
        'take_profit': round(random.uniform(0.01, 0.10), 4)
    }

# Backtest a strategy across multiple timeframes
def backtest_strategy_multitimeframe(data_dict, chromosome, multiplier=5, initial_balance=5000):
    """
    Backtest a single strategy across multiple timeframes.

    Parameters:
    - data_dict: Dictionary of DataFrames for each timeframe.
    - chromosome: Strategy parameters.
    - multiplier: PnL multiplier per point movement.
    - initial_balance: Starting capital.

    Returns:
    - Aggregated performance metrics across all timeframes.
    """
    aggregated_metrics = {
        'total_return_pct': 0,
        'sharpe_ratio': 0,
        'max_drawdown_pct': 0
    }
    num_timeframes = len(data_dict)

    for timeframe, data in data_dict.items():
        df = data.copy()

        # Calculate moving averages
        df['short_ma'] = df['close'].rolling(window=chromosome['short_ma'], min_periods=1).mean()
        df['long_ma'] = df['close'].rolling(window=chromosome['long_ma'], min_periods=1).mean()

        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=chromosome['rsi_period'], min_periods=1).mean()
        avg_loss = loss.rolling(window=chromosome['rsi_period'], min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(100)  # Handle division by zero if avg_loss is 0

        # Generate signals
        buy_condition = (df['short_ma'] > df['long_ma']) & (df['rsi'] < chromosome['rsi_oversold'])
        sell_condition = (df['short_ma'] < df['long_ma']) & (df['rsi'] > chromosome['rsi_overbought'])
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1

        # Implement stop loss and take profit
        position = 0
        entry_price = 0
        pnl = []
        for idx, row in df.iterrows():
            signal = row['signal']
            price_change = row['close'] - entry_price if position != 0 else 0

            # Check for stop loss or take profit
            if position != 0:
                if (position == 1 and (price_change <= -chromosome['stop_loss'] or price_change >= chromosome['take_profit'])) or \
                   (position == -1 and (-price_change <= -chromosome['stop_loss'] or -price_change >= chromosome['take_profit'])):
                    pnl.append(price_change * multiplier * position)
                    position = 0
                    entry_price = 0
                    continue

            if signal == 1 and position == 0:
                position = 1
                entry_price = row['close']
            elif signal == -1 and position == 0:
                position = -1
                entry_price = row['close']
            pnl.append(0)

        df['pnl'] = pnl
        df['cumulative_pnl'] = df['pnl'].cumsum() + initial_balance

        total_return = df['pnl'].sum()
        total_return_pct = (total_return / initial_balance) * 100
        max_drawdown = ((df['cumulative_pnl'].cummax() - df['cumulative_pnl']).max() / initial_balance) * 100
        sharpe_ratio = (df['pnl'].mean() / df['pnl'].std()) * np.sqrt(252) if df['pnl'].std() != 0 else 0

        # Aggregate metrics
        aggregated_metrics['total_return_pct'] += total_return_pct
        aggregated_metrics['sharpe_ratio'] += sharpe_ratio
        aggregated_metrics['max_drawdown_pct'] += max_drawdown

    # Average metrics across timeframes
    aggregated_metrics = {k: v / num_timeframes for k, v in aggregated_metrics.items()}

    return aggregated_metrics

# Tournament selection based on prioritized metrics
def tournament_selection(population, fitness_scores, tournament_size=3):
    selected = []
    population_fitness = list(zip(population, fitness_scores))
    for _ in range(len(population)):
        participants = random.sample(population_fitness, tournament_size)
        winner = max(participants, key=lambda x: (
            x[1]['sharpe_ratio'],
            x[1]['total_return_pct'],
            -x[1]['max_drawdown_pct']
        ))
        selected.append(winner[0])
    return selected

# Crossover two parents to produce two children
def crossover(parent1, parent2):
    child1, child2 = parent1.copy(), parent2.copy()
    for key in parent1.keys():
        if random.random() < 0.5:
            child1[key], child2[key] = child2[key], child1[key]
    return child1, child2

# Mutate a chromosome
def mutate(chromosome, mutation_rate=0.1):
    if random.random() < mutation_rate:
        chromosome['short_ma'] = random.randint(5, 50)
    if random.random() < mutation_rate:
        chromosome['long_ma'] = random.randint(chromosome['short_ma'] + 1, 200)
    if random.random() < mutation_rate:
        chromosome['rsi_period'] = random.randint(5, 30)
    if random.random() < mutation_rate:
        chromosome['rsi_overbought'] = random.randint(70, 90)
    if random.random() < mutation_rate:
        chromosome['rsi_oversold'] = random.randint(10, 30)
    if random.random() < mutation_rate:
        chromosome['stop_loss'] = round(random.uniform(0.01, 0.05), 4)
    if random.random() < mutation_rate:
        chromosome['take_profit'] = round(random.uniform(0.01, 0.10), 4)
    return chromosome

# Parallel backtesting
def parallel_backtest_multitimeframe(data_dict, population, multiplier=5, initial_balance=5000, n_jobs=mp.cpu_count()):
    with mp.Pool(processes=n_jobs) as pool:
        func = partial(backtest_strategy_multitimeframe, data_dict, multiplier=multiplier, initial_balance=initial_balance)
        fitness_scores = pool.map(func, population)
    return fitness_scores

# Main execution
if __name__ == "__main__":
    try:
        # Load data for all specified timeframes
        data_dict = load_data(TIMEFRAMES)
        population_size = 1000  # Increased population size
        generations = 100  # Increased number of generations
        elitism_size = 50  # Number of top strategies to carry over
        mutation_rate = 0.2  # Mutation probability per gene

        # Initialize population
        population = [generate_chromosome() for _ in range(population_size)]

        # Track best strategies
        best_strategies = []

        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")

            # Evaluate population in parallel across multiple timeframes
            fitness_scores = parallel_backtest_multitimeframe(data_dict, population, n_jobs=mp.cpu_count())

            # Combine for evaluation
            results = pd.DataFrame([
                {**population[i], **fitness_scores[i]} for i in range(population_size)
            ])

            # Sort by prioritized metrics: Sharpe Ratio (desc), Total Return (desc), Max Drawdown (asc)
            results_sorted = results.sort_values(
                by=['sharpe_ratio', 'total_return_pct', 'max_drawdown_pct'],
                ascending=[False, False, True]
            ).reset_index(drop=True)

            # Save top strategies from this generation
            top_generation = results_sorted.head(10)
            best_strategies.append(top_generation)

            print(top_generation[['sharpe_ratio', 'total_return_pct', 'max_drawdown_pct']].head(5))

            # Elitism: carry forward top strategies
            elites = results_sorted.head(elitism_size)[[
                'short_ma', 'long_ma', 'rsi_period', 'rsi_overbought', 'rsi_oversold', 'stop_loss', 'take_profit'
            ]].to_dict('records')

            # Selection: Tournament selection based on prioritized metrics
            selected = tournament_selection(population, fitness_scores)

            # Crossover & Mutation to create new population
            next_population = elites.copy()
            while len(next_population) < population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)
                next_population.extend([child1, child2])

            # Ensure population size
            population = next_population[:population_size]

        # After all generations, compile the best strategies
        all_best = pd.concat(best_strategies).sort_values(
            by=['sharpe_ratio', 'total_return_pct', 'max_drawdown_pct'],
            ascending=[False, False, True]
        ).drop_duplicates().reset_index(drop=True)
        top_overall = all_best.head(100)  # Top 100 strategies

        print("\nTop 100 Strategies:")
        print(top_overall[['short_ma', 'long_ma', 'rsi_period', 'rsi_overbought',
                           'rsi_oversold', 'stop_loss', 'take_profit',
                           'sharpe_ratio', 'total_return_pct', 'max_drawdown_pct']])

        # Optionally, save the best strategies to a CSV file
        top_overall.to_csv('best_trading_strategies_multitimeframe.csv', index=False)
        print("Best strategies saved to 'best_trading_strategies_multitimeframe.csv'")

    except FileNotFoundError as e:
        print(e)