import pandas as pd
import numpy as np
import random
import os

# Load data function with correct columns
def load_data(timeframe):
    file_name = f'es_{timeframe}_data.csv'
    if os.path.exists(file_name):
        data = pd.read_csv(file_name, parse_dates=['date'])
        data.rename(columns={'date': 'datetime'}, inplace=True)
        data['returns'] = data['close'].pct_change().fillna(0)
        return data[['datetime', 'open', 'high', 'low', 'close', 'volume', 'returns']]
    else:
        raise FileNotFoundError(f"File {file_name} not found.")

# Generate a random chromosome
def generate_chromosome():
    return {
        'short_ma': random.randint(5, 50),
        'long_ma': random.randint(51, 200),
        'rsi_period': random.randint(5, 30),
        'rsi_overbought': random.randint(70, 90),
        'rsi_oversold': random.randint(10, 30),
        'stop_loss': round(random.uniform(0.01, 0.05), 4),
        'take_profit': round(random.uniform(0.01, 0.10), 4)
    }

# Backtest a strategy
def backtest_strategy(data, chromosome, multiplier=5):
    data['short_ma'] = data['close'].rolling(chromosome['short_ma']).mean()
    data['long_ma'] = data['close'].rolling(chromosome['long_ma']).mean()
    data['rsi'] = 100 - (100 / (1 + data['returns'].rolling(chromosome['rsi_period']).mean()))

    data['signal'] = np.where(
        (data['short_ma'] > data['long_ma']) & (data['rsi'] < chromosome['rsi_oversold']), 1,
        np.where(
            (data['short_ma'] < data['long_ma']) & (data['rsi'] > chromosome['rsi_overbought']), -1, 0
        )
    )

    # Simulate PnL with $5 per point movement
    data['pnl'] = data['signal'].shift(1) * (data['close'].diff()) * multiplier
    
    total_return = data['pnl'].sum()
    initial_balance = 5000  # Example starting capital

    # Calculate performance metrics
    total_return_pct = (total_return / initial_balance) * 100
    cumulative_pnl = data['pnl'].cumsum() + initial_balance
    max_drawdown = ((cumulative_pnl.cummax() - cumulative_pnl).max() / initial_balance) * 100

    sharpe_ratio = data['pnl'].mean() / data['pnl'].std() if data['pnl'].std() != 0 else 0

    return {
        'total_return_pct': total_return_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown
    }

# Main execution
if __name__ == "__main__":
    timeframe = '1h'  # Change this to your desired timeframe
    try:
        es_data = load_data(timeframe)
        population_size = 10
        generations = 50
        
        # Initialize population
        population = [generate_chromosome() for _ in range(population_size)]

        for generation in range(generations):
            # Evaluate population
            fitness_scores = [backtest_strategy(es_data.copy(), chromo) for chromo in population]

            # Combine for evaluation
            results = pd.DataFrame([
                {**population[i], **fitness_scores[i]} for i in range(population_size)
            ])
            print(f"Generation {generation + 1}")
            print(results.sort_values(by='total_return_pct', ascending=False).head(3))

            # Selection: Top 2 strategies
            top_strategies = results.nlargest(2, 'total_return_pct').to_dict('records')

            # Crossover & Mutation
            population = []
            for _ in range(population_size // 2):
                parent1, parent2 = random.sample(top_strategies, 2)
                child1, child2 = parent1.copy(), parent2.copy()

                # Crossover
                crossover_point = random.choice(list(child1.keys()))
                child1[crossover_point], child2[crossover_point] = child2[crossover_point], child1[crossover_point]

                # Mutation
                if random.random() < 0.2:  # 20% mutation chance
                    child1 = generate_chromosome()
                if random.random() < 0.2:
                    child2 = generate_chromosome()

                population.extend([child1, child2])

    except FileNotFoundError as e:
        print(e)