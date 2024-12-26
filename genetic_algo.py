import pandas as pd
import numpy as np
import random
import os
import time
from tqdm import tqdm  # For progress bars
import multiprocessing as mp
from functools import partial
from numba import njit

# Define the timeframe you want to test
TIMEFRAME = '1m'  # Only using 1-minute data for precise crossover detection

# Load data function with correct columns
def load_data(timeframe):
    """
    Load CSV data for the specified timeframe.

    Parameters:
    - timeframe: Timeframe string (e.g., '1m').

    Returns:
    - data: DataFrame containing the loaded data.
    """
    file_name = f'es_{timeframe}_data.csv'
    if os.path.exists(file_name):
        data = pd.read_csv(file_name, parse_dates=['date'])
        data.rename(columns={'date': 'datetime'}, inplace=True)
        data['returns'] = data['close'].pct_change().fillna(0)
        data = data[['datetime', 'open', 'high', 'low', 'close', 'volume', 'returns']]
        print(f"Loaded data for timeframe: {timeframe}")
        return data
    else:
        error_msg = f"File {file_name} not found."
        print(error_msg)
        raise FileNotFoundError(error_msg)

# Generate a random chromosome ensuring logical parameter constraints
def generate_chromosome():
    """
    Generate a random chromosome with strategy parameters.

    Returns:
    - chromosome: Dictionary containing strategy parameters.
    """
    short_ema = random.choice(range(5, 51, 5))  # Generate multiples of 5 between 5 and 50
    long_ema = random.choice(range(short_ema + 5, 205, 5))  # Ensure long_ema > short_ema

    stop_loss = random.randint(2, 25)     # Stop loss in points as a whole number
    take_profit = random.randint(2, 50)  # Take profit in points as a whole number
    return {
        'short_ema': short_ema,
        'long_ema': long_ema,
        'stop_loss': stop_loss,
        'take_profit': take_profit
    }

@njit
def calculate_ema(prices, span):
    """
    Calculate Exponential Moving Average (EMA) for given prices and span.

    Parameters:
    - prices: Numpy array of prices.
    - span: Integer representing the EMA span.

    Returns:
    - ema: Numpy array of EMA values.
    """
    ema = np.empty(len(prices))
    ema[0] = prices[0]
    alpha = 2.0 / (span + 1)
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
    return ema

def detect_crossover(prices, short_span, long_span):
    """
    Detect EMA crossovers on 1-minute data based on EMA spans.

    Parameters:
    - prices: Numpy array of 1-minute closing prices.
    - short_span: Short EMA span.
    - long_span: Long EMA span.

    Returns:
    - signals: Numpy array of signals (1 for bullish, -1 for bearish, 0 otherwise).
    """
    ema_short = calculate_ema(prices, short_span)
    ema_long = calculate_ema(prices, long_span)

    # Initialize signals array
    signals = np.zeros(len(prices), dtype=np.int8)

    for i in range(1, len(prices)):
        if ema_short[i-1] <= ema_long[i-1] and ema_short[i] > ema_long[i]:
            signals[i] = 1  # Bullish Crossover
        elif ema_short[i-1] >= ema_long[i-1] and ema_short[i] < ema_long[i]:
            signals[i] = -1  # Bearish Crossover
    return signals

@njit
def backtest_strategy(prices, signals, stop_loss, take_profit, multiplier, initial_balance):
    """
    Backtest the strategy based on EMA crossover signals.

    Parameters:
    - prices: Numpy array of 1-minute closing prices.
    - signals: Numpy array of signals (1, -1, 0) for each minute.
    - stop_loss: Stop loss in points.
    - take_profit: Take profit in points.
    - multiplier: PnL multiplier per point movement.
    - initial_balance: Starting capital.

    Returns:
    - total_return_pct: Total return in percentage.
    - sharpe_ratio: Sharpe Ratio.
    - win_rate: Win rate of trades.
    - max_drawdown: Maximum drawdown percentage.
    """
    position = 0  # 1 for long, -1 for short, 0 for no position
    entry_price = 0.0
    cumulative_pnl = initial_balance
    peak = initial_balance
    max_drawdown = 0.0
    wins = 0
    total_trades = 0

    pnl_sum = 0.0
    pnl_sq_sum = 0.0

    for i in range(len(prices)):
        signal = signals[i]
        price = prices[i]

        # Check for existing position
        if position != 0:
            if position == 1:
                change = price - entry_price
                if change <= -stop_loss:
                    trade_pnl = -stop_loss * multiplier
                    cumulative_pnl += trade_pnl
                    pnl_sum += trade_pnl
                    pnl_sq_sum += trade_pnl * trade_pnl
                    if trade_pnl > 0:
                        wins += 1
                    total_trades += 1
                    position = 0
                    entry_price = 0.0
                    if cumulative_pnl > peak:
                        peak = cumulative_pnl
                    drawdown = (peak - cumulative_pnl) / peak * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    continue
                elif change >= take_profit:
                    trade_pnl = take_profit * multiplier
                    cumulative_pnl += trade_pnl
                    pnl_sum += trade_pnl
                    pnl_sq_sum += trade_pnl * trade_pnl
                    if trade_pnl > 0:
                        wins += 1
                    total_trades += 1
                    position = 0
                    entry_price = 0.0
                    if cumulative_pnl > peak:
                        peak = cumulative_pnl
                    drawdown = (peak - cumulative_pnl) / peak * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    continue
            elif position == -1:
                change = entry_price - price
                if change <= -stop_loss:
                    trade_pnl = -stop_loss * multiplier
                    cumulative_pnl += trade_pnl
                    pnl_sum += trade_pnl
                    pnl_sq_sum += trade_pnl * trade_pnl
                    if trade_pnl > 0:
                        wins += 1
                    total_trades += 1
                    position = 0
                    entry_price = 0.0
                    if cumulative_pnl > peak:
                        peak = cumulative_pnl
                    drawdown = (peak - cumulative_pnl) / peak * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    continue
                elif change >= take_profit:
                    trade_pnl = take_profit * multiplier
                    cumulative_pnl += trade_pnl
                    pnl_sum += trade_pnl
                    pnl_sq_sum += trade_pnl * trade_pnl
                    if trade_pnl > 0:
                        wins += 1
                    total_trades += 1
                    position = 0
                    entry_price = 0.0
                    if cumulative_pnl > peak:
                        peak = cumulative_pnl
                    drawdown = (peak - cumulative_pnl) / peak * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    continue

        # Open new position based on signal
        if signal == 1 and position == 0:
            position = 1
            entry_price = price
        elif signal == -1 and position == 0:
            position = -1
            entry_price = price

        # Update peak and drawdown
        if cumulative_pnl > peak:
            peak = cumulative_pnl
        drawdown = (peak - cumulative_pnl) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Final PnL
    total_return = cumulative_pnl - initial_balance
    total_return_pct = (total_return / initial_balance) * 100

    # Calculate Sharpe Ratio
    if total_trades > 0:
        mean_pnl = pnl_sum / total_trades
        var_pnl = (pnl_sq_sum / total_trades) - (mean_pnl) ** 2
        if var_pnl > 0:
            std_pnl = np.sqrt(var_pnl)
            sharpe_ratio = (mean_pnl / std_pnl) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    # Win Rate
    if total_trades > 0:
        win_rate = wins / total_trades
    else:
        win_rate = 0.0

    # Aggregate metrics
    return total_return_pct, sharpe_ratio, win_rate, max_drawdown

def evaluate_chromosome(chromosome, prices, initial_balance, multiplier):
    """
    Evaluate a single chromosome by detecting crossovers and backtesting the strategy.

    Parameters:
    - chromosome: Dictionary containing strategy parameters.
    - prices: Numpy array of 1-minute closing prices.
    - initial_balance: Starting capital.
    - multiplier: PnL multiplier per point movement.

    Returns:
    - metrics: Tuple containing (total_return_pct, sharpe_ratio, win_rate, max_drawdown).
    """
    short_span = chromosome['short_ema']
    long_span = chromosome['long_ema']
    stop_loss = chromosome['stop_loss']
    take_profit = chromosome['take_profit']

    # Detect crossover signals for this chromosome
    signals = detect_crossover(prices, short_span, long_span)

    # Backtest the strategy
    metrics = backtest_strategy(prices, signals, stop_loss, take_profit, multiplier, initial_balance)
    return metrics

def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Perform tournament selection on the population based on fitness scores.

    Parameters:
    - population: List of chromosomes.
    - fitness_scores: List of fitness scores corresponding to the population.
    - tournament_size: Number of participants in each tournament.

    Returns:
    - selected: List of selected chromosomes.
    """
    selected = []
    population_fitness = list(zip(population, fitness_scores))
    for _ in range(len(population)):
        participants = random.sample(population_fitness, tournament_size)
        # Prioritize Sharpe Ratio, then Total Return, then Win Rate, then Max Drawdown
        winner = max(participants, key=lambda x: (
            x[1][1],  # sharpe_ratio
            x[1][0],  # total_return_pct
            x[1][2],  # win_rate
            -x[1][3]  # max_drawdown_pct
        ))
        selected.append(winner[0])
    return selected

def crossover(parent1, parent2):
    """
    Perform crossover between two parent chromosomes to produce two children.

    Parameters:
    - parent1: First parent chromosome (dict).
    - parent2: Second parent chromosome (dict).

    Returns:
    - child1: First child chromosome (dict).
    - child2: Second child chromosome (dict).
    """
    child1, child2 = parent1.copy(), parent2.copy()
    
    # Swap EMAs with 50% probability
    if random.random() < 0.5:
        child1['short_ema'], child2['short_ema'] = child2['short_ema'], child1['short_ema']
        child1['long_ema'], child2['long_ema'] = child2['long_ema'], child1['long_ema']
    
    # Swap other genes independently
    for key in ['stop_loss', 'take_profit']:
        if random.random() < 0.5:
            child1[key], child2[key] = child2[key], child1[key]
    
    # Ensure long_ema > short_ema
    for child in [child1, child2]:
        if child['long_ema'] <= child['short_ema']:
            child['long_ema'] = child['short_ema'] + 5  # Adjust to maintain constraint
    
    return child1, child2

def mutate(chromosome, mutation_rate=0.1):
    """
    Mutate a chromosome with a given mutation rate.

    Parameters:
    - chromosome: Chromosome to mutate (dict).
    - mutation_rate: Probability of mutating each gene.

    Returns:
    - mutated_chromosome: Mutated chromosome (dict).
    """
    if random.random() < mutation_rate:
        chromosome['short_ema'] = random.choice(range(5, 51, 5))
        # After mutating short_ema, ensure long_ema > short_ema
        chromosome['long_ema'] = random.choice(range(chromosome['short_ema'] + 5, 205, 5))
    if random.random() < mutation_rate:
        chromosome['long_ema'] = random.choice(range(chromosome['short_ema'] + 5, 205, 5))
    if random.random() < mutation_rate:
        chromosome['stop_loss'] = random.randint(2, 25)  # As whole number
    if random.random() < mutation_rate:
        chromosome['take_profit'] = random.randint(2, 50)  # As whole number
    return chromosome

# Main execution
if __name__ == "__main__":
    try:
        start_time = time.time()
        # Load data for the specified timeframe
        print("Loading data...")
        data = load_data(TIMEFRAME)
        print("Data loaded successfully.")

        # Validate data
        if data.isnull().values.any():
            print("Data contains missing values. Dropping missing data...")
            data = data.dropna()

        # Extract 1-minute data
        close_prices = data['close'].values.astype(np.float64)
        datetime_series = data['datetime'].values  # For potential future use

        # Genetic Algorithm parameters
        population_size = 1000  # Adjusted population size for performance
        generations = 100       # Number of generations
        elitism_size = 100      # Number of top strategies to carry over
        mutation_rate = 0.2     # Mutation probability per gene
        multiplier = 5          # PnL multiplier per point movement
        initial_balance = 5000  # Starting capital

        # Initialize population
        print("Initializing population...")
        population = [generate_chromosome() for _ in range(population_size)]
        print(f"Initial population of {population_size} chromosomes generated.")

        # Track best strategies
        best_strategies = []

        # Prepare partial function for multiprocessing
        evaluate_func = partial(evaluate_chromosome, prices=close_prices,
                                initial_balance=initial_balance, multiplier=multiplier)

        # Initialize tqdm progress bar for generations
        with tqdm(total=generations, desc="Generations", unit="gen") as pbar:
            for generation in range(generations):
                # Start timing for the current generation
                gen_start_time = time.time()

                print(f"\nGeneration {generation + 1}/{generations}")

                # Initialize multiprocessing pool
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    # Initialize inner progress bar for chromosome evaluations
                    with tqdm(total=population_size, desc="Evaluating Chromosomes", leave=False, unit="chr") as inner_pbar:
                        # Map chromosomes to the evaluation function
                        results = []
                        for result in pool.imap(evaluate_func, population):
                            results.append(result)
                            inner_pbar.update(1)

                # Convert results to DataFrame for easier handling
                results_df = pd.DataFrame(results, columns=['total_return_pct', 'sharpe_ratio', 'win_rate', 'max_drawdown'])

                # Combine population with results
                population_df = pd.DataFrame(population)
                combined = pd.concat([population_df, results_df], axis=1)

                # Sort by prioritized metrics: Sharpe Ratio (desc), Total Return (desc), Win Rate (desc), Max Drawdown (asc)
                combined_sorted = combined.sort_values(
                    by=['sharpe_ratio', 'total_return_pct', 'win_rate', 'max_drawdown'],
                    ascending=[False, False, False, True]
                ).reset_index(drop=True)

                # Save top strategies from this generation
                top_generation = combined_sorted.head(elitism_size)
                best_strategies.append(top_generation)

                # Display top 5 strategies
                print("Top 5 strategies this generation:")
                print(top_generation[['short_ema', 'long_ema', 'stop_loss', 'take_profit',
                                      'sharpe_ratio', 'total_return_pct', 'win_rate', 'max_drawdown']].head(5))

                # Display average and best metrics
                avg_sharpe = combined_sorted['sharpe_ratio'].mean()
                avg_return = combined_sorted['total_return_pct'].mean()
                avg_win_rate = combined_sorted['win_rate'].mean()
                avg_drawdown = combined_sorted['max_drawdown'].mean()

                max_sharpe = combined_sorted['sharpe_ratio'].max()
                max_return = combined_sorted['total_return_pct'].max()
                max_win_rate = combined_sorted['win_rate'].max()
                min_drawdown = combined_sorted['max_drawdown'].min()

                print(f"Average Sharpe Ratio: {avg_sharpe:.4f}")
                print(f"Average Total Return: {avg_return:.2f}%")
                print(f"Average Win Rate: {avg_win_rate:.2f}")
                print(f"Average Max Drawdown: {avg_drawdown:.2f}%")
                print(f"Best Sharpe Ratio: {max_sharpe:.4f}")
                print(f"Best Total Return: {max_return:.2f}%")
                print(f"Best Win Rate: {max_win_rate:.2f}")
                print(f"Best (Min) Max Drawdown: {min_drawdown:.2f}%")

                # Elitism: carry forward top strategies
                elites = top_generation[['short_ema', 'long_ema', 'stop_loss', 'take_profit']].to_dict('records')

                # Selection: Tournament selection based on prioritized metrics
                # Prepare fitness_scores as list of tuples: (total_return_pct, sharpe_ratio, win_rate, max_drawdown)
                fitness_scores = list(zip(combined_sorted['total_return_pct'],
                                          combined_sorted['sharpe_ratio'],
                                          combined_sorted['win_rate'],
                                          combined_sorted['max_drawdown']))
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

                # Update the progress bar
                pbar.update(1)

                # Display generation time
                gen_end_time = time.time()
                gen_elapsed = gen_end_time - gen_start_time
                print(f"Generation {generation + 1} completed in {gen_elapsed / 60:.2f} minutes.")

    except FileNotFoundError as e:
        print(e)
    except Exception as ex:
        print(f"An error occurred: {ex}")

    try:
        # After all generations, compile the best strategies
        all_best = pd.concat(best_strategies).sort_values(
            by=['sharpe_ratio', 'total_return_pct', 'win_rate', 'max_drawdown'],
            ascending=[False, False, False, True]
        ).drop_duplicates().reset_index(drop=True)
        top_overall = all_best.head(100)  # Top 100 strategies

        print("\nTop 100 Overall Strategies:")
        print(top_overall[['short_ema', 'long_ema', 'stop_loss', 'take_profit',
                           'sharpe_ratio', 'total_return_pct',
                           'win_rate', 'max_drawdown']])

        # Add timeframe information
        top_overall['timeframe'] = TIMEFRAME

        # Reorder columns to include timeframe
        top_overall = top_overall[['timeframe', 'short_ema', 'long_ema', 'stop_loss', 'take_profit',
                                   'sharpe_ratio', 'total_return_pct',
                                   'win_rate', 'max_drawdown']]

        # Save the best strategies to a CSV file
        top_overall.to_csv('best_trading_strategies_1m_ema.csv', index=False)
        print("Best strategies saved to 'best_trading_strategies_1m_ema.csv'")
    except Exception as ex:
        print(f"An error occurred while compiling best strategies: {ex}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time / 60:.2f} minutes")
