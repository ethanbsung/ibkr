import pandas as pd
import numpy as np
import random
import os
import time
from tqdm import tqdm  # For progress bars
from numba import njit, prange

# Define the list of timeframes you want to test, including '1m'
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h']

# Load data function with correct columns
def load_data(timeframes):
    """
    Load CSV data for each specified timeframe.

    Parameters:
    - timeframes: List of timeframe strings (e.g., ['1m', '5m']).

    Returns:
    - data_dict: Dictionary mapping timeframe to its DataFrame.
    """
    data_dict = {}
    for timeframe in timeframes:
        file_name = f'es_{timeframe}_data.csv'
        if os.path.exists(file_name):
            data = pd.read_csv(file_name, parse_dates=['date'])
            data.rename(columns={'date': 'datetime'}, inplace=True)
            data['returns'] = data['close'].pct_change().fillna(0)
            data_dict[timeframe] = data[['datetime', 'open', 'high', 'low', 'close', 'volume', 'returns']]
            print(f"Loaded data for timeframe: {timeframe}")
        else:
            error_msg = f"File {file_name} not found."
            print(error_msg)
            raise FileNotFoundError(error_msg)
    return data_dict

# Generate a random chromosome ensuring logical parameter constraints
def generate_chromosome():
    """
    Generate a random chromosome with strategy parameters.

    Returns:
    - chromosome: Dictionary containing strategy parameters.
    """
    short_ema = random.randint(5, 50)
    long_ema = random.randint(short_ema + 1, 200)  # Ensure long_ema > short_ema
    stop_loss = round(random.uniform(2, 25), 1)     # Stop loss in points
    take_profit = round(random.uniform(4, 50), 1)  # Take profit in points
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

@njit
def detect_crossover(close_prices, short_span, long_span):
    """
    Detect EMA crossovers in 1-minute data based on higher timeframe EMAs.

    Parameters:
    - close_prices: Numpy array of 1-minute closing prices.
    - short_span: Short EMA span for higher timeframe.
    - long_span: Long EMA span for higher timeframe.

    Returns:
    - signals: Numpy array of signals (1 for bullish, -1 for bearish, 0 otherwise).
    """
    ema_short = calculate_ema(close_prices, short_span)
    ema_long = calculate_ema(close_prices, long_span)
    
    # Initialize signals array
    signals = np.zeros(len(close_prices), dtype=np.int8)
    
    for i in range(1, len(close_prices)):
        if ema_short[i-1] <= ema_long[i-1] and ema_short[i] > ema_long[i]:
            signals[i] = 1  # Bullish Crossover
        elif ema_short[i-1] >= ema_long[i-1] and ema_short[i] < ema_long[i]:
            signals[i] = -1  # Bearish Crossover
    return signals

@njit
def backtest_single_timeframe_signal(close_prices, signals, stop_loss, take_profit, multiplier, initial_balance):
    """
    Backtest a single higher timeframe using pre-mapped signals.

    Parameters:
    - close_prices: Numpy array of closing prices for the higher timeframe.
    - signals: Numpy array of signals (1, -1, 0) for each candle.
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
    
    for i in range(len(close_prices)):
        signal = signals[i]
        price = close_prices[i]
        
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

@njit(parallel=True)
def backtest_strategy_multitimeframe_numba(close_prices_list, chromosomes_np, signals_list, multiplier, initial_balance, num_timeframes):
    """
    Backtest multiple chromosomes across multiple timeframes using pre-mapped signals.

    Parameters:
    - close_prices_list: List of Numpy arrays containing close prices for each timeframe.
    - chromosomes_np: Structured Numpy array of chromosome parameters.
    - signals_list: List of Numpy arrays containing signals for each timeframe.
    - multiplier: PnL multiplier per point movement.
    - initial_balance: Starting capital.
    - num_timeframes: Number of higher timeframes.

    Returns:
    - results: Numpy array of shape (chromosomes, 4) containing performance metrics.
    """
    results = np.zeros((len(chromosomes_np), 4))  # Columns: total_return_pct, sharpe_ratio, win_rate, max_drawdown_pct
    
    for c in prange(len(chromosomes_np)):
        total_return = 0.0
        sharpe = 0.0
        win_rate = 0.0
        max_drawdown = 0.0
        
        for t in range(1, num_timeframes):  # Start from index 1 to skip '1m'
            close_prices = close_prices_list[t]
            signals = signals_list[t]
            stop_loss = chromosomes_np['stop_loss'][c]
            take_profit = chromosomes_np['take_profit'][c]
            
            ret = backtest_single_timeframe_signal(close_prices, signals, stop_loss, take_profit, multiplier, initial_balance)
            
            total_return += ret[0]
            sharpe += ret[1]
            win_rate += ret[2]
            max_drawdown += ret[3]
        
        # Average metrics across timeframes
        results[c, 0] = total_return / (num_timeframes - 1)
        results[c, 1] = sharpe / (num_timeframes - 1)
        results[c, 2] = win_rate / (num_timeframes - 1)
        results[c, 3] = max_drawdown / (num_timeframes - 1)
    
    return results

# Tournament selection based on prioritized metrics
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

# Crossover two parents to produce two children
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
    for key in parent1.keys():
        if random.random() < 0.5:
            child1[key], child2[key] = child2[key], parent1[key]
    return child1, child2

# Mutate a chromosome
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
        chromosome['short_ema'] = random.randint(5, 50)
    if random.random() < mutation_rate:
        chromosome['long_ema'] = random.randint(chromosome['short_ema'] + 1, 200)
    if random.random() < mutation_rate:
        chromosome['stop_loss'] = round(random.uniform(2, 25), 2)
    if random.random() < mutation_rate:
        chromosome['take_profit'] = round(random.uniform(4, 50), 2)
    return chromosome

def map_signals_to_higher_timeframes(data_dict, crossover_signals_1m):
    """
    Map 1-minute EMA crossover signals to higher timeframes.

    Parameters:
    - data_dict: Dictionary containing DataFrames for each timeframe.
    - crossover_signals_1m: Numpy array of signals from 1-minute data.

    Returns:
    - signals_list: List of Numpy arrays containing signals for each timeframe, ordered as TIMEFRAMES.
    """
    signals_list = []
    one_min_df = data_dict['1m'].copy()
    one_min_df['crossover_signal'] = crossover_signals_1m
    one_min_df['datetime'] = pd.to_datetime(one_min_df['datetime'])
    
    for timeframe in TIMEFRAMES:
        if timeframe == '1m':
            # 1-minute signals are directly used
            signals = one_min_df['crossover_signal'].values
            signals_list.append(signals)
            continue
        
        higher_tf_df = data_dict[timeframe].copy()
        higher_tf_df['datetime'] = pd.to_datetime(higher_tf_df['datetime'])
        
        # Initialize signal array
        signals = np.zeros(len(higher_tf_df), dtype=np.int8)
        
        # Iterate over higher timeframe candles
        for i in range(len(higher_tf_df)):
            start_time = higher_tf_df.at[i, 'datetime']
            # Calculate end_time based on timeframe
            if timeframe.endswith('m'):
                minutes = int(timeframe.rstrip('m'))
                end_time = start_time + pd.Timedelta(minutes=minutes)
            elif timeframe.endswith('h'):
                hours = int(timeframe.rstrip('h'))
                end_time = start_time + pd.Timedelta(hours=hours)
            else:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # Find 1-minute signals within this higher timeframe period
            mask = (one_min_df['datetime'] >= start_time) & (one_min_df['datetime'] < end_time)
            if mask.any():
                # Assign the first signal occurrence within the period
                first_signal = one_min_df.loc[mask, 'crossover_signal'].values[0]
                signals[i] = first_signal
            else:
                signals[i] = 0  # No signal
        
        signals_list.append(signals)
    
    return signals_list

# Main execution
if __name__ == "__main__":
    try:
        start_time = time.time()
        # Load data for all specified timeframes
        print("Loading data...")
        data_dict = load_data(TIMEFRAMES)
        print("Data loaded successfully.")

        # Extract 1-minute data
        one_min_df = data_dict['1m']
        close_prices_1m = one_min_df['close'].values.astype(np.float64)
        # Example spans; these will be optimized by the genetic algorithm
        short_span_1m = 10  # Short EMA span for higher timeframe
        long_span_1m = 50   # Long EMA span for higher timeframe

        # Detect EMA crossovers on 1-minute data
        print("Detecting EMA crossovers on 1-minute data...")
        crossover_signals_1m = detect_crossover(close_prices_1m, short_span_1m, long_span_1m)
        print("Crossover detection completed.")

        # Map 1-minute signals to higher timeframes
        print("Mapping 1-minute signals to higher timeframes...")
        signals_list = map_signals_to_higher_timeframes(data_dict, crossover_signals_1m)
        print("Signal mapping completed.")

        # Genetic Algorithm parameters
        population_size = 500  # Adjusted population size for performance
        generations = 100       # Number of generations
        elitism_size = 50       # Number of top strategies to carry over
        mutation_rate = 0.2     # Mutation probability per gene
        multiplier = 5          # PnL multiplier per point movement
        initial_balance = 5000  # Starting capital
        num_timeframes = len(TIMEFRAMES)

        # Initialize population
        print("Initializing population...")
        population = [generate_chromosome() for _ in range(population_size)]
        print(f"Initial population of {population_size} chromosomes generated.")

        # Convert population to structured Numpy array for Numba
        chromosomes_np = np.empty(population_size, dtype=np.dtype([
            ('short_ema', np.int32),
            ('long_ema', np.int32),
            ('stop_loss', np.float64),
            ('take_profit', np.float64)
        ]))
        for i, chrom in enumerate(population):
            chromosomes_np['short_ema'][i] = chrom['short_ema']
            chromosomes_np['long_ema'][i] = chrom['long_ema']
            chromosomes_np['stop_loss'][i] = chrom['stop_loss']
            chromosomes_np['take_profit'][i] = chrom['take_profit']
        
        # Track best strategies
        best_strategies = []

        # Initialize tqdm progress bar for generations
        with tqdm(total=generations, desc="Generations", unit="gen") as pbar:
            for generation in range(generations):
                # Start timing for the current generation
                gen_start_time = time.time()

                print(f"\nGeneration {generation + 1}/{generations}")

                # Run the Numba backtest function
                metrics = backtest_strategy_multitimeframe_numba(
                    close_prices_list=[data_dict[tf]['close'].values.astype(np.float64) for tf in TIMEFRAMES],
                    chromosomes_np=chromosomes_np,
                    signals_list=signals_list,
                    multiplier=multiplier,
                    initial_balance=initial_balance,
                    num_timeframes=num_timeframes
                )

                # Extract metrics into DataFrame
                results = pd.DataFrame(metrics, columns=['total_return_pct', 'sharpe_ratio', 'win_rate', 'max_drawdown_pct'])
                results['short_ema'] = chromosomes_np['short_ema']
                results['long_ema'] = chromosomes_np['long_ema']
                results['stop_loss'] = chromosomes_np['stop_loss']
                results['take_profit'] = chromosomes_np['take_profit']

                # Calculate additional metrics for progress tracking
                avg_sharpe = results['sharpe_ratio'].mean()
                avg_return = results['total_return_pct'].mean()
                avg_win_rate = results['win_rate'].mean()
                avg_drawdown = results['max_drawdown_pct'].mean()

                max_sharpe = results['sharpe_ratio'].max()
                max_return = results['total_return_pct'].max()
                max_win_rate = results['win_rate'].max()
                min_drawdown = results['max_drawdown_pct'].min()

                # Sort by prioritized metrics: Sharpe Ratio (desc), Total Return (desc), Win Rate (desc), Max Drawdown (asc)
                results_sorted = results.sort_values(
                    by=['sharpe_ratio', 'total_return_pct', 'win_rate', 'max_drawdown_pct'],
                    ascending=[False, False, False, True]
                ).reset_index(drop=True)

                # Save top strategies from this generation
                top_generation = results_sorted.head(10)
                best_strategies.append(top_generation)

                # Display top 5 strategies
                print("Top 5 strategies this generation:")
                print(top_generation[['sharpe_ratio', 'total_return_pct', 'win_rate', 'max_drawdown_pct']].head(5))

                # Display average and best metrics
                print(f"Average Sharpe Ratio: {avg_sharpe:.4f}")
                print(f"Average Total Return: {avg_return:.2f}%")
                print(f"Average Win Rate: {avg_win_rate:.2f}")
                print(f"Average Max Drawdown: {avg_drawdown:.2f}%")
                print(f"Best Sharpe Ratio: {max_sharpe:.4f}")
                print(f"Best Total Return: {max_return:.2f}%")
                print(f"Best Win Rate: {max_win_rate:.2f}")
                print(f"Best (Min) Max Drawdown: {min_drawdown:.2f}%")

                # Elitism: carry forward top strategies
                elites = results_sorted.head(elitism_size)[[
                    'short_ema', 'long_ema', 'stop_loss', 'take_profit'
                ]].to_dict('records')

                # Selection: Tournament selection based on prioritized metrics
                selected = tournament_selection(population, metrics.tolist())

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

                # Update chromosomes_np
                for i, chrom in enumerate(population):
                    chromosomes_np['short_ema'][i] = chrom['short_ema']
                    chromosomes_np['long_ema'][i] = chrom['long_ema']
                    chromosomes_np['stop_loss'][i] = chrom['stop_loss']
                    chromosomes_np['take_profit'][i] = chrom['take_profit']

                # Save progress periodically
                if (generation + 1) % 10 == 0:
                    intermediate_best = pd.concat(best_strategies).sort_values(
                        by=['sharpe_ratio', 'total_return_pct', 'win_rate', 'max_drawdown_pct'],
                        ascending=[False, False, False, True]
                    ).drop_duplicates().reset_index(drop=True)
                    intermediate_best.to_csv(f'best_strategies_gen_{generation + 1}.csv', index=False)
                    print(f"Intermediate best strategies saved to 'best_strategies_gen_{generation + 1}.csv'")

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
            by=['sharpe_ratio', 'total_return_pct', 'win_rate', 'max_drawdown_pct'],
            ascending=[False, False, False, True]
        ).drop_duplicates().reset_index(drop=True)
        top_overall = all_best.head(100)  # Top 100 strategies

        print("\nTop 100 Overall Strategies:")
        print(top_overall[['short_ema', 'long_ema', 'stop_loss', 'take_profit',
                           'sharpe_ratio', 'total_return_pct',
                           'win_rate', 'max_drawdown_pct']])

        # Save the best strategies to a CSV file
        top_overall.to_csv('best_trading_strategies_multitimeframe_ema.csv', index=False)
        print("Best strategies saved to 'best_trading_strategies_multitimeframe_ema.csv'")
    except Exception as ex:
        print(f"An error occurred while compiling best strategies: {ex}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time / 60:.2f} minutes")
