import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import timedelta, time as dt_time
import sys
import random
import time  # Import time module for tracking runtime

# -------------------------------------------------------------
#                GA CONFIGURATION
# -------------------------------------------------------------
POPULATION_SIZE = 20
NUM_GENERATIONS = 10  # Increased for more thorough optimization
TOURNAMENT_SIZE = 4
CROSSOVER_RATE  = 0.8
MUTATION_RATE   = 0.2
RANDOM_SEED     = 42

# --- Set random seeds for reproducibility ---
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------------------------------------------
#           BACKTEST CONFIGURATION (defaults)
# -------------------------------------------------------------
INITIAL_CASH    = 5000
ES_MULTIPLIER   = 5       # 1 ES point = $5 profit/loss per contract (ES)
POSITION_SIZE   = 1       # can be fractional if desired
COMMISSION      = 1.24    # commission per trade

# We'll discover these via the GA:
# STOP_LOSS_POINTS, TAKE_PROFIT_POINTS, ROLLING_WINDOW

# We'll assume 1 tick = 0.25 for ES. Feel free to adjust.
ONE_TICK = 0.25

# -------------------------------------------------------------
#                       LOGGING SETUP
# -------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------
#               STEP 1: LOAD THE 1-MINUTE DATA
# -------------------------------------------------------------
def load_data(csv_file):
    """
    Loads 1-minute intraday data from CSV, parses the Time column as datetime, 
    sorts by time, and does some basic cleanup.
    """
    try:
        print(f"Loading data from CSV: {csv_file}")
        start_time = time.time()
        df = pd.read_csv(
            csv_file,
            parse_dates=['Time'],
            dayfirst=False,  # Adjust if your data uses day-first format
            na_values=['', 'NA', 'NaN']  # Handle missing values
        )
        load_duration = time.time() - start_time
        print(f"Data loading completed in {load_duration:.2f} seconds.")
        
        # Check if 'Time' column exists
        if 'Time' not in df.columns:
            logger.error("The CSV file does not contain a 'Time' column.")
            sys.exit(1)
        
        # Verify 'Time' column data type
        if not np.issubdtype(df['Time'].dtype, np.datetime64):
            logger.error("The 'Time' column was not parsed as datetime. Check the date format.")
            sys.exit(1)
        
        # Remove timezone if present
        if df['Time'].dt.tz is not None:
            df['Time'] = df['Time'].dt.tz_convert(None)
        
        # Sort by 'Time' and set as index
        df.sort_values('Time', inplace=True)
        df.set_index('Time', inplace=True)
        
        # Drop 'Symbol' if present
        if 'Symbol' in df.columns:
            df.drop(columns=['Symbol'], inplace=True)
        
        # Rename 'Last' to 'Close' if needed
        if 'Last' in df.columns:
            df.rename(columns={'Last': 'Close'}, inplace=True)
        
        # Optional: drop columns you don't need
        unnecessary_cols = ['Change', '%Chg', 'Open Int']
        df.drop(columns=[col for col in unnecessary_cols if col in df.columns],
                inplace=True, errors='ignore')
        
        # Print data range and size
        print(f"Data loaded successfully. Data range: {df.index.min()} to {df.index.max()}")
        print(f"Total rows loaded: {len(df)}")
        
        return df
    
    except FileNotFoundError:
        logger.error(f"File not found: {csv_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error("The CSV file is empty.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        logger.error(f"Parser error while reading the CSV: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading the CSV: {e}")
        sys.exit(1)


# -------------------------------------------------------------
#        STEP 2: BACKTEST FUNCTION (1-MINUTE LOGIC)
# -------------------------------------------------------------
def perform_backtest(
    df_intraday,
    stop_loss_points,
    take_profit_points,
    rolling_window,
    initial_cash,
    es_multiplier,
    position_size,
    commission,
    start_date,
    end_date
):
    """
    1) Filters df_intraday by [start_date, end_date].
    2) Resamples that data to 30-min bars to compute a rolling high over `rolling_window` bars.
    3) Forward-fills that Rolling High back into the original 1-minute data.
    4) On the 1-minute timeframe:
       - Enter long 1 tick above rolling high if the 1-min High >= rolling_high + one_tick (during 09:30-16:00).
       - Use 1-min Low & High to manage Stop Loss and Take Profit.
    5) Return the annualized Sharpe ratio for this parameter set.
    """
    ONE_TICK_LOCAL = ONE_TICK  # for clarity, you can rename or inline

    print(f"Starting backtest: SL={stop_loss_points}, TP={take_profit_points}, RW={rolling_window}, Period={start_date} to {end_date}")
    start_bt = time.time()
    
    # Filter data by date range
    start_time = pd.to_datetime(start_date)
    end_time   = pd.to_datetime(end_date)
    df_filtered = df_intraday.loc[start_time:end_time].copy()
    if df_filtered.empty:
        print("No data available for the specified date range.")
        return -999.0  # penalize no data scenario

    # Step A: Compute Rolling High on 30-min bars
    df_30m = df_filtered.resample('30min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Rolling high of the past `rolling_window` bars, excluding current bar => shift(1)
    df_30m['Rolling_High'] = (
        df_30m['High'].shift(1)
                      .rolling(window=rolling_window, min_periods=rolling_window)
                      .max()
    )
    # Drop any rows lacking Rolling_High
    df_30m.dropna(subset=['Rolling_High'], inplace=True)
    
    # If still empty after rolling
    if df_30m.empty:
        print("No data available after applying rolling window. Adjust parameters.")
        return -999.0
    
    # Step B: Forward-fill Rolling High to 1-min
    df_filtered['Rolling_High'] = df_30m['Rolling_High'].reindex(df_filtered.index, method='ffill')
    df_filtered.dropna(subset=['Rolling_High'], inplace=True)
    
    # Step C: 1-Minute Trading Logic
    cash = initial_cash
    position = None
    trade_results = []
    balance_series = [cash]
    balance_dates  = [df_filtered.index[0]]
    
    for current_time, row in df_filtered.iterrows():
        rolling_high_value = row['Rolling_High']
        if pd.isna(rolling_high_value):
            continue
        
        if position is None:
            # Enter if within RTH (09:30 to 16:00)
            if dt_time(9, 30) <= current_time.time() < dt_time(16, 0):
                breakout_price = rolling_high_value + ONE_TICK_LOCAL
                # If 1-min High >= breakout_price => fill
                if row['High'] >= breakout_price:
                    entry_price = breakout_price
                    stop_price  = entry_price - stop_loss_points
                    target_price= entry_price + take_profit_points
                    position = {
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_price,
                        'take_profit': target_price
                    }
                    logger.debug(f"Entered position at {entry_price} on {current_time}")
        else:
            # Manage open position
            current_high = row['High']
            current_low  = row['Low']
            exit_time    = current_time
            
            # Stop Loss
            if current_low <= position['stop_loss']:
                exit_price = position['stop_loss']
                pnl = ((exit_price - position['entry_price']) 
                       * position_size * es_multiplier) - commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                balance_dates.append(exit_time)
                logger.debug(f"Exited position at {exit_price} on {exit_time} with PnL={pnl}")
                position = None
            # Take Profit
            elif current_high >= position['take_profit']:
                exit_price = position['take_profit']
                pnl = ((exit_price - position['entry_price']) 
                       * position_size * es_multiplier) - commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                balance_dates.append(exit_time)
                logger.debug(f"Exited position at {exit_price} on {exit_time} with PnL={pnl}")
                position = None
        
        # If position closed, record equity
        if position is None:
            if len(balance_series) == len(balance_dates):
                balance_series.append(cash)
                balance_dates.append(current_time)
    
    # Step D: Compute Sharpe Ratio
    balance_df = pd.DataFrame({
        'Datetime': balance_dates,
        'Equity': balance_series
    }).set_index('Datetime').sort_index()
    
    returns = balance_df['Equity'].pct_change().dropna()
    if returns.std() == 0 or len(returns) < 2:
        sharpe_ratio = -999.0
    else:
        # Approximate annualization for 1-min data
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5 * 60)
    
    end_bt = time.time()
    print(f"Backtest completed in {end_bt - start_bt:.2f} seconds. Sharpe Ratio: {sharpe_ratio:.4f}")
    
    return sharpe_ratio


def run_backtest(
    df_intraday,
    stop_loss_points,
    take_profit_points,
    rolling_window,
    initial_cash=INITIAL_CASH,
    es_multiplier=ES_MULTIPLIER,
    position_size=POSITION_SIZE,
    commission=COMMISSION,
    train_start_date="2012-01-01",
    train_end_date="2019-12-31",
    val_start_date="2020-01-01",
    val_end_date="2024-12-23"
):
    """
    Runs the new 1-min + rolling high breakout backtest on:
      - Train dataset
      - Validation dataset
    Returns the combined Sharpe ratio (train + val) / 2.
    """
    # 1) Train
    print(f"Starting training backtest: {train_start_date} to {train_end_date}")
    train_start = time.time()
    train_sharpe = perform_backtest(
        df_intraday=df_intraday,
        stop_loss_points=stop_loss_points,
        take_profit_points=take_profit_points,
        rolling_window=rolling_window,
        initial_cash=initial_cash,
        es_multiplier=es_multiplier,
        position_size=position_size,
        commission=commission,
        start_date=train_start_date,
        end_date=train_end_date
    )
    train_end = time.time()
    print(f"Training backtest completed in {train_end - train_start:.2f} seconds. Sharpe Ratio: {train_sharpe:.4f}")
    
    # 2) Validation
    print(f"Starting validation backtest: {val_start_date} to {val_end_date}")
    val_start = time.time()
    val_sharpe = perform_backtest(
        df_intraday=df_intraday,
        stop_loss_points=stop_loss_points,
        take_profit_points=take_profit_points,
        rolling_window=rolling_window,
        initial_cash=initial_cash,
        es_multiplier=es_multiplier,
        position_size=position_size,
        commission=commission,
        start_date=val_start_date,
        end_date=val_end_date
    )
    val_end = time.time()
    print(f"Validation backtest completed in {val_end - val_start:.2f} seconds. Sharpe Ratio: {val_sharpe:.4f}")
    
    # If either fails, penalize
    if train_sharpe == -999.0 or val_sharpe == -999.0:
        return -999.0
    
    combined_sharpe = (train_sharpe + val_sharpe) / 2.0
    print(f"Combined Sharpe Ratio: {combined_sharpe:.4f}")
    return combined_sharpe


# -------------------------------------------------------------
#      STEP 3: GENETIC ALGORITHM CODE (UNCHANGED)
# -------------------------------------------------------------
def create_individual():
    """
    Create a single individual (a parameter set).
    Adjust the ranges to fit your strategy's possible parameter space.
    """
    stop_loss_points   = random.randint(1, 10)
    take_profit_points = random.randint(5, 30)
    rolling_window     = random.randint(5, 30)
    return (stop_loss_points, take_profit_points, rolling_window)

def mutate(individual):
    """
    Mutate an individual's parameters with some probability.
    """
    stop_loss_points, take_profit_points, rolling_window = individual
    
    if random.random() < MUTATION_RATE:
        stop_loss_points = random.randint(1, 10)
    if random.random() < MUTATION_RATE:
        take_profit_points = random.randint(5, 30)
    if random.random() < MUTATION_RATE:
        rolling_window = random.randint(5, 30)
    
    return (stop_loss_points, take_profit_points, rolling_window)

def crossover(parent1, parent2):
    """
    Single-point crossover: randomly choose one param to swap.
    """
    sl1, tp1, rw1 = parent1
    sl2, tp2, rw2 = parent2
    
    cx_point = random.randint(0, 2)  # among 3 genes
    if cx_point == 0:
        child1 = (sl1, tp2, rw2)
        child2 = (sl2, tp1, rw1)
    elif cx_point == 1:
        child1 = (sl1, tp1, rw2)
        child2 = (sl2, tp2, rw1)
    else:
        child1 = (sl1, tp2, rw1)
        child2 = (sl2, tp1, rw2)
    
    return child1, child2

def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE):
    """
    Tournament selection: pick k random individuals, return the best.
    """
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected_sorted = sorted(selected, key=lambda x: x[1], reverse=True)
    return selected_sorted[0][0]  # best individual

def evolve_population(population, fitnesses):
    """
    Create a new population via selection, crossover, mutation.
    """
    new_population = []
    pop_size = len(population)
    
    # Elitism: keep a fraction of the best individuals
    elite_size = max(1, pop_size // 5)
    combined = list(zip(population, fitnesses))
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
    
    # Keep the best as elites
    elites = [ind for ind, fit in combined_sorted[:elite_size]]
    new_population.extend(elites)
    
    # Fill the rest
    while len(new_population) < pop_size:
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)
        
        if random.random() < CROSSOVER_RATE:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2
        
        child1 = mutate(child1)
        child2 = mutate(child2)
        
        new_population.append(child1)
        if len(new_population) < pop_size:
            new_population.append(child2)
    
    return new_population


# -------------------------------------------------------------
#     MAIN SCRIPT: LOAD DATA, GA LOOP, PRINT BEST SOLUTION
# -------------------------------------------------------------
def main():
    overall_start_time = time.time()
    print("=== Starting Genetic Algorithm Optimization ===")
    
    # 1) Load the 1-minute data
    csv_file = 'es_1m_data.csv'
    df_intraday = load_data(csv_file)
    
    # 2) Initialize population
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    print(f"\nInitialized population with {POPULATION_SIZE} individuals.")
    
    best_individual = None
    best_fitness    = float('-inf')
    
    # Train/Validation Date Ranges (adjust as needed)
    train_start_date = "2012-01-01"
    train_end_date   = "2019-12-31"
    val_start_date   = "2020-01-01"
    val_end_date     = "2024-12-23"
    
    # 3) Evolve population
    for gen in range(NUM_GENERATIONS):
        print(f"\n--- Generation {gen+1}/{NUM_GENERATIONS} ---")
        gen_start_time = time.time()
        
        fitnesses = []
        for idx, ind in enumerate(population):
            stop_loss_points, take_profit_points, rolling_window = ind
            print(f" Evaluating Individual {idx+1}/{POPULATION_SIZE}: SL={stop_loss_points}, TP={take_profit_points}, RW={rolling_window}")
            sharpe = run_backtest(
                df_intraday=df_intraday,
                stop_loss_points=stop_loss_points,
                take_profit_points=take_profit_points,
                rolling_window=rolling_window,
                initial_cash=INITIAL_CASH,
                es_multiplier=ES_MULTIPLIER,
                position_size=POSITION_SIZE,
                commission=COMMISSION,
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                val_start_date=val_start_date,
                val_end_date=val_end_date
            )
            fitnesses.append(sharpe)
            
            # Optionally, print individual fitness
            print(f"  Sharpe Ratio: {sharpe:.4f}")
            
            # Track global best
            if sharpe > best_fitness:
                best_fitness    = sharpe
                best_individual = ind
        
        gen_end_time = time.time()
        gen_duration = gen_end_time - gen_start_time
        print(f" Generation {gen+1} completed in {gen_duration:.2f} seconds.")
        print(f"  Best Sharpe this generation: {max(fitnesses):.4f}")
        print(f"  Mean Sharpe this generation: {np.mean(fitnesses):.4f}")
        print(f"  Global Best Sharpe so far: {best_fitness:.4f}  Params: {best_individual}")
        
        # Evolve to the next generation
        population = evolve_population(population, fitnesses)
    
    # 4) Final Best
    print("\n=== Genetic Algorithm Optimization Complete ===")
    print(f"Best Individual Found: Stop Loss={best_individual[0]}, "
          f"Take Profit={best_individual[1]}, Rolling Window={best_individual[2]}")
    print(f"Best Combined Sharpe Ratio: {best_fitness:.4f}")
    
    # 5) Optional: Final check on validation set
    print("\n--- Running Final Validation Backtest with Best Parameters ---")
    final_start = time.time()
    final_sharpe = perform_backtest(
        df_intraday=df_intraday,
        stop_loss_points=best_individual[0],
        take_profit_points=best_individual[1],
        rolling_window=best_individual[2],
        initial_cash=INITIAL_CASH,
        es_multiplier=ES_MULTIPLIER,
        position_size=POSITION_SIZE,
        commission=COMMISSION,
        start_date=val_start_date,
        end_date=val_end_date
    )
    final_end = time.time()
    final_duration = final_end - final_start
    print(f"Final Validation Backtest completed in {final_duration:.2f} seconds. Sharpe Ratio: {final_sharpe:.4f}")
    
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    print(f"\nTotal Optimization Time: {total_duration / 60:.2f} minutes.")
    

# -------------------------------------------------------------
#                   RUN SCRIPT
# -------------------------------------------------------------
if __name__ == '__main__':
    main()