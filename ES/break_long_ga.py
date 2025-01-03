import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import random
import time as time_module  # Renamed to avoid conflict with datetime.time
from datetime import timedelta, time as dt_time

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
ONE_TICK        = 0.25    # Tick size for ES
SLIPPAGE        = 1       # Slippage in points

# We'll discover these via the GA:
# STOP_LOSS_POINTS, TAKE_PROFIT_POINTS, ROLLING_WINDOW

# -------------------------------------------------------------
#                       LOGGING SETUP
# -------------------------------------------------------------
logging.basicConfig(level=logging.WARNING,  # Set to INFO or DEBUG for more verbosity
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
#               STEP 1: LOAD DATA
# -------------------------------------------------------------
def load_data(csv_file, data_type='1m'):
    """
    Loads intraday data from CSV, parses the Time column as datetime,
    sorts by time, sets index, and performs basic cleanup.

    Parameters:
    - csv_file: Path to the CSV file.
    - data_type: '1m' for 1-minute data, '30m' for 30-minute data.

    Returns:
    - df: Cleaned DataFrame.
    """
    try:
        print(f"Loading {data_type}-Minute data from CSV: {csv_file}")
        start_time = time_module.time()
        df = pd.read_csv(
            csv_file,
            parse_dates=['Time'],
            dayfirst=False,  # Adjust if your data uses day-first format
            na_values=['', 'NA', 'NaN']
        )
        load_duration = time_module.time() - start_time
        print(f"Data loading completed in {load_duration:.2f} seconds.")

        # Check if 'Time' column exists
        if 'Time' not in df.columns:
            logger.error(f"CSV {csv_file} does not contain a 'Time' column.")
            sys.exit(1)

        # Verify 'Time' column data type
        if not np.issubdtype(df['Time'].dtype, np.datetime64):
            logger.error(f"'Time' column in {csv_file} not parsed as datetime. Check the date format.")
            sys.exit(1)

        # Remove timezone if present
        if df['Time'].dt.tz is not None:
            df['Time'] = df['Time'].dt.tz_convert(None)
            logger.debug("Removed timezone from 'Time' column.")

        # Sort by 'Time' and set as index
        df.sort_values('Time', inplace=True)
        df.set_index('Time', inplace=True)

        # Drop 'Symbol' if present
        if 'Symbol' in df.columns:
            df.drop(columns=['Symbol'], inplace=True)
            logger.debug("Dropped 'Symbol' column.")

        # Rename 'Last' to 'Close' if present
        if 'Last' in df.columns:
            df.rename(columns={'Last': 'Close'}, inplace=True)
            logger.debug("Renamed 'Last' column to 'Close'.")

        # Optional: drop columns you don't need
        unnecessary_cols = ['Change', '%Chg', 'Open Int']
        df.drop(columns=[col for col in unnecessary_cols if col in df.columns],
                inplace=True, errors='ignore')

        # Print data range and size
        print(f"{data_type}-Minute Data loaded successfully. Data range: {df.index.min()} to {df.index.max()}")
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
#   STEP 2: PREPARE DATA USING 30-MIN DATA FOR ROLLING HIGH
# -------------------------------------------------------------
def prepare_data(df_1m, df_30m, rolling_window, backtest_start_date=None):
    """
    Prepares 1-minute data by calculating Rolling_High and Prev_30m_High
    based on the specified rolling_window.

    Parameters:
    - df_1m: DataFrame of 1-minute data.
    - df_30m: DataFrame of 30-minute data.
    - rolling_window: Number of 30-minute bars for the rolling high calculation.
    - backtest_start_date: The start date of the backtest to reset Rolling_High.

    Returns:
    - df_1m: Updated 1-minute DataFrame.
    """
    # Compute Rolling_High over the previous rolling_window 30-minute bars
    df_30m['Rolling_High'] = (
        df_30m['High']
        .shift(1)  # Exclude the current bar
        .rolling(window=rolling_window, min_periods=rolling_window)
        .max()
    )

    # Compute Prev_30m_High as the High of the previous 30-minute bar
    df_30m['Prev_30m_High'] = df_30m['High'].shift(1)

    # Drop rows where Rolling_High or Prev_30m_High is NaN
    df_30m.dropna(subset=['Rolling_High', 'Prev_30m_High'], inplace=True)

    logger.info(f"Computed Rolling_High and Prev_30m_High with {len(df_30m)} valid 30-minute bars.")

    # Forward-fill Rolling_High and Prev_30m_High into 1-minute data
    df_1m['Rolling_High'] = df_30m['Rolling_High'].reindex(df_1m.index, method='pad')
    df_1m['Prev_30m_High'] = df_30m['Prev_30m_High'].reindex(df_1m.index, method='pad')

    # Reset Rolling_High and Prev_30m_High at the start of the backtest period
    if backtest_start_date:
        backtest_start = pd.to_datetime(backtest_start_date)
        df_1m.loc[:backtest_start, ['Rolling_High', 'Prev_30m_High']] = np.nan

    # Add a '30m_bar' column indicating the 30-minute bar each 1-minute bar belongs to
    df_1m['30m_bar'] = df_1m.index.floor('30min')

    logger.debug(f"Prepared data with Rolling_High and Prev_30m_High using a rolling window of {rolling_window}.")
    return df_1m

# -------------------------------------------------------------
#             STEP 3: BACKTEST FUNCTION
# -------------------------------------------------------------
def perform_backtest(
    df_1m,
    df_30m,
    stop_loss_points,
    take_profit_points,
    rolling_window,
    initial_cash,
    es_multiplier,
    position_size,
    commission,
    slippage,
    start_date,
    end_date
):
    """
    Runs the backtest with enhanced logic to ensure only one trade per 30-minute bar.
    Returns the annualized Sharpe ratio for this parameter set.

    Parameters:
    - df_1m: 1-minute DataFrame.
    - df_30m: 30-minute DataFrame.
    - stop_loss_points: Stop loss in points.
    - take_profit_points: Take profit in points.
    - rolling_window: Number of 30-minute bars in the rolling window.
    - initial_cash: Starting cash.
    - es_multiplier: ES multiplier.
    - position_size: Number of contracts per trade.
    - commission: Commission per trade.
    - slippage: Slippage in points.
    - start_date: Backtest start date.
    - end_date: Backtest end date.

    Returns:
    - sharpe_ratio: Annualized Sharpe ratio or -999.0 if invalid.
    """
    ONE_TICK_LOCAL = ONE_TICK  # for clarity

    print(f"\nStarting backtest: SL={stop_loss_points}, TP={take_profit_points}, RW={rolling_window}, Period={start_date} to {end_date}")
    start_bt = time_module.time()

    # Step A: Prepare data
    df_prepared = prepare_data(df_1m, df_30m, rolling_window=rolling_window, backtest_start_date=start_date)
    df_prepared.dropna(subset=['Rolling_High', 'Prev_30m_High'], inplace=True)

    # Filter date range
    start_time = pd.to_datetime(start_date)
    end_time   = pd.to_datetime(end_date)
    df_filtered = df_prepared.loc[start_time:end_time].copy()

    if df_filtered.empty:
        print(f"No data available for the specified date range: {start_date} to {end_date}")
        return -999.0  # penalize no-data scenario

    logger.info(f"Backtesting from {start_time} to {end_time} with {len(df_filtered)} 1-minute bars.")

    # Initialize backtest variables
    cash = initial_cash
    position = None
    trade_results = []
    balance_series = [cash]
    balance_dates  = [df_filtered.index[0]]

    total_bars = len(df_filtered)
    active_trades = 0  # For measuring "exposure"

    # Initialize last_trade_30m_bar and previous_rolling_high
    last_trade_30m_bar = None
    previous_rolling_high = -np.inf

    # For plotting/debugging purposes, store points where breakout should occur
    breakout_points = []

    for idx, (current_time, row) in enumerate(df_filtered.iterrows()):
        rolling_high_value = row['Rolling_High']
        prev_30m_high = row['Prev_30m_High']
        current_30m_bar = row['30m_bar']

        # Skip if Rolling High or Prev_30m_High is NaN (shouldn't happen if we've forward-filled + dropped NaN)
        if pd.isna(rolling_high_value) or pd.isna(prev_30m_high):
            logger.debug(f"Skipped Time: {current_time} due to NaN in Rolling_High or Prev_30m_High.")
            continue

        # Determine current breakout price
        breakout_price = rolling_high_value + ONE_TICK_LOCAL

        # Check eligibility:
        # 1. Breakout Price > Prev_30m_High
        # 2. No trade has been taken in the current 30-minute bar
        # 3. Rolling_High has increased since the last trade
        eligible_for_entry = (breakout_price > prev_30m_high) and \
                              (current_30m_bar != last_trade_30m_bar) and \
                              (rolling_high_value > previous_rolling_high)

        # Debug logs for current bar
        logger.debug(f"Time: {current_time}")
        logger.debug(f"30m_bar: {current_30m_bar}")
        logger.debug(f"Rolling_High: {rolling_high_value}, Prev_30m_High: {prev_30m_high}")
        logger.debug(f"Breakout_Price: {breakout_price}")
        logger.debug(f"Eligible for Entry: {eligible_for_entry}")

        position_closed = False  # Flag to track if a position was closed in this iteration

        if position is not None:
            # Manage open position
            current_high = row['High']
            current_low  = row['Low']
            exit_time    = current_time

            # Check Stop Loss
            if current_low <= position['stop_loss']:
                exit_price = position['stop_loss']
                pnl = ((exit_price - position['entry_price']) 
                       * position_size * es_multiplier) - commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                balance_dates.append(exit_time)
                logger.info(f"[STOP LOSS] Exit at {exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
                position = None
                position_closed = True  # Set flag
                # No need to reset last_trade_30m_bar
                # No need to reset previous_rolling_high

            # If still open, check Take Profit
            elif current_high >= position['take_profit']:
                exit_price = position['take_profit']
                pnl = ((exit_price - position['entry_price']) 
                       * position_size * es_multiplier) - commission
                cash += pnl
                trade_results.append(pnl)
                balance_series.append(cash)
                balance_dates.append(exit_time)
                logger.info(f"[TAKE PROFIT] Exit at {exit_price} on {exit_time}, PnL: ${pnl:,.2f}")
                position = None
                position_closed = True  # Set flag
                # No need to reset last_trade_30m_bar
                # No need to reset previous_rolling_high

        if not position_closed and position is None and eligible_for_entry:
            # Only trade during Regular Trading Hours (09:30 - 16:00)
            if dt_time(9, 30) <= current_time.time() < dt_time(16, 0):
                # Entry Condition: High price >= breakout_price
                if row['High'] >= breakout_price:
                    # Debugging: Mark the attempt to enter trade
                    logger.debug(f"Attempting to enter trade at {current_time} with breakout_price {breakout_price}")

                    entry_price = breakout_price + slippage
                    stop_price  = entry_price - stop_loss_points
                    target_price= entry_price + take_profit_points

                    position = {
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_price,
                        'take_profit': target_price
                    }
                    # No placeholder in trade_results; PnL is recorded upon exit
                    balance_series.append(cash)
                    balance_dates.append(current_time)
                    active_trades += 1
                    last_trade_30m_bar = current_30m_bar  # Update last trade 30m bar
                    previous_rolling_high = rolling_high_value  # Update previous rolling high
                    logger.info(f"[ENTRY] Long entered at {entry_price} on {current_time}")

                    # For debugging: mark this breakout
                    breakout_points.append(current_time)
                else:
                    logger.debug(f"No entry: High {row['High']} < Breakout Price {breakout_price}")
            else:
                logger.debug(f"No entry: Outside Regular Trading Hours at {current_time.time()}")

        # Record equity if no position or if we just closed
        if position is None and not position_closed:
            if len(balance_series) == len(balance_dates):
                balance_series.append(cash)
                balance_dates.append(current_time)

    exposure_time_percentage = (active_trades / total_bars) * 100
    logger.info(f"Total Bars: {total_bars}, Active Trades (Trades Entered): {active_trades}")
    logger.info(f"Exposure Time Percentage: {exposure_time_percentage:.2f}%")

    balance_df = pd.DataFrame({
        'Datetime': balance_dates,
        'Equity': balance_series
    }).set_index('Datetime').sort_index()

    # For debugging: Save breakout points to a CSV
    if breakout_points:
        breakout_df = pd.DataFrame({'Breakout_Time': breakout_points})
        breakout_df.to_csv('breakout_points_ga.csv', index=False)
        logger.info(f"Saved {len(breakout_points)} breakout points to 'breakout_points_ga.csv'.")

    # Step D: Compute Sharpe Ratio
    returns = balance_df['Equity'].pct_change().dropna()
    if returns.std() == 0 or len(returns) < 2:
        sharpe_ratio = -999.0
    else:
        # Approximate annualization for 1-min data
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5 * 60)

    end_bt = time_module.time()
    print(f"Backtest completed in {end_bt - start_bt:.2f} seconds. Sharpe Ratio: {sharpe_ratio:.4f}")

    return sharpe_ratio

# -------------------------------------------------------------
#             STEP 4: RUN_BACKTEST FUNCTION
# -------------------------------------------------------------
def run_backtest(
    df_intraday,
    df_intraday_30m,
    stop_loss_points,
    take_profit_points,
    rolling_window,
    initial_cash=INITIAL_CASH,
    es_multiplier=ES_MULTIPLIER,
    position_size=POSITION_SIZE,
    commission=COMMISSION,
    slippage=SLIPPAGE,
    train_start_date="2012-01-01",
    train_end_date="2019-12-31",
    val_start_date="2020-01-01",
    val_end_date="2024-12-23"
):
    """
    Runs the updated 1-min + rolling high breakout backtest on:
      - Train dataset
      - Validation dataset
    Returns the combined Sharpe ratio (train + val) / 2.

    Parameters:
    - df_intraday: 1-minute DataFrame.
    - df_intraday_30m: 30-minute DataFrame.
    - stop_loss_points: Stop loss in points.
    - take_profit_points: Take profit in points.
    - rolling_window: Number of 30-minute bars in the rolling window.
    - initial_cash: Starting cash.
    - es_multiplier: ES multiplier.
    - position_size: Number of contracts per trade.
    - commission: Commission per trade.
    - slippage: Slippage in points.
    - train_start_date: Training backtest start date.
    - train_end_date: Training backtest end date.
    - val_start_date: Validation backtest start date.
    - val_end_date: Validation backtest end date.

    Returns:
    - combined_sharpe: Average Sharpe ratio of train and validation backtests.
    """
    # 1) Train Backtest
    print(f"\nStarting training backtest: {train_start_date} to {train_end_date}")
    train_start = time_module.time()
    train_sharpe = perform_backtest(
        df_1m=df_intraday,
        df_30m=df_intraday_30m,
        stop_loss_points=stop_loss_points,
        take_profit_points=take_profit_points,
        rolling_window=rolling_window,
        initial_cash=initial_cash,
        es_multiplier=es_multiplier,
        position_size=position_size,
        commission=commission,
        slippage=slippage,
        start_date=train_start_date,
        end_date=train_end_date
    )
    train_end = time_module.time()
    print(f"Training backtest completed in {train_end - train_start:.2f} seconds. Sharpe Ratio: {train_sharpe:.4f}")

    # 2) Validation Backtest
    print(f"\nStarting validation backtest: {val_start_date} to {val_end_date}")
    val_start = time_module.time()
    val_sharpe = perform_backtest(
        df_1m=df_intraday,
        df_30m=df_intraday_30m,
        stop_loss_points=stop_loss_points,
        take_profit_points=take_profit_points,
        rolling_window=rolling_window,
        initial_cash=initial_cash,
        es_multiplier=es_multiplier,
        position_size=position_size,
        commission=commission,
        slippage=slippage,
        start_date=val_start_date,
        end_date=val_end_date
    )
    val_end = time_module.time()
    print(f"Validation backtest completed in {val_end - val_start:.2f} seconds. Sharpe Ratio: {val_sharpe:.4f}")

    # If either fails, penalize
    if train_sharpe == -999.0 or val_sharpe == -999.0:
        return -999.0

    combined_sharpe = (train_sharpe + val_sharpe) / 2.0
    print(f"Combined Sharpe Ratio: {combined_sharpe:.4f}")
    return combined_sharpe

# -------------------------------------------------------------
#      STEP 5: GENETIC ALGORITHM SUPPORT FUNCTIONS
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
    overall_start_time = time_module.time()
    print("=== Starting Genetic Algorithm Optimization ===")

    # 1) Load the 1-minute and 30-minute data
    csv_file_1m = 'Data/es_1m_data.csv'
    csv_file_30m = 'Data/es_30m_data.csv'
    df_intraday_1m = load_data(csv_file_1m, data_type='1m')
    df_intraday_30m = load_data(csv_file_30m, data_type='30m')

    # 2) Define backtest periods
    train_start_date = "2022-01-08"
    train_end_date   = "2024-12-23"  # Adjusted to match backtest_end
    val_start_date   = "2025-01-04"  # Assuming continuation beyond backtest_end
    val_end_date     = "2025-12-31"  # Example dates; adjust as needed

    # 3) Initialize population
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    print(f"\nInitialized population with {POPULATION_SIZE} individuals.")

    best_individual = None
    best_fitness    = float('-inf')

    # 4) Evolve population
    for gen in range(NUM_GENERATIONS):
        print(f"\n--- Generation {gen+1}/{NUM_GENERATIONS} ---")
        gen_start_time = time_module.time()

        fitnesses = []
        for idx, ind in enumerate(population):
            stop_loss_points, take_profit_points, rolling_window = ind
            print(f" Evaluating Individual {idx+1}/{POPULATION_SIZE}: SL={stop_loss_points}, TP={take_profit_points}, RW={rolling_window}")
            sharpe = run_backtest(
                df_intraday=df_intraday_1m,
                df_intraday_30m=df_intraday_30m,
                stop_loss_points=stop_loss_points,
                take_profit_points=take_profit_points,
                rolling_window=rolling_window,
                initial_cash=INITIAL_CASH,
                es_multiplier=ES_MULTIPLIER,
                position_size=POSITION_SIZE,
                commission=COMMISSION,
                slippage=SLIPPAGE,
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

        gen_end_time = time_module.time()
        gen_duration = gen_end_time - gen_start_time
        print(f" Generation {gen+1} completed in {gen_duration:.2f} seconds.")
        if fitnesses:
            print(f"  Best Sharpe this generation: {max(fitnesses):.4f}")
            print(f"  Mean Sharpe this generation: {np.mean(fitnesses):.4f}")
            print(f"  Global Best Sharpe so far: {best_fitness:.4f}  Params: {best_individual}")
        else:
            print("  No fitness scores calculated for this generation.")

        # Evolve to the next generation
        population = evolve_population(population, fitnesses)

    # 5) Final Best
    print("\n=== Genetic Algorithm Optimization Complete ===")
    if best_individual:
        print(f"Best Individual Found: Stop Loss={best_individual[0]}, "
              f"Take Profit={best_individual[1]}, Rolling Window={best_individual[2]}")
        print(f"Best Combined Sharpe Ratio: {best_fitness:.4f}")
    else:
        print("No valid individuals were found during optimization.")

    # 6) Optional: Final check on validation set
    if best_individual and best_fitness != -999.0:
        print("\n--- Running Final Validation Backtest with Best Parameters ---")
        final_start = time_module.time()
        final_sharpe = perform_backtest(
            df_1m=df_intraday_1m,
            df_30m=df_intraday_30m,
            stop_loss_points=best_individual[0],
            take_profit_points=best_individual[1],
            rolling_window=best_individual[2],
            initial_cash=INITIAL_CASH,
            es_multiplier=ES_MULTIPLIER,
            position_size=POSITION_SIZE,
            commission=COMMISSION,
            slippage=SLIPPAGE,
            start_date=val_start_date,
            end_date=val_end_date
        )
        final_end = time_module.time()
        final_duration = final_end - final_start
        print(f"Final Validation Backtest completed in {final_duration:.2f} seconds. Sharpe Ratio: {final_sharpe:.4f}")
    else:
        print("\nNo valid best individual to run final validation backtest.")

    overall_end_time = time_module.time()
    total_duration = overall_end_time - overall_start_time
    print(f"\nTotal Optimization Time: {total_duration / 60:.2f} minutes.")

# -------------------------------------------------------------
#     GENETIC ALGORITHM SUPPORT FUNCTIONS (Defined Above)
# -------------------------------------------------------------
# (Functions: create_individual, mutate, crossover, tournament_selection, evolve_population)

# -------------------------------------------------------------
#                   RUN SCRIPT
# -------------------------------------------------------------
if __name__ == '__main__':
    main()