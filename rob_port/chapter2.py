from .chapter1 import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np

def calculate_notional_exposure(multiplier, price, fx=1.0):
    """
    Calculate the notional exposure in base currency.

    Parameters:
    - multiplier (float): Futures contract multiplier.
    - price (float): Current price of the instrument.
    - fx (float): FX rate if the instrument is priced in another currency.

    Returns:
    - float: Notional exposure in base currency.
    """
    return multiplier * price * fx

def calculate_contract_risk(multiplier, price, annualized_std_percentage, fx=1.0):
    """
    Calculate the annualized standard deviation (risk) of a single contract position in base currency.

    Formula:
        σ(Contract, Base currency) = Notional exposure × σ_%

    Parameters:
    - multiplier (float): Futures contract multiplier.
    - price (float): Current price.
    - annualized_std_percentage (float): Annualized std dev as a decimal (e.g., 0.16 for 16%).
    - fx (float): FX rate if the instrument is priced in another currency.

    Returns:
    - float: Annualized standard deviation of a single contract in base currency.
    """
    notional_exposure = calculate_notional_exposure(multiplier, price, fx)
    return notional_exposure * annualized_std_percentage

def calculate_position_risk(num_contracts, contract_risk):
    """
    Calculate the total risk (annualized standard deviation) of the entire position in base currency.

    Formula:
        σ(Position, Base currency) = σ(Contract, Base currency) × N

    Parameters:
    - num_contracts (float): Number of contracts held.
    - contract_risk (float): Annualized standard deviation of a single contract.

    Returns:
    - float: Annualized standard deviation of the entire position in base currency.
    """
    return num_contracts * contract_risk

def calculate_target_risk(capital, risk_target):
    """
    Calculate the target risk (annualized standard deviation) in base currency.

    Formula:
        σ(Target, Base currency) = Capital × τ

    Parameters:
    - capital (float): Total capital in base currency.
    - risk_target (float): Risk fraction of capital (e.g., 0.2 for 20%).

    Returns:
    - float: Target risk in base currency.
    """
    return capital * risk_target

def calculate_position_size(capital, multiplier, price, annualized_std_percentage, risk_target=0.2, fx=1.0):
    """
    Calculate the required number of contracts (position size) to achieve a target risk.

    Formula:
        N = (Capital × τ) ÷ (Multiplier × Price × FX rate × σ_%)

    Parameters:
    - capital (float): Total capital in base currency.
    - multiplier (float): Futures contract multiplier.
    - price (float): Current price of the instrument.
    - annualized_std_percentage (float): Annualized std dev as a decimal (e.g., 0.16 for 16%).
    - risk_target (float): Target risk fraction (e.g., 0.2 = 20%).
    - fx (float): FX rate if the instrument is priced in another currency.

    Returns:
    - float: The required number of contracts (can be fractional, typically rounded).
    """
    contract_risk = calculate_contract_risk(multiplier, price, annualized_std_percentage, fx)
    target_risk = calculate_target_risk(capital, risk_target)
    
    return target_risk / contract_risk

def calculate_contract_leverage_ratio(notional_exposure, capital):
    """
    Calculate the contract leverage ratio.

    Formula:
        Contract Leverage Ratio = Notional Exposure per Contract ÷ Capital

    Parameters:
    - notional_exposure (float): The notional exposure per contract.
    - capital (float): Total capital in base currency.

    Returns:
    - float: The leverage ratio (how many times the capital is leveraged).
    """
    return notional_exposure / capital

def calculate_volatility_ratio(risk_target, annualized_std_percentage):
    """
    Calculate the volatility ratio.

    Formula:
        Volatility Ratio = τ ÷ σ_%

    Parameters:
    - risk_target (float): The target risk fraction (e.g., 0.2 for 20%).
    - annualized_std_percentage (float): Annualized standard deviation as a decimal (e.g., 0.16 for 16%).

    Returns:
    - float: The volatility ratio (used for determining risk per unit volatility).
    """
    return risk_target / annualized_std_percentage

def calculate_leverage_ratio(total_notional_exposure, capital):
    """
    Calculate the overall leverage ratio.

    Formula:
        Leverage Ratio = Total Notional Exposure ÷ Capital

    Parameters:
    - total_notional_exposure (float): Total notional exposure of position.
    - capital (float): Total capital in base currency.

    Returns:
    - float: The overall leverage ratio.
    """
    return total_notional_exposure / capital

def calculate_maximum_contracts(capital, margin_per_contract, fx=1.0):
    """
    Calculate the maximum number of contracts that can be bought based on available capital 
    and margin requirements.

    Formula:
        Maximum N = Capital ÷ (Margin per contract × FX)

    Parameters:
    - capital (float): Total trading capital in base currency.
    - margin_per_contract (float): Margin required per contract.
    - fx (float): FX rate if the margin is in a different currency.

    Returns:
    - float: The maximum number of contracts that can be bought.
    """
    return capital / (margin_per_contract * fx)

def calculate_maximum_risk_target(multiplier, price, annualized_std_percentage, margin_per_contract):
    """
    Calculate the maximum possible risk target (τ) based on margin level.

    Formula:
        Maximum τ = (Multiplier × Price × σ_%) ÷ (Margin per contract)

    Parameters:
    - multiplier (float): Futures contract multiplier.
    - price (float): Current price of the instrument.
    - annualized_std_percentage (float): Annualized standard deviation as a decimal (e.g., 0.16 for 16%).
    - margin_per_contract (float): Margin required per contract.

    Returns:
    - float: The maximum possible risk target.
    """
    return (multiplier * price * annualized_std_percentage) / margin_per_contract

def calculate_min_capital_1_contract(multiplier, price, annualized_std_percentage, risk_target, fx=1.0):
    """
    Calculate minimum capital required to trade 1 contract at given risk target.

    Formula:
        Minimum capital = (Multiplier × Price × FX × σ_%) ÷ τ

    Parameters:
    - multiplier (float): Futures contract multiplier.
    - price (float): Current price of the instrument.
    - annualized_std_percentage (float): Annualized std dev as a decimal.
    - risk_target (float): Target risk fraction.
    - fx (float): FX rate if needed.

    Returns:
    - float: Minimum capital required for 1 contract.
    """
    contract_risk = calculate_contract_risk(multiplier, price, annualized_std_percentage, fx)
    return contract_risk / risk_target

def calculate_min_capital_n_contracts(n_contracts, multiplier, price, annualized_std_percentage, risk_target, fx=1.0):
    """
    Calculate minimum capital required to trade N contracts at given risk target.

    Formula:
        Minimum capital = (N × Multiplier × Price × FX × σ_%) ÷ τ

    Parameters:
    - n_contracts (int): Number of contracts to trade.
    - multiplier (float): Futures contract multiplier.
    - price (float): Current price of the instrument.
    - annualized_std_percentage (float): Annualized std dev as a decimal.
    - risk_target (float): Target risk fraction.
    - fx (float): FX rate if needed.

    Returns:
    - float: Minimum capital required for N contracts.
    """
    contract_risk = calculate_contract_risk(multiplier, price, annualized_std_percentage, fx)
    return (n_contracts * contract_risk) / risk_target

def calculate_min_capital_4_contracts(multiplier, price, annualized_std_percentage, risk_target, fx=1.0):
    """
    Calculate minimum capital required to trade 4 contracts (book's recommended minimum).

    Parameters:
    - multiplier (float): Futures contract multiplier.
    - price (float): Current price of the instrument.
    - annualized_std_percentage (float): Annualized std dev as a decimal.
    - risk_target (float): Target risk fraction.
    - fx (float): FX rate if needed.

    Returns:
    - float: Minimum capital required for 4 contracts.
    """
    return calculate_min_capital_n_contracts(4, multiplier, price, annualized_std_percentage, risk_target, fx)

def calculate_optimal_risk_target_kelly(expected_sharpe_ratio):
    """
    Calculate optimal risk target using Kelly criterion.

    Formula:
        Optimal τ = Expected Sharpe Ratio

    Parameters:
    - expected_sharpe_ratio (float): Expected Sharpe ratio of the strategy.

    Returns:
    - float: Optimal risk target.
    """
    return expected_sharpe_ratio

def calculate_conservative_risk_target_half_kelly(expected_sharpe_ratio):
    """
    Calculate conservative risk target using half-Kelly criterion.

    Formula:
        Conservative τ = 0.5 × Expected Sharpe Ratio

    Parameters:
    - expected_sharpe_ratio (float): Expected Sharpe ratio of the strategy.

    Returns:
    - float: Conservative risk target.
    """
    return 0.5 * expected_sharpe_ratio

def plot_chapter2_risk_scaling_demo(save_path='results/chapter2_risk_scaling.png'):
    """
    Plot Chapter 2: Dynamic Risk Scaled Strategy (Strategy 2) vs. Static Buy & Hold (Strategy 1).
    Strategy 2: $100k initial capital, daily position resizing based on current equity,
                  rolling volatility (256-day window), and 20% risk target.
    Strategy 1: Minimum capital for 1 contract (based on 1st-year vol), static 1 contract position.
                  Used for metrics comparison only, not plotted.
    
    Parameters:
        save_path (str): Path to save the plot.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        df = pd.read_csv('Data/mes_daily_data.csv', parse_dates=['Time'])
        df.set_index('Time', inplace=True)
        df['price'] = df['Last'] # Use 'Last' as the price series
        df['daily_price_change_pct'] = df['price'].pct_change()
        df = df.dropna(subset=['daily_price_change_pct']) # Drop first row with NaN return
        
        instruments_df = load_instrument_data('Data/instruments.csv')
        mes_specs = get_instrument_specs('MES', instruments_df)
        multiplier = mes_specs['multiplier']
        risk_target = 0.20
        rolling_vol_window = 256

        # --- Strategy 1 Setup (Static, for metrics only) ---
        initial_period_days = 256
        if len(df) < initial_period_days:
            s1_initial_vol_series = df['daily_price_change_pct']
        else:
            s1_initial_vol_series = df['daily_price_change_pct'].iloc[:initial_period_days]
        s1_initial_annualized_vol = s1_initial_vol_series.std() * np.sqrt(256)
        s1_first_price = df['price'].iloc[0]
        
        s1_min_capital = calculate_min_capital_1_contract(
            multiplier, s1_first_price, s1_initial_annualized_vol, risk_target
        )
        s1_capital = s1_min_capital
        s1_contracts = 1.0 # Static position
        
        # Calculate Strategy 1 returns
        s1_dollar_returns = df['daily_price_change_pct'] * df['price'].shift(1) * multiplier * s1_contracts
        s1_percentage_returns = s1_dollar_returns / s1_capital
        s1_percentage_returns = s1_percentage_returns.dropna()
        s1_equity_curve = build_account_curve(s1_percentage_returns, s1_capital)
        s1_perf = calculate_comprehensive_performance(s1_equity_curve, s1_percentage_returns)

        print(f"--- Strategy 1 (Static Buy & Hold for comparison) ---")
        print(f"Initial Capital: ${s1_capital:,.0f}")
        print(f"Contracts Held: {s1_contracts}")
        print(f"Sized with Initial Vol (1st yr): {s1_initial_annualized_vol:.2%}")
        print("First 5 daily % returns for S1:", s1_percentage_returns.head().values)

        # --- Strategy 2 Setup (Dynamic Daily Resizing) ---
        s2_initial_capital = 100000.0
        s2_equity = pd.Series(index=df.index, dtype=float)
        s2_equity.iloc[0] = s2_initial_capital
        s2_contracts_held = pd.Series(index=df.index, dtype=float)
        s2_daily_percentage_returns = pd.Series(index=df.index, dtype=float)

        print(f"--- Strategy 2 (Dynamic Daily Risk Scaling) ---")
        print(f"Initial Capital: ${s2_initial_capital:,.0f}")
        print(f"Risk Target: {risk_target:.0%}")
        print(f"Rolling Vol Window: {rolling_vol_window} days")

        for i in range(1, len(df)):
            current_date = df.index[i]
            prev_date = df.index[i-1]
            
            capital_for_sizing = s2_equity.loc[prev_date]
            price_for_sizing = df['price'].loc[prev_date]
            
            # Calculate rolling volatility up to prev_date
            start_idx_vol = max(0, i - rolling_vol_window)
            rolling_returns_series = df['daily_price_change_pct'].iloc[start_idx_vol:i]

            if len(rolling_returns_series) < 2: # Need at least 2 points for std dev
                current_rolling_annual_vol = s1_initial_annualized_vol # Corrected variable name
            else:
                current_rolling_annual_vol = rolling_returns_series.std() * np.sqrt(256)
            
            if current_rolling_annual_vol == 0: # Avoid division by zero if vol is zero
                 num_contracts = 0 # Or handle as per strategy rules, e.g. hold previous position
            else:
                num_contracts = calculate_position_size(
                    capital_for_sizing, multiplier, price_for_sizing,
                    current_rolling_annual_vol, risk_target
                )
            
            s2_contracts_held.loc[current_date] = num_contracts
            
            # Calculate P&L for current day i
            price_change_points_today = df['daily_price_change_pct'].iloc[i] * df['price'].shift(1).iloc[i]
            dollar_return_today = price_change_points_today * multiplier * num_contracts
            
            percentage_return_today = dollar_return_today / capital_for_sizing if capital_for_sizing > 0 else 0
            s2_daily_percentage_returns.loc[current_date] = percentage_return_today
            s2_equity.loc[current_date] = capital_for_sizing * (1 + percentage_return_today)

            if i < 5: # Print details for the first few days for S2
                print(f"--- Day {i} (S2) ---")
                print(f"Date: {current_date.date()}")
                print(f"Capital for Sizing: {capital_for_sizing:,.2f}")
                print(f"Price for Sizing: {price_for_sizing:,.2f}")
                print(f"Rolling Vol: {current_rolling_annual_vol:.4f}")
                print(f"Num Contracts: {num_contracts:.4f}")
                print(f"Dollar Return Today: {dollar_return_today:,.2f}")
                print(f"Percentage Return Today: {percentage_return_today:.6f}")

        s2_equity_curve = s2_equity.dropna()
        s2_daily_percentage_returns = s2_daily_percentage_returns.dropna()
        s2_perf = calculate_comprehensive_performance(s2_equity_curve, s2_daily_percentage_returns)

        # --- Plotting ---
        plt.figure(figsize=(12, 8))
        
        # Plot Strategy 2 Equity Curve
        plt.subplot(2, 1, 1)
        plt.plot(s2_equity_curve.index, s2_equity_curve.values/1000, 
                'b-', linewidth=1.5, 
                label=f'Strategy 2: Dynamic Risk Scaled 20% (${s2_initial_capital/1000:.0f}K capital)')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($K)', fontsize=12)
        plt.title('Chapter 2: Dynamic Risk Scaled Strategy (Daily Rebalancing)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot Strategy 2 Drawdown
        plt.subplot(2, 1, 2)
        s2_drawdown = calculate_maximum_drawdown(s2_equity_curve)['drawdown_series'] * 100
        plt.fill_between(s2_drawdown.index, s2_drawdown.values, 0, 
                        color='blue', alpha=0.3, label='Strategy 2 Drawdown')
        plt.plot(s2_drawdown.index, s2_drawdown.values, 'b-', linewidth=1)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.title('Drawdown Comparison', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        for ax in plt.gcf().get_axes():
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Performance Text
        start_date = df.index[0].strftime('%Y-%m-%d')
        end_date = df.index[-1].strftime('%Y-%m-%d')
        
        textstr = f"""Performance Comparison (Real MES Data):
Strategy 1 (Static B&H, 1 Contract - For Metrics Only):
  Capital: ${s1_capital:,.0f}
  Total Return: {s1_perf['total_return']:.1%}
  Annualized Return: {s1_perf['annualized_return']:.1%}
  Strategy Volatility: {s1_perf['annualized_volatility']:.1%}
  Sharpe Ratio: {s1_perf['sharpe_ratio']:.3f}
  Max Drawdown: {s1_perf['max_drawdown_pct']:.1f}%

Strategy 2 (Dynamic Risk Scaled 20%, Daily Rebalance):
  Initial Capital: ${s2_initial_capital:,.0f}
  Total Return: {s2_perf['total_return']:.1%}
  Annualized Return: {s2_perf['annualized_return']:.1%}
  Strategy Volatility: {s2_perf['annualized_volatility']:.1%}
  Sharpe Ratio: {s2_perf['sharpe_ratio']:.3f}
  Max Drawdown: {s2_perf['max_drawdown_pct']:.1f}%

Backtest Period: {start_date} to {end_date}
Rolling Vol Window for Strategy 2: {rolling_vol_window} days"""
        
        plt.figtext(0.02, 0.01, textstr, fontsize=8, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.43)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Chapter 2 dynamic strategy comparison saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting Chapter 2 demo: {e}")
        import traceback
        traceback.print_exc()

def get_recommended_risk_targets():
    """
    Get recommended risk targets from the book for different scenarios.

    Returns:
    - dict: Dictionary of recommended risk targets.
    """
    return {
        'margin_constraint_sp500': 3.13,  # Maximum based on margin: 313%
        'large_loss_constraint': 0.267,   # Based on surviving 1987 crash: 26.7%
        'personal_appetite_conservative': 0.20,  # Book's conservative recommendation: 20%
        'personal_appetite_aggressive': 0.50,    # Higher risk appetite
        'optimal_kelly_sp500': 0.47,      # Full Kelly criterion for S&P 500: 47%
        'half_kelly_sp500': 0.235         # Conservative half-Kelly: 23.5%
    }

def calculate_continuous_position_size(capital, multiplier, price, annualized_std_percentage, risk_target=0.2, fx=1.0):
    """
    Calculate the exact position size for continuous trading (allows fractional contracts).

    Parameters:
    - capital (float): Current trading capital.
    - multiplier (float): Futures contract multiplier.
    - price (float): Current price of the instrument.
    - annualized_std_percentage (float): Annualized std dev as a decimal.
    - risk_target (float): Target risk fraction.
    - fx (float): FX rate if needed.

    Returns:
    - float: Exact position size (can be fractional).
    """
    return calculate_position_size(capital, multiplier, price, annualized_std_percentage, risk_target, fx)

def calculate_discrete_position_size(capital, multiplier, price, annualized_std_percentage, risk_target=0.2, fx=1.0):
    """
    Calculate the discrete position size (rounded to nearest whole contract).

    Parameters:
    - capital (float): Current trading capital.
    - multiplier (float): Futures contract multiplier.
    - price (float): Current price of the instrument.
    - annualized_std_percentage (float): Annualized std dev as a decimal.
    - risk_target (float): Target risk fraction.
    - fx (float): FX rate if needed.

    Returns:
    - int: Rounded position size (whole contracts only).
    """
    continuous_size = calculate_position_size(capital, multiplier, price, annualized_std_percentage, risk_target, fx)
    return round(continuous_size)

def main():
    """
    Main function to test the risk scaling calculations using examples from the book.
    """
    print("=" * 60)
    print("CHAPTER 2: BUY AND HOLD WITH RISK SCALING")
    print("=" * 60)

    # Load instruments data
    instruments_df = load_instrument_data('Data/instruments.csv')
    
    # Get MES specifications
    mes_specs = get_instrument_specs('MES', instruments_df)
    multiplier = mes_specs['multiplier']
    
    # Example from the book: S&P 500 micro future
    print("\n----- Example: S&P 500 Micro Future (MES) -----")
    price = 4500  # Current S&P 500 price (book example)
    annualized_std_pct = 0.16  # 16% annual volatility (book example)
    capital = 100000  # $100,000 capital (book example)
    risk_target = 0.20  # 20% risk target (conservative book recommendation)
    
    print(f"Instrument: {mes_specs['name']}")
    print(f"Multiplier: ${multiplier}")
    print(f"Current Price: {price}")
    print(f"Annual Volatility: {annualized_std_pct:.1%}")
    print(f"Capital: ${capital:,.0f}")
    print(f"Risk Target: {risk_target:.1%}")
    
    # Calculate notional exposure
    notional_exposure = calculate_notional_exposure(multiplier, price)
    print(f"\nNotional Exposure per Contract: ${notional_exposure:,.0f}")
    
    # Calculate contract risk
    contract_risk = calculate_contract_risk(multiplier, price, annualized_std_pct)
    print(f"Risk per Contract (annual $): ${contract_risk:,.0f}")
    
    # Calculate target risk
    target_risk = calculate_target_risk(capital, risk_target)
    print(f"Target Risk (annual $): ${target_risk:,.0f}")
    
    # Calculate position size
    position_size = calculate_position_size(capital, multiplier, price, annualized_std_pct, risk_target)
    discrete_position = calculate_discrete_position_size(capital, multiplier, price, annualized_std_pct, risk_target)
    
    print(f"\nOptimal Position Size: {position_size:.3f} contracts")
    print(f"Discrete Position Size: {discrete_position} contracts")
    
    # Verify book calculation: N = (100,000 × 0.2) ÷ (5 × 4500 × 1 × 0.16) = 5.556
    book_position_calc = (100000 * 0.2) / (5 * 4500 * 1 * 0.16)
    print(f"Book verification: {book_position_calc:.3f} contracts (expected: 5.556)")
    
    # Calculate leverage ratios
    contract_leverage = calculate_contract_leverage_ratio(notional_exposure, capital)
    volatility_ratio = calculate_volatility_ratio(risk_target, annualized_std_pct)
    
    print(f"\nContract Leverage Ratio: {contract_leverage:.3f}")
    print(f"Volatility Ratio: {volatility_ratio:.3f}")
    print(f"Position Size (alternative): {volatility_ratio / contract_leverage:.3f}")
    
    # Verify ratios match book:
    # Contract leverage = 22500 / 100000 = 0.225
    # Volatility ratio = 0.2 / 0.16 = 1.25
    # Position = 1.25 / 0.225 = 5.556 → but this seems wrong in the book
    book_contract_leverage = 22500 / 100000
    book_volatility_ratio = 0.2 / 0.16
    print(f"Book contract leverage: {book_contract_leverage:.3f} (expected: 0.225)")
    print(f"Book volatility ratio: {book_volatility_ratio:.3f} (expected: 1.25)")
    
    # Minimum capital calculations
    print("\n----- Minimum Capital Requirements -----")
    min_cap_1 = calculate_min_capital_1_contract(multiplier, price, annualized_std_pct, risk_target)
    min_cap_4 = calculate_min_capital_4_contracts(multiplier, price, annualized_std_pct, risk_target)
    
    print(f"Minimum Capital for 1 Contract: ${min_cap_1:,.0f}")
    print(f"Minimum Capital for 4 Contracts: ${min_cap_4:,.0f}")
    
    # Verify book calculation: (5 × 4500 × 1.0 × 0.16) ÷ 0.2 = $18,000
    book_min_calc = (5 * 4500 * 1.0 * 0.16) / 0.2
    print(f"Book verification: ${book_min_calc:,.0f} (should be $18,000)")
    
    # Risk target recommendations
    print("\n----- Risk Target Recommendations -----")
    risk_targets = get_recommended_risk_targets()
    
    for scenario, target in risk_targets.items():
        print(f"{scenario.replace('_', ' ').title()}: {target:.1%}")
    
    # Test with different capital amounts (examples from book)
    print("\n----- Position Sizing for Different Capital Levels -----")
    capital_levels = [5000, 22500, 45000, 100000, 250000]  # Book examples
    
    for cap in capital_levels:
        pos_size = calculate_position_size(cap, multiplier, price, annualized_std_pct, risk_target)
        discrete_pos = calculate_discrete_position_size(cap, multiplier, price, annualized_std_pct, risk_target)
        min_cap_needed = calculate_min_capital_1_contract(multiplier, price, annualized_std_pct, risk_target)
        
        print(f"Capital: ${cap:>7,.0f} | Position: {pos_size:>6.2f} | Discrete: {discrete_pos:>2} | "
              f"Can Trade: {'Yes' if cap >= min_cap_needed else 'No'}")
    
    # Example with margin constraints (from book)
    print("\n----- Margin Constraint Analysis -----")
    margin_per_contract = 1150  # S&P 500 micro future margin (book example)
    
    # Use higher capital for margin example
    large_capital = 100000  # Use standard example capital
    max_contracts = calculate_maximum_contracts(large_capital, margin_per_contract)
    max_risk_target = calculate_maximum_risk_target(multiplier, price, annualized_std_pct, margin_per_contract)
    
    print(f"Margin per Contract: ${margin_per_contract}")
    print(f"Capital for margin example: ${large_capital:,.0f}")
    print(f"Maximum Contracts (margin limit): {max_contracts:.1f}")
    print(f"Maximum Risk Target (margin limit): {max_risk_target:.1%}")
    
    # Compare with our target
    large_position = calculate_position_size(large_capital, multiplier, price, annualized_std_pct, risk_target)
    actual_position = min(large_position, max_contracts)
    print(f"Optimal Position Size: {large_position:.2f} contracts")
    print(f"Actual Position (considering margin): {actual_position:.2f} contracts")
    
    # Test with other instruments from the file
    print("\n----- Risk Scaling for Other Instruments -----")
    test_instruments = ['MYM', 'MNQ', 'VIX', 'ZN']  # Mix of equity, volatility, bond
    
    for symbol in test_instruments:
        try:
            specs = get_instrument_specs(symbol, instruments_df)
            # Use generic price and volatility estimates
            test_price = 1000 if symbol == 'VIX' else 4000
            test_vol = 0.25 if symbol == 'VIX' else 0.15
            
            pos_size = calculate_position_size(capital, specs['multiplier'], test_price, test_vol, risk_target)
            min_cap = calculate_min_capital_1_contract(specs['multiplier'], test_price, test_vol, risk_target)
            
            print(f"{symbol} ({specs['name'][:30]}): {pos_size:>6.2f} contracts | Min Cap: ${min_cap:>8,.0f}")
            
        except Exception as e:
            print(f"{symbol}: Error - {e}")
    
    # Plot Chapter 2 risk scaling demonstration
    plot_chapter2_risk_scaling_demo()  # Uses real data and calculates minimum capital

if __name__ == "__main__":
    main()

