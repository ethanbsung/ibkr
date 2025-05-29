from chapter1 import *

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

def get_recommended_risk_targets():
    """
    Get recommended risk targets from the book for different scenarios.

    Returns:
    - dict: Dictionary of recommended risk targets.
    """
    return {
        'margin_constraint_sp500': 3.13,  # Maximum based on margin (very high risk)
        'large_loss_constraint': 0.267,   # Based on surviving 30% crash
        'personal_appetite_conservative': 0.20,  # Book's conservative recommendation
        'personal_appetite_aggressive': 0.50,    # Higher risk appetite
        'optimal_kelly_sp500': 0.47,      # Based on Kelly criterion for S&P 500
        'half_kelly_sp500': 0.235         # Conservative half-Kelly
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
    instruments_df = load_instrument_data()
    
    # Get MES specifications
    mes_specs = get_instrument_specs('MES', instruments_df)
    multiplier = mes_specs['multiplier']
    
    # Example from the book: S&P 500 micro future
    print("\n----- Example: S&P 500 Micro Future (MES) -----")
    price = 4500  # Current S&P 500 price
    annualized_std_pct = 0.16  # 16% annual volatility (book example)
    capital = 100000  # $100,000 capital
    risk_target = 0.20  # 20% risk target
    
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
    
    # Calculate leverage ratios
    contract_leverage = calculate_contract_leverage_ratio(notional_exposure, capital)
    volatility_ratio = calculate_volatility_ratio(risk_target, annualized_std_pct)
    
    print(f"\nContract Leverage Ratio: {contract_leverage:.3f}")
    print(f"Volatility Ratio: {volatility_ratio:.3f}")
    print(f"Position Size (alternative): {volatility_ratio / contract_leverage:.3f}")
    
    # Minimum capital calculations
    print("\n----- Minimum Capital Requirements -----")
    min_cap_1 = calculate_min_capital_1_contract(multiplier, price, annualized_std_pct, risk_target)
    min_cap_4 = calculate_min_capital_4_contracts(multiplier, price, annualized_std_pct, risk_target)
    
    print(f"Minimum Capital for 1 Contract: ${min_cap_1:,.0f}")
    print(f"Minimum Capital for 4 Contracts: ${min_cap_4:,.0f}")
    
    # Risk target recommendations
    print("\n----- Risk Target Recommendations -----")
    risk_targets = get_recommended_risk_targets()
    
    for scenario, target in risk_targets.items():
        print(f"{scenario.replace('_', ' ').title()}: {target:.1%}")
    
    # Test with different capital amounts
    print("\n----- Position Sizing for Different Capital Levels -----")
    capital_levels = [5000, 25000, 50000, 100000, 250000]
    
    for cap in capital_levels:
        pos_size = calculate_position_size(cap, multiplier, price, annualized_std_pct, risk_target)
        discrete_pos = calculate_discrete_position_size(cap, multiplier, price, annualized_std_pct, risk_target)
        min_cap_needed = calculate_min_capital_1_contract(multiplier, price, annualized_std_pct, risk_target)
        
        print(f"Capital: ${cap:>7,.0f} | Position: {pos_size:>6.2f} | Discrete: {discrete_pos:>2} | "
              f"Can Trade: {'Yes' if cap >= min_cap_needed else 'No'}")
    
    # Example with margin constraints
    print("\n----- Margin Constraint Analysis -----")
    margin_per_contract = 1150  # Typical MES margin requirement
    
    max_contracts = calculate_maximum_contracts(capital, margin_per_contract)
    max_risk_target = calculate_maximum_risk_target(multiplier, price, annualized_std_pct, margin_per_contract)
    
    print(f"Margin per Contract: ${margin_per_contract}")
    print(f"Maximum Contracts (margin limit): {max_contracts:.1f}")
    print(f"Maximum Risk Target (margin limit): {max_risk_target:.1%}")
    
    # Compare with our target
    actual_position = min(discrete_position, int(max_contracts))
    print(f"Actual Position (considering margin): {actual_position} contracts")
    
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

if __name__ == "__main__":
    main()

