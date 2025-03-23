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
    notional_exposure = calculate_notional_exposure(multiplier, price, fx)
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

# N = Volatility Ratio / Contract Leverage Ratio
# Leverage Ratio = Volatility Ratio
# Required amount of leverage is equal to the volatility ratio

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

def calculate_min_capital_4_contracts(multiplier, price, annualized_std_percentage, risk_target, fx=1):
    contract_risk = calculate_contract_risk(multiplier, price, annualized_std_percentage, fx)
    return (4 * contract_risk) / risk_target

