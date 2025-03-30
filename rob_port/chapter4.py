from chapter3 import *

def calculate_min_capital_4_contracts_with_idm(multiplier, price, fx_rate, sigma_pct, idm, weight, risk_target):
    """
    Calculate the minimum capital required for 4 contracts incorporating IDM.
    
    Formula:
    = (4 × Multiplier × Price × FX rate × σ_%) ÷ (IDM × Weight × τ)
    
    Parameters:
    - multiplier (float): Futures contract multiplier
    - price (float): Current price of the instrument
    - fx_rate (float): FX conversion rate (default 1.0 for same currency)
    - sigma_pct (float): Annualized standard deviation as decimal (e.g., 0.16 for 16%)
    - idm (float): Instrument Diversification Multiplier
    - weight (float): Weight of the instrument in the portfolio
    - risk_target (float): Risk target as decimal (e.g., 0.2 for 20%)
    
    Returns:
    - float: Minimum capital required for 4 contracts
    """
    numerator = 4 * multiplier * price * fx_rate * sigma_pct
    denominator = idm * weight * risk_target
    return numerator / denominator

