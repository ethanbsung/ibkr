from chapter3 import *

def calculate_position_size_per_instrument(instrument_code: str, weight: float, capital: float, multiplier: float, fx_rate: float, risk_target: float, sigma_pct: float):
    """Calculate position size per instrument based on risk targeting."""
    # Calculate position size using the formula
    position_size = (capital * weight * risk_target) / (multiplier * fx_rate * sigma_pct)
    return position_size

