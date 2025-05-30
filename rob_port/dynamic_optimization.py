import numpy as np
import pandas as pd
from chapter3 import *
from chapter1 import *
import warnings
warnings.filterwarnings('ignore')

# Import specific functions we need that might not be available through star imports
try:
    from chapter4 import calculate_idm_from_correlations, calculate_position_size_with_idm
except ImportError:
    # Fallback implementation if not available
    def calculate_idm_from_correlations(weights, correlation_matrix):
        """
        Fallback implementation of IDM calculation.
        """
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        
        # Align weights with correlation matrix
        aligned_weights = weights.reindex(correlation_matrix.index).fillna(0)
        w = aligned_weights.values
        
        # Calculate portfolio variance: w^T * Σ * w
        portfolio_variance = np.dot(w, np.dot(correlation_matrix.values, w))
        
        # IDM = 1 / sqrt(portfolio_variance)
        idm = 1.0 / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 1.0
        
        return idm
    
    def calculate_position_size_with_idm(capital, weight, idm, multiplier, price, fx_rate, sigma_pct, risk_target=0.2):
        """
        Fallback implementation of position sizing with IDM.
        """
        if sigma_pct <= 0 or np.isnan(sigma_pct):
            return 0
        
        numerator = capital * idm * weight * risk_target
        denominator = multiplier * price * fx_rate * sigma_pct
        
        return numerator / denominator

#####   DYNAMIC OPTIMIZATION CORE FUNCTIONS   #####

def calculate_optimal_unrounded_position(capital, combined_forecast, idm, weight, sigma_pct, 
                                       multiplier, price, fx_rate=1.0, risk_target=0.2):
    """
    Calculate optimal unrounded position for an instrument.
    
    Formula: N_i = (Capped combined forecast × Capital × IDM × Weight_i × τ) ÷ 
                   (10 × Multiplier_i × Price_i × FX_i × σ_i)
    
    Parameters:
        capital (float): Trading capital
        combined_forecast (float): Combined forecast (capped at ±20)
        idm (float): Instrument diversification multiplier
        weight (float): Instrument weight in portfolio
        sigma_pct (float): Volatility forecast as decimal
        multiplier (float): Contract multiplier
        price (float): Current price
        fx_rate (float): FX rate to base currency
        risk_target (float): Target risk fraction
    
    Returns:
        float: Optimal unrounded position
    """
    if sigma_pct <= 0 or np.isnan(sigma_pct) or price <= 0:
        return 0.0
    
    # Cap forecast at ±20 as per book
    capped_forecast = max(-20, min(20, combined_forecast))
    
    numerator = capped_forecast * capital * idm * weight * risk_target
    denominator = multiplier * price * fx_rate * sigma_pct
    
    return numerator / denominator

def calculate_weight_per_contract(multiplier, price, fx_rate, capital):
    """
    Calculate weight per contract.
    
    Formula: Weight per contract = Notional exposure per contract ÷ Capital
             = (Multiplier × Price × FX rate) ÷ Capital
    
    Parameters:
        multiplier (float): Contract multiplier
        price (float): Current price
        fx_rate (float): FX rate
        capital (float): Trading capital
    
    Returns:
        float: Weight per contract
    """
    notional_exposure = multiplier * price * fx_rate
    return notional_exposure / capital

def calculate_optimal_weight(optimal_position, weight_per_contract):
    """
    Calculate optimal portfolio weight.
    
    Formula: w_i = N_i × Weight per contract_i
    
    Parameters:
        optimal_position (float): Optimal unrounded position
        weight_per_contract (float): Weight per contract
    
    Returns:
        float: Optimal portfolio weight
    """
    return optimal_position * weight_per_contract

def calculate_tracking_error_weight(optimal_weight, current_weight):
    """
    Calculate tracking error weight.
    
    Formula: e_i = w_i* - w_i
    
    Parameters:
        optimal_weight (float): Optimal weight
        current_weight (float): Current weight
    
    Returns:
        float: Tracking error weight
    """
    return optimal_weight - current_weight

def calculate_tracking_error_std(tracking_error_weights, covariance_matrix):
    """
    Calculate standard deviation of tracking error portfolio.
    
    Formula: √(e^T.Σ.e)
    
    Parameters:
        tracking_error_weights (pd.Series): Tracking error weights
        covariance_matrix (pd.DataFrame): Covariance matrix
    
    Returns:
        float: Standard deviation of tracking error
    """
    # Align weights with covariance matrix
    aligned_weights = tracking_error_weights.reindex(covariance_matrix.index).fillna(0)
    
    # Calculate e^T.Σ.e
    tracking_variance = np.dot(aligned_weights.values, 
                              np.dot(covariance_matrix.values, aligned_weights.values))
    
    return np.sqrt(max(0, tracking_variance))

#####   COVARIANCE MATRIX CALCULATION   #####

def calculate_percentage_returns_covariance(returns_matrix, ewma_span=32):
    """
    Calculate covariance matrix from percentage returns using EWMA.
    
    Parameters:
        returns_matrix (pd.DataFrame): Daily percentage returns matrix
        ewma_span (int): EWMA span for volatility (32 days as per book)
    
    Returns:
        pd.DataFrame: Covariance matrix
    """
    # Calculate EWMA volatilities
    volatilities = {}
    for symbol in returns_matrix.columns:
        ewma_vol = calculate_ewma_volatility(returns_matrix[symbol], span=ewma_span, annualize=True)
        volatilities[symbol] = ewma_vol.iloc[-1] if not ewma_vol.empty else 0.16  # fallback
    
    # Calculate correlation matrix using last 6 months of weekly returns
    # Convert to weekly returns
    weekly_returns = returns_matrix.resample('W').sum()
    correlation_matrix = weekly_returns.tail(26).corr()  # ~6 months of weekly data
    
    # Create covariance matrix: Σ = D * ρ * D where D is diagonal of volatilities
    vol_series = pd.Series(volatilities).reindex(correlation_matrix.index).fillna(0.16)
    vol_matrix = np.outer(vol_series.values, vol_series.values)
    covariance_matrix = vol_matrix * correlation_matrix.values
    
    return pd.DataFrame(covariance_matrix, index=correlation_matrix.index, columns=correlation_matrix.columns)

#####   COST CALCULATIONS   #####

def calculate_cost_in_weight_terms(spread_cost_currency, commission_currency, capital, weight_per_contract):
    """
    Calculate cost in weight terms.
    
    Formula: w_i^c = (C_i ÷ Capital) ÷ Weight per contract_i
    
    Parameters:
        spread_cost_currency (float): Spread cost in currency
        commission_currency (float): Commission in currency
        capital (float): Trading capital
        weight_per_contract (float): Weight per contract
    
    Returns:
        float: Cost in weight terms
    """
    total_cost = spread_cost_currency + commission_currency
    if weight_per_contract == 0:
        return 0
    return (total_cost / capital) / weight_per_contract

def calculate_trade_in_weight_terms(previous_weight, current_weight):
    """
    Calculate trade size in weight terms.
    
    Formula: Δ_i = abs(w_i^p - w_i)
    
    Parameters:
        previous_weight (float): Previous portfolio weight
        current_weight (float): Current portfolio weight
    
    Returns:
        float: Trade size in weight terms
    """
    return abs(previous_weight - current_weight)

def calculate_total_cost_weight_terms(trades_weight_terms, costs_weight_terms):
    """
    Calculate total cost of all trades in weight terms.
    
    Formula: δ = sum(Δ_0*w_0^c + Δ_1*w_1^c + Δ_2*w_2^c + ...)
    
    Parameters:
        trades_weight_terms (list): Trade sizes in weight terms
        costs_weight_terms (list): Costs in weight terms
    
    Returns:
        float: Total cost in weight terms
    """
    return sum(trade * cost for trade, cost in zip(trades_weight_terms, costs_weight_terms))

def calculate_tracking_error_with_cost_penalty(tracking_error_std, total_cost_weight_terms, cost_multiplier=50):
    """
    Calculate tracking error with cost penalty.
    
    Formula: Tracking error = √(e^T.Σ.e) + 50δ
    
    Parameters:
        tracking_error_std (float): Standard deviation of tracking error
        total_cost_weight_terms (float): Total cost in weight terms
        cost_multiplier (float): Cost penalty multiplier (50 as per book)
    
    Returns:
        float: Tracking error with cost penalty
    """
    return tracking_error_std + cost_multiplier * total_cost_weight_terms

#####   GREEDY ALGORITHM IMPLEMENTATION   #####

def greedy_optimization_step(current_positions, optimal_positions, weight_per_contract, 
                           covariance_matrix, cost_weight_terms, cost_multiplier=50):
    """
    Perform one step of the greedy algorithm.
    
    Parameters:
        current_positions (dict): Current positions {symbol: position}
        optimal_positions (dict): Optimal unrounded positions {symbol: position}
        weight_per_contract (dict): Weight per contract {symbol: weight}
        covariance_matrix (pd.DataFrame): Covariance matrix
        cost_weight_terms (dict): Cost per trade in weight terms {symbol: cost}
        cost_multiplier (float): Cost penalty multiplier
    
    Returns:
        tuple: (best_symbol, best_increment, best_tracking_error)
    """
    best_symbol = None
    best_increment = 0
    best_tracking_error = float('inf')
    
    current_best_solution = current_positions.copy()
    
    for symbol in optimal_positions.keys():
        optimal_pos = optimal_positions[symbol]
        current_pos = current_positions.get(symbol, 0)
        
        # Try incrementing by +1 or -1 based on optimal direction
        if optimal_pos > current_pos:
            increment = 1
        elif optimal_pos < current_pos:
            increment = -1
        else:
            continue  # Already at optimal
        
        # Test this increment
        test_positions = current_positions.copy()
        test_positions[symbol] = current_pos + increment
        
        # Calculate tracking error for this solution
        tracking_error = calculate_solution_tracking_error(
            test_positions, optimal_positions, weight_per_contract, 
            covariance_matrix, cost_weight_terms, cost_multiplier
        )
        
        if tracking_error < best_tracking_error:
            best_tracking_error = tracking_error
            best_symbol = symbol
            best_increment = increment
    
    return best_symbol, best_increment, best_tracking_error

def calculate_solution_tracking_error(positions, optimal_positions, weight_per_contract, 
                                    covariance_matrix, cost_weight_terms, cost_multiplier=50):
    """
    Calculate tracking error for a given position solution.
    
    Parameters:
        positions (dict): Current positions {symbol: position}
        optimal_positions (dict): Optimal positions {symbol: position}
        weight_per_contract (dict): Weight per contract {symbol: weight}
        covariance_matrix (pd.DataFrame): Covariance matrix
        cost_weight_terms (dict): Cost per trade in weight terms
        cost_multiplier (float): Cost penalty multiplier
    
    Returns:
        float: Total tracking error including cost penalty
    """
    # Calculate tracking error weights
    tracking_weights = {}
    trades = {}
    
    for symbol in optimal_positions.keys():
        current_pos = positions.get(symbol, 0)
        optimal_pos = optimal_positions[symbol]
        wpc = weight_per_contract.get(symbol, 0)
        
        current_weight = current_pos * wpc
        optimal_weight = optimal_pos * wpc
        
        tracking_weights[symbol] = optimal_weight - current_weight
        trades[symbol] = abs(optimal_weight - current_weight)
    
    # Calculate tracking error standard deviation
    tracking_weights_series = pd.Series(tracking_weights)
    tracking_error_std = calculate_tracking_error_std(tracking_weights_series, covariance_matrix)
    
    # Calculate cost penalty
    total_cost = sum(trades[symbol] * cost_weight_terms.get(symbol, 0) 
                    for symbol in trades.keys())
    
    return tracking_error_std + cost_multiplier * total_cost

def run_greedy_algorithm(optimal_positions, initial_positions, weight_per_contract, 
                        covariance_matrix, cost_weight_terms, cost_multiplier=50, max_iterations=1000):
    """
    Run the complete greedy algorithm to find optimal integer positions.
    
    Parameters:
        optimal_positions (dict): Optimal unrounded positions
        initial_positions (dict): Starting positions (usually all zeros)
        weight_per_contract (dict): Weight per contract for each instrument
        covariance_matrix (pd.DataFrame): Covariance matrix
        cost_weight_terms (dict): Cost per trade in weight terms
        cost_multiplier (float): Cost penalty multiplier
        max_iterations (int): Maximum iterations to prevent infinite loops
    
    Returns:
        dict: Optimized integer positions
    """
    current_positions = initial_positions.copy()
    current_best_tracking_error = float('inf')
    
    for iteration in range(max_iterations):
        # Try one step of improvement
        best_symbol, best_increment, best_tracking_error = greedy_optimization_step(
            current_positions, optimal_positions, weight_per_contract, 
            covariance_matrix, cost_weight_terms, cost_multiplier
        )
        
        # Check if we found an improvement
        if best_tracking_error < current_best_tracking_error and best_symbol is not None:
            current_positions[best_symbol] = current_positions.get(best_symbol, 0) + best_increment
            current_best_tracking_error = best_tracking_error
        else:
            # No improvement found, algorithm converged
            break
    
    return current_positions

#####   BUFFERING IMPLEMENTATION   #####

def calculate_tracking_error_buffer(average_risk_target, buffer_fraction=0.05):
    """
    Calculate tracking error buffer.
    
    Formula: B_σ = 0.05τ (5% of risk target as per book)
    
    Parameters:
        average_risk_target (float): Average risk target (τ)
        buffer_fraction (float): Buffer fraction (0.05 = 5%)
    
    Returns:
        float: Tracking error buffer
    """
    return buffer_fraction * average_risk_target

def calculate_adjustment_factor(current_tracking_error, buffer, risk_target):
    """
    Calculate adjustment factor for buffered trading.
    
    Formula: α = max([T - B_σ] ÷ T, 0)
    
    Parameters:
        current_tracking_error (float): Current tracking error
        buffer (float): Tracking error buffer
        risk_target (float): Risk target
    
    Returns:
        float: Adjustment factor (0 to 1)
    """
    if risk_target <= 0:
        return 0
    
    return max((current_tracking_error - buffer) / risk_target, 0)

def calculate_required_trades_with_buffering(current_positions, optimal_positions, adjustment_factor):
    """
    Calculate required trades with buffering adjustment.
    
    Formula: Required trade_i = round(α × [N_i* - P_i])
    
    Parameters:
        current_positions (dict): Current positions
        optimal_positions (dict): Optimal positions
        adjustment_factor (float): Adjustment factor
    
    Returns:
        dict: Required trades for each instrument
    """
    required_trades = {}
    
    for symbol in optimal_positions.keys():
        current_pos = current_positions.get(symbol, 0)
        optimal_pos = optimal_positions[symbol]
        
        trade_amount = adjustment_factor * (optimal_pos - current_pos)
        required_trades[symbol] = round(trade_amount)
    
    return required_trades

#####   DYNAMIC OPTIMIZATION STRATEGY   #####

def calculate_dynamic_portfolio_positions(instruments_data, capital, current_positions, 
                                        portfolio_weights, returns_matrix, risk_target=0.2, 
                                        cost_multiplier=50, use_buffering=True, buffer_fraction=0.05):
    """
    Calculate dynamically optimized portfolio positions for a given day.
    
    Parameters:
        instruments_data (dict): Instrument data {symbol: {'price': price, 'volatility': vol, 'specs': specs}}
        capital (float): Current trading capital
        current_positions (dict): Current positions {symbol: contracts}
        portfolio_weights (dict): Portfolio weights {symbol: weight}
        returns_matrix (pd.DataFrame): Historical returns matrix for covariance calculation
        risk_target (float): Target risk fraction
        cost_multiplier (float): Cost penalty multiplier
        use_buffering (bool): Whether to use buffering
        buffer_fraction (float): Buffer fraction for tracking error
    
    Returns:
        dict: Optimized positions and diagnostics
    """
    try:
        # Calculate covariance matrix
        covariance_matrix = calculate_percentage_returns_covariance(returns_matrix)
        
        # Calculate IDM from portfolio weights and covariance
        if len(portfolio_weights) > 1:
            correlation_matrix = covariance_matrix / np.outer(np.sqrt(np.diag(covariance_matrix)), 
                                                             np.sqrt(np.diag(covariance_matrix)))
            correlation_matrix = pd.DataFrame(correlation_matrix, 
                                            index=covariance_matrix.index, 
                                            columns=covariance_matrix.columns)
            valid_portfolio_weights = {symbol: weight for symbol, weight in portfolio_weights.items() if symbol in covariance_matrix.index}
            idm = calculate_idm_from_correlations(valid_portfolio_weights, correlation_matrix)
        else:
            idm = 1.0
        
        # Calculate optimal unrounded positions and weight per contract
        optimal_positions = {}
        weight_per_contract = {}
        cost_weight_terms = {}
        
        # Track total target weight to prevent excessive leverage
        total_target_weight = 0
        
        # Filter instruments_data to only include those with positive price and volatility
        valid_instruments_for_optimization = {}
        for symbol, data_item in instruments_data.items():
            if (data_item.get('price') is not None and not pd.isna(data_item['price']) and data_item['price'] > 0 and
                data_item.get('volatility') is not None and not pd.isna(data_item['volatility']) and data_item['volatility'] > 0 and
                data_item['specs'].get('multiplier') is not None and data_item['specs']['multiplier'] > 0):
                valid_instruments_for_optimization[symbol] = data_item
        
        for symbol, weight in valid_portfolio_weights.items():
            if symbol in valid_instruments_for_optimization:
                data = valid_instruments_for_optimization[symbol]
                specs = data['specs']
                
                # Use a default combined_forecast of 1.0 as neutral forecast
                # The book's "10" factor in the formula is a scaling convention difference
                combined_forecast = 1.0
                
                # Calculate optimal position size (can be fractional)
                optimal_pos = calculate_optimal_unrounded_position(
                    capital, combined_forecast, idm, weight, data['volatility'],
                    specs['multiplier'], data['price'], 1.0, risk_target
                )
                optimal_positions[symbol] = optimal_pos
                
                wpc = calculate_weight_per_contract(
                    specs['multiplier'], data['price'], 1.0, capital
                )
                weight_per_contract[symbol] = wpc
                
                # Calculate cost in weight terms (using SR cost as approximation)
                cost_pct = specs['sr_cost'] * data['volatility']  # Approximate cost
                cost_weight_terms[symbol] = cost_pct / wpc if wpc != 0 else float('inf') # If wpc is 0, cost is effectively infinite
                
                # Track total target weight
                total_target_weight += weight
        
        if not optimal_positions: # If no instruments are left after validation
            raise ValueError("No valid instruments for dynamic optimization after price/volatility checks.")
        
        # Scale risk target if total weight exceeds 100% to prevent excessive leverage
        if total_target_weight > 1.0:
            risk_scaling_factor = 1.0 / total_target_weight
            print(f"DEBUG: Scaling risk target by {risk_scaling_factor:.3f} to prevent leverage (total weight: {total_target_weight:.1%})")
            # Recalculate positions with scaled risk target
            scaled_risk_target = risk_target * risk_scaling_factor
            for symbol in optimal_positions.keys():
                data = valid_instruments_for_optimization[symbol]
                optimal_pos = calculate_optimal_unrounded_position(
                    capital, combined_forecast, idm, valid_portfolio_weights[symbol], data['volatility'],
                    data['specs']['multiplier'], data['price'], 1.0, scaled_risk_target
                )
                optimal_positions[symbol] = optimal_pos
        
        # Run greedy algorithm only with instruments that have defined optimal_positions
        initial_positions_for_greedy = {s: current_positions.get(s, 0) for s in optimal_positions.keys()}
        
        # Ensure covariance matrix is subset to only instruments in optimal_positions
        opt_symbols = list(optimal_positions.keys())
        covariance_matrix_subset = covariance_matrix.loc[opt_symbols, opt_symbols]

        optimized_positions_greedy = run_greedy_algorithm(
            optimal_positions, initial_positions_for_greedy, weight_per_contract,
            covariance_matrix_subset, cost_weight_terms, cost_multiplier
        )
        
        # Apply buffering if requested
        if use_buffering:
            # Calculate current tracking error
            current_tracking_error = calculate_solution_tracking_error(
                current_positions, optimal_positions, weight_per_contract,
                covariance_matrix, cost_weight_terms, cost_multiplier
            )
            
            # Calculate buffer and adjustment factor
            buffer = calculate_tracking_error_buffer(risk_target, buffer_fraction)
            adjustment_factor = calculate_adjustment_factor(current_tracking_error, buffer, risk_target)
            
            # Calculate required trades with buffering
            required_trades = calculate_required_trades_with_buffering(
                current_positions, optimized_positions_greedy, adjustment_factor
            )
            
            # Apply trades to current positions
            final_positions = current_positions.copy()
            for symbol, trade in required_trades.items():
                final_positions[symbol] = final_positions.get(symbol, 0) + trade
        else:
            final_positions = optimized_positions_greedy
            adjustment_factor = 1.0
            buffer = 0.0
            current_tracking_error = 0.0
        
        return {
            'positions': final_positions,
            'optimal_unrounded': optimal_positions,
            'idm': idm,
            'tracking_error': current_tracking_error if use_buffering else 0.0,
            'adjustment_factor': adjustment_factor if use_buffering else 1.0,
            'buffer': buffer if use_buffering else 0.0,
            'covariance_matrix': covariance_matrix,
            'weight_per_contract': weight_per_contract
        }
        
    except Exception as e:
        # Fallback to simple position sizing if dynamic optimization fails
        print(f"Dynamic optimization failed: {e}, falling back to simple sizing")
        fallback_positions = {}
        for symbol, weight in portfolio_weights.items():
            if symbol in instruments_data:
                data_item = instruments_data[symbol] # Renamed from 'data' to avoid confusion
                
                # Safety check for price and volatility before calculating fallback position
                if (data_item['price'] is not None and not pd.isna(data_item['price']) and data_item['price'] > 0 and
                    data_item['volatility'] is not None and not pd.isna(data_item['volatility']) and data_item['volatility'] > 0 and
                    data_item['specs']['multiplier'] > 0):
                    
                    position = calculate_position_size_with_idm(
                        capital, weight, 1.0, data_item['specs']['multiplier'], 
                        data_item['price'], 1.0, data_item['volatility'], risk_target
                    )
                    fallback_positions[symbol] = round(position)
                else:
                    print(f"DEBUG: Fallback - Skipping {symbol} due to invalid price/vol/multiplier. Price: {data_item['price']}, Vol: {data_item['volatility']}, Mult: {data_item['specs']['multiplier']}")
                    fallback_positions[symbol] = 0 # Assign 0 if data is problematic
            else:
                 fallback_positions[symbol] = 0 # Also 0 if not in instruments_data
        
        return {
            'positions': fallback_positions,
            'optimal_unrounded': fallback_positions,
            'idm': 1.0,
            'tracking_error': 0.0,
            'adjustment_factor': 1.0,
            'buffer': 0.0,
            'error': str(e)
        }

#####   MAIN TESTING FUNCTION   #####

def main():
    """
    Test dynamic optimization with example data.
    """
    print("=" * 70)
    print("DYNAMIC OPTIMIZATION TESTING")
    print("=" * 70)
    
    # Example from the book: 3-instrument portfolio
    print("\n----- Book Example: 3-Instrument Portfolio -----")
    
    # Simulate book example data
    capital = 500000
    instruments = ['US_5Y', 'US_10Y', 'SP500']
    
    # Example positions and data
    optimal_positions = {'US_5Y': 0.4, 'US_10Y': 0.9, 'SP500': 3.1}
    weight_per_contract = {'US_5Y': 0.22, 'US_10Y': 0.24, 'SP500': 0.04}
    
    # Example covariance matrix from book
    correlation_data = {
        'US_5Y': [1.0, 0.9, -0.1],
        'US_10Y': [0.9, 1.0, -0.1], 
        'SP500': [-0.1, -0.1, 1.0]
    }
    
    volatilities = {'US_5Y': 0.052, 'US_10Y': 0.082, 'SP500': 0.171}
    
    # Create covariance matrix
    correlation_matrix = pd.DataFrame(correlation_data, index=instruments, columns=instruments)
    vol_series = pd.Series(volatilities)
    vol_matrix = np.outer(vol_series.values, vol_series.values)
    covariance_matrix = pd.DataFrame(vol_matrix * correlation_matrix.values, 
                                   index=instruments, columns=instruments)
    
    print("Covariance Matrix:")
    print(covariance_matrix)
    
    # Calculate optimal weights
    optimal_weights = {}
    for symbol in instruments:
        optimal_weights[symbol] = calculate_optimal_weight(
            optimal_positions[symbol], weight_per_contract[symbol]
        )
    
    print(f"\nOptimal Weights: {optimal_weights}")
    
    # Test tracking error calculation
    current_positions = {'US_5Y': 0, 'US_10Y': 0, 'SP500': 0}
    tracking_weights = {}
    for symbol in instruments:
        current_weight = current_positions[symbol] * weight_per_contract[symbol]
        tracking_weights[symbol] = optimal_weights[symbol] - current_weight
    
    tracking_weights_series = pd.Series(tracking_weights)
    tracking_error = calculate_tracking_error_std(tracking_weights_series, covariance_matrix)
    
    print(f"Tracking Error (zero positions): {tracking_error:.6f}")
    print(f"Expected from book: ~0.0267")
    
    # Test greedy algorithm
    print(f"\n----- Testing Greedy Algorithm -----")
    
    cost_weight_terms = {'US_5Y': 0.0005, 'US_10Y': 0.0010, 'SP500': 0.0044}
    initial_positions = {'US_5Y': 0, 'US_10Y': 0, 'SP500': 0}
    
    optimized_positions = run_greedy_algorithm(
        optimal_positions, initial_positions, weight_per_contract,
        covariance_matrix, cost_weight_terms, cost_multiplier=50
    )
    
    print(f"Optimized Positions: {optimized_positions}")
    
    # Calculate final tracking error
    final_tracking_error = calculate_solution_tracking_error(
        optimized_positions, optimal_positions, weight_per_contract,
        covariance_matrix, cost_weight_terms, cost_multiplier=50
    )
    
    print(f"Final Tracking Error: {final_tracking_error:.6f}")
    
    # Test buffering
    print(f"\n----- Testing Buffering -----")
    
    risk_target = 0.20
    buffer = calculate_tracking_error_buffer(risk_target, 0.05)
    adjustment_factor = calculate_adjustment_factor(final_tracking_error, buffer, risk_target)
    
    print(f"Risk Target: {risk_target:.3f}")
    print(f"Buffer: {buffer:.6f}")
    print(f"Adjustment Factor: {adjustment_factor:.6f}")
    
    required_trades = calculate_required_trades_with_buffering(
        current_positions, optimized_positions, adjustment_factor
    )
    
    print(f"Required Trades: {required_trades}")

if __name__ == "__main__":
    main() 