# Dynamic Optimization for Portfolio Strategies

This guide explains how to use the dynamic optimization implementation based on Chapter 25 of "Advanced Futures Trading Strategies" by Robert Carver.

## Overview

Dynamic optimization solves the problem of optimal position sizing when you can't trade the exact fractional positions calculated by traditional portfolio theory. Instead of using fixed weights, it dynamically optimizes integer positions daily to minimize tracking error while accounting for transaction costs.

## Key Benefits

1. **Better Capital Utilization**: Optimizes integer positions rather than rounding, leading to better use of available capital
2. **Cost-Aware Optimization**: Includes transaction costs directly in the optimization process
3. **Adaptive Rebalancing**: Uses buffering to reduce unnecessary turnover
4. **Correlation-Aware**: Considers instrument correlations when optimizing positions

## Quick Start

### Basic Usage

```python
from chapter4 import *
from dynamic_optimization import *

# Load your data and create portfolio weights as usual
strategies, data, instruments_df, asset_classes = run_portfolio_comparison()
portfolio_weights = strategies['Risk Parity']  # Or any strategy

# Use dynamic optimization instead of static
dynamic_result = backtest_portfolio_with_dynamic_optimization(
    portfolio_weights=portfolio_weights,
    data=data,
    instruments_df=instruments_df,
    capital=100000,
    risk_target=0.2,
    rebalance_frequency='weekly',  # 'daily', 'weekly', or 'monthly'
    cost_multiplier=50,
    use_buffering=True
)

# Results include additional dynamic optimization metrics
performance = dynamic_result['performance']
print(f"Annual Return: {performance['annualized_return']:.2%}")
print(f"Annual Turnover: {performance['annual_turnover']:.0f} contracts")
print(f"Average Tracking Error: {performance['avg_tracking_error']:.6f}")
```

### Comparing Static vs Dynamic

```python
# Compare static and dynamic optimization
comparison_results = compare_static_vs_dynamic_optimization(
    capital=100000,
    risk_target=0.2,
    max_instruments=15,
    cost_multiplier=50,
    use_buffering=True
)
```

## Key Parameters

### rebalance_frequency
- **'daily'**: Reoptimize positions every day (highest performance, highest computation)
- **'weekly'**: Reoptimize weekly (good balance of performance and efficiency)
- **'monthly'**: Reoptimize monthly (lower turnover, suitable for large portfolios)

### cost_multiplier
- **50**: Default value from the book
- **25-100**: Typical range depending on your actual transaction costs
- Higher values reduce turnover but may increase tracking error

### use_buffering
- **True**: Enable buffering to reduce unnecessary trades (recommended)
- **False**: Trade to exact optimal positions every rebalancing period

### buffer_fraction
- **0.05**: Default (5% of risk target)
- **0.02-0.10**: Typical range, smaller = more trades, larger = less trades

## Performance Metrics

Dynamic optimization provides additional metrics beyond standard performance measures:

### Tracking Error Metrics
- `avg_tracking_error`: Average tracking error between optimal and actual positions
- `avg_adjustment_factor`: Average buffering adjustment factor (0-1)

### Turnover Metrics
- `total_turnover`: Total number of contracts traded over backtest period
- `avg_daily_turnover`: Average contracts traded per day
- `annual_turnover`: Annualized turnover rate

### Optimization Settings
- `rebalance_frequency`: How often positions were reoptimized
- `cost_multiplier`: Cost penalty multiplier used
- `use_buffering`: Whether buffering was enabled

## Implementation Details

### The Greedy Algorithm

The system uses a greedy algorithm to find optimal integer positions:

1. Start with zero positions for all instruments
2. Calculate optimal unrounded positions using combined forecasts
3. For each instrument, try adding/subtracting one contract
4. Choose the trade that most reduces tracking error (including cost penalty)
5. Repeat until no improvement can be found

### Cost Penalty

Transaction costs are incorporated using the formula:
```
Total Cost = √(e^T.Σ.e) + λδ
```
Where:
- `e` is the tracking error vector
- `Σ` is the covariance matrix
- `λ` is the cost multiplier (default 50)
- `δ` is the total cost in weight terms

### Buffering

Buffering prevents excessive trading when tracking error is small:

1. Calculate tracking error buffer: `B = 0.05 × τ` (5% of risk target)
2. Calculate adjustment factor: `α = max([T - B] ÷ T, 0)`
3. Scale required trades: `Required trade = round(α × [Optimal - Current])`

## When to Use Dynamic Optimization

### Recommended For:
- **Large Portfolios**: >$500k capital with >20 instruments
- **Professional Operations**: When minimizing tracking error is crucial
- **Cost-Sensitive Strategies**: When transaction costs significantly impact performance
- **Complex Portfolios**: Multi-asset portfolios with varying correlations

### Use Static Optimization For:
- **Small Portfolios**: <$100k capital with <10 instruments
- **Simple Strategies**: Single-asset or highly correlated instruments
- **Low-Frequency Trading**: Monthly or quarterly rebalancing
- **Educational/Research**: When simplicity is preferred

## Example Scenarios

### Scenario 1: Medium Portfolio ($250k, 15 instruments)
```python
dynamic_result = backtest_portfolio_with_dynamic_optimization(
    portfolio_weights=weights,
    data=data,
    instruments_df=instruments_df,
    capital=250000,
    rebalance_frequency='weekly',  # Good balance
    cost_multiplier=50,
    use_buffering=True,
    buffer_fraction=0.05
)
```

### Scenario 2: Large Portfolio ($1M, 50 instruments)
```python
dynamic_result = backtest_portfolio_with_dynamic_optimization(
    portfolio_weights=weights,
    data=data,
    instruments_df=instruments_df,
    capital=1000000,
    rebalance_frequency='daily',   # Can afford daily optimization
    cost_multiplier=75,            # Higher cost sensitivity
    use_buffering=True,
    buffer_fraction=0.03           # Tighter buffer for large portfolio
)
```

### Scenario 3: Conservative Approach ($100k, 8 instruments)
```python
dynamic_result = backtest_portfolio_with_dynamic_optimization(
    portfolio_weights=weights,
    data=data,
    instruments_df=instruments_df,
    capital=100000,
    rebalance_frequency='monthly', # Lower frequency
    cost_multiplier=100,           # High cost penalty
    use_buffering=True,
    buffer_fraction=0.10           # Large buffer to minimize trades
)
```

## Monitoring and Diagnostics

### Key Metrics to Monitor

1. **avg_tracking_error**: Should be small relative to strategy volatility
2. **annual_turnover**: Balance between optimization and transaction costs
3. **avg_adjustment_factor**: Indicates how often buffering is active

### Typical Values
- Tracking Error: 0.001-0.01 (0.1%-1% of strategy volatility)
- Adjustment Factor: 0.3-0.8 (buffering active 30-80% of time)
- Annual Turnover: 50-500 contracts depending on portfolio size

### Red Flags
- Very high turnover (>1000 contracts/year for small portfolios)
- Consistently high tracking error (>0.02)
- Adjustment factor always 1.0 (buffering not working)

## Troubleshooting

### Common Issues

1. **"Dynamic optimization failed" errors**
   - Check that you have sufficient instruments (>1) with correlation data
   - Ensure returns matrix has adequate history (>50 days)
   - Verify all instrument data is properly formatted

2. **Zero returns from dynamic optimization**
   - Often due to insufficient data overlap between instruments
   - Try reducing the number of instruments or extending data history
   - Check that rebalance dates have adequate data

3. **Extremely high turnover**
   - Increase cost_multiplier (try 100-200)
   - Enable buffering if not already enabled
   - Increase buffer_fraction to 0.08-0.10
   - Consider less frequent rebalancing

### Performance Tips

1. **Start Simple**: Begin with weekly rebalancing and adjust based on results
2. **Calibrate Costs**: Adjust cost_multiplier based on your actual transaction costs
3. **Monitor Metrics**: Track turnover and tracking error to optimize settings
4. **Scale Gradually**: Test with smaller portfolios before applying to large ones

## Advanced Usage

### Custom Cost Modeling

You can modify the cost calculation in `calculate_cost_in_weight_terms()` to reflect your specific transaction cost structure.

### Alternative Optimization Algorithms

The greedy algorithm can be replaced with more sophisticated optimization methods by modifying `run_greedy_algorithm()`.

### Machine Learning Integration

The system can be extended to use machine learning for:
- Forecast combination
- Dynamic cost estimation
- Adaptive buffering parameters

## References

- **Chapter 25**: "Dynamic optimisation (for when you can't trade the Jumbo portfolio)" in "Advanced Futures Trading Strategies" by Robert Carver
- **Chapter 4**: Static portfolio optimization foundations
- **Mathematical Background**: Portfolio theory, tracking error minimization, integer programming

---

For questions or issues, refer to the test scripts:
- `test_dynamic_optimization.py`: Simple examples and component testing
- `chapter4.py`: Full integration with portfolio strategies 