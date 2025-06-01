# Systematic Trading Strategies Implementation

## Overview

This repository implements systematic trading strategies based on the book "Systematic Trading" by Robert Carver. The implementation covers multi-instrument portfolio strategies with sophisticated risk management and position sizing.

## Implemented Strategies

### Strategy 4: Multi-Instrument Variable Risk Portfolio

**Key Features:**
- **75 instruments** across multiple asset classes
- **Handcrafted weighting algorithm** based on cost efficiency, performance, and diversification
- **Variable risk scaling** using blended volatility forecasts
- **Asset class diversification** with strategic allocation targets

**Performance (Full Backtest 2000-2025):**
- **Total Return:** 446.88%
- **Annualized Return:** 6.96%
- **Sharpe Ratio:** 0.466
- **Max Drawdown:** -56.3%
- **Outperforms:** Equal weighting (+278.79% total return) and single-instrument strategies

## Handcrafted Weighting Algorithm

The algorithm uses a sophisticated multi-factor approach:

### Scoring Components:
1. **Cost Efficiency (30%)** - Favors instruments with low risk-adjusted costs
2. **Volatility Scaling (30%)** - Inverse volatility weighting for risk parity  
3. **Performance (20%)** - Considers Sharpe ratio and positive skewness
4. **Base Allocation (20%)** - Ensures diversification across all instruments

### Asset Class Allocation:
- **Bonds (25%):** Treasury notes, government bonds
- **Equities (30%):** Stock indices (US, European, Asian)
- **FX (15%):** Currency pairs
- **Commodities (15%):** Metals, energy
- **Agriculture (8%):** Agricultural futures
- **Volatility (2%):** VIX and volatility indices
- **Crypto (3%):** Bitcoin and Ethereum

### Top Weighted Instruments:
1. **MNQ (2.7%)** - Nasdaq micro futures (low cost, high performance)
2. **GBX (2.6%)** - German Buxl (good performance, reasonable cost)
3. **MYM (2.5%)** - Dow micro futures (strong performance)
4. **GBL (2.4%)** - German Bund (excellent Sharpe ratio)
5. **MES (2.4%)** - S&P 500 micro futures (good performance)

## File Structure

```
rob_port/
├── chapter1.py          # Basic portfolio functions and utilities
├── chapter2.py          # Fixed position sizing strategies
├── chapter3.py          # Variable risk scaling strategies
├── chapter4.py          # Multi-instrument portfolio strategies ⭐
├── tests/
│   ├── test_chapter3.py # Tests for variable risk strategies
│   ├── test_chapter4.py # Tests for Strategy 4 ⭐
│   └── ...
└── README.md
```

## Usage

### Running Strategy 4

```python
# Run the complete Strategy 4 backtest
python rob_port/chapter4.py

# Compare equal vs handcrafted weighting
results = compare_weighting_methods()

# Run specific backtest
results = backtest_multi_instrument_strategy(
    data_dir='Data',
    capital=50000000,
    risk_target=0.2,
    weight_method='handcrafted'  # or 'equal'
)
```

### Running Tests

```bash
# Test Strategy 4 implementation
python rob_port/tests/test_chapter4.py

# Test Chapter 3 strategies
python rob_port/tests/test_chapter3.py
```

## Key Functions

### Strategy 4 (Chapter 4)

- `backtest_multi_instrument_strategy()` - Main backtest function
- `calculate_handcrafted_weights()` - Sophisticated weighting algorithm
- `calculate_portfolio_position_size()` - Multi-instrument position sizing
- `load_all_instrument_data()` - Data loading for all instruments

### Testing Framework

- `TestStrategy4` - Comprehensive unit tests
- `run_strategy4_tests()` - Complete test suite
- Performance comparison tests between weighting methods

## Data Requirements

The system expects daily price data files in the format:
- `{symbol_lowercase}_daily_data.csv`
- Columns: `Time`, `Last` (and other OHLC data)
- Located in `Data/` directory

Example: `mes_daily_data.csv`, `mnq_daily_data.csv`

## Performance Comparison

| Method | Total Return | Ann. Return | Sharpe | Max DD |
|--------|-------------|-------------|---------|---------|
| **Handcrafted** | **446.88%** | **6.96%** | **0.466** | -56.3% |
| Equal Weight | 168.09% | 3.98% | 0.318 | -48.0% |
| MES Only | 353.75% | 6.19% | 0.406 | -56.1% |

## Key Advantages

✅ **Superior Returns** - 3x better than equal weighting
✅ **Cost Efficiency** - Favors low-cost, high-quality instruments  
✅ **Diversification** - Systematic allocation across asset classes
✅ **Risk Management** - Variable position sizing with volatility forecasts
✅ **Robustness** - Comprehensive testing framework
✅ **Scalability** - Handles 75+ instruments seamlessly

## Implementation Notes

- **No instrument selection** - Uses all available instruments
- **Dynamic weighting** - Adjusts based on performance and costs
- **Risk parity components** - Inverse volatility scaling
- **Defensive features** - Protects against infinite position sizes
- **Comprehensive testing** - Unit tests, integration tests, performance comparisons

## Future Enhancements

- Portfolio optimization techniques
- Dynamic asset allocation
- Advanced risk models
- Transaction cost modeling
- Real-time position management 