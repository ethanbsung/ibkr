# Quantitative Trading System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Interactive Brokers](https://img.shields.io/badge/Broker-Interactive%20Brokers-green.svg)](https://www.interactivebrokers.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive algorithmic trading system implementing systematic trading strategies with real-time execution, extensive backtesting, and Monte Carlo simulation capabilities. This project demonstrates enterprise-level quantitative finance software engineering practices.

## 🚀 Key Features

### Live Trading Engine
- **Real-time execution** via Interactive Brokers API with multi-strategy portfolio management
- **Dynamic position sizing** with automated rebalancing across 8 strategy allocations
- **Risk management** with 3x leveraged positions and threshold-based rebalancing (5% drift tolerance)
- **Market hours awareness** with automated scheduling and portfolio state persistence
- **Multi-asset support** across ES, YM, GC, NQ futures with proper contract specifications

### Advanced Analytics
- **Monte Carlo simulations** with bootstrap resampling and GARCH volatility modeling
- **Comprehensive backtesting framework** implementing multiple systematic strategies
- **Performance attribution** with drawdown analysis, Sharpe ratios, and fat-tail statistics
- **Strategy comparison tools** with visual equity curve analysis

### Data Infrastructure
- **75 instruments** with 2-25 years of daily OHLCV data per instrument
- **Automated data collection** and validation pipelines
- **Multi-timeframe support** (1-minute, 5-minute, daily bars)
- **Real-time market data integration** with historical data alignment

## 📊 Trading Strategies

### Strategy Portfolio
1. **IBS (Internal Bar Strength)** - Mean reversion strategy across 4 futures contracts
2. **Williams %R** - Momentum strategy with 2-day lookback periods
3. **Trend Following** - Multi-timeframe momentum with forecast scaling
4. **Long-Short Equity** - Market-neutral statistical arbitrage

### Risk Management
- Portfolio-level position sizing based on volatility forecasting
- Dynamic capital allocation with monthly rebalancing
- Commission and slippage modeling for realistic P&L attribution
- Drawdown controls with maximum position limits

## 🔧 Technical Architecture

### Core Components
```
├── ES/agg_port_live.py          # Live trading engine
├── rob_port/                    # Systematic trading framework
│   ├── chapter*.py              # Strategy implementations
│   └── tests/                   # Comprehensive test suite
├── Data/                        # Market data warehouse
│   ├── instruments.csv          # Contract specifications
│   └── *_daily_data.csv         # Historical price data
├── results/                     # Backtest outputs and visualizations
└── account_snapshots/           # Live trading performance tracking
```

### Key Technologies
- **Python 3.8+** with NumPy, Pandas for quantitative analysis
- **ib_insync** for Interactive Brokers API integration
- **ARCH/GARCH** models for volatility forecasting
- **Matplotlib/Seaborn** for performance visualization
- **JSON/CSV** for data persistence and configuration management

## 📈 Performance Metrics

### Live Trading Results
- **Multi-strategy portfolio** with dynamic allocation across 8 strategies
- **Real-time P&L tracking** with unrealized/realized profit attribution
- **Automated position management** with market hours integration
- **Daily portfolio snapshots** for performance monitoring

### Backtesting Framework
- **Comprehensive strategy testing** with walk-forward validation
- **Monte Carlo simulation** with 1000+ iterations for statistical significance
- **Performance metrics**: Sharpe ratio, Calmar ratio, maximum drawdown analysis
- **Strategy comparison tools** with visual performance attribution

## 🎯 Data Management

### Market Data Collection
- **75 futures instruments** across asset classes:
  - **Equity Indices**: ES, NQ, YM, Russell 2000
  - **Fixed Income**: Treasury bonds, notes across yield curve
  - **Commodities**: Gold, Silver, Oil, Natural Gas, Agricultural
  - **FX**: Major currency pairs (EUR, JPY, GBP, etc.)
  - **Volatility**: VIX, VSTOXX indices

### Data Quality & Validation
- **2-25 years** of historical data per instrument
- **Daily OHLCV bars** with volume validation
- **Corporate actions** and contract roll adjustments
- **Missing data interpolation** and outlier detection

## 🧪 Monte Carlo Simulation

### Methodology
- **Bootstrap resampling** with configurable block sizes
- **GARCH volatility modeling** for realistic return distributions
- **Parameter uncertainty** analysis across strategy configurations
- **Stress testing** with extreme market scenario generation

### Implementation Features
```python
# Example: Monte Carlo simulation with 1000 iterations
num_iterations = 1000
block_size = 5  # Bootstrap block size
results = run_monte_carlo_simulation(
    strategy_returns, 
    iterations=num_iterations,
    garch_model=True
)
```

## 🔄 Live Trading Execution

### Portfolio Management
```python
# Dynamic position sizing based on current equity
def calculate_position_size(current_equity, target_allocation_pct, 
                          price, multiplier, risk_multiplier=3.0):
    target_dollar_amount = current_equity * target_allocation_pct * risk_multiplier
    contract_value = price * multiplier
    return max(1, round(target_dollar_amount / contract_value))
```

### Strategy Allocation
- **50% IBS strategies** (12.5% each: ES, YM, GC, NQ)
- **50% Williams %R strategies** (12.5% each: ES, YM, GC, NQ)
- **Automated rebalancing** when allocation drift exceeds 5%
- **Monthly rebalancing** regardless of drift

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Python dependencies
pip install pandas numpy matplotlib ib_insync arch
```

### Interactive Brokers Setup
1. Install IB Gateway or Trader Workstation
2. Enable API connections (port 4002 for paper, 4001 for live)
3. Configure API settings and permissions

### Running the System
```bash
# Start live trading engine
python ES/agg_port_live.py

# Run backtests
python rob_port/chapter4.py

# Generate Monte Carlo analysis
python ES/ibs_monte.py

# Monitor account status
python account_summary.py
```

## 📊 Performance Visualization

The system generates comprehensive performance reports including:
- **Equity curves** with drawdown overlays
- **Strategy comparison charts** with performance attribution
- **Monte Carlo confidence intervals** for expected returns
- **Risk metrics dashboards** with Sharpe ratios and volatility analysis

## 🚨 Risk Disclosure

This is a demonstration system for educational and professional development purposes. All trading involves risk of loss. The strategies and code provided are for illustrative purposes and should not be used for actual trading without proper risk management and regulatory compliance.

## 📝 Development Roadmap

- [ ] **Options strategies** integration with Greeks calculation
- [ ] **Machine learning** signal generation with feature engineering
- [ ] **Real-time risk monitoring** with VaR calculations
- [ ] **Multi-broker support** beyond Interactive Brokers
- [ ] **Cloud deployment** with AWS/Azure integration

---

*This project demonstrates production-ready quantitative finance software development practices including systematic strategy research, robust backtesting frameworks, real-time execution systems, and comprehensive risk management.*
