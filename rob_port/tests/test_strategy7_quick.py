#!/usr/bin/env python3
"""
Quick test of Strategy 7 to check forecast values.
"""

import sys
sys.path.append('rob_port')
from chapter7 import *

# Run a quick test of Strategy 7
print('Testing Strategy 7 forecast calculation...')
results = backtest_forecast_trend_strategy(
    data_dir='Data',
    capital=50000000,
    risk_target=0.2,
    start_date='2024-01-01',
    end_date='2024-12-31'
)
print('Strategy 7 completed')
print(f'Avg forecast: {results["performance"]["avg_forecast"]:.3f}')
print(f'Avg abs forecast: {results["performance"]["avg_abs_forecast"]:.3f}')

# Look at the portfolio data
portfolio_df = results['portfolio_data']
print(f'\nPortfolio data shape: {portfolio_df.shape}')
print(f'Columns: {list(portfolio_df.columns)}')

# Check forecast columns
forecast_cols = [col for col in portfolio_df.columns if 'forecast' in col.lower()]
print(f'Forecast columns: {forecast_cols}')

if len(forecast_cols) > 0:
    first_forecast_col = forecast_cols[0]
    forecasts = portfolio_df[first_forecast_col]
    print(f'\n{first_forecast_col} stats:')
    print(f'  Min: {forecasts.min():.3f}')
    print(f'  Max: {forecasts.max():.3f}')
    print(f'  Mean: {forecasts.mean():.3f}')
    print(f'  Days at +20: {(forecasts == 20).sum()}')
    print(f'  Days at -20: {(forecasts == -20).sum()}')
    print(f'  % at caps: {((forecasts == 20) | (forecasts == -20)).mean()*100:.1f}%') 