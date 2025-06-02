import pandas as pd
import numpy as np
import sys
import os
sys.path.append('rob_port')
from chapter9 import *

def analyze_strategy9_viability():
    """Comprehensive viability analysis for Strategy 9."""
    print('=' * 80)
    print('STRATEGY 9 VIABILITY ASSESSMENT')
    print('=' * 80)
    
    try:
        # Load Strategy 9 results
        strategy9_config = {
            'capital': 50000000,
            'risk_target': 0.2,
            'weight_method': 'handcrafted',
            'forecast_combination': 'five_filters'
        }
        
        strategy9_results = load_strategy_results('strategy9', strategy9_config)
        if strategy9_results is None:
            print("Strategy 9 results not found. Need to run first.")
            return
            
        s9_data = strategy9_results['portfolio_data']
        
        # Time periods for analysis
        periods = [
            ('Early 2000s', '2000-01-01', '2004-12-31'),
            ('Mid 2000s', '2005-01-01', '2009-12-31'),
            ('2010s', '2010-01-01', '2014-12-31'),
            ('Mid 2010s', '2015-01-01', '2019-12-31'),
            ('2020s', '2020-01-01', '2024-12-31')
        ]
        
        print('\n=== TEMPORAL PERFORMANCE BREAKDOWN ===')
        print('Period           Ann.Return  Sharpe   MaxDD   Vol     Trades/Day  Years')
        print('-' * 75)
        
        period_performances = []
        
        for period_name, start, end in periods:
            period_data = s9_data[(s9_data.index >= start) & (s9_data.index <= end)]
            if len(period_data) > 20:
                returns = period_data['strategy_returns'].dropna()
                if len(returns) > 0:
                    ann_ret = returns.mean() * 256
                    sharpe = returns.mean() / returns.std() * np.sqrt(256) if returns.std() > 0 else 0
                    vol = returns.std() * np.sqrt(256)
                    cum_ret = (1 + returns).cumprod()
                    max_dd = ((cum_ret / cum_ret.cummax()) - 1).min() * 100
                    avg_trades = period_data['total_trades'].mean()
                    years = len(returns) / 256
                    
                    print(f'{period_name:<15} {ann_ret:>8.1%} {sharpe:>7.2f} {max_dd:>7.1f}% {vol:>6.1%} {avg_trades:>9.1f} {years:>6.1f}')
                    
                    period_performances.append({
                        'period': period_name,
                        'return': ann_ret,
                        'sharpe': sharpe,
                        'max_dd': max_dd,
                        'volatility': vol,
                        'trades': avg_trades
                    })
        
        # Recent performance focus
        print('\n=== RECENT PERFORMANCE CRISIS ===')
        recent_periods = [
            ('2018-2019', '2018-01-01', '2019-12-31'),
            ('2020-2021', '2020-01-01', '2021-12-31'),
            ('2022-2023', '2022-01-01', '2023-12-31'),
            ('2024+', '2024-01-01', '2025-12-31')
        ]
        
        print('Period      Ann.Return  Sharpe   MaxDD   Comment')
        print('-' * 55)
        
        for period_name, start, end in recent_periods:
            period_data = s9_data[(s9_data.index >= start) & (s9_data.index <= end)]
            if len(period_data) > 10:
                returns = period_data['strategy_returns'].dropna()
                if len(returns) > 0:
                    ann_ret = returns.mean() * 256
                    sharpe = returns.mean() / returns.std() * np.sqrt(256) if returns.std() > 0 else 0
                    cum_ret = (1 + returns).cumprod()
                    max_dd = ((cum_ret / cum_ret.cummax()) - 1).min() * 100
                    
                    if ann_ret < 0:
                        comment = "LOSS PERIOD"
                    elif sharpe < 0.3:
                        comment = "POOR RISK-ADJ"
                    elif max_dd < -30:
                        comment = "HIGH DRAWDOWN"
                    else:
                        comment = "Acceptable"
                    
                    print(f'{period_name:<10} {ann_ret:>8.1%} {sharpe:>7.2f} {max_dd:>7.1f}% {comment}')
        
        # Performance decay analysis
        print('\n=== PERFORMANCE DECAY ANALYSIS ===')
        if len(period_performances) >= 3:
            early_performance = np.mean([p['return'] for p in period_performances[:2]])
            recent_performance = np.mean([p['return'] for p in period_performances[-2:]])
            performance_decay = recent_performance - early_performance
            
            early_sharpe = np.mean([p['sharpe'] for p in period_performances[:2]])
            recent_sharpe = np.mean([p['sharpe'] for p in period_performances[-2:]])
            sharpe_decay = recent_sharpe - early_sharpe
            
            print(f"Early Performance (2000-2009): {early_performance:.1%} annual")
            print(f"Recent Performance (2015-2024): {recent_performance:.1%} annual")
            print(f"Performance Decay: {performance_decay:.1%}")
            print(f"Early Sharpe: {early_sharpe:.2f}")
            print(f"Recent Sharpe: {recent_sharpe:.2f}")
            print(f"Sharpe Decay: {sharpe_decay:.2f}")
        
        # Market structure changes
        print('\n=== MARKET STRUCTURE ANALYSIS ===')
        
        # Calculate rolling average returns to identify trend
        s9_data['rolling_6m_return'] = s9_data['strategy_returns'].rolling(128).mean() * 256
        s9_data['rolling_vol'] = s9_data['strategy_returns'].rolling(64).std() * np.sqrt(256)
        
        # Recent vs historical comparison
        historical_data = s9_data[s9_data.index < '2015-01-01']
        recent_data = s9_data[s9_data.index >= '2020-01-01']
        
        if len(historical_data) > 0 and len(recent_data) > 0:
            hist_trades_freq = historical_data['total_trades'].mean()
            recent_trades_freq = recent_data['total_trades'].mean()
            
            hist_vol = historical_data['strategy_returns'].std() * np.sqrt(256)
            recent_vol = recent_data['strategy_returns'].std() * np.sqrt(256)
            
            print(f"Historical trading frequency: {hist_trades_freq:.1f} trades/day")
            print(f"Recent trading frequency: {recent_trades_freq:.1f} trades/day")
            print(f"Trading frequency change: {((recent_trades_freq/hist_trades_freq)-1)*100:+.1f}%")
            print(f"Historical volatility: {hist_vol:.1%}")
            print(f"Recent volatility: {recent_vol:.1%}")
            print(f"Volatility change: {((recent_vol/hist_vol)-1)*100:+.1f}%")
        
        # Assessment of current viability
        print('\n=== VIABILITY ASSESSMENT ===')
        
        # Get most recent 2 years performance
        very_recent = s9_data[s9_data.index >= '2022-01-01']
        if len(very_recent) > 0:
            recent_returns = very_recent['strategy_returns'].dropna()
            recent_ann_ret = recent_returns.mean() * 256
            recent_sharpe = recent_returns.mean() / recent_returns.std() * np.sqrt(256) if recent_returns.std() > 0 else 0
            recent_max_dd = ((recent_returns.cumsum() / recent_returns.cumsum().cummax()) - 1).min() * 100
            
            print(f"Recent 2-Year Performance (2022+):")
            print(f"  Annual Return: {recent_ann_ret:.1%}")
            print(f"  Sharpe Ratio: {recent_sharpe:.2f}")
            print(f"  Max Drawdown: {recent_max_dd:.1f}%")
            
            # Viability scores
            viability_scores = []
            
            # Return score (target: >5% annual)
            return_score = min(100, max(0, (recent_ann_ret / 0.05) * 100))
            viability_scores.append(('Return', return_score, recent_ann_ret >= 0.03))
            
            # Sharpe score (target: >0.4)
            sharpe_score = min(100, max(0, (recent_sharpe / 0.4) * 100))
            viability_scores.append(('Sharpe', sharpe_score, recent_sharpe >= 0.3))
            
            # Drawdown score (target: <-25%)
            dd_score = min(100, max(0, (25 + recent_max_dd) / 25 * 100))
            viability_scores.append(('Drawdown', dd_score, recent_max_dd > -30))
            
            print(f"\n=== VIABILITY SCORES ===")
            print("Metric      Score  Pass   Threshold")
            print("-" * 35)
            
            total_score = 0
            pass_count = 0
            
            for metric, score, passed in viability_scores:
                total_score += score
                if passed:
                    pass_count += 1
                    status = "✓"
                else:
                    status = "✗"
                    
                if metric == 'Return':
                    threshold = ">3% annual"
                elif metric == 'Sharpe':
                    threshold = ">0.3"
                else:
                    threshold = ">-30%"
                    
                print(f"{metric:<10} {score:>5.0f}  {status:<5} {threshold}")
            
            avg_score = total_score / len(viability_scores)
            
            print(f"\nOverall Score: {avg_score:.0f}/100")
            print(f"Tests Passed: {pass_count}/{len(viability_scores)}")
            
            # Final assessment
            print(f"\n=== FINAL ASSESSMENT ===")
            
            if avg_score >= 70 and pass_count >= 2:
                assessment = "VIABLE - Strategy shows adequate performance"
                color = "GREEN"
            elif avg_score >= 50 and pass_count >= 1:
                assessment = "MARGINAL - Caution advised, consider modifications"
                color = "YELLOW"
            else:
                assessment = "NOT VIABLE - Significant performance issues"
                color = "RED"
            
            print(f"Status: {assessment}")
            
            # Recommendations
            print(f"\n=== RECOMMENDATIONS ===")
            
            if color == "RED":
                print("• IMMEDIATE ACTION REQUIRED:")
                print("  - Strategy shows significant performance decay")
                print("  - Consider halting live trading")
                print("  - Investigate regime changes in underlying markets")
                print("  - Research alternative parameters or filters")
                
            elif color == "YELLOW":
                print("• PROCEED WITH CAUTION:")
                print("  - Reduce position sizing")
                print("  - Implement stricter risk management")
                print("  - Monitor performance weekly")
                print("  - Consider adaptive parameters")
                
            else:
                print("• STRATEGY APPEARS VIABLE:")
                print("  - Monitor quarterly performance")
                print("  - Continue with current parameters")
                print("  - Consider minor optimizations")
            
            # Market factors contributing to decay
            print(f"\n=== LIKELY CAUSES OF DECAY ===")
            print("1. ALGORITHMIC TRADING PROLIFERATION:")
            print("   - More sophisticated trend-following algorithms")
            print("   - Faster execution eroding edge")
            print("   - Increased correlation during stress periods")
            
            print("\n2. CENTRAL BANK INTERVENTION:")
            print("   - QE policies reducing trend persistence")
            print("   - Fed put reducing sustained downtrends")
            print("   - Coordinated global monetary policy")
            
            print("\n3. MARKET STRUCTURE CHANGES:")
            print("   - High-frequency trading")
            print("   - ETF proliferation")
            print("   - Options market impact")
            
            print("\n4. CROWDING EFFECT:")
            print("   - Too many trend followers")
            print("   - Strategy alpha being arbitraged away")
            print("   - Self-defeating prophecy")
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_strategy9_viability() 