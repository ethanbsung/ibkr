# Futures Trading Calculations Reference

This document organizes all formulas and concepts from the raw `calcs.txt` into logical groups, with plain-English explanations and Python examples for anything non-obvious.

---

## Table of Contents

1. [Core Contract Economics](#1-core-contract-economics)
2. [Returns Calculation](#2-returns-calculation)
3. [Risk & Volatility](#3-risk--volatility)
4. [Position Sizing](#4-position-sizing)
5. [Trading Costs](#5-trading-costs)
6. [Instrument & Portfolio Weights](#6-instrument--portfolio-weights)
7. [EWMAC Trend Signals & Forecasting](#7-ewmac-trend-signals--forecasting)
8. [Combining Multiple Forecasts](#8-combining-multiple-forecasts)
9. [Buffering (Reducing Unnecessary Trades)](#9-buffering-reducing-unnecessary-trades)
10. [Dynamic Portfolio Optimisation](#10-dynamic-portfolio-optimisation)

---

## 1. Core Contract Economics

These are the building blocks — how to translate a futures contract into real dollar exposure and risk.

| Concept | Formula |
|---|---|
| Notional exposure per contract | `multiplier × price × fx_rate` |
| Tick value | `multiplier × tick_size` |
| Contract leverage ratio | `notional_exposure_per_contract / capital` |

**Plain English:**
- **Notional exposure** is the dollar value you're effectively controlling with one contract. It's what the contract is "worth", not what it costs.
- **Tick value** is how many dollars you gain or lose for each minimum price move.

```python
# Example: ES (S&P 500 futures), price=5000, multiplier=50, fx_rate=1.0 (USD)
multiplier = 50
price = 5000
fx_rate = 1.0

notional_exposure = multiplier * price * fx_rate  # $250,000
tick_size = 0.25
tick_value = multiplier * tick_size               # $12.50

capital = 100_000
contract_leverage_ratio = notional_exposure / capital  # 2.5x leverage
```

---

## 2. Returns Calculation

A chain of return types, building from raw price moves to percentage returns suitable for comparison.

### 2.1 Return Decomposition

```
Total Return      = Spot Return + Dividends
Excess Return     = Total Return - Interest Rate
                  = Spot Return + Carry          (where Carry = Dividends - Interest)
```

Futures naturally embed carry — you don't receive dividends directly, but they're baked into the price differential between contracts.

### 2.2 Return in Price Points

This is the most granular step: how many "points" did the price move, scaled by contracts held.

```
R_points_t  =  N_{t-1} × (P_t - P_{t-1})
Cumulative  =  P_t - P_0
```

### 2.3 Return in Currency Terms

```
R_instrument_t  = R_points_t × multiplier
R_base_t        = R_instrument_t × fx_rate
R_%_t           = 100 × R_base_t / capital_{t-1}
```

```python
import pandas as pd

# prices: back-adjusted price series
# N: number of contracts held (from prior day)
# multiplier, fx_rate: contract specs

def calculate_returns(prices: pd.Series, N: pd.Series, multiplier: float,
                      fx_rate: float, capital: pd.Series) -> pd.DataFrame:
    r_points = N.shift(1) * prices.diff()
    r_instr  = r_points * multiplier
    r_base   = r_instr * fx_rate
    r_pct    = 100 * r_base / capital.shift(1)
    return pd.DataFrame({"r_points": r_points, "r_base": r_base, "r_pct": r_pct})
```

---

## 3. Risk & Volatility

Volatility is the single most important input to position sizing. Everything downstream depends on getting this right.

### 3.1 Standard Deviation (Simple Rolling)

The standard "look back N days and compute std dev" approach:

```
σ_N,t = sqrt( (1/N) × Σ (r_{t-i} - r̄_t)² )   for i = 0 … N-1
```

```python
returns.rolling(window=N).std()   # pandas does this directly
```

### 3.2 EWMA (Exponentially Weighted Moving Average) Returns

Instead of a flat rolling window, EWMA gives more weight to recent observations. The `lambda` parameter controls the decay speed.

```
r*_t  =  λ·r_t  +  λ(1-λ)·r_{t-1}  +  λ(1-λ)²·r_{t-2}  + ...
```

To convert a span in days `N` to lambda:

```
λ = 2 / (N + 1)
```

```python
# pandas span= parameter maps directly to this lambda formula
ewma_returns = returns.ewm(span=N).mean()
```

### 3.3 Exponentially Weighted Standard Deviation

Same idea as EWMA returns, but for volatility:

```
σ_exp_t = sqrt( λ(r_t - r*_t)² + λ(1-λ)(r_{t-1} - r*_t)² + ... )
```

```python
sigma_exp = returns.ewm(span=N).std()   # pandas ewm().std() handles this
```

### 3.4 Blended Volatility Estimate

A stability improvement: blend short-term EWMA vol with a long-term average. This prevents over-reacting to temporary vol spikes while still being responsive.

```
σ_blend,t = 0.3 × (10-year average of σ_t) + 0.7 × σ_t
```

```python
sigma_longrun = sigma_daily.rolling(window=252*10).mean()
sigma_blend   = 0.3 * sigma_longrun + 0.7 * sigma_daily
```

### 3.5 Volatility in Dollar Terms

Convert percentage volatility into a dollar risk figure for the instrument and position.

```
σ_contract ($)  = notional_exposure ($) × σ_%
σ_position ($)  = σ_contract ($) × N               (N = number of contracts)
σ_target ($)    = capital × target_risk
```

### 3.6 Daily vs Annualised Volatility

```
Daily σ in price points:   σ_p = (price × σ_%) / 16
```

The `16` is an approximation of `sqrt(256)` — there are roughly 256 trading days per year, and annualised vol ÷ 16 ≈ daily vol.

```python
# Convert annual % vol to daily price-point vol
sigma_p_daily = (price * sigma_pct) / 16

# Or go the other direction:
sigma_pct_annual = sigma_p_daily * 16 / price
```

### 3.7 Fat Tail Ratios

Used to assess whether a return distribution has fatter-than-normal tails (leptokurtosis).

```
Lower percentile ratio  = 1st  percentile / 30th percentile
Upper percentile ratio  = 99th percentile / 70th percentile

Relative lower fat tail = lower percentile ratio / 4.43
Relative upper fat tail = upper percentile ratio / 4.43
```

The `4.43` is the theoretical ratio for a normal distribution. A value > 1 indicates fatter tails than normal.

```python
import numpy as np

def fat_tail_ratios(returns: pd.Series) -> dict:
    p1, p30, p70, p99 = np.percentile(returns, [1, 30, 70, 99])
    lower_ratio = p1 / p30
    upper_ratio = p99 / p70
    normal_ref  = 4.43
    return {
        "relative_lower_fat_tail": lower_ratio / normal_ref,
        "relative_upper_fat_tail": upper_ratio / normal_ref,
    }
```

---

## 4. Position Sizing

Given a volatility target, how many contracts should you hold?

### 4.1 Core Position Formula

The fundamental equation — solve for N contracts such that your position's dollar risk equals your target risk.

```
σ_target ($)  =  σ_position ($)
Capital × target_risk  =  N × multiplier × price × fx_rate × σ_%

→  N = (Capital × target_risk) / (multiplier × price × fx_rate × σ_%)
```

Equivalently, using daily price-point volatility `σ_p` (avoids the `/16` issue):

```
N = (Capital × target_risk) / (multiplier × fx_rate × σ_p × 16)
```

```python
def optimal_contracts(capital, target_risk, multiplier, price, fx_rate, sigma_pct):
    return (capital * target_risk) / (multiplier * price * fx_rate * sigma_pct)

# Or using daily sigma_p directly:
def optimal_contracts_daily(capital, target_risk, multiplier, fx_rate, sigma_p):
    return (capital * target_risk) / (multiplier * fx_rate * sigma_p * 16)
```

### 4.2 Volatility Ratio Shortcut

```
volatility_ratio    = target_risk / σ_%
N                   = volatility_ratio / contract_leverage_ratio
```

This is just a rearrangement of the formula above, but useful for intuition: you're scaling how much leverage you want (target risk) relative to how much the contract naturally gives you (contract leverage ratio).

### 4.3 Portfolio-Level Leverage

```
Leverage ratio  =  total_notional_exposure / capital
                =  volatility_ratio              (at target risk)
```

### 4.4 Capital & Margin Constraints

The maximum number of contracts you can hold is capped by available margin:

```
Max N               = capital / (margin_per_contract / fx_rate)
Max target_risk     = (multiplier × price × σ_%) / margin_per_contract
                    = σ_% × max_leverage_ratio
                    = σ_% × (max_capital_loss / worst_return)
```

### 4.5 Minimum Capital Requirements

```
Min capital for 1 contract:  (multiplier × price × fx_rate × σ_%) / target_risk
Min capital for 4 contracts: 4 × min_capital_for_1   (practical diversification floor)
```

For a portfolio with multiple instruments (adds IDM and instrument weight):

```
Min capital for 4 contracts of instrument i =
    (4 × multiplier_i × price_i × fx_rate_i × σ_%_i) / (IDM × weight_i × target_risk)
```

```python
def min_capital(multiplier, price, fx_rate, sigma_pct, target_risk, n=1):
    return n * (multiplier * price * fx_rate * sigma_pct) / target_risk

def min_capital_portfolio(multiplier, price, fx_rate, sigma_pct,
                          target_risk, idm, weight, n=4):
    return (n * multiplier * price * fx_rate * sigma_pct) / (idm * weight * target_risk)
```

### 4.6 Kelly Criterion

```
Half Kelly = 0.5 × Expected Sharpe Ratio
```

Half Kelly is the standard practical risk target — using full Kelly is too aggressive and leads to large drawdowns.

---

## 5. Trading Costs

Costs must be measured in "SR units" (Sharpe Ratio units) so they're comparable across instruments with different volatilities.

### 5.1 Spread and Commission Costs

```
Spread cost (price points)  = (Bid - Offer) / 2
Spread cost (currency)      = multiplier × spread_cost_pp
                            = tick_value × (spread_in_ticks / 2)
Total cost per trade ($)    = spread_cost_currency + commission_per_contract
Total cost per trade (%)    = total_cost_per_trade ($) / (price × multiplier)
```

```python
def cost_per_trade_currency(bid, offer, multiplier, commission):
    spread_pp  = (offer - bid) / 2
    spread_ccy = multiplier * spread_pp
    return spread_ccy + commission

def cost_per_trade_pct(bid, offer, multiplier, commission, price):
    return cost_per_trade_currency(bid, offer, multiplier, commission) / (price * multiplier)
```

### 5.2 Risk-Adjusted Costs (SR Units)

This converts a cost in % terms into how much Sharpe Ratio it "eats" per trade. This is the critical comparison metric.

```
Risk adjusted cost per trade  = total_cost_% / σ_%      (i.e. cost / annualised vol)
```

**Limits to stay under:**
- `risk_adjusted_cost_per_trade < 0.01`
- `annual_risk_adjusted_cost < 0.10`  (instruments)
- `annual_risk_adjusted_cost < 0.15`  (trading rule variations)

### 5.3 Annual Risk-Adjusted Costs

```
Risk adjusted transaction cost  = risk_adj_cost_per_trade × annual_turnover
Risk adjusted holding cost      = risk_adj_cost_per_trade × rolls_per_year × 2

Annual risk adjusted cost       = transaction_cost + holding_cost
                                = (rolls_per_year × 2 + turnover) × risk_adj_cost_per_trade
```

```python
def annual_risk_adjusted_cost(risk_adj_cost_per_trade, turnover, rolls_per_year):
    return (rolls_per_year * 2 + turnover) * risk_adj_cost_per_trade
```

### 5.4 Historical Cost Adjustment for Backtesting

Past spreads were different — adjust historical costs to be comparable:

```
Historical trading cost = current_cost × (historical_σ_p / current_σ_p)
```

### 5.5 Liquidity Check: Average Daily Volume in USD Risk

```
ADV_USD_risk = fx_rate × avg_daily_volume × σ_% × price × multiplier
```

Used to filter out instruments that can't absorb your desired position size.

### 5.6 Maximum Allowable Turnover Per Trading Rule

To stay under the 0.15 SR unit cost limit:

```
Turnover < [0.15 - (cost_per_trade × rolls_per_year × 2)] / cost_per_trade
```

```python
def max_turnover(risk_adj_cost_per_trade, rolls_per_year, limit=0.15):
    holding_cost = risk_adj_cost_per_trade * rolls_per_year * 2
    return (limit - holding_cost) / risk_adj_cost_per_trade
```

---

## 6. Instrument & Portfolio Weights

### 6.1 Portfolio Sharpe Ratio

```
SR_i (instrument)  = SR* - (T × c_i)
  where SR*  = assumed identical pre-cost SR for all instruments
        T    = estimated annual turnover
        c_i  = risk-adjusted cost per trade for instrument i

Mean_i (annual)    = target_risk × [SR* - T × c_i]

Portfolio mean     = Σ_i (weight_i × IDM × target_risk × [SR* - T × c_i])
Portfolio σ        = IDM × target_risk × sqrt(w · Σ · w')
Portfolio SR       = Σ_i (weight_i × [SR* - T × c_i]) / sqrt(w · Σ · w')
```

Where `Σ` is the correlation matrix of instrument returns and `w` is the weight vector.

```python
import numpy as np

def portfolio_sr(weights, sr_star, turnover, costs, corr_matrix):
    """
    weights     : np.array of instrument weights (sum to 1)
    sr_star     : float, assumed pre-cost SR
    turnover    : float, estimated annual turnover
    costs       : np.array of risk-adjusted costs per instrument
    corr_matrix : np.ndarray, instrument correlation matrix
    """
    net_srs    = sr_star - turnover * costs
    numerator  = np.dot(weights, net_srs)
    denominator = np.sqrt(weights @ corr_matrix @ weights)
    return numerator / denominator
```

### 6.2 Instrument Diversification Multiplier (IDM)

IDM accounts for the fact that a diversified portfolio can run at higher risk than the sum of individual risks. It's typically calculated from the correlation matrix, but approximate lookup values are:

| Instruments | IDM |
|---|---|
| 1 | 1.00 |
| 2 | 1.20 |
| 3 | 1.48 |
| 4 | 1.56 |
| 5 | 1.70 |
| 6 | 1.90 |
| 7 | 2.10 |
| 8–14 | 2.20 |
| 15–24 | 2.30 |
| 25–29 | 2.40 |
| 30+ | 2.50 |

```
Rough IDM = target_risk / aggregate_risk
```

### 6.3 Position Sizing with IDM and Weights

For a portfolio of instruments:

```
N_i,t = (Capital × IDM × weight_i × target_risk) / (multiplier_i × price_i × fx_i × σ_%_i)
```

```python
def position_size(capital, idm, weight, target_risk, multiplier, price, fx, sigma_pct):
    return (capital * idm * weight * target_risk) / (multiplier * price * fx * sigma_pct)
```

### 6.4 Handcrafting Weights (Top-Down Allocation)

When you don't want to rely on an optimizer:

1. **Level 1 — Asset classes:** Divide 100% equally among asset classes (e.g., equities, bonds, commodities each get 33%).
2. **Level 2 — Groups within asset class:** Divide each asset class's share equally among groups within it.
3. **Level 3 — Instruments within group:** Divide each group's share equally among instruments in it.

This avoids overfitting and ensures no single instrument dominates unexpectedly.

### 6.5 Instrument Selection Algorithm

Greedy forward-selection — add instruments one at a time, keeping only those that improve the portfolio SR:

```
1. Start with a single instrument (pick the one with the best post-cost SR given your expected final portfolio size).
2. Current portfolio = {instrument_1}.
3. For each candidate instrument not yet in the portfolio:
   a. Build a trial portfolio = current + candidate
   b. Assign weights, compute IDM
   c. Check minimum capital for all instruments — skip if not met
   d. Compute expected portfolio SR (net of costs)
4. Add the candidate that produced the highest trial portfolio SR.
5. If the new portfolio SR is >10% lower than the best SR seen so far → STOP.
   Otherwise return to step 3.
```

---

## 7. EWMAC Trend Signals & Forecasting

EWMAC (Exponentially Weighted Moving Average Crossover) is the core trend signal. The idea: if the fast EWMA is above the slow EWMA, the instrument is trending up — go long. Normalize by volatility to make forecasts comparable across instruments.

### 7.1 EWMA Definition

```
EWMA(N=64,  λ=0.031)_t  =  0.031·p_t  +  0.031(0.969)·p_{t-1}  +  ...
EWMA(N=256, λ=0.0078)_t =  0.0078·p_t + 0.0078(0.9922)·p_{t-1} + ...
```

Lambda is calculated as: `λ = 2 / (N + 1)`

```python
def ewma(prices: pd.Series, span: int) -> pd.Series:
    return prices.ewm(span=span, adjust=False).mean()
```

### 7.2 Raw Forecast

```
raw_forecast = (fast_EWMA - slow_EWMA) / σ_p
```

`σ_p` is the daily standard deviation of price changes (in price points). Dividing by `σ_p` makes the forecast dimensionless and comparable across instruments.

```python
def raw_forecast(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    sigma_p = prices.diff().ewm(span=32).std()   # 32-day EWMA vol of price changes
    return (ewma(prices, fast) - ewma(prices, slow)) / sigma_p
```

### 7.3 Scaled Forecast

Raw forecasts have different absolute magnitudes for different EWMAC speeds. We scale so the average absolute value = 10:

```
forecast_scalar = 10 / mean(|raw_forecast|)
scaled_forecast = raw_forecast × forecast_scalar
```

Pre-estimated forecast scalars (use these fixed values to avoid look-ahead bias in backtests):

| Rule | Forecast Scalar |
|---|---|
| EWMAC2  | 12.1  |
| EWMAC4  | 8.53  |
| EWMAC8  | 5.95  |
| EWMAC16 | 4.10  |
| EWMAC32 | 2.79  |
| EWMAC64 | 1.91  |

```python
FORECAST_SCALARS = {2: 12.1, 4: 8.53, 8: 5.95, 16: 4.10, 32: 2.79, 64: 1.91}

def scaled_forecast(raw: pd.Series, fast_span: int) -> pd.Series:
    return raw * FORECAST_SCALARS[fast_span]
```

### 7.4 Capped Forecast

Prevent extreme forecasts from causing outsized positions:

```
capped_forecast = max(min(scaled_forecast, +20), -20)
```

```python
def cap_forecast(scaled: pd.Series, cap: float = 20.0) -> pd.Series:
    return scaled.clip(-cap, cap)
```

### 7.5 Position Sizing with a Single Forecast

A forecast of +10 means "hold the average long position". +20 means "hold double". 0 means flat.

```
N_i,t = capped_forecast × Capital × IDM × weight_i × target_risk
        / (10 × multiplier_i × price_i × fx_i × σ_%_i)
```

The `10` in the denominator normalizes for the fact that the average absolute forecast is 10.

```python
def position_with_forecast(forecast, capital, idm, weight, target_risk,
                            multiplier, price, fx, sigma_pct):
    return (forecast * capital * idm * weight * target_risk) / \
           (10 * multiplier * price * fx * sigma_pct)
```

### 7.6 EWMAC Rule Variations

Six speeds, from fast (reactive, noisy) to slow (smooth, less responsive):

| Rule | Fast Span | Slow Span | Character |
|---|---|---|---|
| EWMAC2  | 2  | 8   | Very fast, noisy |
| EWMAC4  | 4  | 16  | Fast |
| EWMAC8  | 8  | 32  | Medium-fast |
| EWMAC16 | 16 | 64  | Medium |
| EWMAC32 | 32 | 128 | Slow |
| EWMAC64 | 64 | 256 | Very slow, smooth |

---

## 8. Combining Multiple Forecasts

Running several EWMAC speeds and blending them gives a more robust signal than any single rule alone.

### 8.1 Per-Instrument, Per-Rule Forecast

For each instrument `i` and rule variation `j`:

```
raw_forecast_i,j,t    = (fast_EWMA_i,j,t - slow_EWMA_i,j,t) / σ_p_i,t
scaled_forecast_i,j,t = raw_forecast_i,j,t × scalar_j
capped_forecast_i,j,t = max(min(scaled_forecast_i,j,t, +20), -20)
```

### 8.2 Weighted Average (Raw Combined Forecast)

```
raw_combined_i,t = Σ_j (w_j × capped_forecast_i,j,t)
```

Forecast weights `w_j` sum to 1 and are set equal for all remaining rules (after removing rules too expensive for a given instrument).

```python
def combined_forecast(forecasts: dict, weights: dict) -> pd.Series:
    """
    forecasts : {rule_name: pd.Series of capped forecasts}
    weights   : {rule_name: float}  (must sum to 1)
    """
    return sum(weights[name] * forecasts[name] for name in forecasts)
```

### 8.3 Forecast Diversification Multiplier (FDM)

Because the combined forecasts are correlated, the blended forecast has lower absolute average than 10. FDM corrects for this, analogous to IDM for instruments.

```
scaled_combined_i,t = raw_combined_i,t × FDM_i
```

FDM values based on which EWMAC rules survive the cost filter:

| Rules Used | Forecast Weight Each | FDM |
|---|---|---|
| EWMAC2, 4, 8, 16, 32, 64 | 0.167 | 1.26 |
| EWMAC4, 8, 16, 32, 64    | 0.200 | 1.19 |
| EWMAC8, 16, 32, 64       | 0.250 | 1.13 |
| EWMAC16, 32, 64          | 0.333 | 1.08 |
| EWMAC32, 64              | 0.500 | 1.03 |
| EWMAC64 only             | 1.000 | 1.00 |

### 8.4 Final Combined Forecast (Capped Again)

After applying FDM, re-cap since diversification can push the value above 20:

```
capped_combined_i,t = max(min(scaled_combined_i,t, +20), -20)
```

### 8.5 Final Position Sizing

```
N_i,t = capped_combined_i,t × Capital × IDM × weight_i × target_risk
        / (10 × multiplier_i × price_i × fx_i × σ_%_i)
```

### 8.6 Forecast Weight Selection Rules

1. **Remove expensive rules first:** Drop any rule where annual risk-adjusted cost > 0.15 SR units.
2. **Equal weights among surviving rules** (within the EWMAC family).
3. **Prefer diversifying rules:** Slowest and fastest rules add the most diversification.
4. **Avoid overfitting:** Don't over-optimize weights to historical data.

---

## 9. Buffering (Reducing Unnecessary Trades)

Buffering prevents constant small re-sizing trades that eat into returns via transaction costs.

### 9.1 Position Buffer

Set a "buffer zone" around the optimal position. Only trade when the current position falls outside this zone.

```
B_i,t (buffer size) = F × Capital × IDM × weight_i × target_risk
                     / (multiplier_i × price_i × fx_i × σ_%_i)

where F = 0.10  (10% of the average expected position)
```

```
Lower bound  = round(N_optimal - B)
Upper bound  = round(N_optimal + B)

Decision:
  lower ≤ current ≤ upper  →  No trade needed
  current < lower          →  Buy (lower - current) contracts
  current > upper          →  Sell (current - upper) contracts
```

```python
def buffer_trade(n_optimal, buffer_fraction, avg_position, current_position):
    B = buffer_fraction * avg_position
    lower = round(n_optimal - B)
    upper = round(n_optimal + B)
    if lower <= current_position <= upper:
        return 0                          # no trade
    elif current_position < lower:
        return lower - current_position   # buy
    else:
        return upper - current_position   # sell (negative = sell)
```

### 9.2 Portfolio-Level Tracking Error Buffer (Dynamic Optimisation)

For dynamic optimisation, buffering operates on the entire portfolio's tracking error rather than individual positions.

```
B_σ = 0.05 × target_risk
```

At a 20% risk target: `B_σ = 1%`

**Steps:**

1. Compute the optimised portfolio (see Section 10).
2. Compute the tracking error `T` of your *current* portfolio vs the optimised portfolio.
3. If `T < B_σ` → no trades required.
4. If `T ≥ B_σ` → trade partially:

```
adjustment_factor a = max((T - B_σ) / T, 0)
required_trade_i    = round(a × (N*_i - current_i))
```

```python
def tracking_error_buffer(current_weights, optimised_weights, cov_matrix, b_sigma):
    e = optimised_weights - current_weights
    T = np.sqrt(e @ cov_matrix @ e)
    if T <= b_sigma:
        return 0.0   # no trade needed
    a = (T - b_sigma) / T
    return a         # multiply required trades by this factor
```

---

## 10. Dynamic Portfolio Optimisation

For smaller accounts that can't hold the full "jumbo" portfolio, dynamic optimisation finds the best integer-contract allocation that minimises tracking error versus the ideal fractional portfolio.

### 10.1 Core Concept

Convert all positions to **portfolio weights** (proportion of capital):

```
weight_per_contract_i      = notional_exposure_i / capital
                           = (multiplier_i × price_i × fx_i) / capital

optimal_unrounded_weight_i = N_i × weight_per_contract_i
```

We want to find integer multiples of `weight_per_contract_i` that minimise:

```
tracking_error_std = sqrt(e^T · Σ · e)

where e = w* - w    (tracking error weight per instrument)
      w* = optimised integer weights
      w  = optimal unrounded weights
      Σ  = covariance matrix of percentage returns
```

### 10.2 Covariance Matrix Inputs

- **Volatility:** EWMA with 32-day span on daily percentage returns.
- **Correlation:** 6-month lookback on weekly percentage returns (longer window = more stable).

```python
def percentage_returns(back_adj_prices: pd.Series, contract_prices: pd.Series) -> pd.Series:
    return back_adj_prices.diff() / contract_prices.shift(1)

# Covariance = correlation * outer product of std devs
def covariance_matrix(pct_returns_df: pd.DataFrame) -> np.ndarray:
    weekly  = pct_returns_df.resample("W").sum()
    corr    = weekly.tail(26).corr().values        # ~6 months of weekly data
    daily_std = pct_returns_df.ewm(span=32).std().iloc[-1].values
    annual_std = daily_std * np.sqrt(256)
    return np.outer(annual_std, annual_std) * corr
```

### 10.3 Greedy Optimisation Algorithm

Start from zero and incrementally add/remove one contract at a time, always choosing the step that most reduces tracking error:

```
1. Start with w* = {0 for all instruments}  (current best solution)
2. Loop:
   a. proposed = current best solution
   b. For each instrument i:
      - If optimal N_i > 0: try adding +1 contract  (w*_i += weight_per_contract_i)
      - If optimal N_i < 0: try removing -1 contract (w*_i -= weight_per_contract_i)
      - Compute tracking error of this incremented solution
      - If better than proposed, update proposed
   c. If proposed is better than current best → current best = proposed
   d. If not → STOP (we're at a local minimum)
3. Convert optimised weights back to contracts:
   N*_i = w*_i / weight_per_contract_i    (always an integer)
```

```python
def greedy_optimise(optimal_weights, weight_per_contract, cov_matrix):
    n_instruments = len(optimal_weights)
    best = np.zeros(n_instruments)

    while True:
        proposed = best.copy()
        best_te  = tracking_error(best, optimal_weights, cov_matrix)

        for i in range(n_instruments):
            delta = weight_per_contract[i] if optimal_weights[i] >= 0 else -weight_per_contract[i]
            trial = best.copy()
            trial[i] += delta
            te = tracking_error(trial, optimal_weights, cov_matrix)
            if te < tracking_error(proposed, optimal_weights, cov_matrix):
                proposed = trial

        if tracking_error(proposed, optimal_weights, cov_matrix) < tracking_error(best, optimal_weights, cov_matrix):
            best = proposed
        else:
            break

    return np.round(best / weight_per_contract).astype(int)

def tracking_error(w_star, w_optimal, cov_matrix):
    e = w_star - w_optimal
    return np.sqrt(e @ cov_matrix @ e)
```

### 10.4 Cost Penalty in Dynamic Optimisation

To avoid excessive turnover, add a cost penalty to the tracking error:

```
cost_in_weight_i   = (C_i / capital) / weight_per_contract_i
trade_as_weight_i  = |previous_weight_i - optimised_weight_i|
total_cost         = Σ_i (trade_i × cost_in_weight_i)

penalised_tracking_error = sqrt(e^T · Σ · e + 50 × total_cost)
```

The multiplier `50` on cost was empirically chosen. It discourages unnecessary trades without completely preventing beneficial rebalancing.

```python
def penalised_tracking_error(w_star, w_optimal, w_prev, cost_per_contract,
                              capital, weight_per_contract, cov_matrix):
    e         = w_star - w_optimal
    variance  = e @ cov_matrix @ e
    cost_w    = (cost_per_contract / capital) / weight_per_contract
    trades    = np.abs(w_prev - w_star)
    total_cost = np.dot(trades, cost_w)
    return np.sqrt(variance + 50 * total_cost)
```

### 10.5 Additional Uses of Dynamic Optimisation

- **Proxy trading:** Generate optimal positions for ~150 instruments but only allow trading in ~100 (due to cost/liquidity). The optimizer will route exposure to correlated tradeable instruments.
- **Position limits:** Set min/max contracts per instrument (e.g., max 10 VIX contracts).
- **Market holidays:** Lock an instrument's position to current level; optimizer trades proxies.
- **Roll management:** Force a rolling instrument to zero; let optimizer decide whether to re-enter or transfer to correlated markets.

---

## Appendix: Quick Reference Checklist

**Before trading an instrument:**
- [ ] `risk_adjusted_cost_per_trade < 0.01`
- [ ] `annual_risk_adjusted_cost < 0.10`
- [ ] Minimum capital requirement met for at least 4 contracts

**Before using a trading rule variation:**
- [ ] `annual_risk_adjusted_cost < 0.15` for that rule on that instrument

**Daily position sizing flow:**
1. Calculate `σ_p` (daily vol in price points) using 32-day EWMA
2. Calculate raw and scaled forecasts for each EWMAC variation
3. Cap individual forecasts at ±20
4. Combine with forecast weights + FDM, cap combined at ±20
5. Compute optimal `N` using the position sizing formula
6. Apply buffer — only trade if outside buffer zone

**Key parameter values:**
- Buffer fraction `F` = 0.10
- Tracking error buffer `B_σ` = 0.05 × target_risk
- Cost penalty multiplier = 50
- Forecast cap = ±20
- Average absolute forecast target = 10
- Annualisation factor = 16 (≈ sqrt(256) trading days)
