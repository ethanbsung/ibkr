Trading Strategy Notes

Overview

Go long or short on one or more instruments with variable risk estimates using a combined forecast from multiple momentum filters.

Instrument(s)

Choose instruments that meet minimum capital, liquidity, and cost thresholds.

Trading Rules Selection

Choose trading rules where:

Turnover < [0.15 - (Cost per trade × Rolls per year)] ÷ Cost per trade

EWMA Calculations

Exponentially Weighted Moving Average (EWMA) calculation for an N-day span using back-adjusted prices:

λ = 2 ÷ (N + 1)

EWMA(N)_i,t = λp_i,t + λ(1 - λ)p_i,t-1 + λ(1 - λ)^2p_i,t-2 + ...

Forecast Calculations

Raw Forecast

Raw forecast(N)_i,t = [EWMA(N)_i,t - EWMA(4N)_i,t] ÷ σ_p

Scaled Forecast

Scaled forecast(N)_i,t = Raw forecast(N)_i,t × forecast scalar(N)

Capped Forecast

Capped forecast(N)_i,t = Max(Min(Scaled forecast(N)_i,t, +20), -20)

Forecast Combination

Given forecasts f_i,j,t with weights w_i,j:

Raw combined forecast_i,t = w_i,1 f_i,1,t + w_i,2 f_i,2,t + w_i,3 f_i,3,t + …

Forecast Diversification

Scaled combined forecast_i,t = Raw combined forecast_i,t × FDM_i

Capped Combined Forecast

Capped combined forecast_i,t = Max(Min(Scaled combined forecast_i,t, +20), -20)

Optimal Position Sizing

Buy or sell N contracts based on:

N_i,t = (Capped combined forecast × Capital × IDM × Weight_i × τ) ÷ (10 × Multiplier_i × Price_i,t × FX_i,t × σ_%,i,t)

Buffer Width

B_i,t = (0.1 × Capital × IDM × Weight_i × τ) ÷ (Multiplier_i × Price_i,t × FX_i,t × σ_%,i,t)

Buffer Zone

Lower buffer, B^L_i,t = round(N_i - B_i,t)
Upper buffer, B^U_i,t = round(N_i + B_i,t)

Trading Decision Logic

Calculate optimal position considering current position C:

No trade: B^U_i,t ≤ C_i,t ≤ B^L_i,t

Buy: C_i,t < B^L_i,t: buy (B^L_i,t - C_i,t) contracts

Sell: C_i,t > B^U_i,t: sell (C_i,t - B^U_i,t) contracts