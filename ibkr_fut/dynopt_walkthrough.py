#!/usr/bin/env python3
"""
dynopt_walkthrough.py

Runs the full dynamic optimisation pipeline for a single day at $100k capital
and prints every intermediate step: tradable set selection, weights, fractional
positions, what the greedy optimiser chose, and the final tracking error.

Run from the repo root:
    python3 ibkr_fut/dynopt_walkthrough.py
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr_fut.jumbo import JUMBO
from ibkr_fut.backtest_dynamic import _build_universe
from ibkr_fut.backtest_ewmac import TARGET_RISK, IDM_CAP
from ibkr_fut.dynamic_opt import (
    optimise_positions,
    tracking_error_std,
    greedy_optimise_weights,
    apply_buffering,
)

# ─── Parameters ───────────────────────────────────────────────────────────────
CAPITAL     = 100_000.0
TARGET_RISK = 0.20          # 20% annualised
COST_LIMIT  = 1.0           # notional per contract ≤ capital * this

# ─── Step 1: Tradable set ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("STEP 1 — TRADABLE SET SELECTION")
print("=" * 72)
print(f"Capital: ${CAPITAL:,.0f}")
print(f"Rule: one contract's notional (mult × price) must be ≤ ${CAPITAL * COST_LIMIT:,.0f}")
print()

# We need raw prices & multipliers to compute notional.  Build the universe first
# then filter — same logic as live_dynamic.build_tradable_set.
print("Building universe (this takes ~30s for correlation matrix + EWMA)...")
all_instruments = list(JUMBO.keys())
uni = _build_universe(all_instruments, tradable_set=None)

if uni is None:
    print("ERROR: universe build failed.")
    sys.exit(1)

names   = uni["names"]
n       = len(names)
idx     = uni["idx"]
t       = len(idx) - 1   # latest date row
as_of   = pd.Timestamp(idx.values[t])

price   = uni["price"]
raw     = uni["raw"]
fx      = uni["fx"]
sigma   = uni["sigma"]
forecast= uni["forecast"]
mult    = uni["mult"]
spread  = uni["spread"]
commission = uni["commission"]
W       = uni["W"]        # handcraft weights
C       = uni["C"]        # static correlation matrix
est     = uni["est"]      # CovarianceEstimator

print(f"\nUniverse date: {as_of.date()}")
print(f"Total instruments with valid signals: {n}")

# Compute notional per contract for each instrument
p_t   = raw[t]          # raw contract prices today
f_t   = fx[t]           # fx rates today
notional = mult * p_t * f_t

# Valid mask: all required fields present
valid = (
    ~np.isnan(price[t]) & ~np.isnan(raw[t]) & (raw[t] > 0)
    & ~np.isnan(sigma[t]) & (sigma[t] > 0)
    & ~np.isnan(fx[t]) & ~np.isnan(forecast[t])
)

# Tradable: valid AND one contract fits capital
tradable_mask = valid & (notional <= CAPITAL * COST_LIMIT)

print(f"\n{'Instrument':<18} {'Class':<8} {'Mult':>6} {'Price':>10} "
      f"{'Notional':>12} {'Fits $100k':>11}  {'Reason if not'}")
print("-" * 80)

not_tradable_reasons = []
for j, nm in enumerate(names):
    cls = JUMBO.get(nm, "?")
    if not valid[j]:
        reason = "missing signal/vol/price"
        not_tradable_reasons.append((nm, cls, reason))
        continue
    fits = tradable_mask[j]
    reason = "" if fits else f"${notional[j]:,.0f} > limit"
    if not fits:
        not_tradable_reasons.append((nm, cls, reason))

tradable_names = [names[j] for j in range(n) if tradable_mask[j]]
print(f"  Tradable instruments: {len(tradable_names)}")
print(f"  Not tradable / no signal: {n - len(tradable_names)}")

print(f"\n{'Instrument':<18} {'Class':<10} {'Mult':>6} {'Price':>10} {'Notional ($)':>14}")
print("-" * 62)
for j in np.flatnonzero(tradable_mask):
    nm  = names[j]
    cls = JUMBO.get(nm, "?")
    print(f"  {nm:<16} {cls:<10} {mult[j]:>6.1f} {raw[t,j]:>10.2f} "
          f"  ${mult[j]*raw[t,j]*fx[t,j]:>12,.0f}")

# ─── Step 2: Live universe, weights, IDM ─────────────────────────────────────
print("\n" + "=" * 72)
print("STEP 2 — LIVE UNIVERSE WEIGHTS & IDM")
print("=" * 72)
print("""
The optimiser needs WEIGHTS (fraction of capital) for each instrument, not just
a flat list. Handcraft weights were computed once from the full correlation matrix.
We now renormalise them to sum to 1 over just the live/valid instruments today.
""")

live_idx = np.flatnonzero(valid)
w_live   = W[live_idx]
ssum     = float(w_live.sum())
w_n      = w_live / ssum if ssum > 0 else np.full(len(live_idx), 1.0 / len(live_idx))

# IDM from the live correlation submatrix
Csub = C[np.ix_(live_idx, live_idx)]
var  = float(w_n @ Csub @ w_n)
idm  = min(1.0 / np.sqrt(var), IDM_CAP) if var > 0 else 1.0

print(f"  Live instruments today:  {len(live_idx)}")
print(f"  Handcraft weight sum before renorm: {ssum:.4f}")
print(f"  IDM (1/sqrt(w'Cw)):     {idm:.3f}")
print(f"  Interpretation: at 20% target risk, portfolio actually runs at "
      f"{20/idm:.1f}% per-instrument risk to achieve 20% combined after diversification.")

print(f"\n  Top 15 live instruments by handcraft weight:")
print(f"  {'Instrument':<18} {'Class':<10} {'Weight':>8}  {'Weight×IDM':>10}")
print(f"  {'-'*52}")
order = np.argsort(w_n)[::-1]
for rank, k in enumerate(order[:15]):
    j   = live_idx[k]
    nm  = names[j]
    cls = JUMBO.get(nm, "?")
    print(f"  {nm:<18} {cls:<10} {w_n[k]:>8.4f}  {w_n[k]*idm:>10.4f}")

# ─── Step 3: Fractional optimal positions ────────────────────────────────────
print("\n" + "=" * 72)
print("STEP 3 — FRACTIONAL OPTIMAL POSITIONS (N_ideal)")
print("=" * 72)
print("""
For each instrument i, the ideal (non-integer) number of contracts:

  N_ideal = (forecast × capital × IDM × weight × target_risk)
            / (10 × multiplier × raw_price × fx × sigma_%)

The "10" normalises so a forecast of +10 means "hold the average long position."
""")

ml  = mult[live_idx]
rl  = raw[t, live_idx]
fl  = fx[t, live_idx]
sl  = sigma[t, live_idx]
fcl = forecast[t, live_idx]

N_unrounded = (fcl * CAPITAL * idm * w_n * TARGET_RISK) / (10.0 * ml * rl * fl * sl)

weight_per_contract = ml * rl * fl / CAPITAL   # each contract as fraction of capital

print(f"  {'Instrument':<18} {'Fcast':>7} {'Weight':>8} {'σ%':>7} "
      f"{'Mult×Price':>12} {'N_ideal':>9} {'wt/contract':>13}")
print(f"  {'-'*78}")

# sort by |N_ideal| descending
order2 = np.argsort(np.abs(N_unrounded))[::-1]
for k in order2:
    j  = live_idx[k]
    nm = names[j]
    print(f"  {nm:<18} {fcl[k]:>+7.2f} {w_n[k]:>8.4f} {sl[k]*100:>6.1f}%"
          f"  ${ml[k]*rl[k]:>11,.0f}  {N_unrounded[k]:>+9.3f}  {weight_per_contract[k]:>13.5f}")

print(f"\n  Key observations:")
print(f"  • Instruments with high forecast + high weight → large N_ideal")
print(f"  • Expensive contracts (large mult×price) → small N_ideal (less room)")
print(f"  • High vol (σ%) → smaller N_ideal (risk-targeted, so more vol = fewer contracts)")
print(f"  • Many instruments have |N_ideal| < 1 — that's where integer rounding matters most")

# ─── Step 4: Portfolio weights (pre-optimisation) ────────────────────────────
print("\n" + "=" * 72)
print("STEP 4 — PORTFOLIO WEIGHTS (IDEAL vs AFFORDABLE)")
print("=" * 72)
print("""
Convert N_ideal into portfolio weights (fraction of capital):
  ideal_weight_i = N_ideal_i × weight_per_contract_i
  (weight_per_contract = mult × price × fx / capital)

A weight of 0.08 means 8% of capital is "in" that instrument.
""")

target_weights = N_unrounded * weight_per_contract

print(f"  {'Instrument':<18} {'Class':<10} {'Ideal weight':>13} {'Tradable':>9}")
print(f"  {'-'*56}")
order3 = np.argsort(np.abs(target_weights))[::-1]
for k in order3:
    j      = live_idx[k]
    nm     = names[j]
    cls    = JUMBO.get(nm, "?")
    is_tr  = "YES" if tradable_mask[j] else "no"
    print(f"  {nm:<18} {cls:<10}   {target_weights[k]:>+10.5f}   {is_tr:>9}")

total_long  = float(np.sum(target_weights[target_weights > 0]))
total_short = float(np.sum(target_weights[target_weights < 0]))
print(f"\n  Total long weight:  {total_long:+.4f}  ({total_long*100:.1f}% of capital)")
print(f"  Total short weight: {total_short:+.4f}  ({total_short*100:.1f}% of capital)")
print(f"  Gross leverage:     {abs(total_long)+abs(total_short):.4f}x")

# ─── Step 5: Covariance matrix (subset) ───────────────────────────────────────
print("\n" + "=" * 72)
print("STEP 5 — COVARIANCE MATRIX")
print("=" * 72)
print("""
The covariance matrix Σ tells the optimiser how instruments move together.
  Σ = D · ρ · D   where D = diagonal matrix of vols, ρ = correlation matrix.
  Σ[i][j] = vol_i × corr(i,j) × vol_j

Tracking error variance = e' · Σ · e   (e = actual_weights - ideal_weights)

Showing the 8 instruments with largest |N_ideal| to keep it readable:
""")

cov = est.covariance_by_index(as_of, live_idx)
top8_k = np.argsort(np.abs(N_unrounded))[::-1][:8]
top8_names = [names[live_idx[k]] for k in top8_k]

print(f"  Annualised vol (diagonal of Σ, as %):")
for k in top8_k:
    j  = live_idx[k]
    nm = names[j]
    vol_pct = np.sqrt(cov[k, k]) * 100
    print(f"    {nm:<18}: {vol_pct:.1f}%  (σ_pct from ewma_vol = {sl[k]*100:.1f}%)")

print(f"\n  Correlation sub-matrix (top 8 by |N_ideal|):")
# Extract the 8×8 sub-matrix
sub_cov = cov[np.ix_(top8_k, top8_k)]
vols    = np.sqrt(np.diag(sub_cov))
outer   = np.outer(vols, vols)
with np.errstate(invalid="ignore"):
    sub_corr = np.where(outer > 0, sub_cov / outer, 0.0)
np.fill_diagonal(sub_corr, 1.0)

header = f"  {'':18}" + "".join(f"{nm[:8]:>10}" for nm in top8_names)
print(header)
for i, nm_i in enumerate(top8_names):
    row = f"  {nm_i:<18}" + "".join(f"{sub_corr[i,j]:>10.2f}" for j in range(len(top8_names)))
    print(row)

print("""
  Values close to +1.0 = move together (e.g. correlated equity indices).
  Values close to -1.0 = move opposite (e.g. equity vs bonds in risk-off).
  Values near 0.0      = largely independent.
  The optimiser prefers diversified positions: holding correlated instruments
  adds less to tracking error than holding uncorrelated ones.
""")

# ─── Step 6: Greedy optimisation ─────────────────────────────────────────────
print("=" * 72)
print("STEP 6 — GREEDY INTEGER OPTIMISATION")
print("=" * 72)
print("""
Start at zero contracts. Each iteration, try adding one contract to one
instrument (in the direction of its forecast). Keep the move that most
reduces the objective: sqrt(e'Σe) + 50×cost_penalty. Repeat until no
single move helps.
""")

tradable_live = np.array([tradable_mask[live_idx[k]] for k in range(len(live_idx))])
cost_per_contract = (spread[live_idx] * ml + commission[live_idx] / 2.0) * fl
cost_in_weight    = (cost_per_contract / CAPITAL) / weight_per_contract

# --- no-costs, no-buffer pass (clean illustration) ---
print("  Pass A: no cost penalty, no buffer (shows pure tracking-error logic)")
prev_weights_zero = np.zeros(len(live_idx))
opt_weights_nocost = greedy_optimise_weights(
    covariance=cov,
    weight_per_contract=weight_per_contract,
    target_weights=target_weights,
    previous_weights=prev_weights_zero,
    cost_in_weight=np.zeros(len(live_idx)),
    locked=~tradable_live,
)
N_nocost = np.round(opt_weights_nocost / weight_per_contract).astype(int)

# --- full pass (costs + buffer) ---
print("  Pass B: with cost penalty + tracking-error buffer (production logic)")
N_star = optimise_positions(
    covariance=cov,
    weight_per_contract=weight_per_contract,
    optimal_unrounded_positions=N_unrounded,
    previous_positions=np.zeros(len(live_idx)),
    cost_per_contract=cost_per_contract,
    capital=CAPITAL,
    target_risk=TARGET_RISK,
    use_costs=True,
    use_buffering=True,
    tradable=tradable_live,
)

# ─── Step 7: Side-by-side result table ───────────────────────────────────────
print("\n" + "=" * 72)
print("STEP 7 — RESULTS: IDEAL vs OPTIMISED POSITIONS")
print("=" * 72)

# Only show instruments with N_star ≠ 0 or N_ideal ≠ 0 (significant)
significant = np.abs(N_unrounded) > 0.05
rows = []
for k in range(len(live_idx)):
    if abs(N_star[k]) == 0 and abs(N_unrounded[k]) < 0.05:
        continue
    j        = live_idx[k]
    nm       = names[j]
    cls      = JUMBO.get(nm, "?")
    n_ideal  = N_unrounded[k]
    n_opt    = int(N_star[k])
    n_nc     = int(N_nocost[k])
    wt_ideal = target_weights[k]
    wt_opt   = n_opt * weight_per_contract[k]
    rows.append((nm, cls, fcl[k], sl[k]*100, n_ideal, n_nc, n_opt,
                 wt_ideal, wt_opt, mult[j]*rl[k]))

rows.sort(key=lambda r: abs(r[4]), reverse=True)

print(f"\n  {'Instrument':<18} {'Class':<8} {'Fcast':>7} {'σ%':>6} "
      f"{'N_ideal':>9} {'N_nocost':>9} {'N_final':>8}  "
      f"{'Wt_ideal':>9} {'Wt_final':>9}  {'Notional':>11}")
print(f"  {'-'*100}")
for (nm, cls, fc, sv, ni, nc, nf, wi, wf, notl) in rows:
    trd = "*" if tradable_mask[names.index(nm)] else " "
    print(f"  {nm:<18} {cls:<8} {fc:>+7.2f} {sv:>5.1f}%"
          f"  {ni:>+9.3f}  {nc:>+9d}  {nf:>+8d}"
          f"   {wi:>+9.5f}  {wf:>+9.5f}  ${notl:>10,.0f}  {trd}")

print(f"\n  * = tradable at $100k (notional ≤ ${CAPITAL*COST_LIMIT:,.0f} per contract)")

# ─── Step 8: Tracking error ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("STEP 8 — TRACKING ERROR ANALYSIS")
print("=" * 72)
print("""
Tracking error = sqrt(e'Σe), where e = actual_weights - ideal_weights.
It measures: if we hold this integer portfolio vs the ideal fractional one,
by how many percentage points per year would our returns be expected to differ?
""")

opt_weights_final = N_star * weight_per_contract

te_vs_ideal = tracking_error_std(opt_weights_final, target_weights, cov)

# Also compute what we'd get rounding each instrument naively (no optimiser)
naive_N = np.round(N_unrounded).astype(float)
naive_N[~tradable_live] = 0.0
naive_weights = naive_N * weight_per_contract
te_naive = tracking_error_std(naive_weights, target_weights, cov)

# Perfect (fractional) tracking error for reference
te_fractional = tracking_error_std(target_weights, target_weights, cov)

print(f"  Tracking error scenarios:")
print(f"    Fractional ideal (no rounding):    {te_fractional*100:.3f}%  (theoretical min = 0)")
print(f"    Naive rounding (no optimiser):     {te_naive*100:.3f}%")
print(f"    Dynamic optimiser (this run):      {te_vs_ideal*100:.3f}%")

print(f"""
  The optimiser achieved {te_vs_ideal*100:.3f}% tracking error vs {te_naive*100:.3f}% naive rounding.
  At a 20% annual risk target, {te_vs_ideal*100:.3f}% tracking error means the portfolio's
  returns could differ from the ideal by ~±{te_vs_ideal*100:.2f}% per year — mostly
  because we can't hold fractional contracts.
""")

# Per-instrument contribution to tracking error
print(f"  Per-instrument weight error and its contribution to total tracking error:")
print(f"  (contribution = how much this instrument's mismatch 'costs' in vol terms)")
print()
e = opt_weights_final - target_weights
g = cov @ e   # Σe
contributions = e * g  # e[i] * (Σe)[i] — each instrument's marginal variance contribution

total_var = float(e @ g)
print(f"  {'Instrument':<18} {'Wt error':>10} {'N_ideal':>9} {'N_held':>8}  "
      f"{'Contrib to var':>15}  {'Share':>7}")
print(f"  {'-'*72}")
contr_order = np.argsort(np.abs(contributions))[::-1]
shown = 0
for k in contr_order:
    if shown >= 20:
        break
    j  = live_idx[k]
    nm = names[j]
    if abs(e[k]) < 1e-7 and abs(contributions[k]) < 1e-7:
        continue
    share = contributions[k] / total_var * 100 if total_var > 0 else 0
    print(f"  {nm:<18}  {e[k]:>+10.5f}  {N_unrounded[k]:>+9.3f}  {int(N_star[k]):>+8d}"
          f"    {contributions[k]*10000:>+12.4f}bps  {share:>6.1f}%")
    shown += 1

print(f"""
  Contributions sum to {sum(contributions)*10000:.4f}bps (variance).
  sqrt of that sum × 10000 = {np.sqrt(max(total_var,0))*100:.3f}%  ← tracking error std dev.

  Why some instruments have zero error: either their N_ideal is already an integer,
  or they're non-tradable so the optimiser transferred their risk to correlated
  instruments that can be traded.
""")

# ─── Step 9: Summary ─────────────────────────────────────────────────────────
print("=" * 72)
print("STEP 9 — FINAL PORTFOLIO SUMMARY")
print("=" * 72)

held = [(names[live_idx[k]], int(N_star[k]),
         mult[live_idx[k]], raw[t, live_idx[k]], fx[t, live_idx[k]],
         JUMBO.get(names[live_idx[k]], "?"))
        for k in range(len(live_idx)) if N_star[k] != 0]
held.sort(key=lambda r: abs(r[1] * r[2] * r[3]), reverse=True)

gross_notional = sum(abs(c * m * p * f) for _, c, m, p, f, _ in held)
net_notional   = sum(c * m * p * f for _, c, m, p, f, _ in held)
n_long  = sum(1 for _, c, *_ in held if c > 0)
n_short = sum(1 for _, c, *_ in held if c < 0)

print(f"\n  Capital:               ${CAPITAL:>12,.0f}")
print(f"  Target risk:            {TARGET_RISK:.0%}")
print(f"  IDM:                    {idm:.3f}")
print(f"  Positions held:         {len(held)}  ({n_long} long, {n_short} short)")
print(f"  Gross notional:        ${gross_notional:>12,.0f}  ({gross_notional/CAPITAL:.2f}x leverage)")
print(f"  Net notional:          ${net_notional:>12,.0f}")
print(f"  Tracking error:         {te_vs_ideal*100:.3f}%")
print(f"\n  {'Instrument':<18} {'Class':<10} {'Contracts':>10} {'Notional ($)':>14}  Direction")
print(f"  {'-'*64}")
for nm, c, m, p, f, cls in held:
    direction = "LONG" if c > 0 else "SHORT"
    print(f"  {nm:<18} {cls:<10}  {c:>+10d}  ${abs(c*m*p*f):>13,.0f}  {direction}")

print(f"\n  Non-tradable instruments (locked at 0, risk transferred to above):")
locked_count = 0
for k in range(len(live_idx)):
    j = live_idx[k]
    if not tradable_mask[j] and abs(N_unrounded[k]) >= 0.05:
        nm  = names[j]
        cls = JUMBO.get(nm, "?")
        print(f"    {nm:<18} {cls:<10}  N_ideal {N_unrounded[k]:>+8.3f}  "
              f"  notional ${mult[j]*raw[t,j]:>12,.0f}  (too big for $100k)")
        locked_count += 1
        if locked_count >= 15:
            print("    ...")
            break
