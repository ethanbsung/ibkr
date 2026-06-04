"""
audit_dynamic.py — forensic audit of the dynamic-optimisation backtest at a single
capital level (default $100k). Step 1 of a full audit of the dynamic-opt implementation.

Re-runs the exact joint daily loop from backtest_dynamic._simulate, but instruments it
to expose what the optimiser is actually doing day-to-day so we can judge whether the
backtest is clean and representative of live trading:

  - positions taken: which instruments, how often, sizes (contract counts)
  - sizing sanity: gross notional / leverage, biggest single positions
  - costs: per-instrument cost & turnover attribution, conservatism of the cost model
  - buffering: how often it actually suppresses a trade vs. rubber-stamps a 1-contract move
  - warmup: positions during the early (thin-history) period
  - per-instrument P&L attribution
  - full detail dumps for a few sample days

Usage:
  source /home/ethanbsung/ibkr/venv/bin/activate && \
    python ibkr_fut/audit_dynamic.py --capital 100000
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_fut.backtest_dynamic import _build_universe
from ibkr_fut.backtest_ewmac import TARGET_RISK, IDM_CAP
from ibkr_fut.dynamic_opt import greedy_optimise_weights, apply_buffering
from ibkr_fut.foundations import ANNUAL_DAYS, performance_stats
from ibkr_fut.jumbo import JUMBO


def audit(uni: dict, capital: float, target_risk: float = TARGET_RISK,
          n_sample_days: int = 3) -> dict:
    names, idx = uni["names"], uni["idx"]
    price, raw, fx = uni["price"], uni["raw"], uni["fx"]
    sigma, forecast = uni["sigma"], uni["forecast"]
    mult, spread, commission = uni["mult"], uni["spread"], uni["commission"]
    W, C, est = uni["W"], uni["C"], uni["est"]
    T, n = len(idx), len(names)
    idx_values = idx.values

    pos = np.zeros(n)
    pnl_list = np.zeros(T)
    total_cost = 0.0
    abs_changes = 0.0
    abs_pos_sum = 0.0

    # per-instrument accumulators
    days_held       = np.zeros(n)       # days with nonzero position
    abs_pos_when_held = np.zeros(n)     # sum |pos| over days held
    max_abs_pos     = np.zeros(n)
    contracts_traded = np.zeros(n)      # one-way contracts
    cost_by_instr   = np.zeros(n)
    pnl_by_instr    = np.zeros(n)

    # per-day diagnostics
    n_held       = np.zeros(T)
    gross_notional = np.zeros(T)        # |pos|*mult*raw*fx  (absolute exposure)
    net_notional = np.zeros(T)
    te_series    = np.full(T, np.nan)
    trades_per_day = np.zeros(T)
    cost_per_day = np.zeros(T)

    # buffering effectiveness
    buffer_suppressed_days = 0          # days where buffering reduced trade vs greedy
    greedy_wanted_trade_days = 0        # days greedy differed from prev
    contracts_saved_by_buffer = 0.0     # |greedy-prev| - |final-prev| summed

    sample_idx = np.linspace(int(T * 0.4), int(T * 0.9), n_sample_days).astype(int)
    samples = []

    first_position_day = None

    for t in range(T):
        p, r, f = price[t], raw[t], fx[t]
        s, fc = sigma[t], forecast[t]
        valid = (
            ~np.isnan(p) & ~np.isnan(r) & (r > 0)
            & ~np.isnan(s) & (s > 0) & ~np.isnan(f) & ~np.isnan(fc)
        )

        daily_pnl = 0.0
        per_instr_pnl = np.zeros(n)
        if t > 0:
            m = valid & ~np.isnan(price[t - 1])
            if m.any():
                dp = p[m] - price[t - 1][m]
                contrib = pos[m] * dp * mult[m] * f[m]
                per_instr_pnl[np.flatnonzero(m)] = contrib
                daily_pnl = float(np.sum(contrib))
        pnl_by_instr += per_instr_pnl

        live_idx = np.flatnonzero(valid)
        if live_idx.size == 0:
            pnl_list[t] = daily_pnl
            abs_pos_sum += np.abs(pos).sum()
            continue

        key = tuple(live_idx.tolist())
        w_live = W[live_idx]
        ssum = float(w_live.sum())
        w_n = w_live / ssum if ssum > 0 else np.full(live_idx.size, 1.0 / live_idx.size)
        Csub = C[np.ix_(live_idx, live_idx)]
        var = float(w_n @ Csub @ w_n)
        idm_t = min(1.0 / np.sqrt(var), IDM_CAP) if var > 0 else 1.0

        ml, rl, fl, sl, fcl = mult[live_idx], r[live_idx], f[live_idx], s[live_idx], fc[live_idx]
        N_unrounded = (fcl * capital * idm_t * w_n * target_risk) / (10.0 * ml * rl * fl * sl)
        weight_per_contract = ml * rl * fl / capital
        cost_per_contract = (spread[live_idx] * ml + commission[live_idx] / 2.0) * fl

        cov = est.covariance_by_index(pd.Timestamp(idx_values[t]), live_idx)
        prev_live = pos[live_idx]

        # --- replicate optimise_positions internals so we can see pre/post buffer ---
        target_weights = N_unrounded * weight_per_contract
        previous_weights = prev_live * weight_per_contract
        cost_in_weight = (cost_per_contract / capital) / weight_per_contract
        opt_weights = greedy_optimise_weights(
            cov, weight_per_contract, target_weights, previous_weights, cost_in_weight
        )
        final_weights = apply_buffering(
            opt_weights, previous_weights, cov, weight_per_contract, target_risk
        )
        N_greedy = np.round(opt_weights / weight_per_contract)
        N_star = np.round(final_weights / weight_per_contract)

        # buffering effectiveness
        greedy_delta = np.abs(N_greedy - prev_live)
        final_delta = np.abs(N_star - prev_live)
        if greedy_delta.sum() > 0:
            greedy_wanted_trade_days += 1
        if final_delta.sum() < greedy_delta.sum() - 1e-9:
            buffer_suppressed_days += 1
            contracts_saved_by_buffer += float(greedy_delta.sum() - final_delta.sum())

        trades = np.abs(N_star - prev_live)
        trade_cost = float(np.sum(trades * cost_per_contract))
        total_cost += trade_cost
        abs_changes += float(trades.sum())

        contracts_traded[live_idx] += trades
        cost_by_instr[live_idx] += trades * cost_per_contract
        trades_per_day[t] = float(trades.sum())
        cost_per_day[t] = trade_cost

        pos[live_idx] = N_star

        nz = np.flatnonzero(N_star != 0)
        if nz.size and first_position_day is None:
            first_position_day = t
        held_global = live_idx[N_star != 0]
        days_held[held_global] += 1
        abs_pos_when_held[held_global] += np.abs(N_star[N_star != 0])
        max_abs_pos[live_idx] = np.maximum(max_abs_pos[live_idx], np.abs(N_star))

        # exposure
        expo = N_star * ml * rl * fl
        gross_notional[t] = float(np.sum(np.abs(expo)))
        net_notional[t] = float(np.sum(expo))
        n_held[t] = int(np.count_nonzero(N_star))

        e = (N_star - N_unrounded) * weight_per_contract
        te_series[t] = float(np.sqrt(max(e @ cov @ e, 0.0)))

        abs_pos_sum += np.abs(pos).sum()
        pnl_list[t] = daily_pnl - trade_cost

        if t in sample_idx:
            order = np.argsort(-np.abs(N_unrounded))
            rows = []
            for k in order[:12]:
                rows.append(dict(
                    instr=names[live_idx[k]],
                    fcast=round(float(fcl[k]), 1),
                    N_ideal=round(float(N_unrounded[k]), 2),
                    N_held=int(N_star[k]),
                    wpc=round(float(weight_per_contract[k]), 5),
                    notional=int(N_star[k] * ml[k] * rl[k] * fl[k]),
                    cost_1=round(float(cost_per_contract[k]), 2),
                ))
            samples.append(dict(
                date=str(pd.Timestamp(idx_values[t]).date()),
                idm=round(idm_t, 3), n_live=int(live_idx.size),
                n_held=int(np.count_nonzero(N_star)),
                gross_notional=int(gross_notional[t]),
                leverage=round(gross_notional[t] / capital, 2),
                te=round(float(te_series[t]) * 100, 2),
                rows=rows,
            ))

    pnl_s = pd.Series(pnl_list, index=idx)
    daily_returns = pnl_s / capital
    equity = capital * (1.0 + daily_returns).cumprod()
    years = T / ANNUAL_DAYS

    nh_series = pd.Series(n_held, index=idx)
    lev_series = pd.Series(gross_notional / capital, index=idx)
    cost_series = pd.Series(cost_per_day, index=idx)
    avg_abs_N = abs_pos_sum / T if T else 1.0
    turnover = (abs_changes / 2.0) / avg_abs_N / years if years and avg_abs_N else 0.0
    costs_pct = (total_cost / capital / years) * 100 if years else 0.0
    stats = performance_stats(equity, daily_returns, costs_pct=costs_pct, turnover=turnover)

    return dict(
        names=names, idx=idx, T=T, n=n, years=years, capital=capital,
        stats=stats, total_cost=total_cost, costs_pct=costs_pct, turnover=turnover,
        days_held=days_held, abs_pos_when_held=abs_pos_when_held, max_abs_pos=max_abs_pos,
        contracts_traded=contracts_traded, cost_by_instr=cost_by_instr, pnl_by_instr=pnl_by_instr,
        n_held=n_held, gross_notional=gross_notional, net_notional=net_notional,
        te_series=te_series, trades_per_day=trades_per_day, cost_per_day=cost_per_day,
        buffer_suppressed_days=buffer_suppressed_days,
        greedy_wanted_trade_days=greedy_wanted_trade_days,
        contracts_saved_by_buffer=contracts_saved_by_buffer,
        first_position_day=first_position_day, samples=samples,
        daily_returns=daily_returns, nh_series=nh_series,
        lev_series=lev_series, cost_series=cost_series,
    )


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def report(a: dict):
    names = a["names"]; T = a["T"]; cap = a["capital"]; years = a["years"]
    st = a["stats"]
    valid_days = int(np.sum(a["n_held"] >= 0))  # all days
    active = a["n_held"] > 0
    n_active_days = int(active.sum())

    print("=" * 100)
    print(f"DYNAMIC-OPT AUDIT  |  capital = {_fmt_money(cap)}  |  {T} days (~{years:.1f}y)  |  "
          f"{a['n']} instruments in universe")
    print("=" * 100)

    print("\n── HEADLINE STATS ──")
    print(f"  Mean return   {st['mean_annual_return_pct']:.2f}%   "
          f"Std dev {st['std_dev_pct']:.2f}%   SR {st['sharpe_ratio']:.3f}   "
          f"MaxDD {st['max_drawdown_pct']:.1f}%   Skew {st['skew']:.2f}")
    print(f"  Costs {a['costs_pct']:.2f}%/yr (total {_fmt_money(a['total_cost'])})   "
          f"Turnover {a['turnover']:.1f}x")

    print("\n── PORTFOLIO SHAPE PER DAY ──")
    nh = a["n_held"][active]
    print(f"  Days with >=1 position: {n_active_days}/{T} ({100*n_active_days/T:.1f}%)")
    print(f"  First nonzero position on day index {a['first_position_day']} "
          f"(date {a['idx'][a['first_position_day']].date() if a['first_position_day'] else 'n/a'})")
    if nh.size:
        print(f"  # instruments held: mean {nh.mean():.1f}  median {np.median(nh):.0f}  "
              f"min {nh.min():.0f}  max {nh.max():.0f}")
    lev = a["gross_notional"][active] / cap
    if lev.size:
        print(f"  Gross leverage (|notional|/capital): mean {lev.mean():.2f}x  "
              f"median {np.median(lev):.2f}x  p95 {np.percentile(lev,95):.2f}x  max {lev.max():.2f}x")
    nn = a["net_notional"][active] / cap
    if nn.size:
        print(f"  Net leverage (signed):              mean {nn.mean():+.2f}x  "
              f"p5 {np.percentile(nn,5):+.2f}x  p95 {np.percentile(nn,95):+.2f}x")
    te = a["te_series"][~np.isnan(a["te_series"])]
    if te.size:
        print(f"  Tracking error vs ideal (ann. std): mean {100*te.mean():.2f}%  "
              f"median {100*np.median(te):.2f}%  p95 {100*np.percentile(te,95):.2f}%")

    print("\n── POSITION SIZES ──")
    mx = a["max_abs_pos"]
    held = a["days_held"] > 0
    avg_size = np.where(a["days_held"] > 0, a["abs_pos_when_held"] / np.maximum(a["days_held"],1), 0)
    print(f"  Max |contracts| over all instr/days: {int(mx.max())} "
          f"({names[int(np.argmax(mx))]})")
    print(f"  Avg |position| when held, across held instruments: "
          f"{avg_size[held].mean():.2f}")
    big = np.argsort(-mx)[:8]
    print("  Largest single positions ever taken:")
    for j in big:
        if mx[j] == 0: continue
        print(f"    {names[j]:<8} max {int(mx[j]):>3} contracts   "
              f"avg-when-held {avg_size[j]:.1f}   held {int(a['days_held'][j])} days")

    print("\n── COST & TURNOVER ATTRIBUTION (top 12 by cost) ──")
    cb = a["cost_by_instr"]; ct = a["contracts_traded"]
    order = np.argsort(-cb)[:12]
    print(f"  {'instr':<8} {'cost$':>10} {'%oftot':>7} {'contracts':>10} {'$/contract':>11} {'daysHeld':>9}")
    for j in order:
        if cb[j] == 0: continue
        print(f"  {names[j]:<8} {cb[j]:>10,.0f} {100*cb[j]/a['total_cost']:>6.1f}% "
              f"{ct[j]:>10,.0f} {cb[j]/max(ct[j],1):>11,.2f} {int(a['days_held'][j]):>9}")

    print("\n── BUFFERING EFFECTIVENESS ──")
    gw = a["greedy_wanted_trade_days"]; bs = a["buffer_suppressed_days"]
    print(f"  Days greedy wanted to trade:        {gw}")
    print(f"  Days buffering reduced the trade:   {bs} ({100*bs/max(gw,1):.1f}% of those)")
    print(f"  Contracts saved by buffering:       {a['contracts_saved_by_buffer']:,.0f}")
    tdays = a["trades_per_day"]
    traded_days = int(np.sum(tdays > 0))
    print(f"  Days with any actual trade:         {traded_days}/{T} ({100*traded_days/T:.1f}%)")
    print(f"  Avg contracts traded / trading day: {tdays[tdays>0].mean():.2f}")

    print("\n── P&L ATTRIBUTION (top/bottom 6 instruments) ──")
    pb = a["pnl_by_instr"]
    order = np.argsort(-pb)
    print("  Best:")
    for j in order[:6]:
        print(f"    {names[j]:<8} {_fmt_money(pb[j]):>14}  ({100*pb[j]/cap/years:+.2f}%/yr)")
    print("  Worst:")
    for j in order[-6:]:
        print(f"    {names[j]:<8} {_fmt_money(pb[j]):>14}  ({100*pb[j]/cap/years:+.2f}%/yr)")

    print("\n── ERA BREAKDOWN (does the headline depend on an unrepresentable early period?) ──")
    dr = a["daily_returns"]; nhs = a["nh_series"]; ls = a["lev_series"]; cs = a["cost_series"]
    eras = [("1970-1989", "1970", "1989"), ("1990-1999", "1990", "1999"),
            ("2000-2009", "2000", "2009"), ("2010-2019", "2010", "2019"),
            ("2020-2025", "2020", "2026"), ("last 10y", None, None)]
    print(f"  {'era':<12}{'ann.ret%':>9}{'vol%':>7}{'SR':>6}{'costs%':>8}"
          f"{'avg#held':>9}{'avgLev':>8}{'maxLev':>8}")
    for label, lo, hi in eras:
        if label == "last 10y":
            cut = a["idx"][-1] - pd.Timedelta(days=3653)
            mask = a["idx"] >= cut
        else:
            mask = (a["idx"] >= lo) & (a["idx"] <= hi)
        if mask.sum() < 60:
            continue
        d = dr[mask]
        ann = d.mean() * ANNUAL_DAYS * 100
        vol = d.std() * np.sqrt(ANNUAL_DAYS) * 100
        sr = ann / vol if vol else 0.0
        cpct = cs[mask].sum() / cap / (mask.sum() / ANNUAL_DAYS) * 100
        active_m = nhs[mask] > 0
        avgh = nhs[mask][active_m].mean() if active_m.any() else 0
        avgl = ls[mask][active_m].mean() if active_m.any() else 0
        maxl = ls[mask][active_m].max() if active_m.any() else 0
        print(f"  {label:<12}{ann:>9.2f}{vol:>7.2f}{sr:>6.2f}{cpct:>8.2f}"
              f"{avgh:>9.1f}{avgl:>8.2f}{maxl:>8.2f}")

    print("\n── SAMPLE DAYS (full detail, top-12 by |ideal N|) ──")
    for s in a["samples"]:
        print(f"\n  {s['date']}  IDM={s['idm']}  live={s['n_live']}  held={s['n_held']}  "
              f"gross_notional={_fmt_money(s['gross_notional'])}  lev={s['leverage']}x  TE={s['te']}%")
        print(f"    {'instr':<8}{'fcast':>7}{'N_ideal':>9}{'N_held':>8}{'wpc':>9}"
              f"{'notional':>12}{'$/contract':>11}")
        for r in s["rows"]:
            print(f"    {r['instr']:<8}{r['fcast']:>7}{r['N_ideal']:>9}{r['N_held']:>8}"
                  f"{r['wpc']:>9}{_fmt_money(r['notional']):>12}{r['cost_1']:>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--sample-days", type=int, default=3)
    args = ap.parse_args()

    print(f"Loading {len(JUMBO)} instruments (Carver Jumbo)...")
    uni = _build_universe(list(JUMBO.keys()))
    if uni is None:
        print("No tradable instruments.")
        return
    a = audit(uni, args.capital, n_sample_days=args.sample_days)
    report(a)


if __name__ == "__main__":
    main()
