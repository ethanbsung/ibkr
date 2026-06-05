"""
backtest_ewmac.py — EWMAC backtest engine.
Phase 3: single-instrument.
Phase 4: full 252-instrument PST portfolio with handcrafted weights.
All equations reference ibkr_fut/calcs.txt (Carver AFTS chapters 7-9).
"""

import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pst_loader import PSTLoader
from ibkr_fut.foundations import (
    PST_CUTOFF,
    ANNUAL_DAYS,
    VOL_SCALAR,
    blended_vol,
    pct_returns_backadjusted,
    sigma_p_from_pct,
    sr_cost_per_trade,
    annual_sr_cost,
    compute_corr_matrix,
    handcraft_weights,
    idm_from_corr,
    idm_from_count,
    performance_stats,
)
from ibkr_fut.ewmac_signals import (
    combined_forecast,
    eligible_speeds,
    _SPEED_TURNOVER,
)

# ── Parameters ───────────────────────────────────────────────────────────────────

CAPITAL          = 100_000   # fixed capital for position sizing throughout
TARGET_RISK      = 0.20
BUFFER_FRAC      = 0.10         # [calcs line 113]
COMMISSION       = 1.5          # round-trip commission per contract
IDM_CAP          = 2.5          # cap IDM for live realism (Carver caps at 2.5)  [calcs line 69]
MAX_INST_SR_COST = 0.10         # [calcs line 58]
MIN_HISTORY_DAYS = 512          # ~2 years

pst = PSTLoader()


# ── Single-instrument backtest ────────────────────────────────────────────────────

@dataclass
class InstrumentSpec:
    """
    Data-agnostic description of one tradable instrument, consumed by the shared
    engine. Decouples the simulation from the data source (PST futures, ETF CSVs):

      prices      back-adjusted price series — the P&L driver. Its day-to-day
                  *changes* carry roll gaps (true point P&L). For cash ETFs there
                  are no rolls, so prices == raw_price (total-return close).
      raw_price   always-positive price — denominator for returns, vol and sizing.
      fx          FX to USD, aligned to prices.index (all 1.0 for USD instruments).
      mult        contract multiplier (1.0 for ETFs: one share, price is notional).
      spread      half-spread in price points (one-way).
      rolls       rolls per year (cost filter only; 0 for ETFs).
      commission  round-trip commission per contract/share.
      long_only   clip position >= 0 (non-shortable instruments).
    """
    name: str
    prices: pd.Series
    raw_price: pd.Series
    fx: pd.Series
    mult: float
    spread: float
    rolls: int
    commission: float = 0.0
    long_only: bool = False


def _pst_spec(instrument: str) -> InstrumentSpec | None:
    """Build an InstrumentSpec from PST futures data (the futures adapter)."""
    try:
        info = pst.instrument_info(instrument)
    except (ValueError, KeyError):
        return None
    mult = float(info["Pointsize"])
    ccy  = str(info["Currency"])

    prices = pst.adjusted_prices(instrument)
    prices = prices[prices.index <= PST_CUTOFF]
    if len(prices) < MIN_HISTORY_DAYS:
        return None

    # Raw front-contract price: always positive. The back-adjusted `prices` series
    # is used only for P&L — its *changes* carry roll gaps, but its *level* can be
    # zero/negative and must never be a divisor.
    try:
        contract = pst.multiple_prices(instrument)["PRICE"]
        contract = contract[contract.index <= PST_CUTOFF].reindex(prices.index).ffill()
    except (FileNotFoundError, KeyError):
        contract = prices  # fallback (should not occur for PST instruments)

    if ccy != "USD":
        try:
            fx_series = pst.fx_rate(ccy).reindex(prices.index, method="ffill")
        except FileNotFoundError:
            fx_series = pd.Series(1.0, index=prices.index)
    else:
        fx_series = pd.Series(1.0, index=prices.index)

    return InstrumentSpec(
        name=instrument, prices=prices, raw_price=contract, fx=fx_series, mult=mult,
        spread=float(info.get("SpreadCost", 0.0)), rolls=_rolls_per_year(info),
        commission=COMMISSION,
    )


def instrument_signals(spec: InstrumentSpec) -> dict | None:
    """
    Shared signal builder (used by both the futures and ETF engines). Returns the
    vol and combined-forecast series for one instrument, plus its eligible speeds,
    or None if the instrument has invalid stats / no cost-eligible speeds.
    """
    ret   = pct_returns_backadjusted(spec.prices, spec.raw_price)
    sigma = blended_vol(ret)
    sp    = sigma_p_from_pct(spec.raw_price, sigma)   # points vol off the raw price

    med_price = float(spec.raw_price[spec.raw_price > 0].median())
    med_sigma = float(sigma.dropna().median())
    if med_sigma <= 0 or med_price <= 0 or np.isnan(med_price):
        return None

    # SR cost per trade (risk-adjusted, dimensionless) at median price/vol. With
    # historical sigma_p scaling this is invariant through time, so eligible speeds
    # computed here hold for the whole backtest.  [calcs line 176]
    c_trade = sr_cost_per_trade(spec.spread, spec.mult, med_price, med_sigma, spec.commission)
    active  = eligible_speeds(c_trade, spec.rolls)
    if not active:
        return None

    sigma_a    = sigma.reindex(spec.prices.index, method="ffill")
    sp_a       = sp.reindex(spec.prices.index, method="ffill")
    forecast_s = combined_forecast(spec.prices, sp_a, active)
    return {"sigma": sigma_a, "forecast": forecast_s, "active": active}


def simulate_instrument(
    spec: InstrumentSpec,
    capital: float = CAPITAL,
    target_risk: float = TARGET_RISK,
    idm: float = 1.0,
    weight: float = 1.0,
    buffer_frac: float = BUFFER_FRAC,
    round_positions: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Standalone single-instrument EWMAC simulation from an InstrumentSpec.

    Position sizing uses fixed capital throughout (Carver's standard approach);
    returns are P&L / capital so std_dev reflects the actual risk target.
    round_positions rounds to whole contracts (futures); ETFs that size in shares
    pass False. Returns dict: equity_curve, pnl_series, daily_returns, stats,
    costs_total, instrument.
    """
    sig = instrument_signals(spec)
    if sig is None:
        return {}
    sigma_a, forecast_s = sig["sigma"], sig["forecast"]
    fx_a = spec.fx.reindex(spec.prices.index, method="ffill")
    mult, spread, commission = spec.mult, spec.spread, spec.commission

    # ── Daily simulation ──────────────────────────────────────────────────────────
    # Each day: (1) P&L from yesterday's held position, (2) new target + buffer,
    # (3) deduct trade cost. Returns are P&L / capital (fixed capital).
    current_pos = 0.0
    total_cost  = 0.0
    abs_changes = 0.0   # sum of abs position changes (one-way)
    abs_pos_sum = 0.0   # sum of abs(current_pos) for turnover denominator
    pnl_list, dates = [], []

    prices_arr   = spec.prices.values
    contract_arr = spec.raw_price.values
    sigma_arr    = sigma_a.values
    fx_arr       = fx_a.values
    forecast_arr = forecast_s.values
    idx          = spec.prices.index

    for i in range(len(idx)):
        price     = float(prices_arr[i])      # back-adjusted: P&L only
        raw_price = float(contract_arr[i])    # raw price: sizing denominator
        sig_pct   = float(sigma_arr[i])
        fx        = float(fx_arr[i])

        if (np.isnan(price) or np.isnan(raw_price) or raw_price <= 0
                or np.isnan(sig_pct) or sig_pct <= 0 or np.isnan(fx)):
            pnl_list.append(0.0)
            dates.append(idx[i])
            abs_pos_sum += abs(current_pos)
            continue

        # Step 1: P&L from yesterday's position (back-adjusted price change)
        daily_pnl = 0.0
        if i > 0:
            prev_price = float(prices_arr[i - 1])
            if not np.isnan(prev_price):
                daily_pnl = current_pos * (price - prev_price) * mult * fx

        # Step 2: target position  [calcs line 107 / 214]
        forecast = float(forecast_arr[i])
        N_target = (forecast * capital * idm * weight * target_risk
                    / (10.0 * mult * raw_price * fx * sig_pct))

        # Step 3: buffer band = buffer_frac × average (forecast=10) position  [calcs 112-115]
        B     = (buffer_frac * capital * idm * weight * target_risk
                 / (mult * raw_price * fx * sig_pct))
        if round_positions:
            lower, upper = round(N_target - B), round(N_target + B)
        else:
            lower, upper = N_target - B, N_target + B

        # Step 4: trade decision  [calcs lines 116-119]
        if current_pos < lower:
            new_pos = lower
        elif current_pos > upper:
            new_pos = upper
        else:
            new_pos = current_pos

        if spec.long_only and new_pos < 0:
            new_pos = 0.0

        trade_size = abs(new_pos - current_pos)
        trade_cost = 0.0
        if trade_size > 0:
            trade_cost  = trade_size * (2.0 * spread * mult + commission) * fx
            total_cost += trade_cost
            abs_changes += trade_size
            current_pos  = new_pos

        abs_pos_sum += abs(current_pos)
        pnl_list.append(daily_pnl - trade_cost)
        dates.append(idx[i])

    pnl_s         = pd.Series(pnl_list, index=dates, name=spec.name)
    daily_returns = pnl_s / capital   # fixed-capital % returns          [calcs line 10]
    # Compounded equity curve — drawdowns bounded at -100% (cannot cross zero).
    equity_curve  = capital * (1.0 + daily_returns).cumprod()

    n_days    = len(dates)
    years     = n_days / ANNUAL_DAYS
    avg_abs_N = abs_pos_sum / n_days if n_days > 0 else 1.0
    turnover  = (abs_changes / 2.0) / avg_abs_N / years if years > 0 and avg_abs_N > 0 else 0.0
    costs_pct = (total_cost / capital / years) * 100   # annual costs as % of capital

    stats = performance_stats(equity_curve, daily_returns, costs_pct=costs_pct, turnover=turnover)

    if verbose:
        print(f"\n=== {spec.name} ===")
        for k, v in stats.items():
            print(f"  {k:<30} {v}")

    return {
        "equity_curve":  equity_curve,
        "pnl_series":    pnl_s,
        "daily_returns": daily_returns,
        "stats":         stats,
        "costs_total":   total_cost,
        "instrument":    spec.name,
    }


def run_single(
    instrument: str,
    capital: float = CAPITAL,
    target_risk: float = TARGET_RISK,
    idm: float = 1.0,
    weight: float = 1.0,
    verbose: bool = True,
) -> dict:
    """EWMAC backtest for one PST futures instrument (thin wrapper over the engine)."""
    spec = _pst_spec(instrument)
    if spec is None:
        return {}
    return simulate_instrument(spec, capital=capital, target_risk=target_risk,
                               idm=idm, weight=weight, verbose=verbose)


# ── Portfolio backtest ────────────────────────────────────────────────────────────

def filter_instruments(asset_classes: list[str] | None = None) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Return (eligible, excluded) instrument lists.
    Eligible: enough history AND annual SR cost <= MAX_INST_SR_COST.  [calcs line 58]
    asset_classes: if provided, restrict to instruments in those classes.
    """
    meta      = pst.instruments_df().set_index("Instrument")
    all_instr = pst.list_instruments()
    if asset_classes:
        all_instr = [
            i for i in all_instr
            if i in meta.index and str(meta.loc[i, "AssetClass"]) in asset_classes
        ]
    eligible  = []
    excluded  = []

    for instr in all_instr:
        try:
            info = pst.instrument_info(instr)
        except ValueError:
            excluded.append((instr, "no metadata"))
            continue

        prices = pst.adjusted_prices(instr)
        prices = prices[prices.index <= PST_CUTOFF]
        if len(prices) < MIN_HISTORY_DAYS:
            excluded.append((instr, f"short history ({len(prices)}d)"))
            continue

        mult   = float(info["Pointsize"])
        spread = float(info.get("SpreadCost", 0.0))
        rolls  = _rolls_per_year(info)

        try:
            contract = pst.multiple_prices(instr)["PRICE"]
            contract = contract[contract.index <= PST_CUTOFF].reindex(prices.index).ffill()
        except (FileNotFoundError, KeyError):
            contract = prices

        ret        = pct_returns_backadjusted(prices, contract)
        sigma      = blended_vol(ret)
        med_price  = float(contract[contract > 0].median())
        med_sigma  = float(sigma.dropna().median())

        if med_sigma <= 0 or med_price <= 0 or np.isnan(med_sigma) or np.isnan(med_price):
            excluded.append((instr, "invalid price/vol"))
            continue

        c    = sr_cost_per_trade(spread, mult, med_price, med_sigma, COMMISSION)
        if c > 0.01:                                     # [calcs line 59] per-trade cost limit
            excluded.append((instr, f"cost per trade too high ({c:.4f})"))
            continue
        ann  = annual_sr_cost(c, rolls, turnover=2.0)   # mid-speed estimate
        if ann > MAX_INST_SR_COST:
            excluded.append((instr, f"too expensive (SR cost {ann:.3f})"))
            continue

        eligible.append(instr)

    return eligible, excluded


def run_portfolio(
    verbose: bool = True,
    asset_classes: list[str] | None = None,
    instruments: dict[str, str] | list[str] | None = None,
    label_override: str | None = None,
) -> dict:
    """
    Full multi-instrument EWMAC portfolio backtest with handcrafted weights.
    asset_classes: if provided, restrict to those asset class names (e.g. ["Equity","Bond"]).
    instruments:   explicit, pre-selected universe (e.g. ibkr_fut.jumbo.JUMBO). If a dict,
                   it maps instrument -> asset-class label; if a list, PST classes are used.
                   When given, every instrument with enough history is traded with NO
                   instrument-level cost exclusion (selection is assumed already done);
                   per-speed cost filtering still applies inside run_single.
    """
    class_map = instruments if isinstance(instruments, dict) else None

    if instruments is not None:
        names = list(instruments.keys()) if isinstance(instruments, dict) else list(instruments)
        label = label_override or "(explicit universe)"
        print(f"Loading {len(names)} pre-selected instruments {label}...")
        eligible, excluded = [], []
        for instr in names:
            try:
                prices = pst.adjusted_prices(instr)
                prices = prices[prices.index <= PST_CUTOFF]
            except Exception:
                excluded.append((instr, "no data"))
                continue
            if len(prices) < MIN_HISTORY_DAYS:
                excluded.append((instr, f"short history ({len(prices)}d)"))
                continue
            eligible.append(instr)
        print(f"  {len(eligible)} tradable, {len(excluded)} unavailable/short")
    else:
        label = f"({', '.join(asset_classes)})" if asset_classes else "(all)"
        print(f"Filtering instruments {label}...")
        eligible, excluded = filter_instruments(asset_classes=asset_classes)
        print(f"  {len(eligible)} eligible, {len(excluded)} excluded")

    if not eligible:
        print("No eligible instruments.")
        return {}

    meta = pst.instruments_df().set_index("Instrument")

    def class_of(instr: str) -> str:
        if class_map and instr in class_map:
            return class_map[instr]
        return str(meta.loc[instr, "AssetClass"]) if instr in meta.index else "Unknown"

    print("Computing correlation matrix...")
    corr_matrix = compute_corr_matrix(eligible, pst)
    weights     = handcraft_weights(eligible, corr_matrix)
    idm         = min(idm_from_corr(weights, corr_matrix), IDM_CAP)
    classes     = sorted(set(class_of(e) for e in eligible))
    print(f"  IDM = {idm:.3f} | {len(eligible)} instruments across {len(classes)} asset classes")

    # Run each instrument STANDALONE (weight=1, IDM=1) for per-class table stats.
    # Each standalone return stream is targeted to TARGET_RISK on its own.
    all_results: dict[str, dict] = {}

    print("Running instrument backtests (standalone)...")
    for i, instr in enumerate(eligible):
        result = run_single(
            instrument=instr,
            capital=CAPITAL,
            target_risk=TARGET_RISK,
            idm=1.0,
            weight=1.0,
            verbose=False,
        )
        if not result:
            continue
        all_results[instr] = result

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(eligible)} processed...")

    print(f"  {len(all_results)} instruments ran successfully.")

    # ── Combine into the portfolio with DYNAMIC reweighting to the live universe ──
    # Each standalone stream is at TARGET_RISK with weight=1, IDM=1. Statically
    # scaling by the full-universe weight/IDM under-risks every period in which
    # not all instruments exist yet (early decades hold only a handful of the 99
    # Jumbo markets), collapsing realized vol far below target and diluting return.
    # Instead, on each day we renormalise the handcrafted weights over the
    # instruments that are actually live and recompute the IDM = 1/sqrt(w'Cw) on
    # that live sub-universe, so every period targets TARGET_RISK.  [calcs line 69]
    ret_df    = pd.DataFrame(
        {instr: r["daily_returns"] for instr, r in all_results.items()}
    ).sort_index()
    instr_list = list(ret_df.columns)
    W = np.array([weights.get(i, 0.0) for i in instr_list])
    C = corr_matrix.loc[instr_list, instr_list].values.copy()
    np.fill_diagonal(C, 1.0)

    live  = ret_df.notna().values          # (T, n): instrument has data that day
    R     = ret_df.fillna(0.0).values      # (T, n): standalone returns
    port  = np.zeros(len(ret_df))
    cache: dict[tuple, tuple | None] = {}
    for t in range(len(ret_df)):
        key = tuple(np.flatnonzero(live[t]))
        if not key:
            continue
        c = cache.get(key, 0)
        if c == 0:
            idxs = np.asarray(key)
            w_a  = W[idxs]
            s    = float(w_a.sum())
            if s <= 0:
                cache[key] = None
                continue
            w_n  = w_a / s
            Csub = C[np.ix_(idxs, idxs)]
            var  = float(w_n @ Csub @ w_n)
            idm_t = 1.0 / np.sqrt(var) if var > 0 else 1.0
            idm_t = min(idm_t, IDM_CAP)   # cap for live realism  [calcs line 69]
            c = cache[key] = (idxs, w_n, idm_t)
        if c is None:
            continue
        idxs, w_n, idm_t = c
        port[t] = idm_t * float(w_n @ R[t, idxs])

    portfolio_returns = pd.Series(port, index=ret_df.index)
    # Compounded equity curve — drawdowns bounded at -100%.  [calcs line 10]
    portfolio_equity  = CAPITAL * (1.0 + portfolio_returns).cumprod()

    # Approximate portfolio annual costs (scale standalone costs by w*IDM then annualise)
    portfolio_years = len(portfolio_equity) / ANNUAL_DAYS
    total_cost    = sum(
        r["costs_total"] * weights.get(instr, 0.0) * idm
        for instr, r in all_results.items()
    )
    n_instruments = len(all_results)
    # Turnover: average round-trips per instrument per year (from standalone)
    avg_turnover  = (
        np.mean([r["stats"]["turnover"] for r in all_results.values()])
        if all_results else 0.0
    )
    costs_pct = (total_cost / CAPITAL / portfolio_years) * 100   # annual %

    portfolio_stats = performance_stats(
        portfolio_equity,
        portfolio_returns,
        costs_pct=costs_pct,
        turnover=avg_turnover,
    )

    # ── Per-asset-class table ──────────────────────────────────────────────────
    class_results: dict[str, list[dict]] = {}
    for instr, res in all_results.items():
        ac = class_of(instr)
        class_results.setdefault(ac, []).append(res["stats"])

    def _safe_median(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return round(float(np.median(vals)), 2) if vals else np.nan

    table_rows = []
    for ac in sorted(class_results.keys()):
        rows = class_results[ac]
        table_rows.append({
            "Asset Class":     ac,
            "N":               len(rows),
            "Mean Ret %":      _safe_median([r["mean_annual_return_pct"] for r in rows]),
            "Costs %":         _safe_median([r["costs_pct"] for r in rows]),
            "Avg DD %":        _safe_median([r["avg_drawdown_pct"] for r in rows]),
            "Max DD %":        _safe_median([r["max_drawdown_pct"] for r in rows]),
            "Std Dev %":       _safe_median([r["std_dev_pct"] for r in rows]),
            "SR":              _safe_median([r["sharpe_ratio"] for r in rows]),
            "Turnover":        _safe_median([r["turnover"] for r in rows]),
            "Skew":            _safe_median([r["skew"] for r in rows]),
            "Lower Tail":      _safe_median([r["lower_tail"] for r in rows]),
            "Upper Tail":      _safe_median([r["upper_tail"] for r in rows]),
        })

    # Aggregate row
    table_rows.append({
        "Asset Class":     "** PORTFOLIO **",
        "N":               n_instruments,
        "Mean Ret %":      portfolio_stats["mean_annual_return_pct"],
        "Costs %":         portfolio_stats["costs_pct"],
        "Avg DD %":        portfolio_stats["avg_drawdown_pct"],
        "Max DD %":        portfolio_stats["max_drawdown_pct"],
        "Std Dev %":       portfolio_stats["std_dev_pct"],
        "SR":              portfolio_stats["sharpe_ratio"],
        "Turnover":        portfolio_stats["turnover"],
        "Skew":            portfolio_stats["skew"],
        "Lower Tail":      portfolio_stats["lower_tail"],
        "Upper Tail":      portfolio_stats["upper_tail"],
    })

    if verbose:
        print("\n" + "=" * 110)
        print("EWMAC PORTFOLIO RESULTS — CARVER AFTS STRATEGY 9")
        print("=" * 110)
        print(tabulate(table_rows, headers="keys", floatfmt=".2f", tablefmt="simple"))
        print(f"\nExcluded instruments by reason:")
        reasons: dict[str, int] = {}
        for _, reason in excluded:
            key = reason.split("(")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<40} {count}")

    # Save equity curve plot
    os.makedirs("ibkr_fut/results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(portfolio_equity.index, portfolio_equity.values / 1e6, linewidth=0.8, color="navy")
    ax.set_yscale("log")
    ax.set_title(
        f"EWMAC Portfolio — {n_instruments} instruments | "
        f"SR={portfolio_stats['sharpe_ratio']:.2f} | "
        f"MaxDD={portfolio_stats['max_drawdown_pct']:.1f}%"
    )
    ax.set_ylabel("Portfolio Value ($M, log scale)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "ibkr_fut/results/ewmac_portfolio.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\nEquity curve saved to {out}")

    return {
        "table":              table_rows,
        "portfolio_equity":   portfolio_equity,
        "portfolio_stats":    portfolio_stats,
        "instrument_results": all_results,
        "eligible":           eligible,
        "excluded":           excluded,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _rolls_per_year(info: dict) -> int:
    """Estimate rolls per year from the hold roll cycle string."""
    return len(str(info.get("HoldRollCycle", "HMUZ")))


# ── Entry points ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EWMAC Backtest")
    parser.add_argument("--single", default="SP500_micro",
                        help="Instrument for single-instrument backtest")
    parser.add_argument("--portfolio", action="store_true",
                        help="Run full portfolio backtest")
    parser.add_argument("--asset-class", nargs="+", default=None,
                        help="Restrict portfolio to these asset classes (e.g. Equity Bond)")
    parser.add_argument("--jumbo", action="store_true",
                        help="Run Carver's Jumbo portfolio (ibkr_fut/jumbo.py)")
    args = parser.parse_args()

    if args.jumbo:
        from ibkr_fut.jumbo import JUMBO
        run_portfolio(instruments=JUMBO, label_override="(Carver Jumbo)")
    elif args.portfolio:
        run_portfolio(asset_classes=args.asset_class)
    else:
        result = run_single(args.single)
        if result:
            eq  = result["equity_curve"]
            sr  = result["stats"]["sharpe_ratio"]
            mdd = result["stats"]["max_drawdown_pct"]
            os.makedirs("ibkr_fut/results", exist_ok=True)
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(eq.index, eq.values / 1e6, linewidth=0.8, color="navy")
            ax.set_title(f"{args.single} EWMAC  SR={sr:.2f} | MaxDD={mdd:.1f}%")
            ax.set_ylabel("Equity ($M)")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out = f"ibkr_fut/results/ewmac_{args.single}.png"
            plt.savefig(out, dpi=120)
            plt.close()
            print(f"\nEquity curve saved to {out}")
