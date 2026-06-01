#!/usr/bin/env python3
"""
ETF universe selection via correlation clustering.

Process:
  1. Load daily returns for all passing ETFs
  2. Compute pairwise correlation (2016-01-01 onward for consistency with 1h data)
  3. Hierarchical clustering on distance = 1 - corr
  4. Cut tree at CORR_THRESHOLD — any two instruments above this are "duplicates"
  5. Within each cluster, keep the best instrument (scored by history × liquidity)
  6. Override: ensure all major risk factors have at least one representative
  7. Output curated universe to Data/etf/etf_universe_curated.json + plots

Usage:
  python3 etf/etf_universe_select.py
  python3 etf/etf_universe_select.py --threshold 0.90  # stricter deduplication
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

# ── Config ────────────────────────────────────────────────────────────────────
CORR_THRESHOLD  = 0.95   # instruments correlated above this are treated as duplicates
CORR_START      = "2016-01-01"   # align with Alpaca 1h start
MIN_OVERLAP     = 1000   # min overlapping daily bars needed to trust a correlation
DATA_DIR        = "Data/etf"
UNIVERSE_FILE   = "Data/etf/etf_universe.json"
OUTPUT_FILE     = "Data/etf/etf_universe_curated.json"
OUTPUT_DIR      = "results"

# ── Force-include overrides ───────────────────────────────────────────────────
# These tickers are always included in the final universe regardless of whether
# the clustering algorithm would have dropped them as duplicates of another instrument.
# Use this for instruments where the correlation is high but the economic distinction
# is meaningful (e.g. QQQ vs SPY: both broad US equity but QQQ has a strong tech tilt
# that produces meaningfully different behaviour in tech-specific regimes).
FORCE_INCLUDE = [
    "QQQ",   # Nasdaq 100 — tech/growth tilt, diverges from SPY in rate regimes
]

# ── Asset class tags ──────────────────────────────────────────────────────────
# Used to ensure diversification across risk factors in the final universe.
ASSET_CLASS = {
    # US Broad Equity
    "SPY":"US_EQUITY","QQQ":"US_EQUITY","IVV":"US_EQUITY","VTI":"US_EQUITY",
    "MDY":"US_EQUITY","IWM":"US_EQUITY",
    # US Factors (style + quality + dividend + low-vol)
    "VTV":"US_FACTOR","VUG":"US_FACTOR","IUSG":"US_FACTOR","IUSV":"US_FACTOR",
    "MTUM":"US_FACTOR","QUAL":"US_FACTOR","VLUE":"US_FACTOR",
    "USMV":"US_FACTOR","DGRO":"US_FACTOR","VIG":"US_FACTOR",
    "HDV":"US_FACTOR","NOBL":"US_FACTOR","COWZ":"US_FACTOR","CALF":"US_FACTOR",
    # US Sectors (sector ETFs + energy infrastructure equity)
    "XLK":"US_SECTOR","XLF":"US_SECTOR","XLV":"US_SECTOR","XLE":"US_SECTOR",
    "XLI":"US_SECTOR","XLU":"US_SECTOR","XLP":"US_SECTOR","XLY":"US_SECTOR",
    "XLB":"US_SECTOR","XLRE":"US_SECTOR","XLC":"US_SECTOR","XBI":"US_SECTOR",
    "SMH":"US_SECTOR","XHB":"US_SECTOR","XRT":"US_SECTOR","KRE":"US_SECTOR",
    "KBE":"US_SECTOR","IBB":"US_SECTOR","IYT":"US_SECTOR","ITA":"US_SECTOR",
    "OIH":"US_SECTOR","XOP":"US_SECTOR","AMLP":"US_SECTOR",
    # International Broad
    "EFA":"INTL_EQUITY","VEA":"INTL_EQUITY","IEFA":"INTL_EQUITY",
    "EEM":"EM_EQUITY","VWO":"EM_EQUITY","IEMG":"EM_EQUITY",
    "ACWI":"INTL_EQUITY","VXUS":"INTL_EQUITY","VSS":"INTL_EQUITY",
    "DWX":"INTL_EQUITY",
    # International Country
    "EWJ":"COUNTRY","EWG":"COUNTRY","EWU":"COUNTRY","EWY":"COUNTRY",
    "EWZ":"COUNTRY","FXI":"COUNTRY","MCHI":"COUNTRY","INDA":"COUNTRY",
    "EWT":"COUNTRY","EWA":"COUNTRY","EWC":"COUNTRY","EWH":"COUNTRY",
    "EWQ":"COUNTRY","EWI":"COUNTRY","EWP":"COUNTRY","EWL":"COUNTRY",
    "EWD":"COUNTRY","EWN":"COUNTRY","EWS":"COUNTRY","EIDO":"COUNTRY",
    "EWW":"COUNTRY","ECH":"COUNTRY","ARGT":"COUNTRY","EZA":"COUNTRY",
    # Treasuries
    "TLT":"TREASURY","IEF":"TREASURY","IEI":"TREASURY","SHY":"TREASURY",
    "BIL":"TREASURY","SGOV":"TREASURY","VGLT":"TREASURY","VGIT":"TREASURY",
    "VGSH":"TREASURY","ZROZ":"TREASURY","EDV":"TREASURY",
    # TIPS
    "TIP":"TIPS","STIP":"TIPS","VTIP":"TIPS","LTPZ":"TIPS","RINF":"TIPS",
    # Corporate Bonds
    "LQD":"CORP_BOND","HYG":"HY_BOND","JNK":"HY_BOND",
    "VCSH":"CORP_BOND","VCIT":"CORP_BOND","VCLT":"CORP_BOND",
    "IGIB":"CORP_BOND","IGSB":"CORP_BOND","IGLB":"CORP_BOND",
    "USHY":"HY_BOND","FLOT":"CORP_BOND","FALN":"HY_BOND",
    # Aggregate / Other Bonds
    "AGG":"AGG_BOND","BND":"AGG_BOND","BNDX":"INTL_BOND","EMB":"EM_BOND",
    "VWOB":"EM_BOND","MUB":"MUNI","HYD":"MUNI","VTEB":"MUNI",
    "BWX":"INTL_BOND","IAGG":"INTL_BOND",
    # Metals
    "GLD":"GOLD","IAU":"GOLD","GLDM":"GOLD","SLV":"SILVER",
    "PPLT":"METALS","PALL":"METALS","DBP":"METALS","DBB":"METALS",
    "GDX":"GOLD_MINERS","GDXJ":"GOLD_MINERS","COPX":"GOLD_MINERS","SILJ":"GOLD_MINERS",
    # Energy (commodity futures only; AMLP is equity → US_SECTOR)
    "USO":"ENERGY","UNG":"NAT_GAS","DBE":"ENERGY",
    # Agriculture
    "CORN":"AGRI","WEAT":"AGRI","SOYB":"AGRI","DBA":"AGRI",
    "CANE":"AGRI","JO":"AGRI","NIB":"AGRI","BAL":"AGRI","RJA":"AGRI",
    # Broad Commodity
    "DBC":"COMMODITY","PDBC":"COMMODITY","GSG":"COMMODITY","COMT":"COMMODITY",
    # Real Estate
    "VNQ":"REIT","IYR":"REIT","REET":"REIT","VNQI":"INTL_REIT",
    "REM":"REIT","O":"REIT","PLD":"REIT","AMT":"REIT",
    # FX
    "FXE":"FX","FXY":"FX","FXB":"FX","FXA":"FX","FXC":"FX","FXF":"FX",
    "UUP":"FX","UDN":"FX","CEW":"FX","DBV":"FX",
    # Volatility
    "VXX":"VOLATILITY","UVXY":"VOLATILITY","SVXY":"VOLATILITY",
    # Crypto/Thematic
    "BITO":"CRYPTO","IBIT":"CRYPTO","FBTC":"CRYPTO","ETHA":"CRYPTO","FETH":"CRYPTO",
    "ARKK":"THEMATIC","BOTZ":"THEMATIC","HACK":"THEMATIC",
    "LIT":"THEMATIC","ICLN":"THEMATIC","TAN":"THEMATIC",
}

# Risk factor labels for final output grouping
FACTOR_LABEL = {
    "US_EQUITY":"US Equity","US_FACTOR":"US Factor","US_SECTOR":"US Sector",
    "INTL_EQUITY":"Intl Equity","EM_EQUITY":"EM Equity","COUNTRY":"Country",
    "TREASURY":"Treasury","TIPS":"TIPS","CORP_BOND":"Corp Bond",
    "HY_BOND":"High Yield","AGG_BOND":"Agg Bond","INTL_BOND":"Intl Bond",
    "EM_BOND":"EM Bond","MUNI":"Muni",
    "GOLD":"Gold","SILVER":"Silver","METALS":"Metals",
    "GOLD_MINERS":"Mining Equity","ENERGY":"Energy","NAT_GAS":"Nat Gas",
    "AGRI":"Agriculture","COMMODITY":"Commodity",
    "REIT":"REIT","INTL_REIT":"Intl REIT",
    "FX":"FX","VOLATILITY":"Volatility",
    "CRYPTO":"Crypto","THEMATIC":"Thematic",
}


def load_returns(tickers: list[str], start: str) -> pd.DataFrame:
    """Load daily close prices, compute returns, align to common date range."""
    series = {}
    for tk in tickers:
        path = os.path.join(DATA_DIR, f"{tk.lower()}_1d_yf.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["time"]).set_index("time")["close"]
        df = df[df.index >= start]
        if len(df) > MIN_OVERLAP:
            series[tk] = df.pct_change()
    prices = pd.DataFrame(series).dropna(how="all")
    return prices


def score_instrument(tk: str, returns: pd.DataFrame, adv: dict) -> float:
    """Score instrument for cluster representative selection.
    Higher = better. Balances history length and liquidity."""
    n_years = len(returns[tk].dropna()) / 252
    adv_m   = adv.get(tk, 1.0) / 1e6
    return n_years * 0.5 + np.log10(max(adv_m, 0.1)) * 0.5


def compute_adv(tickers: list[str], start: str) -> dict:
    """Compute average daily dollar volume for each ticker."""
    adv = {}
    for tk in tickers:
        path = os.path.join(DATA_DIR, f"{tk.lower()}_1d_yf.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
        df = df[df.index >= start]
        if "volume" in df.columns and "close" in df.columns:
            adv[tk] = (df["volume"] * df["close"]).mean()
    return adv


def select_clusters(returns: pd.DataFrame, threshold: float,
                    adv: dict) -> tuple[dict, pd.DataFrame]:
    """
    Cluster instruments and pick the best representative from each cluster.
    Returns (cluster_map, corr_matrix) where cluster_map is
      {representative: [all members]}.
    """
    tickers = list(returns.columns)

    # Pairwise correlation on overlapping bars (pairwise=True handles different start dates)
    corr = returns.corr(method="pearson", min_periods=MIN_OVERLAP)

    # Distance matrix: 1 - corr.  Clip negatives (don't want negative distances).
    dist_sq = np.clip(1 - corr.values, 0, 2)
    np.fill_diagonal(dist_sq, 0)

    # Condensed distance matrix required by scipy linkage
    dist_cond = squareform(dist_sq, checks=False)
    Z = linkage(dist_cond, method="average")

    # Cut tree: instruments in the same cluster have corr >= threshold
    dist_cut = 1 - threshold
    labels = fcluster(Z, t=dist_cut, criterion="distance")

    # Map cluster id → list of tickers
    clusters: dict[int, list[str]] = {}
    for tk, lbl in zip(tickers, labels):
        clusters.setdefault(lbl, []).append(tk)

    # From each cluster, pick the best representative
    cluster_map = {}
    for lbl, members in clusters.items():
        if len(members) == 1:
            rep = members[0]
        else:
            scores = {tk: score_instrument(tk, returns, adv) for tk in members}
            rep = max(scores, key=scores.get)
        cluster_map[rep] = sorted(members)

    return cluster_map, corr, Z, tickers, labels


def plot_results(corr: pd.DataFrame, Z, tickers: list, labels: np.ndarray,
                 selected: list[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ── Correlation heatmap sorted by cluster ─────────────────────────────────
    order = np.argsort(labels)
    sorted_tickers = [tickers[i] for i in order]
    corr_sorted = corr.loc[sorted_tickers, sorted_tickers]

    fig, ax = plt.subplots(figsize=(24, 22))
    im = ax.imshow(corr_sorted.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sorted_tickers)))
    ax.set_yticks(range(len(sorted_tickers)))
    ax.set_xticklabels(sorted_tickers, rotation=90, fontsize=5.5)
    ax.set_yticklabels(sorted_tickers, fontsize=5.5)
    # Mark selected representatives with a box
    sel_set = set(selected)
    for i, tk in enumerate(sorted_tickers):
        if tk in sel_set:
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor="navy", lw=1.5))
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    ax.set_title(f"ETF correlation matrix (sorted by cluster)  —  "
                 f"{len(selected)} selected (navy boxes)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "etf_correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"  Saved results/etf_correlation_heatmap.png")

    # ── Dendrogram ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(18, len(tickers) * 0.18), 8))
    dend = dendrogram(Z, labels=tickers, ax=ax, leaf_rotation=90,
                      leaf_font_size=6, color_threshold=1 - CORR_THRESHOLD)
    ax.axhline(1 - CORR_THRESHOLD, color="crimson", linestyle="--", alpha=0.7,
               label=f"cut @ corr={CORR_THRESHOLD}")
    ax.set_title("ETF hierarchical clustering dendrogram")
    ax.set_ylabel("Distance (1 − correlation)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "etf_dendrogram.png"), dpi=150)
    plt.close()
    print(f"  Saved results/etf_dendrogram.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=CORR_THRESHOLD,
                    help=f"Correlation threshold for deduplication (default {CORR_THRESHOLD})")
    args = ap.parse_args()

    with open(UNIVERSE_FILE) as f:
        universe = json.load(f)
    tickers = universe["passed"]
    print(f"Loaded {len(tickers)} passing ETFs from {UNIVERSE_FILE}")

    print(f"Computing ADV and loading returns from {CORR_START}…")
    adv     = compute_adv(tickers, CORR_START)
    returns = load_returns(tickers, CORR_START)

    # Drop tickers with too little overlap
    tickers_ok = [tk for tk in tickers if tk in returns.columns]
    print(f"  {len(tickers_ok)} tickers have sufficient data post {CORR_START}")

    returns = returns[tickers_ok]

    print(f"\nClustering at correlation threshold = {args.threshold}…")
    cluster_map, corr, Z, tk_list, labels = select_clusters(
        returns, args.threshold, adv)

    selected = sorted(cluster_map.keys())

    # Apply force-include overrides: if a forced ticker was dropped as a duplicate,
    # add it back and note which cluster it came from.
    forced_added = []
    for tk in FORCE_INCLUDE:
        if tk not in selected and tk in returns.columns:
            selected.append(tk)
            forced_added.append(tk)
            # Find which cluster it belongs to so we can report it
            for rep, members in cluster_map.items():
                if tk in members:
                    cluster_map[tk] = [tk]   # give it its own "cluster" entry
                    print(f"  Force-include: {tk} (was merged into {rep})")
                    break
    selected = sorted(selected)

    print(f"  {len(cluster_map) - len(forced_added)} clusters → "
          f"{len(selected)} selected ({len(forced_added)} force-included)\n")

    # ── Print cluster report ──────────────────────────────────────────────────
    print("=" * 80)
    print(f"  CLUSTER REPORT  (threshold={args.threshold})")
    print("=" * 80)

    # Sort clusters by asset class of representative
    def sort_key(rep):
        return (ASSET_CLASS.get(rep, "ZZZ"), rep)

    for rep in sorted(cluster_map.keys(), key=sort_key):
        members = cluster_map[rep]
        ac = ASSET_CLASS.get(rep, "?")
        label = FACTOR_LABEL.get(ac, ac)
        n_years = len(returns[rep].dropna()) / 252
        adv_m = adv.get(rep, 0) / 1e6
        if len(members) == 1:
            print(f"  {rep:<8} [{label:<14}]  {n_years:.1f}y  ADV ${adv_m:.0f}M")
        else:
            dropped = [m for m in members if m != rep]
            print(f"  {rep:<8} [{label:<14}]  {n_years:.1f}y  ADV ${adv_m:.0f}M"
                  f"  ← deduped: {', '.join(dropped)}")

    # ── Summary by asset class ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SELECTED UNIVERSE BY ASSET CLASS")
    print("=" * 80)
    by_class: dict[str, list] = {}
    for rep in selected:
        ac = ASSET_CLASS.get(rep, "OTHER")
        by_class.setdefault(ac, []).append(rep)
    for ac in sorted(by_class):
        label = FACTOR_LABEL.get(ac, ac)
        tks = ", ".join(sorted(by_class[ac]))
        print(f"  {label:<16}  {tks}")

    print(f"\n  Total: {len(selected)} instruments")

    # ── Save curated universe ─────────────────────────────────────────────────
    out = {
        "selected": selected,
        "n_selected": len(selected),
        "threshold": args.threshold,
        "corr_start": CORR_START,
        "clusters": {rep: members for rep, members in cluster_map.items()},
        "asset_classes": {rep: ASSET_CLASS.get(rep, "OTHER") for rep in selected},
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Curated universe saved to {OUTPUT_FILE}")

    plot_results(corr, Z, tk_list, labels, selected, OUTPUT_DIR)


if __name__ == "__main__":
    main()
