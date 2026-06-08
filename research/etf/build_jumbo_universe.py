#!/usr/bin/env python3
"""
build_jumbo_universe.py — the "Jumbo ETF" universe: trade EVERYTHING available,
no performance-based selection.

Rationale
---------
Carver's greedy static optimisation (etf_greedy_select.py) exists to answer
"which subset can I afford given a minimum position size?" — a constraint that
DOES NOT apply to fractional ETF shares (the greedy config itself sets
size_penalty = 0). With fractional shares you can hold $20 of every ETF, so the
only thing greedy buys you is (a) a full-sample correlation matrix baked into the
selection (mild look-ahead) and (b) throwing away free diversification. Since the
gross-leverage cap binds ~98% of days, more uncorrelated instruments is the single
biggest lever on a cap-constrained book. So: trade the whole curated pool, let
handcraft clustering + IDM discount redundant names, and apply ONLY mechanical
(return-blind) screens.

Mechanical screens (nothing here looks at realised returns/SR):
  1. Vol-ETP blocklist  — daily-reset leveraged/inverse (UVXY 1.5x, SVXY -0.5x)
     and structural-decay roll products (VXX/VIXY). These violate the EWMAC
     assumption that price compounds like a normal asset. Clean vol exposure is
     the VIX *future* in the futures book, not an ETF wrapper. See AUDIT/memory.
  2. Leveraged-name guard — regex for 2x/3x/ultra/inverse, in case the pool grows.
  3. Cash-like screen   — annualised price vol < CASH_VOL_THRESHOLD; sized to a
     vol target these take absurd leverage on a near-riskless drift.
  4. Min history        — at least MIN_HISTORY_DAYS bars.

Pool: Data/etf/etf_universe_curated.json (101, human-vetted, has asset_classes).

Output: Data/etf/etf_universe_jumbo.json  (schema ewmac_carver.py consumes:
        "selected" + "asset_classes"; plus "hard_excluded"/"meta" for audit).

Usage:
    python3 etf/build_jumbo_universe.py
"""

import json
import os
import re

import numpy as np
import pandas as pd

DATA_DIR      = "Data/etf"
CURATED_FILE  = "Data/etf/etf_universe_curated.json"
OUTPUT_FILE   = "Data/etf/etf_universe_jumbo.json"

CASH_VOL_THRESHOLD = 0.03    # annualised price vol floor; below = cash-like
MIN_HISTORY_DAYS   = 512
ANNUAL_DAYS        = 256

# Explicit, documented blocklist — leveraged/inverse/structural-decay vol wrappers.
# Kept as a named list (not just the regex) so the decision is visible & reversible.
VOL_ETP_BLOCKLIST = {"UVXY", "SVXY", "VXX", "VIXY", "VIXM", "SVIX", "UVIX"}

# Catches 2x/3x/ultra/inverse share-class naming if the candidate pool ever grows.
LEVERAGED_NAME_RE = re.compile(r"(?:^|[^A-Z])(2X|3X|ULTRA|INVERSE|SHORT|BEAR|BULL)", re.I)


def load_close(tk: str) -> pd.Series:
    path = os.path.join(DATA_DIR, f"{tk.lower()}_1d_yf.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    s = pd.read_csv(path, parse_dates=["time"]).set_index("time")["close"]
    return s[~s.index.duplicated(keep="last")].sort_index()


def annualised_vol(s: pd.Series) -> float:
    r = s.pct_change().dropna()
    return float(r.std() * np.sqrt(ANNUAL_DAYS)) if len(r) else 0.0


def main():
    cur = json.load(open(CURATED_FILE))
    pool = cur["selected"]
    asset_classes = cur.get("asset_classes", {})

    selected, excluded = [], {}
    for tk in pool:
        # 1+2. vol / leveraged wrappers
        if tk.upper() in VOL_ETP_BLOCKLIST:
            excluded[tk] = "vol_etp"
            continue
        if LEVERAGED_NAME_RE.search(tk):
            excluded[tk] = "leveraged_name"
            continue
        s = load_close(tk)
        # 4. history
        if len(s) < MIN_HISTORY_DAYS:
            excluded[tk] = f"short_history_{len(s)}b"
            continue
        # 3. cash-like
        vol = annualised_vol(s)
        if vol < CASH_VOL_THRESHOLD:
            excluded[tk] = f"cash_like_{vol*100:.2f}pct"
            continue
        selected.append(tk)

    # asset classes for survivors only (handcraft needs every name grouped)
    ac_out = {tk: asset_classes.get(tk, "OTHER") for tk in selected}
    unmapped = [tk for tk in selected if tk not in asset_classes]

    from collections import Counter
    out = {
        "selected": selected,
        "n_selected": len(selected),
        "asset_classes": ac_out,
        "hard_excluded": excluded,
        "meta": {
            "source_pool": CURATED_FILE,
            "n_pool": len(pool),
            "method": "jumbo_no_selection",
            "screens": {
                "vol_etp_blocklist": sorted(VOL_ETP_BLOCKLIST),
                "cash_vol_threshold": CASH_VOL_THRESHOLD,
                "min_history_days": MIN_HISTORY_DAYS,
            },
            "note": ("Trade-everything universe; NO performance/correlation-based "
                     "selection. Greedy subset is unnecessary for fractional ETFs "
                     "(no min position size). Vol ETPs dropped: daily-reset "
                     "leveraged/inverse + roll decay break EWMAC's compounding "
                     "assumption; clean vol exposure is the VIX future in futures."),
        },
    }
    json.dump(out, open(OUTPUT_FILE, "w"), indent=2)

    print(f"Pool (curated):  {len(pool)}")
    print(f"Selected (jumbo): {len(selected)}")
    print(f"Excluded:        {len(excluded)}")
    for tk, why in sorted(excluded.items(), key=lambda kv: kv[1]):
        print(f"    {tk:<6} {why}")
    if unmapped:
        print(f"\n  WARNING — no asset class (→ OTHER): {unmapped}")
    print(f"\n  Asset classes: {dict(Counter(ac_out.values()))}")
    print(f"\n  Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
