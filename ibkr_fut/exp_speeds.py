#!/usr/bin/env python3
"""
exp_speeds.py — per-EWMAC-speed analysis on the Jumbo futures portfolio.

(1) Each EWMAC speed run ALONE across the whole Jumbo portfolio (forced onto every
    instrument, FDM=1.0 for a single rule) — shows the standalone quality + cost
    drag of each rule.
(2) Full Jumbo with the normal cost-eligible speed set, vs. the same with EWMAC2
    removed everywhere — Carver's "EWMAC2 costs too much, I'd start at EWMAC4."

Works by monkeypatching backtest_ewmac.eligible_speeds (a module global that
instrument_signals looks up at call time), then calling run_portfolio unchanged.
"""
import sys, os, io, contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ibkr_fut.backtest_ewmac as bt
from ibkr_fut.ewmac_signals import eligible_speeds as eligible_orig, EWMAC_SPEEDS
from ibkr_fut.jumbo import JUMBO

ANNUAL = 256
ALL_SPEEDS = [f for f, _ in EWMAC_SPEEDS]   # [2,4,8,16,32,64]


def _run(label):
    """Call run_portfolio on the Jumbo, swallow its prints, return key stats."""
    with contextlib.redirect_stdout(io.StringIO()):
        r = bt.run_portfolio(verbose=False, instruments=JUMBO, label_override=label)
    eq  = r["portfolio_equity"]
    ret = eq.pct_change().dropna()
    s   = r["portfolio_stats"]
    full = ret.mean() / ret.std() * np.sqrt(ANNUAL)
    sub  = ret[ret.index >= "2010-01-01"]
    sr10 = sub.mean() / sub.std() * np.sqrt(ANNUAL)
    return dict(label=label, sr_full=full, sr_2010=sr10,
                ret=s["mean_annual_return_pct"], vol=s["std_dev_pct"],
                maxdd=s["max_drawdown_pct"], costs=s["costs_pct"],
                turnover=s["turnover"], skew=s["skew"], n=s.get("_n"))


def patched(fn):
    """Context: temporarily replace eligible_speeds with fn(cost, rolls)."""
    class _C:
        def __enter__(self): bt.eligible_speeds = fn
        def __exit__(self, *a): bt.eligible_speeds = eligible_orig
    return _C()


def main():
    rows = []

    # ── (1) each speed alone, forced on every instrument ───────────────────────
    print("Running each EWMAC speed ALONE across the Jumbo (forced on all)...")
    for sp in ALL_SPEEDS:
        with patched(lambda c, r, _sp=sp: [_sp]):
            rows.append(_run(f"EWMAC{sp} only"))
        print(f"  EWMAC{sp} done")

    # ── (2) baseline (cost-eligible) vs. no-EWMAC2 ─────────────────────────────
    print("Running baseline (cost-eligible speeds)...")
    with patched(eligible_orig):
        base = _run("BASELINE (cost-eligible)")
    rows.append(base)

    print("Running baseline minus EWMAC2...")
    def no2(c, r):
        e = [s for s in eligible_orig(c, r) if s != 2]
        return e if e else eligible_orig(c, r)   # never empty
    with patched(no2):
        rows.append(_run("NO EWMAC2"))

    # how many instruments actually had EWMAC2 in their eligible set?
    n_with2 = 0
    n_total = 0
    for instr in JUMBO:
        spec = bt._pst_spec(instr)
        if spec is None:
            continue
        sig = bt.instrument_signals(spec)
        if sig is None:
            continue
        n_total += 1
        if 2 in sig["active"]:
            n_with2 += 1

    # ── report ─────────────────────────────────────────────────────────────────
    hdr = f"{'Configuration':<26}{'SR(full)':>9}{'SR(2010+)':>10}{'Ret%':>8}{'Vol%':>7}{'MaxDD%':>8}{'Costs%':>8}{'Turn':>7}{'Skew':>7}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for r in rows:
        if r["label"].startswith("BASELINE"):
            print("-" * len(hdr))
        print(f"{r['label']:<26}{r['sr_full']:>9.3f}{r['sr_2010']:>10.3f}"
              f"{r['ret']:>8.2f}{r['vol']:>7.2f}{r['maxdd']:>8.1f}"
              f"{r['costs']:>8.3f}{r['turnover']:>7.1f}{r['skew']:>7.2f}")
    print("=" * len(hdr))
    print(f"\nInstruments where EWMAC2 is cost-eligible: {n_with2}/{n_total}")


if __name__ == "__main__":
    main()
