#!/usr/bin/env python3
"""
Rebuild paper/ledgers/ibkr_dynamic/daily.csv from IBKR Flex (BUG-26 backfill).

WHY
---
BUG-26 tied the daily-ledger flush to daemon lifetime, so 24 of 39 business days
between 2026-06-11 and 2026-08-04 were never written. Worse, `log_daily` derives
`daily_pnl = equity - state.last_equity`, so each *surviving* row absorbed the P&L
of every skipped day: its `ret` is a multi-day return mislabelled as daily, and
every statistic computed from the column (vol, Sharpe, drawdown) is wrong.

WHY A FULL REBUILD, NOT AN INSERT
---------------------------------
Flex covers all 39 days, but it disagrees with the 15 surviving rows (mean -$294,
worst $3,456). That is NOT an error — account/currency/column were verified — it is
a measurement-basis difference:

    Flex reportDate   = IBKR's settled end-of-day NAV
    ledger 22:1x UTC  = NetLiquidation just after the futures reopen, while
                        Asian/European positions are still moving

Keeping the 15 and inserting 24 would splice two bases into one column with a ~$300
mean step at every boundary — the exact corruption this backfill exists to remove.
Since Flex spans the whole window, take it wholesale: one authoritative basis.

Consequence (intended): headline cumulative restates -1.88% -> -1.52%, which matches
the same-period backtest. Existing equity values move by up to $3,456.

USAGE
-----
    python3 scripts/backfill_daily_ledger.py              # dry-run, prints full diff
    python3 scripts/backfill_daily_ledger.py --apply      # writes (after backup!)

Idempotent: rebuilds from source every run, so re-running is safe and the dry-run
diff always reflects what --apply would do.
"""

import argparse
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from paper.dyn_ledger import (  # noqa: E402
    DAILY_COLS, DAILY_PATH, STATE_PATH, _fills_on, _load_state,
)

ENV_PATH   = os.path.join(REPO, ".env")
FLEX_BASE  = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"
UA         = {"User-Agent": "Mozilla/5.0"}

# Window being rebuilt. Start = inception of the current $250k book (the 06-11 reseed).
START_DATE = "2026-06-11"
END_DATE   = "2026-08-04"

EXPECT_ACCOUNT  = "DUA295747"
EXPECT_CURRENCY = "USD"

# Guard rails from the read-only probe (2026-08-05). If the restatement distribution
# drifts far from this, the query changed underneath us — abort rather than write.
PROBE_MEAN_DIFF  = -294.49
PROBE_WORST_DIFF = 3456.14
DIFF_TOLERANCE   = 3.0        # allow 3x the probe's worst before refusing


# ── Flex fetch ────────────────────────────────────────────────────────────────

def _read_env(path=ENV_PATH):
    env = {}
    if not os.path.exists(path):
        return env
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def fetch_flex_xml(token, query_id, *, max_wait=300, verbose=True):
    """
    Two-step Flex Web Service fetch with polling.

    ib_insync.FlexReport is NOT used here: it raises on error 1019 ("statement
    generation in progress"), which is the normal first response for a query of
    this size. The two-step SendRequest -> poll GetStatement flow handles it.
    """
    resp = _get(f"{FLEX_BASE}/SendRequest?t={token}&q={query_id}&v=3")
    root = ET.fromstring(resp)
    status = (root.findtext("Status") or "").strip()
    if status != "Success":
        raise RuntimeError(
            f"Flex SendRequest failed: status={status} "
            f"code={root.findtext('ErrorCode')} msg={root.findtext('ErrorMessage')}")

    ref = root.findtext("ReferenceCode").strip()
    url = root.findtext("Url").strip()
    if verbose:
        print(f"  Flex referenceCode={ref}")

    deadline = time.time() + max_wait
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        body = _get(f"{url}?q={ref}&t={token}&v=3")
        if "<ErrorCode>1019</ErrorCode>" in body or "generation in progress" in body:
            if verbose:
                print(f"  attempt {attempt}: still generating, waiting 15s…")
            time.sleep(15)
            continue
        if "<ErrorCode>" in body:
            er = ET.fromstring(body)
            raise RuntimeError(
                f"Flex GetStatement error {er.findtext('ErrorCode')}: "
                f"{er.findtext('ErrorMessage')}")
        if verbose:
            print(f"  Flex statement retrieved ({len(body):,} bytes)")
        return body
    raise TimeoutError(f"Flex statement not ready after {max_wait}s")


# ── Parse + validate ──────────────────────────────────────────────────────────

def _iso(d):
    return f"{d[:4]}-{d[4:6]}-{d[6:]}" if d and len(d) == 8 and d.isdigit() else d


def parse_equity_series(xml_text):
    """{iso_date: total_nav} from EquitySummaryByReportDateInBase, with scope checks."""
    root = ET.fromstring(xml_text)
    rows = root.findall(".//EquitySummaryByReportDateInBase")
    if not rows:
        raise RuntimeError("no EquitySummaryByReportDateInBase rows — check the "
                           "Flex query includes 'Net Asset Value (NAV) in Base'")
    series = {}
    for r in rows:
        acct, ccy = r.get("accountId"), r.get("currency")
        if acct != EXPECT_ACCOUNT:
            raise RuntimeError(f"unexpected accountId {acct!r} (want {EXPECT_ACCOUNT!r})")
        if ccy != EXPECT_CURRENCY:
            raise RuntimeError(f"unexpected currency {ccy!r} (want {EXPECT_CURRENCY!r})")
        series[_iso(r.get("reportDate"))] = float(r.get("total"))
    return series


def parse_change_in_nav(xml_text):
    """{iso_to_date: dict} from ChangeInNAV — the independent cross-check section."""
    root = ET.fromstring(xml_text)
    out = {}
    for c in root.findall(".//ChangeInNAV"):
        out[_iso(c.get("toDate"))] = {
            k: float(c.get(k) or 0.0)
            for k in ("startingValue", "mtm", "realized", "changeInUnrealized",
                      "commissions", "endingValue")
        }
    return out


def cross_check_change_in_nav(equity, cin, tol=1.0):
    """
    ChangeInNAV.endingValue must agree with EquitySummary.total on the same date.
    Two independently-generated sections disagreeing means a parse error — the whole
    point of having selected both.
    """
    problems, checked = [], 0
    for d, v in sorted(cin.items()):
        if d in equity:
            checked += 1
            diff = v["endingValue"] - equity[d]
            if abs(diff) > tol:
                problems.append((d, v["endingValue"], equity[d], diff))
    return checked, problems


def trading_days(start, end):
    """Business days in [start, end], holidays excluded via the repo calendar."""
    days = [d.date().isoformat() for d in pd.bdate_range(start, end)]
    try:
        from ibkr_fut.trading_calendar import is_trading_day  # type: ignore
        return [d for d in days if is_trading_day(pd.Timestamp(d))]
    except Exception:
        # Calendar helper unavailable/incompatible — fall back to plain weekdays and
        # let the coverage check below surface any date Flex genuinely lacks.
        return days


# ── Rebuild ───────────────────────────────────────────────────────────────────

def rebuild(equity, dates):
    """Chain ret/nav off the equity path; costs from trades.csv per date."""
    rows, prev_eq, nav = [], None, 1.0
    for d in dates:
        eq = equity[d]
        if prev_eq is None:
            pnl, ret = 0.0, 0.0
        else:
            pnl = eq - prev_eq
            ret = pnl / prev_eq if prev_eq else 0.0
            nav = nav * (1 + ret)
        n_tr, comm, slip = _fills_on(d)
        rows.append({
            "date":           d,
            "equity":         round(eq, 2),
            "daily_pnl_usd":  round(pnl, 2),
            "ret":            round(ret, 8),
            "nav":            round(nav, 6),
            "n_trades":       n_tr,
            "commission":     round(comm, 4),
            "slippage_usd":   round(slip, 2),
            "n_positions":    "",
            "gross_leverage": "",
            "source":         "flex",
        })
        prev_eq = eq
    return pd.DataFrame(rows)


def carry_forward_diagnostics(new_df, old_df):
    """Preserve n_positions / gross_leverage from the old rows where we have them."""
    if old_df.empty:
        return new_df
    old = old_df.set_index("date")
    for col in ("n_positions", "gross_leverage"):
        if col in old.columns:
            new_df[col] = new_df["date"].map(old[col]).fillna("")
    return new_df


def write_atomic(df, path, cols):
    tmp = f"{path}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False, columns=cols)
    os.replace(tmp, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="write the rebuilt ledger (default: dry-run diff only)")
    ap.add_argument("--xml", help="use a saved Flex XML file instead of fetching")
    ap.add_argument("--save-xml", help="save the fetched XML here")
    args = ap.parse_args()

    print("=" * 78)
    print("  BUG-26 BACKFILL — rebuild daily.csv from IBKR Flex")
    print(f"  window {START_DATE} → {END_DATE}   mode: "
          f"{'APPLY' if args.apply else 'DRY-RUN'}")
    print("=" * 78)

    # 1. Fetch -------------------------------------------------------------
    if args.xml:
        print(f"\n[1] Reading saved XML: {args.xml}")
        xml_text = open(args.xml).read()
    else:
        env = _read_env()
        token, qid = env.get("FLEX_QUERY_TOKEN"), env.get("FLEX_QUERY_ID")
        if not token or not qid:
            sys.exit("ERROR: FLEX_QUERY_TOKEN / FLEX_QUERY_ID missing from .env")
        print(f"\n[1] Fetching Flex query {qid}…")
        xml_text = fetch_flex_xml(token, qid)
        if args.save_xml:
            open(args.save_xml, "w").write(xml_text)
            print(f"  saved raw XML → {args.save_xml}")

    # 2. Parse + scope validation ------------------------------------------
    print("\n[2] Parsing + validating scope…")
    equity = parse_equity_series(xml_text)
    cin    = parse_change_in_nav(xml_text)
    ks = sorted(equity)
    print(f"  account {EXPECT_ACCOUNT} / {EXPECT_CURRENCY} confirmed")
    print(f"  EquitySummary rows: {len(equity)}  ({ks[0]} … {ks[-1]})")
    print(f"  ChangeInNAV rows:   {len(cin)}")

    # 3. Cross-check the two sections against each other --------------------
    print("\n[3] Cross-check: ChangeInNAV.endingValue vs EquitySummary.total…")
    checked, problems = cross_check_change_in_nav(equity, cin)
    if problems:
        print(f"  FAIL — {len(problems)} dates disagree (parse error suspected):")
        for d, a, b, diff in problems[:10]:
            print(f"    {d}  changeInNav={a:,.2f}  equitySummary={b:,.2f}  diff={diff:,.2f}")
        sys.exit("ABORT: the two Flex sections disagree; not rewriting the ledger.")
    print(f"  PASS — {checked} dates agree within $1.00")

    # 4. Coverage -----------------------------------------------------------
    print(f"\n[4] Coverage over {START_DATE} → {END_DATE}…")
    dates = trading_days(START_DATE, END_DATE)
    missing = [d for d in dates if d not in equity]
    print(f"  business days: {len(dates)}   present in Flex: {len(dates) - len(missing)}")
    if missing:
        sys.exit(f"ABORT: Flex is missing {len(missing)} required dates: {missing}")
    print("  PASS — every required date present")

    # 5. Restatement sanity gate -------------------------------------------
    old_df = pd.read_csv(DAILY_PATH) if os.path.exists(DAILY_PATH) else pd.DataFrame()
    print("\n[5] Restatement vs the existing rows…")
    if not old_df.empty:
        old_eq = dict(zip(old_df["date"].astype(str), old_df["equity"].astype(float)))
        diffs = pd.Series({d: equity[d] - old_eq[d] for d in old_eq if d in equity})
        print(f"  n={len(diffs)}  mean {diffs.mean():+,.2f}  "
              f"median {diffs.median():+,.2f}  worst |diff| {diffs.abs().max():,.2f}")
        print(f"  (probe measured: mean {PROBE_MEAN_DIFF:+,.2f}, "
              f"worst {PROBE_WORST_DIFF:,.2f})")
        limit = PROBE_WORST_DIFF * DIFF_TOLERANCE
        if diffs.abs().max() > limit:
            sys.exit(f"ABORT: restatement ({diffs.abs().max():,.2f}) exceeds {limit:,.2f} — "
                     f"the Flex query may have changed. Investigate before writing.")
        print("  PASS — restatement matches the expected basis difference")

    # 6. Rebuild ------------------------------------------------------------
    print("\n[6] Rebuilding all rows…")
    new_df = carry_forward_diagnostics(rebuild(equity, dates), old_df)

    e0, e1 = new_df["equity"].iloc[0], new_df["equity"].iloc[-1]
    nav_end, nav_expect = new_df["nav"].iloc[-1], e1 / e0
    vol = new_df["ret"].std() * (252 ** 0.5)
    print(f"  rows {len(new_df)}   {new_df['date'].iloc[0]} → {new_df['date'].iloc[-1]}")
    print(f"  equity {e0:,.2f} → {e1:,.2f}")
    print(f"  cumulative {(nav_end - 1) * 100:+.4f}%   ann vol {vol * 100:.2f}%")

    # Chain integrity. NOTE: necessary but NOT sufficient — the chain telescopes to
    # E_n/E_0, so corrupted intermediates still land on the right cumulative figure.
    # Value correctness comes from steps [3] and [5], not from this.
    if abs(nav_end - nav_expect) > 1e-6:
        sys.exit(f"ABORT: nav chain broken ({nav_end:.8f} vs {nav_expect:.8f})")
    if new_df["date"].duplicated().any():
        sys.exit("ABORT: duplicate dates in rebuilt frame")
    if not new_df["date"].is_monotonic_increasing:
        sys.exit("ABORT: rebuilt frame is not date-sorted")
    print("  PASS — chain integrity (telescopes to E_n/E_0), sorted, no duplicates")

    # 7. Diff ---------------------------------------------------------------
    print("\n[7] DIFF (old → new)")
    print(f"  {'date':12s} {'old_equity':>12s} {'new_equity':>12s} {'delta':>10s} "
          f"{'new_ret':>9s}  src")
    old_eq = (dict(zip(old_df["date"].astype(str), old_df["equity"].astype(float)))
              if not old_df.empty else {})
    for _, r in new_df.iterrows():
        d = r["date"]
        if d in old_eq:
            print(f"  {d:12s} {old_eq[d]:>12,.2f} {r['equity']:>12,.2f} "
                  f"{r['equity'] - old_eq[d]:>10,.2f} {r['ret'] * 100:>8.3f}%  changed")
        else:
            print(f"  {d:12s} {'—':>12s} {r['equity']:>12,.2f} {'—':>10s} "
                  f"{r['ret'] * 100:>8.3f}%  NEW")

    if not old_df.empty:
        old_vol = old_df["ret"].astype(float).std() * (252 ** 0.5)
        old_cum = old_df["equity"].iloc[-1] / old_df["equity"].iloc[0] - 1
        print(f"\n  rows      {len(old_df)} → {len(new_df)}")
        print(f"  cumulative {old_cum * 100:+.4f}% → {(nav_end - 1) * 100:+.4f}%")
        print(f"  ann vol    {old_vol * 100:.2f}% → {vol * 100:.2f}%   "
              f"(old was inflated by multi-day returns labelled daily)")

    # 8. Write --------------------------------------------------------------
    if not args.apply:
        print("\n" + "=" * 78)
        print("  DRY-RUN — nothing written. Re-run with --apply to commit.")
        print("=" * 78)
        return

    print("\n[8] Writing…")
    write_atomic(new_df, DAILY_PATH, DAILY_COLS)
    print(f"  wrote {DAILY_PATH} ({len(new_df)} rows)")

    # state.json must move in the same operation: nav chains off the LAST row in
    # file order, so a stale anchor silently re-creates the BUG-26 failure mode.
    state = _load_state()
    state["initial_equity"] = float(new_df["equity"].iloc[0])
    state["last_equity"]    = float(new_df["equity"].iloc[-1])
    state["last_date"]      = str(new_df["date"].iloc[-1])
    import json
    tmp = f"{STATE_PATH}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, STATE_PATH)
    print(f"  updated {STATE_PATH}: last_equity={state['last_equity']:,.2f} "
          f"last_date={state['last_date']}")
    print("\n" + "=" * 78)
    print("  DONE — verify with: python3 paper/dyn_ledger.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
