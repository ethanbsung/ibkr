#!/usr/bin/env python3
"""
ibkr_fut/risk_check.py

Risk controls for the EWMAC dynamic-optimisation live executor. Three gates,
checked by live_dynamic.py before orders are submitted:

  1. check_order_vol     — per-instrument annualised dollar-vol cap (skip 1 order)
  2. check_gross_leverage — portfolio gross-leverage cap (skip 1 cycle)
  3. check_daily_loss    — intraday equity drawdown circuit breaker (halt daemon)

The circuit breaker writes a halt file (risk_halt.txt) and exits. The daemon and
run_execution.sh both refuse to start while that file exists, so trading stays
halted until the file is manually removed (and the loss investigated).

All checks return (ok, reason): ok=True means proceed; ok=False means take the
gate's action and `reason` explains why.
"""

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── Thresholds ────────────────────────────────────────────────────────────────
MAX_GROSS_LEV        = 12.0  # skip cycle if portfolio gross leverage exceeds this.
                             # Combined carry+trend @ target risk 0.25 backtests to
                             # mean 3.4x / p95 5.9x / max 10.9x gross; 12.0 clears the
                             # legitimate peak while still catching a genuine blow-up.
MAX_INSTR_VOL_FRAC   = 0.50  # skip order if single-instr annual $ vol > 50% of capital
DAILY_LOSS_HALT_FRAC = 0.08  # halt daemon if equity dropped >8% from compute-time capital

HERE      = Path(__file__).parent
HALT_FILE = HERE / "risk_halt.txt"
_REPO     = HERE.parent


# ── Discord alert (self-contained — mirrors scripts/daily_report.py) ──────────

def _discord_webhook_url() -> str:
    """Read DISCORD_WEBHOOK_URL from env, falling back to .env (no overwrite)."""
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if url:
        return url
    env_path = _REPO / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "DISCORD_WEBHOOK_URL":
                    return v.strip().strip('"').strip("'")
    return ""


def _send_discord(text: str) -> None:
    """Best-effort Discord alert; swallows errors so it never blocks a halt."""
    url = _discord_webhook_url()
    if not url:
        return
    try:
        data = json.dumps({"content": f"```\n{text}\n```"}).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json",
                     "User-Agent": "DiscordBot (private, 1.0)"},
        )
        with urllib.request.urlopen(req, timeout=30):
            pass
    except Exception as e:
        print(f"[RISK] Discord alert failed: {e}")


# ── Checks ────────────────────────────────────────────────────────────────────

def check_halt_file() -> tuple[bool, str]:
    """False = halt file present; daemon/wrapper must refuse to start."""
    if HALT_FILE.exists():
        return False, HALT_FILE.read_text().strip()
    return True, ""


def check_gross_leverage(gross_lev: float) -> tuple[bool, str]:
    """False = portfolio gross leverage too high; skip this reconcile cycle."""
    if gross_lev > MAX_GROSS_LEV:
        return False, f"gross_lev={gross_lev:.2f} > {MAX_GROSS_LEV}"
    return True, ""


def check_order_vol(qty: int, mult: float, price: float, fx: float, sigma: float,
                    capital: float) -> tuple[bool, str]:
    """
    False = single-instrument annualised dollar vol exceeds MAX_INSTR_VOL_FRAC
    of capital. dollar_vol = |qty| * mult * price * fx * sigma, where fx converts
    the instrument's local-currency notional to USD (USD notional = mult*price*fx,
    matching the sizing math in compute_targets). `qty` is the net TARGET position,
    so position-reducing rolls/closes are never blocked.
    """
    dollar_vol = abs(qty) * mult * price * fx * sigma
    frac = dollar_vol / capital if capital else 0.0
    if frac > MAX_INSTR_VOL_FRAC:
        return False, (f"annual $ vol ${dollar_vol:,.0f} = {frac:.1%} of capital "
                       f"(limit {MAX_INSTR_VOL_FRAC:.0%})")
    return True, ""


def check_daily_loss(ib, snapshot_capital: float) -> tuple[bool, str]:
    """
    False = intraday equity drop exceeds DAILY_LOSS_HALT_FRAC. On breach: write
    the halt file and fire a Discord alert. Queries live IBKR NetLiquidation;
    compares against snapshot_capital (equity at compute time, ~6 PM ET).
    """
    try:
        # Match the baseline: compute_targets reads NetLiquidation @ currency=="USD"
        # (live_dynamic.py). Using the same field keeps current vs baseline comparable.
        vals = {v.tag: v.value for v in ib.accountValues()
                if v.tag == "NetLiquidation" and v.currency == "USD"}
        equity = float(vals.get("NetLiquidation", snapshot_capital))
    except Exception as e:
        # If we can't read equity, don't false-trip the breaker — let the cycle run.
        print(f"[RISK] could not read NetLiquidation ({e}) — skipping loss check")
        return True, ""

    if not snapshot_capital:
        return True, ""

    drop_frac = (snapshot_capital - equity) / snapshot_capital
    if drop_frac > DAILY_LOSS_HALT_FRAC:
        reason = (f"equity ${equity:,.0f} vs baseline ${snapshot_capital:,.0f} "
                  f"({drop_frac:.1%} drop)")
        ts = datetime.now(timezone.utc).isoformat()
        HALT_FILE.write_text(f"{ts} | circuit breaker — {reason}\n")
        _send_discord(
            f"[RISK] CIRCUIT BREAKER TRIGGERED\n{reason}\n"
            f"Daemon halted; positions left untouched.\n"
            f"Remove ibkr_fut/risk_halt.txt to resume trading next session."
        )
        return False, reason
    return True, ""
