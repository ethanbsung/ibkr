"""
EWMAC trend-following signal generator.

Reuses the exact same functions as ewmac_backtest.py so live signals
match backtest signals precisely.  Returns {ticker: target_usd} for
today's close.

Adding a new strategy: create a new file with a class that has:
    name: str
    def get_signals(self, capital: float) -> dict[str, float]
    def get_metadata(self, capital: float) -> dict[str, dict]
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# Allow imports from the etf/ parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ewmac_backtest import (
    EWMAC_VARIANTS,
    FORECAST_TARGET,
    IDM_CAP,
    DATA_DIR,
    WARMUP_BARS,
    GROSS_LEVERAGE_CAP,   # single source of truth — Reg-T 1.9x gross cap
    blended_vol,
    normalised_ewmac,
    estimate_scalar,
    build_forecasts,
    compute_handcraft_weights,
    compute_idm,
    size_targets,         # shared single-bar sizing (live == backtest)
    load_prices,
)

UNIVERSE_FILE = "Data/etf/etf_universe_greedy.json"


class EWMACStrategy:
    """
    Daily EWMAC trend-following on the 87-instrument ETF universe.
    Signal: 6 EWMAC speeds blended with FDM, handcraft equal weights.
    """

    name = "ewmac"

    def __init__(self, vol_target: float = 0.25, history_start: str = "2010-01-01",
                 gross_leverage_cap: float = GROSS_LEVERAGE_CAP):
        self.vol_target         = vol_target
        self.history_start      = history_start
        self.gross_leverage_cap = gross_leverage_cap
        self._state_cache       = None   # CSV-only base state, built once per instance
        self._live_cache        = None   # (today_prices key, state) — reused within a run

        with open(UNIVERSE_FILE) as f:
            u = json.load(f)
        self.tickers       = u["selected"]
        self.asset_classes = u.get("asset_classes", {})

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_base_state(self) -> dict:
        """
        Load full yfinance history from CSVs.  Cached on the instance — this
        is the expensive part (I/O + full EWM computation over 15+ years).
        today_prices are never written here; CSVs stay clean.
        """
        if self._state_cache is not None:
            return self._state_cache

        prices  = load_prices(self.tickers, self.history_start)
        tickers = list(prices.columns)
        returns = prices.pct_change()
        vols    = pd.DataFrame({tk: blended_vol(returns[tk]) for tk in tickers})

        scalars = {}
        for fast, slow in EWMAC_VARIANTS:
            raw = pd.concat(
                [normalised_ewmac(prices[tk], fast, slow, vols[tk]) for tk in tickers]
            ).dropna()
            scalars[(fast, slow)] = estimate_scalar(raw)

        _, combined_fc = build_forecasts(prices, vols, scalars)
        weights        = compute_handcraft_weights(tickers, self.asset_classes)
        idm            = compute_idm(weights, returns)

        self._state_cache = dict(prices=prices, vols=vols, combined_fc=combined_fc,
                                 weights=weights, idm=idm, tickers=tickers)
        return self._state_cache

    def _build_state(self, today_prices: dict[str, float] | None = None) -> dict:
        """
        Returns the full state used for signal computation.

        If today_prices is provided (live prices fetched at ~3:58 PM), they are
        appended as today's row to the in-memory price series before recomputing
        vols and forecasts.  The CSV files are never modified.

        If today_prices is None, returns the cached CSV-based state (last bar =
        yesterday's adjusted close).
        """
        base = self._build_base_state()
        if not today_prices:
            return base

        # Reuse within a run: get_signals() and get_metadata() are called
        # back-to-back with the same today_prices — rebuild the (expensive)
        # 87-ticker forecast only once.
        key = tuple(sorted(today_prices.items()))
        if self._live_cache is not None and self._live_cache[0] == key:
            return self._live_cache[1]

        # Append live prices as today's row — in memory only
        today_ts = pd.Timestamp.today().normalize()
        prices   = base["prices"].copy()
        row      = pd.Series(
            {tk: today_prices.get(tk, float("nan")) for tk in prices.columns},
            name=today_ts,
        )
        prices = pd.concat([prices, row.to_frame().T])
        prices = prices[~prices.index.duplicated(keep="last")]

        tickers = list(prices.columns)
        returns = prices.pct_change()
        vols    = pd.DataFrame({tk: blended_vol(returns[tk]) for tk in tickers})

        scalars = {}
        for fast, slow in EWMAC_VARIANTS:
            raw = pd.concat(
                [normalised_ewmac(prices[tk], fast, slow, vols[tk]) for tk in tickers]
            ).dropna()
            scalars[(fast, slow)] = estimate_scalar(raw)

        _, combined_fc = build_forecasts(prices, vols, scalars)

        state = dict(
            prices     = prices,
            vols       = vols,
            combined_fc= combined_fc,
            weights    = base["weights"],   # weights/IDM don't change with one price row
            idm        = base["idm"],
            tickers    = tickers,
        )
        self._live_cache = (key, state)
        return state

    # ── Public interface ──────────────────────────────────────────────────────

    def get_signals(self, capital: float,
                    today_prices: dict[str, float] | None = None) -> dict[str, float]:
        """
        Returns {ticker: target_position_usd}.
        Pass today_prices (live Alpaca prices at ~3:58 PM) to compute signals
        on the current price rather than yesterday's close.  Those prices are
        used in memory only and never written to CSV files.
        """
        state   = self._build_state(today_prices)
        targets = self._compute_targets(capital, state)
        if today_prices is not None:
            # Hold (do not rebalance) any ticker we couldn't get a live price
            # for: drop it from targets entirely so the executor never touches
            # it.  Emitting 0.0 here would instead liquidate the position.
            targets = {tk: v for tk, v in targets.items() if tk in today_prices}
        return targets

    def get_metadata(self, capital: float,
                     today_prices: dict[str, float] | None = None) -> dict[str, dict]:
        """
        Returns per-instrument signal metadata for ledger logging.
        Pass the same today_prices used in get_signals() for consistency.
        """
        state   = self._build_state(today_prices)
        targets = self._compute_targets(capital, state)
        today_fc  = state["combined_fc"].iloc[-1]
        today_vol = state["vols"].iloc[-1]

        meta = {}
        for tk in state["tickers"]:
            meta[tk] = {
                "forecast":    float(today_fc.get(tk, np.nan)),
                "annual_vol":  float(today_vol.get(tk, np.nan)),
                "weight":      float(state["weights"].get(tk, 0.0)),
                "idm":         float(state["idm"]),
                "target_usd":  float(targets.get(tk, 0.0)),
                "asset_class": self.asset_classes.get(tk, "OTHER"),
            }
        return meta

    def _compute_targets(self, capital: float, state: dict) -> dict[str, float]:
        # Shared single-bar sizer (ewmac_backtest.size_targets) — the formula and
        # the gross-leverage cap live in one place so live can't drift from the
        # backtest.
        targets, scale = size_targets(
            capital, state["tickers"],
            state["combined_fc"].iloc[-1], state["vols"].iloc[-1],
            state["weights"], state["idm"],
            self.vol_target, self.gross_leverage_cap,
        )
        self._last_scale = scale

        return targets
