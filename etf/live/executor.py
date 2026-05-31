"""
Executor — translates target positions into Alpaca orders.

Order mechanics (Alpaca constraint-aware):
  • LONG side  — fractional `notional` orders (capital-efficient; lets a small
                 account hold many small fractional longs across the universe).
  • SHORT side — whole-share `qty` orders (Alpaca rejects notional/fractional
                 short sales).  A short whose target is smaller than one share
                 ROUNDS TO ZERO and is skipped — the book is long/short but
                 realised somewhat long-biased on a small account.
  • FLIP       — a target that crosses zero (long→short or short→long) is split
                 into two orders: close the existing leg, then open the new leg.
                 No single order ever crosses zero.

Per instrument:
  1. delta vs current Alpaca position
  2. 10% position buffer + MIN_TRADE_USD floor (skip tiny adjustments)
  3. independent per-order notional ceiling (defense-in-depth vs a bad signal)
  4. place order(s), poll for fill confirmation (errors logged, never swallowed)
  5. return fill record(s) for the ledger

Dry-run mode simulates fills at the reference price without placing real orders.
"""

import os
import time
from datetime import date, datetime
from typing import Optional

from etf.live._env import load_dotenv

BUFFER_FRACTION   = 0.10     # skip trade if within 10% of target (matches backtest)
MIN_TRADE_USD     = 5.00     # skip if |delta| < $5 (Alpaca fractional min ~$1, buffer)
FILL_TIMEOUT      = 30       # seconds to wait for fill confirmation
POLL_INTERVAL     = 2        # seconds between fill status polls
TERMINAL_OK       = ("filled", "partially_filled")
TERMINAL_BAD      = ("canceled", "expired", "rejected", "done_for_day")


class Executor:
    """
    Wraps Alpaca TradingClient and DataClient to execute position deltas.
    All fills are returned as dicts ready for Ledger.record_fill().
    """

    def __init__(self, dry_run: bool = True, max_order_notional: Optional[float] = None):
        load_dotenv()
        api_key    = os.environ.get("ALPACA_API_KEY", "")
        api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not api_secret:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )

        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient

        self.trading = TradingClient(api_key, api_secret, paper=True)
        self.data    = StockHistoricalDataClient(api_key, api_secret)
        self.dry_run = dry_run
        # Hard ceiling on any single order's notional.  None disables the check.
        self.max_order_notional = max_order_notional

    # ── Account / market state ────────────────────────────────────────────────

    def get_account_equity(self) -> float:
        """Returns Alpaca account equity in dollars."""
        acct = self.trading.get_account()
        return float(acct.equity)

    def is_market_open(self) -> bool:
        """True if the market is open right now per Alpaca's clock."""
        try:
            return bool(self.trading.get_clock().is_open)
        except Exception as e:
            print(f"  WARNING: could not fetch market clock ({e})")
            return False

    def get_current_positions(self) -> dict[str, float]:
        """Returns {ticker: market_value_usd} for all open Alpaca positions."""
        positions = self.trading.get_all_positions()
        return {p.symbol: float(p.market_value) for p in positions}

    def get_current_shares(self) -> dict[str, float]:
        """
        Returns {ticker: shares} for all open Alpaca positions.
        Shares are NEGATIVE for short positions.
        """
        positions = self.trading.get_all_positions()
        return {p.symbol: float(p.qty) for p in positions}

    def get_positions_detail(self) -> dict[str, dict]:
        """
        Returns {ticker: {shares, avg_entry_price, market_value}} for all open
        positions.  avg_entry_price is Alpaca's authoritative cost basis (handles
        adds/partial closes correctly), used for unrealized-P&L accounting.
        Shares/market_value are NEGATIVE for shorts.
        """
        out = {}
        for p in self.trading.get_all_positions():
            out[p.symbol] = {
                "shares":          float(p.qty),
                "avg_entry_price": float(p.avg_entry_price) if p.avg_entry_price else 0.0,
                "market_value":    float(p.market_value) if p.market_value else 0.0,
            }
        return out

    # ── Quote fetching (execution-quality diagnostics) ─────────────────────────

    def get_quote(self, ticker: str) -> dict:
        """
        Returns {bid, ask, mid, spread_bps} from the latest NBBO quote.
        Used only for slippage/spread diagnostics on traded names.
        """
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            req  = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            resp = self.data.get_stock_latest_quote(req)
            q    = resp[ticker]
            bid  = float(q.bid_price) if q.bid_price and q.bid_price > 0 else None
            ask  = float(q.ask_price) if q.ask_price and q.ask_price > 0 else None
            mid  = (bid + ask) / 2 if bid and ask else None
            spread_bps = ((ask - bid) / mid * 10_000
                          if mid and mid > 0 else None)
            return {"bid": bid, "ask": ask, "mid": mid, "spread_bps": spread_bps}
        except Exception:
            return {"bid": None, "ask": None, "mid": None, "spread_bps": None}

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Returns latest Alpaca trade price for a single ticker (fallback ref)."""
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            req  = StockLatestTradeRequest(symbol_or_symbols=ticker)
            resp = self.data.get_stock_latest_trade(req)
            px   = float(resp[ticker].price)
            return px if px > 0 else None
        except Exception:
            return None

    # ── Core execution ────────────────────────────────────────────────────────

    def execute_targets(
        self,
        target_positions: dict[str, float],    # {ticker: target_usd}
        ref_prices: dict[str, float],           # {ticker: current price} for sizing
        close_tickers: Optional[set[str]] = None,  # held names to fully unwind
        strategy: str = "ewmac",
    ) -> list[dict]:
        """
        Rebalance every name in target_positions toward its target_usd, and fully
        close any name in close_tickers (e.g. dropped from the universe).
        Returns a list of fill dicts ready for Ledger.record_fill().

        Names absent from target_positions are NOT touched (a missing live price
        upstream means "hold"), unless they appear in close_tickers.
        """
        current_shares = self.get_current_shares()
        fills: list[dict] = []

        # 1) Rebalance current universe names that have a valid reference price.
        for ticker, target_usd in target_positions.items():
            price = ref_prices.get(ticker)
            if not price or price <= 0:
                # No price -> cannot size safely; hold the existing position.
                print(f"  HOLD {ticker}: no reference price, leaving position unchanged")
                continue
            cur = current_shares.get(ticker, 0.0)
            fills += self._rebalance_one(ticker, cur, target_usd, price, strategy)

        # 2) Close names held on Alpaca but no longer in the universe.
        for ticker in (close_tickers or set()):
            cur = current_shares.get(ticker, 0.0)
            if cur == 0:
                continue
            price = ref_prices.get(ticker) or self.get_latest_price(ticker)
            if not price or price <= 0:
                print(f"  WARN {ticker}: dropped from universe but no price to close; skipping")
                continue
            print(f"  CLOSE {ticker}: removed from universe, unwinding {cur:+.4f} sh")
            fills += self._close_one(ticker, cur, price, strategy)

        return fills

    def _over_ceiling(self, ticker: str, target_usd: float) -> bool:
        if self.max_order_notional is not None and abs(target_usd) > self.max_order_notional:
            print(f"  CAP {ticker}: target ${target_usd:,.0f} exceeds ceiling "
                  f"${self.max_order_notional:,.0f}; skipping (check signal sizing)")
            return True
        return False

    def _rebalance_one(self, ticker: str, cur: float, target_usd: float,
                       price: float, strategy: str) -> list[dict]:
        """Route a single instrument's rebalance into long/short/flip orders."""
        if self._over_ceiling(ticker, target_usd):
            return []

        cur_usd = cur * price
        buffer  = max(abs(target_usd) * BUFFER_FRACTION, MIN_TRADE_USD)

        # ── Target is LONG (or flat) ──────────────────────────────────────────
        if target_usd >= 0:
            if cur >= 0:
                # Stays long: fractional notional adjustment.
                delta_usd = target_usd - cur_usd
                if abs(delta_usd) < buffer:
                    return []
                side = "buy" if delta_usd > 0 else "sell"
                return self._order_notional(ticker, side, abs(delta_usd),
                                            price, target_usd, strategy)
            # Currently short -> target long: FLIP (close short, then open long).
            fills  = self._order_qty(ticker, "buy", abs(cur), price, 0.0, strategy)
            if target_usd >= MIN_TRADE_USD:
                fills += self._order_notional(ticker, "buy", target_usd,
                                              price, target_usd, strategy)
            return fills

        # ── Target is SHORT ───────────────────────────────────────────────────
        tgt_short = round(target_usd / price)   # negative int, or 0 if < ~half a share
        if tgt_short == 0:
            # Rounds to flat: just close whatever we hold (no new short opened).
            if cur != 0:
                return self._close_one(ticker, cur, price, strategy)
            return []

        if cur <= 0:
            # Stays short (or opens from flat): whole-share delta.
            delta_shares = tgt_short - cur          # both <= 0
            if abs(delta_shares * price) < buffer:
                return []
            side = "sell" if delta_shares < 0 else "buy"
            return self._order_qty(ticker, side, abs(delta_shares),
                                   price, target_usd, strategy)

        # Currently long -> target short: FLIP (close long, then open short).
        fills  = self._close_one(ticker, cur, price, strategy)
        fills += self._order_qty(ticker, "sell", abs(tgt_short),
                                 price, target_usd, strategy)
        return fills

    def _close_one(self, ticker: str, cur: float, price: float,
                   strategy: str) -> list[dict]:
        """Fully flatten an existing position."""
        if cur > 0:
            return self._order_notional(ticker, "sell", cur * price,
                                        price, 0.0, strategy)
        if cur < 0:
            return self._order_qty(ticker, "buy", abs(cur), price, 0.0, strategy)
        return []

    # ── Order primitives ──────────────────────────────────────────────────────

    def _order_notional(self, ticker: str, side: str, notional: float,
                        price: float, target_usd: float, strategy: str) -> list[dict]:
        """Place a fractional notional order (long side only)."""
        notional = round(notional, 2)
        if notional < MIN_TRADE_USD:
            return []
        est_shares = notional / price * (1 if side == "buy" else -1)
        return self._dispatch(ticker, side, price, target_usd, strategy,
                              notional=notional, est_shares=est_shares)

    def _order_qty(self, ticker: str, side: str, qty: int,
                   price: float, target_usd: float, strategy: str) -> list[dict]:
        """Place a whole-share qty order (used for the short side)."""
        qty = int(qty)
        if qty <= 0:
            return []
        est_shares = qty * (1 if side == "buy" else -1)
        return self._dispatch(ticker, side, price, target_usd, strategy,
                              qty=qty, est_shares=est_shares)

    def _dispatch(self, ticker: str, side: str, price: float, target_usd: float,
                  strategy: str, notional: Optional[float] = None,
                  qty: Optional[int] = None, est_shares: float = 0.0) -> list[dict]:
        """Simulate (dry-run) or submit a single order and build the fill dict."""
        quote = self.get_quote(ticker)
        mid   = quote["mid"]
        dt    = date.today().isoformat()
        size_str = (f"${notional:,.2f}" if notional is not None else f"{qty} sh")

        if self.dry_run:
            fill_price  = mid or price
            fill_shares = est_shares
            order_id    = f"DRY-{ticker}-{datetime.utcnow().strftime('%H%M%S%f')}"
            status      = "dry_run"
            print(f"  DRY  {ticker:<6} {side:<4} {fill_shares:>9.4f} sh"
                  f"  @ ${fill_price:>8.2f}  ({size_str})")
        else:
            result = self._place_and_wait(ticker, side, notional=notional, qty=qty)
            if result is None:
                return []
            sign        = 1 if side == "buy" else -1
            fill_price  = result["fill_price"] or (mid or price)
            fill_shares = result["filled_shares"] * sign
            order_id    = result["order_id"]
            status      = result["status"]
            flag = "" if status in ("filled",) else f"  [{status}]"
            print(f"  FILL {ticker:<6} {side:<4} {fill_shares:>9.4f} sh"
                  f"  @ ${fill_price:>8.2f}  ({size_str}){flag}")

        return [{
            "date":          dt,
            "strategy":      strategy,
            "ticker":        ticker,
            "side":          side,
            "target_shares": (target_usd / price) if price else None,
            "filled_shares": fill_shares,
            "fill_price":    fill_price,
            "fill_value":    abs(fill_shares * fill_price),
            "signal_price":  price,
            "bid":           quote["bid"],
            "ask":           quote["ask"],
            "spread_bps":    quote["spread_bps"],
            "order_id":      order_id,
            "order_status":  status,
            "is_dry_run":    self.dry_run,
        }]

    def _place_and_wait(self, ticker: str, side: str,
                        notional: Optional[float] = None,
                        qty: Optional[int] = None) -> Optional[dict]:
        """
        Submit a market order and poll for fill.  Errors are logged, never
        silently swallowed.  Returns a result dict whose `status` is one of:
          filled / partially_filled / <terminal-bad> / unconfirmed.
        Returns None only when the order could not be submitted at all.
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        kwargs = dict(
            symbol        = ticker,
            side          = OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force = TimeInForce.DAY,
        )
        if notional is not None:
            kwargs["notional"] = round(notional, 2)
        else:
            kwargs["qty"] = int(qty)

        try:
            order = self.trading.submit_order(MarketOrderRequest(**kwargs))
        except Exception as e:
            print(f"  ERROR submit {ticker} {side}: {e}")
            return None

        deadline   = time.time() + FILL_TIMEOUT
        last_known = None
        while time.time() < deadline:
            time.sleep(POLL_INTERVAL)
            try:
                o = self.trading.get_order_by_id(order.id)
            except Exception as e:
                print(f"  WARN poll {ticker} ({e}); retrying")
                continue
            status = o.status.value
            if status in TERMINAL_OK:
                return {
                    "fill_price":    float(o.filled_avg_price or 0) or None,
                    "filled_shares": float(o.filled_qty or 0),
                    "order_id":      str(o.id),
                    "status":        status,
                }
            if status in TERMINAL_BAD:
                print(f"  ORDER {ticker} {status}")
                return {
                    "fill_price":    float(o.filled_avg_price or 0) or None,
                    "filled_shares": float(o.filled_qty or 0),
                    "order_id":      str(o.id),
                    "status":        status,
                }
            last_known = o

        # Timed out without a terminal status — do NOT silently drop it.  Record
        # whatever filled and flag it unconfirmed so reconciliation can catch it.
        print(f"  TIMEOUT {ticker}: order {order.id} not confirmed in {FILL_TIMEOUT}s "
              f"(last status: {last_known.status.value if last_known else 'unknown'})")
        return {
            "fill_price":    float(last_known.filled_avg_price or 0) or None if last_known else None,
            "filled_shares": float(last_known.filled_qty or 0) if last_known else 0.0,
            "order_id":      str(order.id),
            "status":        "unconfirmed",
        }
