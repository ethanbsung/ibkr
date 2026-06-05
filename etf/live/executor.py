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
  • NON-SHORTABLE — symbols flagged in Data/etf/etf_shortability.json as
                 shortable=False are silently treated as long-only: any short
                 target is skipped (position stays flat or long).

Execution is two-phase to avoid sequential blocking and stay under API rate limits:
  Phase 1 — submit all orders quickly (0.35s spacing), no blocking poll per order.
             A local-clock post-market guard skips orders if it's already past 4 PM ET.
  Phase 2 — batch-poll get_orders(status=open) every 3s until all submitted orders
             settle or a 60s timeout expires. One API call per cycle vs N per cycle.

Quotes are batch-fetched upfront (1 call for all tickers) for slippage diagnostics.

Per instrument:
  1. delta vs current Alpaca position
  2. 10% position buffer + MIN_TRADE_USD floor (skip tiny adjustments)
  3. independent per-order notional ceiling (defense-in-depth vs a bad signal)
  4. place order(s) in phase 1, confirm fills in phase 2
  5. return fill record(s) for the ledger
"""

import json
import os
import time
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

from etf.live._env import load_dotenv

BUFFER_FRACTION    = 0.10    # skip trade if within 10% of target (matches backtest)
MIN_TRADE_USD      = 5.00    # skip if |delta| < $5 (Alpaca fractional min ~$1, buffer)
SUBMIT_SPACING     = 0.35    # seconds between order submissions (keeps us ~156 req/min)
POLL_INTERVAL      = 3       # seconds between batch poll cycles
POLL_TIMEOUT       = 60      # seconds to wait for all orders to settle
TERMINAL_OK        = ("filled", "partially_filled")
TERMINAL_BAD       = ("canceled", "expired", "rejected", "done_for_day")
MARKET_CLOSE_ET    = (16, 0)  # (hour, minute) — skip submits at/after this local time
ET                 = ZoneInfo("America/New_York")

SHORTABILITY_FILE  = "Data/etf/etf_shortability.json"


def _load_shortability() -> dict[str, bool]:
    """Returns {ticker: shortable} from the pre-built shortability JSON. Defaults to True."""
    if not os.path.exists(SHORTABILITY_FILE):
        return {}
    with open(SHORTABILITY_FILE) as f:
        data = json.load(f)
    return {sym: d.get("shortable", True) for sym, d in data.get("symbols", {}).items()}


def _market_closed() -> bool:
    """True if current ET wall-clock time is at or past 4:00 PM (no API call)."""
    now = datetime.now(ET)
    return (now.hour, now.minute) >= MARKET_CLOSE_ET


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

        self.trading           = TradingClient(api_key, api_secret, paper=True)
        self.data              = StockHistoricalDataClient(api_key, api_secret)
        self.dry_run           = dry_run
        self.max_order_notional = max_order_notional
        self._shortable        = _load_shortability()

        n_ns = sum(1 for v in self._shortable.values() if not v)
        if n_ns:
            ns_list = sorted(k for k, v in self._shortable.items() if not v)
            print(f"  Shortability: {n_ns} non-shortable symbols loaded: {ns_list}")

    # ── Account / market state ────────────────────────────────────────────────

    def get_account_equity(self) -> float:
        acct = self.trading.get_account()
        return float(acct.equity)

    def is_market_open(self) -> bool:
        try:
            return bool(self.trading.get_clock().is_open)
        except Exception as e:
            print(f"  WARNING: could not fetch market clock ({e})")
            return False

    def get_current_positions(self) -> dict[str, float]:
        positions = self.trading.get_all_positions()
        return {p.symbol: float(p.market_value) for p in positions}

    def get_current_shares(self) -> dict[str, float]:
        positions = self.trading.get_all_positions()
        return {p.symbol: float(p.qty) for p in positions}

    def get_positions_detail(self) -> dict[str, dict]:
        out = {}
        for p in self.trading.get_all_positions():
            out[p.symbol] = {
                "shares":          float(p.qty),
                "avg_entry_price": float(p.avg_entry_price) if p.avg_entry_price else 0.0,
                "market_value":    float(p.market_value) if p.market_value else 0.0,
            }
        return out

    # ── Quote fetching ────────────────────────────────────────────────────────

    def get_quotes_batch(self, tickers: list[str]) -> dict[str, dict]:
        """
        Fetch latest NBBO quotes for all tickers in a single API call.
        Returns {ticker: {bid, ask, mid, spread_bps}}.
        """
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            req  = StockLatestQuoteRequest(symbol_or_symbols=tickers)
            resp = self.data.get_stock_latest_quote(req)
            out  = {}
            for tk in tickers:
                q = resp.get(tk)
                if q is None:
                    out[tk] = {"bid": None, "ask": None, "mid": None, "spread_bps": None}
                    continue
                bid = float(q.bid_price) if q.bid_price and q.bid_price > 0 else None
                ask = float(q.ask_price) if q.ask_price and q.ask_price > 0 else None
                mid = (bid + ask) / 2 if bid and ask else None
                spread_bps = ((ask - bid) / mid * 10_000 if mid and mid > 0 else None)
                out[tk] = {"bid": bid, "ask": ask, "mid": mid, "spread_bps": spread_bps}
            return out
        except Exception as e:
            print(f"  WARN batch quote fetch failed ({e}); diagnostics will be missing")
            return {tk: {"bid": None, "ask": None, "mid": None, "spread_bps": None}
                    for tk in tickers}

    def get_latest_price(self, ticker: str) -> Optional[float]:
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
        target_positions: dict[str, float],
        ref_prices: dict[str, float],
        close_tickers: Optional[set[str]] = None,
        strategy: str = "ewmac",
    ) -> list[dict]:
        """
        Two-phase execution:
          Phase 1 — build the order list, then submit all quickly with SUBMIT_SPACING delay.
                    Skip any submit if the local ET clock has passed market close.
          Phase 2 — batch-poll open orders until settled or POLL_TIMEOUT expires.

        Returns a list of fill dicts ready for Ledger.record_fill().
        """
        current_shares = self.get_current_shares()

        # Batch-fetch all quotes upfront (1 API call for all tickers).
        all_tickers = list(target_positions) + list(close_tickers or [])
        quotes = self.get_quotes_batch(all_tickers) if not self.dry_run else {}

        # ── Build order specs (no submission yet) ─────────────────────────────
        order_specs: list[dict] = []

        for ticker, target_usd in target_positions.items():
            price = ref_prices.get(ticker)
            if not price or price <= 0:
                print(f"  HOLD {ticker}: no reference price, leaving position unchanged")
                continue
            cur = current_shares.get(ticker, 0.0)
            specs = self._build_order_specs(ticker, cur, target_usd, price, strategy,
                                            quotes.get(ticker, {}))
            order_specs.extend(specs)

        for ticker in (close_tickers or set()):
            cur = current_shares.get(ticker, 0.0)
            if cur == 0:
                continue
            price = ref_prices.get(ticker) or self.get_latest_price(ticker)
            if not price or price <= 0:
                print(f"  WARN {ticker}: dropped from universe but no price to close; skipping")
                continue
            print(f"  CLOSE {ticker}: removed from universe, unwinding {cur:+.4f} sh")
            specs = self._build_order_specs(ticker, cur, 0.0, price, strategy,
                                            quotes.get(ticker, {}))
            order_specs.extend(specs)

        if not order_specs:
            return []

        # ── Phase 1: submit all orders ────────────────────────────────────────
        if self.dry_run:
            return self._dry_run_fills(order_specs, ref_prices)

        submitted: dict[str, dict] = {}   # order_id → {spec, order}
        skipped_post_close = 0

        for spec in order_specs:
            if _market_closed():
                skipped_post_close += 1
                continue
            order_id = self._submit_one(spec)
            if order_id:
                submitted[order_id] = spec
            if submitted or skipped_post_close < len(order_specs):
                time.sleep(SUBMIT_SPACING)

        if skipped_post_close:
            print(f"  ⚠ POST-MARKET: skipped {skipped_post_close} order(s) — market closed")

        print(f"  Placed {len(submitted)} orders")

        # ── Phase 2: batch-poll until all settled ─────────────────────────────
        fills = self._batch_poll(submitted)
        return fills

    # ── Order spec builder ────────────────────────────────────────────────────

    def _build_order_specs(self, ticker: str, cur: float, target_usd: float,
                           price: float, strategy: str, quote: dict) -> list[dict]:
        """Build order spec dicts without submitting. Mirrors the old _rebalance_one logic."""
        if self.max_order_notional and abs(target_usd) > self.max_order_notional:
            print(f"  CAP {ticker}: target ${target_usd:,.0f} exceeds ceiling; skipping")
            return []

        cur_usd = cur * price
        buffer  = max(abs(target_usd) * BUFFER_FRACTION, MIN_TRADE_USD)
        specs   = []

        if target_usd >= 0:
            if cur >= 0:
                delta_usd = target_usd - cur_usd
                if abs(delta_usd) < buffer:
                    return []
                side = "buy" if delta_usd > 0 else "sell"
                if side == "sell" and abs(delta_usd) / price >= cur - 1e-6:
                    # Selling notional can round up past available shares; use exact qty.
                    s = self._qty_spec(ticker, "sell", cur, price,
                                       target_usd, strategy, quote)
                else:
                    s = self._notional_spec(ticker, side, abs(delta_usd), price,
                                            target_usd, strategy, quote)
                if s:
                    specs.append(s)
            else:
                # Flip: close short, then open long.
                s = self._qty_spec(ticker, "buy", abs(cur), price, 0.0, strategy, quote)
                if s:
                    specs.append(s)
                if target_usd >= MIN_TRADE_USD:
                    s = self._notional_spec(ticker, "buy", target_usd, price,
                                            target_usd, strategy, quote)
                    if s:
                        specs.append(s)
        else:
            # Short target.
            if not self._shortable.get(ticker, True):
                print(f"  SKIP {ticker}: not shortable on Alpaca (signal={target_usd:+.0f})")
                return []

            tgt_short = round(target_usd / price)
            if tgt_short == 0:
                if cur != 0:
                    specs += self._build_order_specs(ticker, cur, 0.0, price,
                                                     strategy, quote)
                return specs

            if cur <= 0:
                delta_shares = tgt_short - cur
                if abs(delta_shares * price) < buffer:
                    return []
                side = "sell" if delta_shares < 0 else "buy"
                s = self._qty_spec(ticker, side, abs(delta_shares), price,
                                   target_usd, strategy, quote)
                if s:
                    specs.append(s)
            else:
                # Flip: close long, then open short.
                s = self._qty_spec(ticker, "sell", cur, price,
                                   0.0, strategy, quote)
                if s:
                    specs.append(s)
                s = self._qty_spec(ticker, "sell", abs(tgt_short), price,
                                   target_usd, strategy, quote)
                if s:
                    specs.append(s)

        return specs

    def _notional_spec(self, ticker, side, notional, price, target_usd,
                       strategy, quote) -> Optional[dict]:
        notional = round(notional, 2)
        if notional < MIN_TRADE_USD:
            return None
        est_shares = notional / price * (1 if side == "buy" else -1)
        return dict(ticker=ticker, side=side, price=price, target_usd=target_usd,
                    strategy=strategy, notional=notional, qty=None,
                    est_shares=est_shares, quote=quote)

    def _qty_spec(self, ticker, side, qty, price, target_usd,
                  strategy, quote) -> Optional[dict]:
        qty = int(qty)
        if qty <= 0:
            return None
        est_shares = qty * (1 if side == "buy" else -1)
        return dict(ticker=ticker, side=side, price=price, target_usd=target_usd,
                    strategy=strategy, notional=None, qty=qty,
                    est_shares=est_shares, quote=quote)

    # ── Submission + polling ──────────────────────────────────────────────────

    def _submit_one(self, spec: dict) -> Optional[str]:
        """Submit a single order. Returns order_id string or None on failure."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        ticker = spec["ticker"]
        side   = spec["side"]
        kwargs = dict(
            symbol        = ticker,
            side          = OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force = TimeInForce.DAY,
        )
        if spec["notional"] is not None:
            kwargs["notional"] = spec["notional"]
        else:
            kwargs["qty"] = spec["qty"]

        size_str = (f"${spec['notional']:,.2f}" if spec["notional"] is not None
                    else f"{spec['qty']} sh")
        try:
            order = self.trading.submit_order(MarketOrderRequest(**kwargs))
            print(f"  SUBMIT {ticker:<6} {side:<4} {size_str}")
            return str(order.id)
        except Exception as e:
            print(f"  ERROR submit {ticker} {side}: {e}")
            return None

    def _batch_poll(self, submitted: dict[str, dict]) -> list[dict]:
        """
        Poll get_orders(status=open) every POLL_INTERVAL seconds.
        When an order ID disappears from the open list it has settled.
        Fetch final state with get_order_by_id for fill price/qty.
        """
        if not submitted:
            return []

        pending = set(submitted)
        deadline = time.time() + POLL_TIMEOUT

        while pending and time.time() < deadline:
            time.sleep(POLL_INTERVAL)
            try:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                open_ids = {
                    str(o.id)
                    for o in self.trading.get_orders(
                        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN,
                                                limit=500)
                    )
                }
                pending &= open_ids   # keep only orders still open
            except Exception as e:
                print(f"  WARN batch poll failed ({e}); retrying")

        if pending:
            print(f"  ⚠ TIMEOUT: {len(pending)} order(s) still open after {POLL_TIMEOUT}s")

        # Fetch final state for every submitted order and build fill records.
        # Small sleep between calls to stay within Alpaca's 200 req/min limit.
        fills = []
        for i, (order_id, spec) in enumerate(submitted.items()):
            if i > 0:
                time.sleep(0.35)
            try:
                o = self.trading.get_order_by_id(order_id)
                status        = o.status.value
                fill_price    = float(o.filled_avg_price or 0) or None
                filled_shares = float(o.filled_qty or 0)
                if spec["side"] == "sell":
                    filled_shares = -filled_shares
            except Exception as e:
                print(f"  WARN could not fetch final status for {order_id}: {e}")
                status, fill_price, filled_shares = "unknown", None, 0.0

            ticker = spec["ticker"]
            quote  = spec["quote"]
            size_str = (f"${spec['notional']:,.2f}" if spec["notional"] is not None
                        else f"{spec['qty']} sh")
            flag = "" if status in TERMINAL_OK else f"  [{status}]"
            print(f"  FILL {ticker:<6} {spec['side']:<4} {filled_shares:>9.4f} sh"
                  f"  @ ${fill_price or 0:>8.2f}  ({size_str}){flag}")

            fills.append({
                "date":          date.today().isoformat(),
                "strategy":      spec["strategy"],
                "ticker":        ticker,
                "side":          spec["side"],
                "target_shares": (spec["target_usd"] / spec["price"]) if spec["price"] else None,
                "filled_shares": filled_shares,
                "fill_price":    fill_price,
                "fill_value":    abs(filled_shares * (fill_price or 0)),
                "signal_price":  spec["price"],
                "bid":           quote.get("bid"),
                "ask":           quote.get("ask"),
                "spread_bps":    quote.get("spread_bps"),
                "order_id":      order_id,
                "order_status":  status,
                "is_dry_run":    False,
            })

        return fills

    # ── Dry-run path ──────────────────────────────────────────────────────────

    def _dry_run_fills(self, order_specs: list[dict],
                       ref_prices: dict[str, float]) -> list[dict]:
        fills = []
        for spec in order_specs:
            ticker     = spec["ticker"]
            side       = spec["side"]
            quote      = spec.get("quote", {})
            fill_price = quote.get("mid") or spec["price"]
            est_sh     = spec["est_shares"]
            size_str   = (f"${spec['notional']:,.2f}" if spec["notional"] is not None
                          else f"{spec['qty']} sh")
            print(f"  DRY  {ticker:<6} {side:<4} {est_sh:>9.4f} sh"
                  f"  @ ${fill_price:>8.2f}  ({size_str})")
            fills.append({
                "date":          date.today().isoformat(),
                "strategy":      spec["strategy"],
                "ticker":        ticker,
                "side":          side,
                "target_shares": (spec["target_usd"] / spec["price"]) if spec["price"] else None,
                "filled_shares": est_sh,
                "fill_price":    fill_price,
                "fill_value":    abs(est_sh * fill_price),
                "signal_price":  spec["price"],
                "bid":           quote.get("bid"),
                "ask":           quote.get("ask"),
                "spread_bps":    quote.get("spread_bps"),
                "order_id":      f"DRY-{ticker}-{datetime.utcnow().strftime('%H%M%S%f')}",
                "order_status":  "dry_run",
                "is_dry_run":    True,
            })
        return fills

    # ── Legacy compatibility ──────────────────────────────────────────────────

    def get_quote(self, ticker: str) -> dict:
        """Single-ticker quote (retained for any external callers)."""
        result = self.get_quotes_batch([ticker])
        return result.get(ticker, {"bid": None, "ask": None, "mid": None, "spread_bps": None})
