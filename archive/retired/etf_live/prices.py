"""
Live price fetch for the ~3:58 PM execution snapshot — via yfinance.

Alpaca's free (IEX) data feed is delayed and incomplete for some ETFs, so the
current price used to compute today's signal is sourced from yfinance instead.
The CSV history is refreshed separately (nightly job), never here.

get_live_prices() returns {ticker: price} and SILENTLY OMITS any ticker it
could not price.  Callers MUST treat a missing ticker as "hold" (do not
rebalance that name) — never as a zero/target, which would liquidate it.
"""

import yfinance as yf


def _extract_last(data, ticker: str, n_tickers: int):
    """Pull the last non-NaN Close from a yf.download frame (single or multi)."""
    try:
        if n_tickers == 1:
            ser = data["Close"]
        else:
            # group_by="ticker" -> columns are (ticker, field)
            ser = data[ticker]["Close"]
        ser = ser.dropna()
        if len(ser):
            px = float(ser.iloc[-1])
            return px if px > 0 else None
    except Exception:
        pass
    return None


def _fast_last(ticker: str):
    """Per-ticker fallback using yfinance fast_info."""
    try:
        fi = yf.Ticker(ticker).fast_info
        for key in ("lastPrice", "last_price"):
            try:
                px = float(fi[key])
            except (KeyError, TypeError):
                continue
            if px and px > 0:
                return px
    except Exception:
        pass
    return None


def get_live_prices(tickers: list[str]) -> dict[str, float]:
    """
    Batch-fetch the latest intraday price for each ticker.  Returns
    {ticker: price}; tickers that could not be priced are omitted.
    """
    prices: dict[str, float] = {}
    if not tickers:
        return prices

    # One batched intraday request.  The Close of the latest 1-minute bar is the
    # most recent traded price at call time (~3:58 PM ET).
    data = None
    try:
        data = yf.download(
            tickers,
            period="1d",
            interval="1m",
            progress=False,
            threads=True,
            group_by="ticker",
            auto_adjust=False,
        )
    except Exception as e:
        print(f"  WARNING: yfinance batch price fetch failed ({e}); per-ticker fallback")

    if data is not None and not data.empty:
        for tk in tickers:
            px = _extract_last(data, tk, len(tickers))
            if px is not None:
                prices[tk] = px

    # Per-ticker fallback for anything the batch missed.
    for tk in (t for t in tickers if t not in prices):
        px = _fast_last(tk)
        if px is not None:
            prices[tk] = px

    return prices
