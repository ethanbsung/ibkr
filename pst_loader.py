"""
Loader for pysystemtrade futures data.

Data lives at: /home/ethanbsung/ibkr/Data/pst/futures/

What's available:
  adjusted_prices_csv/  - 252 instruments, back-adjusted continuous prices (hourly)
  multiple_prices_csv/  - 252 instruments, front/carry/forward contract prices (hourly)
  roll_calendars_csv/   - 276 instruments, contract roll schedule
  fx_prices_csv/        - 12 FX rates (daily)
  csvconfig/            - instrument metadata, spread costs, roll config

Usage:
    from pst_loader import PSTLoader
    pst = PSTLoader()

    prices = pst.adjusted_prices("SP500")             # daily Series
    prices = pst.adjusted_prices("SP500", freq="1h")  # hourly Series
    multi  = pst.multiple_prices("SP500")             # daily DataFrame (PRICE/CARRY/FORWARD)
    rolls  = pst.roll_calendar("SP500")               # roll schedule DataFrame
    fx     = pst.fx_rate("EUR")                       # daily EURUSD Series
    info   = pst.instrument_info("SP500")             # dict of all metadata
    meta   = pst.instruments_df()                     # full metadata for all 252 instruments
    syms   = pst.list_instruments()                   # list of all 252 instrument codes
"""

import os
import glob
import pandas as pd

PST_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "pst", "futures")


class PSTLoader:
    def __init__(self, base_path: str = PST_BASE):
        self.base = base_path
        self._meta_cache = None

    # ------------------------------------------------------------------ #
    # Instrument list                                                       #
    # ------------------------------------------------------------------ #

    def list_instruments(self) -> list[str]:
        """Return sorted list of all instruments with adjusted price data."""
        files = glob.glob(os.path.join(self.base, "adjusted_prices_csv", "*.csv"))
        return sorted(os.path.basename(f).replace(".csv", "") for f in files)

    # ------------------------------------------------------------------ #
    # Price data                                                           #
    # ------------------------------------------------------------------ #

    def adjusted_prices(self, instrument: str, freq: str = "daily") -> pd.Series:
        """
        Back-adjusted continuous price series.

        freq: "daily" (last price of each calendar day) or "1h" (raw hourly).

        Returns a pd.Series indexed by datetime, named 'price'.
        Negative prices are expected for older data due to Panama back-adjustment.
        """
        fp = self._path("adjusted_prices_csv", instrument)
        series = pd.read_csv(fp, parse_dates=["DATETIME"], index_col="DATETIME")["price"]
        series.index = pd.DatetimeIndex(series.index)
        return self._resample(series, freq)

    def multiple_prices(self, instrument: str, freq: str = "daily") -> pd.DataFrame:
        """
        Front / carry / forward contract prices for each date.

        Columns returned:
          PRICE           - front contract (current) price
          PRICE_CONTRACT  - front contract expiry (YYYYMM)
          CARRY           - carry contract price
          CARRY_CONTRACT  - carry contract expiry
          FORWARD         - next/forward contract price
          FORWARD_CONTRACT- forward contract expiry

        Needed for carry signal: carry = (CARRY - PRICE) / (PRICE * vol)

        freq: "daily" or "1h"
        """
        fp = self._path("multiple_prices_csv", instrument)
        df = pd.read_csv(fp, parse_dates=["DATETIME"], index_col="DATETIME")
        df.index = pd.DatetimeIndex(df.index)
        if freq == "daily":
            price_cols = [c for c in df.columns if "CONTRACT" not in c]
            contract_cols = [c for c in df.columns if "CONTRACT" in c]
            prices_daily = df[price_cols].resample("D").last()
            contracts_daily = df[contract_cols].resample("D").last()
            df = pd.concat([prices_daily, contracts_daily], axis=1).dropna(how="all")
        return df

    def roll_calendar(self, instrument: str) -> pd.DataFrame:
        """
        Roll schedule: when to switch from current to next contract.

        Columns: current_contract, next_contract, carry_contract (YYYYMM integers)
        """
        fp = self._path("roll_calendars_csv", instrument)
        df = pd.read_csv(fp, parse_dates=["DATE_TIME"], index_col="DATE_TIME")
        df.index = pd.DatetimeIndex(df.index)
        return df

    # ------------------------------------------------------------------ #
    # FX rates                                                             #
    # ------------------------------------------------------------------ #

    def fx_rate(self, currency: str, freq: str = "daily") -> pd.Series:
        """
        FX rate as units of currency per 1 USD... actually stored as currency/USD.

        Available currencies: AUD, CAD, CHF, CNH, EUR, GBP, HKD, JPY, KRW, MXP, SEK, SGD

        For a non-USD instrument multiply P&L by fx_rate(currency) to convert to USD.
        Returns a pd.Series indexed by datetime.
        """
        fname = f"{currency}USD.csv"
        fp = os.path.join(self.base, "fx_prices_csv", fname)
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"No FX file for {currency}. Available: "
                + str(self.list_fx_currencies())
            )
        series = pd.read_csv(fp, parse_dates=["DATETIME"], index_col="DATETIME")["PRICE"]
        series.index = pd.DatetimeIndex(series.index)
        series.name = f"{currency}USD"
        return self._resample(series, freq)

    def list_fx_currencies(self) -> list[str]:
        files = glob.glob(os.path.join(self.base, "fx_prices_csv", "*.csv"))
        return sorted(os.path.basename(f).replace("USD.csv", "") for f in files)

    # ------------------------------------------------------------------ #
    # Metadata                                                             #
    # ------------------------------------------------------------------ #

    def instruments_df(self) -> pd.DataFrame:
        """
        Full metadata DataFrame for all 252 instruments with data files.

        Columns:
          Instrument, Description, Pointsize, Currency, AssetClass,
          PerBlock, Percentage, PerTrade, Region,
          SpreadCost,
          HoldRollCycle, RollOffsetDays, CarryOffset, PricedRollCycle, ExpiryOffset

        SpreadCost is in price units (half-spread per trade).
        SR_cost per trade ≈ 2 * SpreadCost / (Pointsize * price * annual_vol)
        """
        if self._meta_cache is None:
            self._meta_cache = self._load_meta()
        return self._meta_cache

    def instrument_info(self, instrument: str) -> dict:
        """Return all metadata for one instrument as a dict."""
        meta = self.instruments_df()
        row = meta[meta["Instrument"] == instrument]
        if row.empty:
            raise ValueError(f"No metadata for '{instrument}'. Check list_instruments().")
        return row.iloc[0].to_dict()

    def sr_cost(self, instrument: str, price: float, annual_vol: float) -> float:
        """
        Estimate SR cost per trade (Carver formula).

        sr_cost = 2 * SpreadCost / (Pointsize * price * annual_vol)

        annual_vol: annualised volatility as a fraction (e.g. 0.16 for 16%)
        """
        info = self.instrument_info(instrument)
        spread = info.get("SpreadCost", float("nan"))
        pointsize = info["Pointsize"]
        return (2 * spread) / (pointsize * price * annual_vol)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _path(self, subdir: str, instrument: str) -> str:
        fp = os.path.join(self.base, subdir, f"{instrument}.csv")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"No file for '{instrument}' in {subdir}/")
        return fp

    def _resample(self, obj, freq: str):
        if freq == "daily":
            if isinstance(obj, pd.Series):
                return obj.resample("D").last().dropna()
            return obj.resample("D").last().dropna(how="all")
        return obj  # "1h" — return as-is

    def _load_meta(self) -> pd.DataFrame:
        cfg = os.path.join(self.base, "csvconfig")
        ic = pd.read_csv(f"{cfg}/instrumentconfig.csv")
        sc = pd.read_csv(f"{cfg}/spreadcosts.csv")
        rc = pd.read_csv(f"{cfg}/rollconfig.csv")
        meta = (
            ic.merge(sc, on="Instrument", how="left")
              .merge(rc, on="Instrument", how="left")
        )
        available = set(self.list_instruments())
        return meta[meta["Instrument"].isin(available)].reset_index(drop=True)
