"""
jumbo.py — Carver's "Jumbo" portfolio from Advanced Futures Trading Strategies
(AFTS, Tables 172-183), mapped to PST instrument names.

Carver's 7 asset classes are used: Bond, Equity, Vol, FX, Metals, Energy, Ags.
European stock-sector index futures are grouped under Equity, exactly as Carver
presents them (the "Equities" heading spans Tables 174-177, including sectors).

Multiplier differences between Carver's listed contract and the PST instrument
(e.g. Dow micro MYM vs PST DOW_mini, oats, cattle) are just contract-size /
price-unit encoding — the underlying market is the same, and a risk-targeted
backtest is invariant to the multiplier.

Not available in PST (3 of Carver's 102, excluded here):
  - India NIFTY            (no PST instrument)
  - US 5y ERIS Swap (LIW)  (no PST instrument)
  - US 10y Swap (N1U)      (no PST instrument)
=> 99 of the 102 Jumbo instruments.
"""

# PST instrument name -> Carver asset class
JUMBO: dict[str, str] = {
    # ── Bonds & rates (19) ───────────────────────────────────────────────────
    "US2": "Bond", "US3": "Bond", "US5": "Bond", "US10": "Bond",
    "US10U": "Bond", "US20": "Bond", "US30": "Bond",
    "SOFR": "Bond",                       # Eurodollar (GE) -> SOFR successor, mult 2500
    "OAT": "Bond", "SHATZ": "Bond", "BOBL": "Bond", "BUND": "Bond", "BUXL": "Bond",
    "BTP3": "Bond", "BTP": "Bond", "JGB": "Bond", "KR3": "Bond", "KR10": "Bond",
    "BONO": "Bond",

    # ── Equity (33: 6 US + 8 Europe + 8 sector + 11 Asia) ────────────────────
    "DOW_mini": "Equity", "NASDAQ_micro": "Equity", "R1000": "Equity",
    "RUSSELL_mini": "Equity", "SP400": "Equity", "SP500_micro": "Equity",
    "AEX": "Equity", "CAC": "Equity", "DAX": "Equity", "SMI": "Equity",
    "DJSTX-SMALL": "Equity", "EU-DIV30": "Equity", "EURO600": "Equity",
    "EUROSTX": "Equity",
    "EU-AUTO": "Equity", "EU-BASIC": "Equity", "EU-HEALTH": "Equity",
    "EU-INSURE": "Equity", "EU-OIL": "Equity", "EU-TECH": "Equity",
    "EU-TRAVEL": "Equity", "EU-UTILS": "Equity",
    "MSCIASIA": "Equity", "FTSECHINAA": "Equity", "FTSECHINAH": "Equity",
    "NIKKEI": "Equity", "NIKKEI400": "Equity", "MUMMY": "Equity", "TOPIX": "Equity",
    "KOSDAQ": "Equity", "KOSPI": "Equity", "MSCISING": "Equity", "FTSETAIWAN": "Equity",

    # ── Volatility (2) ────────────────────────────────────────────────────────
    "VIX": "Vol", "V2X": "Vol",

    # ── FX (17: 9 major + 8 cross/EM) ────────────────────────────────────────
    "AUD": "FX", "CAD": "FX", "CHF": "FX", "EUR": "FX", "GBP": "FX",
    "JPY": "FX", "NOK": "FX", "NZD": "FX", "SEK": "FX",
    "GBPEUR": "FX", "YENEUR": "FX", "BRE": "FX", "CNH": "FX",
    "INR": "FX", "MXP": "FX", "SGD": "FX",

    # ── Metals & crypto (9) ──────────────────────────────────────────────────
    "ALUMINIUM": "Metals", "COPPER": "Metals", "GOLD_micro": "Metals",
    "IRON": "Metals", "PALLAD": "Metals", "PLAT": "Metals", "SILVER": "Metals",
    "BITCOIN": "Metals", "ETHEREUM": "Metals",

    # ── Energy (6) ────────────────────────────────────────────────────────────
    "BRENT-LAST": "Energy", "CRUDE_W_mini": "Energy", "GAS-LAST": "Energy",
    "GASOILINE": "Energy", "GAS_US_mini": "Energy", "HEATOIL": "Energy",

    # ── Ags (13) ──────────────────────────────────────────────────────────────
    "BBCOMM": "Ags", "CHEESE": "Ags", "CORN": "Ags", "FEEDCOW": "Ags",
    "LEANHOG": "Ags", "LIVECOW": "Ags", "OATIES": "Ags", "REDWHEAT": "Ags",
    "RICE": "Ags", "SOYBEAN": "Ags", "SOYMEAL": "Ags", "SOYOIL": "Ags", "WHEAT": "Ags",
}
