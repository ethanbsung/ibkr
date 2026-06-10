"""
instrument_universe.py — Full tradable universe from Carver's IB data subscriptions.

Carver's monthly subscriptions (~£15/month total):
  Cboe One              — CBOE / CFE (VIX futures, AMERIBOR)
  CFE Enhanced          — CFE
  Eurex Core            — EUREX
  Eurex Retail Europe   — EUREX
  Euronext Basic Bundle — FTA / EURONEXT / MATIF / MEFFRV
  Korea Stock Exchange  — KSE
  Singapore             — SGX
  Osaka                 — OSE.JPN

Free (fee waivers):
  US + EU Bond Quotes   — CME, CBOT (bonds), EUREX (bonds)
  Physical Metals/Comms — COMEX, NYMEX, CBOT (ags), NYBOT
  IDEAL FX              — CME FX futures

NOT included (would cost $100+/month):
  ICE / ICEEU / ICEEUSOFT / IPE  — excluded
  HKFE                           — excluded (Hong Kong, separate subscription)
  LMEOTC                         — excluded (LME OTC metals)
  CDE                            — excluded (Canadian)

UNIVERSE dict: instrument → asset class (matching JUMBO classification where
instruments overlap, extended for non-JUMBO instruments).
"""

# PST instrument code → asset class
# All instruments have price data in adjusted_prices_csv/ AND are configured
# in ib_config_futures.csv for the subscribed exchanges listed above.
UNIVERSE: dict[str, str] = {

    # ── US Bonds & Rates (CME / CBOT) ─────────────────────────────────────
    "US2":          "Bond",
    "US3":          "Bond",
    "US5":          "Bond",
    "US10":         "Bond",
    "US10U":        "Bond",
    "US20":         "Bond",
    "US30":         "Bond",
    "SOFR":         "Bond",    # SOFR 3-month (successor to Eurodollar)
    # "BB3M" removed: BSBY rate discontinued 2024-03-28, data will never update
    "FED":          "Bond",    # Fed Funds futures

    # ── European Bonds (EUREX) ─────────────────────────────────────────────
    "SHATZ":        "Bond",
    "BOBL":         "Bond",
    "BUND":         "Bond",
    "BUXL":         "Bond",
    "OAT":          "Bond",    # French OAT
    "BTP3":         "Bond",    # Italian 3yr BTP
    "BTP":          "Bond",    # Italian 10yr BTP
    "BONO":         "Bond",    # Spanish Bono
    "CH10":         "Bond",    # Swiss 10yr Conf (EUREX)
    "EURIBOR":      "Bond",    # 3-month Euribor (EUREX — likely fails vol floor)

    # ── Asian Bonds (KSE) ──────────────────────────────────────────────────
    "KR3":          "Bond",
    "KR10":         "Bond",

    # ── US Equities (CME / CBOT) ──────────────────────────────────────────
    "SP500_micro":  "Equity",
    "NASDAQ_micro": "Equity",
    "DOW_mini":     "Equity",
    "R1000":        "Equity",
    "RUSSELL_mini": "Equity",
    "SP400":        "Equity",

    # ── European Equities (EUREX) ─────────────────────────────────────────
    "DAX":          "Equity",
    "EUROSTX":      "Equity",
    "EURO600":      "Equity",
    "SMI":          "Equity",
    "DJSTX-SMALL":  "Equity",
    "EU-DIV30":     "Equity",
    # "MSCIWORLD" removed: EUREX delisted, no active security def as of 2025-12
    "OMX":          "Equity",  # OMX Helsinki 25 (EUREX)

    # ── European Equities (MEFFRV) ────────────────────────────────────────
    "IBEX_mini":    "Equity",  # Spanish IBEX 35 mini

    # ── European Equities (Euronext/FTA) ──────────────────────────────────
    "AEX":          "Equity",
    "CAC":          "Equity",

    # ── Eurex Sector Indices ───────────────────────────────────────────────
    "EU-AUTO":      "Equity",
    "EU-BASIC":     "Equity",
    "EU-HEALTH":    "Equity",
    "EU-INSURE":    "Equity",
    "EU-OIL":       "Equity",
    "EU-TECH":      "Equity",
    "EU-TRAVEL":    "Equity",
    "EU-UTILS":     "Equity",

    # ── Asian Equities (SGX) ──────────────────────────────────────────────
    "FTSECHINAA":   "Equity",  # FTSE China A50
    "MSCISING":     "Equity",  # MSCI Singapore
    # "MSCIASIA" removed: EUREX delisted, no active security def as of 2024-12
    "FTSEINDO":     "Equity",  # FTSE Indonesia Index (SGX)
    "FTSEVIET":     "Equity",  # FTSE Vietnam 30 (SGX)

    # ── Asian Bonds (OSE.JPN / Osaka) ─────────────────────────────────────
    "JGB":          "Bond",    # Japanese Government Bond Mini (OSE.JPN)

    # ── Asian Equities (OSE.JPN / Osaka) ──────────────────────────────────
    "NIKKEI":       "Equity",  # Nikkei 225 Mini
    "NIKKEI400":    "Equity",  # JPX-Nikkei 400
    "TOPIX":        "Equity",  # TOPIX Mini
    "MUMMY":        "Equity",  # TSE Mothers (small-cap Japan)

    # ── Asian Equities (KSE) ──────────────────────────────────────────────
    "KOSPI":        "Equity",  # KOSPI 200
    "KOSDAQ":       "Equity",  # KOSDAQ 150

    # ── Other Asian (SGX) ─────────────────────────────────────────────────
    "FTSECHINAH":   "Equity",  # FTSE China H (SGX)
    "FTSETAIWAN":   "Equity",  # FTSE Taiwan (SGX)

    # ── Volatility (CFE / EUREX) ──────────────────────────────────────────
    "VIX":          "Vol",     # CBOE VIX (CFE)
    "V2X":          "Vol",     # Eurostoxx VIX (EUREX)

    # ── FX majors (CME — free via IDEAL FX waiver) ────────────────────────
    "AUD":          "FX",
    "CAD":          "FX",
    "CHF":          "FX",
    "EUR":          "FX",
    "GBP":          "FX",
    "JPY":          "FX",
    "NOK":          "FX",
    "NZD":          "FX",
    "SEK":          "FX",

    # ── FX crosses / EM (CME) ─────────────────────────────────────────────
    "GBPEUR":       "FX",
    "YENEUR":       "FX",
    "BRE":          "FX",      # Brazilian Real
    "CNH":          "FX",      # Chinese Yuan
    "INR":          "FX",      # Indian Rupee
    "MXP":          "FX",      # Mexican Peso
    "SGD":          "FX",      # Singapore Dollar (CME)
    "DX":           "FX",      # US Dollar Index (NYBOT)
    "TWD-mini":     "FX",      # Taiwan Dollar mini (SGX)

    # ── Metals (COMEX — free via Physical Metals waiver) ──────────────────
    "IRON":         "Metals",  # Iron Ore futures (SGX)
    "GOLD_micro":   "Metals",
    "SILVER":       "Metals",
    "COPPER":       "Metals",
    "PLAT":         "Metals",
    "PALLAD":       "Metals",
    "ALUMINIUM":    "Metals",  # COMEX aluminium

    # ── Crypto (CME) ──────────────────────────────────────────────────────
    "BITCOIN":      "Metals",   # grouped with metals per Carver convention
    "ETHEREUM":     "Metals",

    # ── Energy (NYMEX — free via Physical Metals/Comms waiver) ───────────
    "CRUDE_W_mini": "Energy",
    "BRENT-LAST":   "Energy",
    "GAS_US_mini":  "Energy",
    "GAS-LAST":     "Energy",
    "HEATOIL":      "Energy",
    "GASOILINE":    "Energy",

    # ── Ags (CBOT / NYBOT — free via Physical Metals/Comms waiver) ────────
    "CORN":         "Ags",
    "WHEAT":        "Ags",
    "SOYBEAN":      "Ags",
    "SOYMEAL":      "Ags",
    "SOYOIL":       "Ags",
    "OATIES":       "Ags",
    "REDWHEAT":     "Ags",
    "RICE":         "Ags",
    "LEANHOG":      "Ags",
    "LIVECOW":      "Ags",
    "FEEDCOW":      "Ags",
    "BBCOMM":       "Ags",     # Bloomberg Commodity Index
    "CHEESE":       "Ags",
    "COFFEE":       "Ags",     # Arabica coffee (NYBOT)
    "COCOA":        "Ags",     # Cocoa NY (NYBOT)
    "SUGAR11":      "Ags",     # Sugar No. 11 (NYBOT)
    "COTTON2":      "Ags",     # Cotton No. 2 (NYBOT)
    "OJ":           "Ags",     # Orange juice (NYBOT)
    "RUBBER":       "Ags",     # TSR 20 rubber (SGX)
}

# Instruments in UNIVERSE that are NOT in the JUMBO 99 (new candidates).
# These get no PST history — prices must be collected via pst_updater.py first.
from ibkr_fut.jumbo import JUMBO
UNIVERSE_ONLY: dict[str, str] = {
    k: v for k, v in UNIVERSE.items() if k not in JUMBO
}

# Asset class → set of instruments
def instruments_by_class(universe: dict[str, str] | None = None) -> dict[str, list[str]]:
    u = universe or UNIVERSE
    out: dict[str, list[str]] = {}
    for instr, cls in u.items():
        out.setdefault(cls, []).append(instr)
    return out


if __name__ == "__main__":
    from ibkr_fut.jumbo import JUMBO
    jumbo_set = set(JUMBO)
    universe_set = set(UNIVERSE)

    print(f"UNIVERSE:          {len(universe_set)} instruments")
    print(f"JUMBO (subset):    {len(jumbo_set)} instruments")
    print(f"In UNIVERSE only:  {len(universe_set - jumbo_set)} instruments")
    print(f"In JUMBO only:     {len(jumbo_set - universe_set)} instruments")
    print()

    print("Non-JUMBO instruments in UNIVERSE (new candidates):")
    for instr in sorted(universe_set - jumbo_set):
        print(f"  {instr:20} {UNIVERSE[instr]}")

    print()
    print("JUMBO instruments NOT in UNIVERSE (may lack subscribed exchange):")
    for instr in sorted(jumbo_set - universe_set):
        print(f"  {instr:20} {JUMBO[instr]}")

    print()
    by_class = instruments_by_class()
    print("Counts by asset class:")
    for cls, instrs in sorted(by_class.items()):
        print(f"  {cls:10} {len(instrs):3}  {sorted(instrs)}")
