#!/usr/bin/env python3
"""
Build Data/settlement_times.csv for the full PST futures universe (584 instruments).

Settlement time = time the exchange publishes the official daily settlement price
used for margin and signal generation. This drives:
  - Order scheduling (trade after settlement so signals use final prices)
  - PST data freshness checks (is today's bar actually available yet?)
  - Multi-timezone scheduling when trading non-US instruments

Sources (primary):
  CME equity index: daily settlement at 14:59:30–15:00:00 CT (VWAP 30-sec window)
    https://www.cmegroup.com/education/courses/introduction-to-equity-index-products/
    understanding-equity-index-daily-and-final-settlement
  CME treasury: 13:59:30–14:00:00 CT
    https://www.cmegroup.com/rulebook/CBOT/II/18.pdf (Chapter 18 T-Bond)
  CME grains: 13:14:00–13:15:00 CT
    https://www.cmegroup.com/trading/agricultural/files/daily-grains-settlement-procedure.pdf
  CME livestock: 12:59:30–13:00:00 CT
    https://www.cmegroup.com/market-data/settlements/files/cme-livestock-futures-daily-settlement.pdf
  CME FX / STIR: 13:59:00–14:00:00 CT
    https://www.cmegroup.com/trading/fx/files/daily-fx-settlements.pdf
  CME crypto: 14:59:00–15:00:00 CT (Globex window)
    https://www.cmegroup.com/articles/faqs/frequently-asked-questions-cryptocurrency-futures.html
  COMEX gold: 13:29:00–13:30:00 ET
  COMEX silver: 12:24:00–12:25:00 CT (= 13:25 ET)
  COMEX copper: 12:59:00–13:00:00 ET
  COMEX aluminium: ~12:00 ET (cash-settles to LME; LME session ends ~17:00 London = 12:00 ET)
  NYMEX energy: 14:28:00–14:30:00 ET
    https://www.cmegroup.com/trading/energy/files/NYMEX_Energy_Futures_Daily_Settlement_Procedure.pdf
  NYMEX platinum: 13:03:00–13:05:00 ET
  NYMEX palladium: 12:58:00–13:00:00 ET
  NYMEX Brent (BZ): cash-settles to ICE Brent, which uses VWAP 2-min window ending 19:30 London
    = 14:30 ET in both summer and winter (coincidentally same as NYMEX energy)
  ICE/NYBOT settlement windows (Feb 2026):
    https://www.ice.com/publicdocs/futures_us/Settlement_Window.pdf
    KC (coffee): 12:53–12:55 ET
    SB (sugar 11): 12:53–12:55 ET
    CC (cocoa): 11:48–11:50 ET
    CT (cotton): 14:14–14:15 ET
    OJ (FCOJ-A): 13:29–13:30 ET
    DX (dollar index): 14:59–15:00 ET
  CFE/VIX: changed Aug 2024 to 15:00 CT (was 16:15 CT)
    https://cdn.cboe.com/resources/release_notes/2024/CFE-Updates-VX-and-VXM-Futures-Daily-Settlement-Process.pdf
  EUREX equity index (DAX/ESTX50 etc.): VWAP 1 min before 17:30 CET
    https://www.eurex.com/ex-en/markets/idx/dax/DAX-Futures-139902
  EUREX fixed income (Bund/Bobl/Schatz/OAT/BTP etc.): VWAP 1 min before 17:15 CET
    https://www.eurex.com/ex-en/markets/int/long-term-interest-rates/fix/government-bonds/Euro-Bund-Futures-137298
  KSE equity (KOSPI/KOSDAQ): 15:45 KST; KTB bonds: 15:30 KST
    https://marketswiki.com/wiki/KRX_KOSPI_200_Index_futures
  OSE.JPN equity (Nikkei/TOPIX): 15:25 JST; JGB: 15:30 JST
    https://www.jpx.co.jp/english/derivatives/products/domestic/225mini/01.html
  SGX: per-product (A50 16:00, Taiwan 16:15, Singapore 17:15, rubber 18:00, etc.)
    https://rulebook.sgx.com/rulebook/practice-note-71112-daily-settlement-price-methodology

Run from repo root:
    python3 ibkr_fut/build_settlement_times.py

Output:
    Data/settlement_times.csv
"""

import os
import pandas as pd
from datetime import datetime, date
from zoneinfo import ZoneInfo

IB_CONFIG   = "Data/pst/ib_config/ib_config_futures.csv"
OUTPUT_FILE = "Data/settlement_times.csv"

# Representative dates for DST offset computation
WINTER_DATE = date(2025, 1, 15)   # EST (UTC-5)
SUMMER_DATE = date(2025, 7, 15)   # EDT (UTC-4)
ET_TZ = ZoneInfo("America/New_York")


def local_to_et(time_str: str, iana_tz: str, ref_date: date) -> str:
    """Convert HH:MM local time to HH:MM ET on the given date."""
    h, m = map(int, time_str.split(":"))
    local_dt = datetime(ref_date.year, ref_date.month, ref_date.day, h, m,
                        tzinfo=ZoneInfo(iana_tz))
    et_dt = local_dt.astimezone(ET_TZ)
    return et_dt.strftime("%H:%M")


# ── Non-CME exchange settlement rules ────────────────────────────────────────
# (settlement_time_local HH:MM, IANA tz, notes)
# Exchanges with per-instrument variation are handled in get_settlement() below.

EXCHANGE_RULES: dict[str, tuple[str, str, str]] = {
    # ── US ───────────────────────────────────────────────────────────────────
    # CFE: VIX/VXM daily settlement changed from 16:15 CT to 15:00 CT in Aug 2024.
    # Source: https://cdn.cboe.com/resources/release_notes/2024/CFE-Updates-VX-and-VXM-Futures-Daily-Settlement-Process.pdf
    "CFE":       ("15:00", "America/Chicago",
                  "CBOE Futures Exchange — VIX/VXM daily settlement at 15:00 CT "
                  "(changed from 16:15 CT in Aug 2024)"),
    "SMFE":      ("17:15", "America/Chicago",
                  "Small Exchange — tracks CME equity session, settles at 17:15 CT"),
    "ICECRYPTO": ("16:00", "America/New_York",
                  "ICE Crypto (BAKKT) — 16:00 ET daily settlement"),
    "CDE":       ("15:00", "America/Toronto",
                  "Montreal Exchange — rate/bond futures 15:00 ET; equity sectors 16:15 ET"),
    "MEXDER":    ("15:00", "America/Mexico_City",
                  "Mexican Derivatives Exchange — IPC, CETES; 15:00 CT Mexico"),
    # ── European ─────────────────────────────────────────────────────────────
    # EUREX is handled per-symbol in get_settlement() (equity 17:30, bonds 17:15)
    "ICEEU":     ("16:30", "Europe/London",
                  "ICE Futures Europe — FTSE100, FTSE250, Gilts, MSCI indices, "
                  "single-stock futures, SONIA rates, Swiss rates"),
    "ICEEUSOFT": ("16:30", "Europe/London",
                  "ICE Futures Europe (soft commodities) — Robusta coffee, white sugar, wheat"),
    "NYSELIFFE": ("16:30", "Europe/London",
                  "NYSE Liffe London — MSCI index futures"),
    "LMEOTC":    ("13:20", "Europe/London",
                  "London Metal Exchange official settlement (Ring trading close) — "
                  "aluminium, copper, lead, nickel, tin, zinc"),
    "IPE":       ("19:30", "Europe/London",
                  "ICE/IPE London energy — Brent crude (COIL), gasoil, heating oil, gasoline; "
                  "19:30 UK is end of main electronic session; daily settlement ~19:30"),
    "MONEP":     ("17:35", "Europe/Paris",
                  "Euronext Paris — CAC40; DSP published at 17:35 CET based on cash close ~17:30"),
    "MATIF":     ("18:30", "Europe/Paris",
                  "Euronext MATIF — European grains (corn EMA, rapeseed ECO, milling wheat EBM); "
                  "French OAT bond settles at 17:30"),
    "FTA":       ("17:30", "Europe/Amsterdam",
                  "Euronext Amsterdam — AEX index; DSP based on cash market close ~17:30 CET"),
    "ENDEX":     ("18:00", "Europe/Amsterdam",
                  "ICE Endex — Dutch TTF natural gas, UK NBP gas"),
    "OMS":       ("18:30", "Europe/Stockholm",
                  "Nasdaq Stockholm — OMX Stockholm 30, OMX Helsinki, OMX ESG"),
    "BELFOX":    ("17:30", "Europe/Brussels",
                  "Euronext Brussels — BEL20"),
    "MEFFRV":    ("17:30", "Europe/Madrid",
                  "MEFF Spanish Exchange — IBEX35 mini; VWAP 17:29–17:30 CET"),
    "IDEM":      ("17:40", "Europe/Rome",
                  "Italian Derivatives Market — FTSE MIB, MIB dividend"),
    # ── Asian ────────────────────────────────────────────────────────────────
    "HKFE":      ("16:30", "Asia/Hong_Kong",
                  "HKEX Futures Exchange — Hang Seng, H-shares, China A50, "
                  "MSCI indices, CNH, single-stock futures; regular session 9:00-16:30 HKT"),
    # OSE.JPN and KSE are handled per-symbol in get_settlement()
    # SGX is handled per-symbol in get_settlement()
    "NSE":       ("15:30", "Asia/Kolkata",
                  "NSE India — NIFTY50, Bank Nifty, USD/INR, EUR/INR, GBP/INR, JPY/INR; "
                  "09:15-15:30 IST equity / 09:00-17:00 IST currency"),
    "SNFE":      ("16:30", "Australia/Sydney",
                  "ASX/SFE Sydney — SPI200 ASX equity index, wheat, barley; "
                  "10:00-16:30 AEST (UTC+10); AEDT (UTC+11) Oct-Apr"),
}


# ── EUREX per-symbol routing ──────────────────────────────────────────────────
# Equity index: VWAP 1 min before 17:30 CET
# Fixed income / rates: VWAP 1 min before 17:15 CET
# Source: https://www.eurex.com (product pages linked in header)

EUREX_BOND_SYMBOLS = {
    # German sovereign
    "GBL",              # Euro-Bund (10yr)
    "GBM",              # Euro-Bobl (5yr)
    "GBS",              # Euro-Schatz (2yr)
    "GBX",              # Euro-Buxl (30yr)
    # French
    "OAT",              # French OAT 10yr
    # Italian
    "BTP", "BTM", "BTS",  # BTP 10yr, 5yr, 3yr
    # Spanish
    "FBON",             # Spanish Bono 10yr
    # Swiss
    "CONF",             # Swiss 10yr Conf
    # Short-term rates (EURIBOR follows bond session timing)
    "EU3",              # 3-month EURIBOR
}


def eurex_settlement(ib_symbol: str) -> tuple[str, str, str]:
    s = ib_symbol.upper()
    if s in EUREX_BOND_SYMBOLS:
        return ("17:15", "Europe/Berlin",
                "EUREX fixed income futures — VWAP 1 min before 17:15 CET/CEST "
                "(Bund/Bobl/Schatz/Buxl/OAT/BTP/Bono/Conf/EURIBOR)")
    return ("17:30", "Europe/Berlin",
            "EUREX equity index futures — VWAP 1 min before 17:30 CET/CEST "
            "(DAX, EURO STOXX 50, sector indices, SMI, OMX, V2X, MSCI)")


# ── KSE per-symbol routing ────────────────────────────────────────────────────
# Equity index futures: 15:45 KST (KOSPI 200, KOSDAQ 150)
# KTB bond futures: 15:30 KST (closing call auction in final 10 min)
# Source: marketswiki.com/wiki/KRX_KOSPI_200_Index_futures

KSE_BOND_SYMBOLS = {"3KTB", "FLKTB"}


def kse_settlement(ib_symbol: str) -> tuple[str, str, str]:
    s = ib_symbol.upper()
    if s in KSE_BOND_SYMBOLS:
        return ("15:30", "Asia/Seoul",
                "Korea Exchange — KTB bond futures (3yr/10yr); "
                "day session closes 15:30 KST with call auction")
    return ("15:45", "Asia/Seoul",
            "Korea Exchange — KOSPI 200/KOSDAQ 150 equity futures; "
            "day session closes 15:45 KST (last trading day 14:50)")


# ── OSE.JPN per-symbol routing ────────────────────────────────────────────────
# Equity index (Nikkei 225 mini, TOPIX mini, Nikkei 400, TSE Mothers): 15:25 JST
# JGB futures: 15:30 JST (day session with call auction 15:28–15:30)
# Source: https://www.jpx.co.jp/english/derivatives/products/domestic/225mini/01.html

OSE_JGB_SYMBOLS = {"JGB"}


def ose_settlement(ib_symbol: str) -> tuple[str, str, str]:
    s = ib_symbol.upper()
    if s in OSE_JGB_SYMBOLS:
        return ("15:30", "Asia/Tokyo",
                "Osaka Exchange — JGB futures; day session closes 15:30 JST "
                "with call auction 15:28–15:30 JST")
    return ("15:25", "Asia/Tokyo",
            "Osaka Exchange — Nikkei 225 mini / TOPIX mini / Nikkei 400 / TSE Mothers; "
            "day session closes 15:25 JST (closing call auction ends 15:25)")


# ── SGX per-symbol routing ────────────────────────────────────────────────────
# Source: https://rulebook.sgx.com/rulebook/practice-note-71112-daily-settlement-price-methodology
# SGX contract specs: sgx.com/derivatives/products

SGX_SETTLEMENT: dict[str, tuple[str, str, str]] = {
    # China A50: T session ends when China market closes (~16:00 SGT / 16:00 CST same clock)
    "XINA50": ("16:00", "Asia/Singapore",
               "SGX FTSE China A50 — T session closes ~16:00 SGT (China A-share market close)"),
    # FTSE Taiwan: T session pre-close at ~16:15 SGT
    "TWN":    ("16:15", "Asia/Singapore",
               "SGX FTSE Taiwan RIC Capped — T session closes ~16:15 SGT"),
    # FTSE China H50: closes with HK market ~16:00 SGT (HKT = SGT)
    "XIN0I":  ("16:00", "Asia/Singapore",
               "SGX FTSE China H50 — T session closes ~16:00 SGT (HK market close)"),
    # FTSE Vietnam: ~16:00 SGT
    "FIVNM30": ("16:00", "Asia/Singapore",
                "SGX FTSE Vietnam 30 — T session closes ~16:00 SGT"),
    # FTSE Indonesia: ~16:15 SGT
    "WIIDN":  ("16:15", "Asia/Singapore",
               "SGX FTSE Indonesia — T session closes ~16:15 SGT (Jakarta market close)"),
    # MSCI Singapore: T session closes 17:15 SGT
    "SSG":    ("17:15", "Asia/Singapore",
               "SGX MSCI Singapore Free — T session closes 17:15 SGT"),
    # SGD futures: FX, closes ~17:00 SGT
    "SND":    ("17:00", "Asia/Singapore",
               "SGX SGD/USD futures — FX, T session closes ~17:00 SGT"),
    # USD/CNH: cash-settles to TMA HK fixing at 11:15 HKT (= 11:15 SGT, same UTC+8)
    "UC":     ("11:15", "Asia/Singapore",
               "SGX USD/CNH — cash-settles to TMA HK official fixing at 11:15 HKT/SGT"),
    # TWD mini: settles to Taiwan CB rate; TWD fixing ~13:30-13:45 SGT
    "TD":     ("13:45", "Asia/Singapore",
               "SGX TWD/USD mini — settles to Taiwan central bank rate ~13:45 SGT"),
    # SICOM Rubber TSR20: T session runs until 18:00 SGT
    "TSR20":  ("18:00", "Asia/Singapore",
               "SGX SICOM Rubber TSR20 — T session closes 18:00 SGT"),
    # Iron ore (SGX/IODEX): monthly-average cash settlement using daily Platts IODEX
    # assessments published at 17:30 SGT; use 17:30 as proxy for daily mark time
    "SCI":    ("17:30", "Asia/Singapore",
               "SGX Iron Ore (Platts IODEX) — monthly average cash settlement; "
               "Platts daily assessment published ~17:30 SGT; no single daily settlement fix"),
}

SGX_DEFAULT = ("17:30", "Asia/Singapore",
               "SGX — default T session close ~17:30 SGT")


def sgx_settlement(ib_symbol: str) -> tuple[str, str, str]:
    return SGX_SETTLEMENT.get(ib_symbol.upper(), SGX_DEFAULT)


# ── CME Group settlement logic ────────────────────────────────────────────────

def cme_settlement(ib_symbol: str, exchange: str) -> tuple[str, str, str]:
    """Return (time_local, iana_tz, notes) for any CME-group instrument."""
    s = ib_symbol.upper()
    tz = "America/Chicago"

    # ─── CBOT treasuries: 14:00 CT ───────────────────────────────────────────
    # Settlement window: 13:59:30–14:00:00 CT. ZQ (fed funds) also 14:00 CT.
    # Source: CBOT Chapter 18 rulebook; CME Confluence Treasury Wiki
    CBOT_TREAS = {
        "ZN","ZB","ZF","ZT","TN","UB","Z3N","ZQ","ZG",
        "20U","25U","30U","35U","40U","45U","50U",
        "N1U","T1U","F1U","B1U",
        "LIY","LIT","LIC","LID","LIW","LIO","LIL","LII","LIB",
        "10Y","2YY","5YY","30Y",
        "10YSME","2YSME","30YSME","5YSME",
    }
    if s in CBOT_TREAS:
        return ("14:00", tz,
                "CBOT treasury / UMBS / IRS / fed-funds futures — "
                "settlement window 13:59:30–14:00:00 CT")

    # ─── CBOT agricultural: 13:15 CT ─────────────────────────────────────────
    # Settlement window: 13:14:00–13:15:00 CT (VWAP of Globex + pit calendar spreads).
    # Source: https://www.cmegroup.com/trading/agricultural/files/daily-grains-settlement-procedure.pdf
    CBOT_AG = {
        "ZC","YC",           # corn
        "ZS","YK",           # soybeans
        "ZM",                # soymeal
        "ZL",                # soyoil
        "ZW","YW","KE",      # wheat (CBOT soft red, HRW Kansas)
        "ZO",                # oats
        "ZR",                # rough rice
        "AC",                # ethanol
        "EMA","TGCN",        # European corn, Japanese corn
        "DJUBS",             # DJ-UBS commodity index
    }
    if s in CBOT_AG:
        return ("13:15", tz,
                "CBOT agricultural futures (grains/oilseeds) — "
                "settlement window 13:14:00–13:15:00 CT")

    # ─── CBOT Bloomberg Commodity Index: 16:00 CT ────────────────────────────
    # Settles to the Bloomberg Commodity Index official closing value published by ~16:00 CT.
    # Source: CBOT Chapter 29 rulebook
    if s in {"AIGCI"}:
        return ("16:00", tz,
                "CBOT Bloomberg Commodity Index futures — "
                "settles to Bloomberg official closing value, deadline 16:00 CT")

    # ─── CBOT US real estate index ────────────────────────────────────────────
    if s in {"DJUSRE"}:
        return ("16:15", tz, "CBOT US Real Estate index futures — 16:15 CT")

    # ─── CME equity index: 15:00 CT ──────────────────────────────────────────
    # Settlement window: 14:59:30–15:00:00 CT for ALL US equity index futures including Dow.
    # Source: CME Group education page; CBOT Chapter 27 (Dow) rulebook
    CME_EQUITY = {
        "ES","MES",                             # S&P 500
        "NQ","MNQ","QCN",                       # Nasdaq-100
        "RTY","M2K","RS1","RSG","RUO","RUJ",    # Russell 2000
        "EMD",                                  # S&P MidCap 400
        "RSV",                                  # Russell 1000 Value
        "YM","MYM","DJIA",                      # Dow Jones (CBOT-listed, same 15:00 CT window)
        "NKD","NIY",                            # Nikkei (CME USD/JPY)
        "IBAA","IBOV",                          # Bovespa on CME
        "GSCI","SPGSCIP",                       # GSCI commodity indices
        "IBXXIBHY","IBXXIBIG",                  # HY/IG bond indices
        "M1CNX",                                # MSCI commodity
        "IXY","IXE","SIXM","IXV","IXI","SPSIINS","IXB","SPSIOP","SIXRE","SPSIRBK",
        "SPSIRE","SPSOX","IXR","SIXT","IXU","SPSIBI",
        "SMC","SGX","SVX","BQX",
        "MXEA","M1EA","MXEF","M1EF",
        "NASBIO-MINI",
        "BOS","CHI","WDC","DEN","LAX","LAV","MIA","NYM","SDG","SFR","CUS",
        "D0","D2","H4",
        "VOLQ","NQH2O",
        "LB","LBR",
    }
    if s in CME_EQUITY or (exchange == "CME" and s.startswith("SP")):
        return ("15:00", tz,
                "CME/CBOT equity index futures — "
                "settlement window 14:59:30–15:00:00 CT (VWAP 30-sec window)")

    # ─── CME FX: 14:00 CT ────────────────────────────────────────────────────
    # Settlement window: 13:59:30–14:00:00 CT.
    # Source: https://www.cmegroup.com/trading/fx/files/daily-fx-settlements.pdf
    CME_FX = {
        "EUR","M6E","E7","EO",
        "GBP","M6B","MP","GB",
        "JPY","M6J","J7","UY","UJ",
        "CHF","MSF",
        "AUD","M6A","AU",
        "CAD","MCD",
        "NZD",
        "NOK","SEK","PLN","PLZ",
        "MXP","6M",
        "SIR","MIR","IU","INR","INX",
        "CNH","RMB","MNH","MCS","MUC","UC","CY",
        "BRE",
        "RY",
        "EJ",
        "SND","US",
        "ACD","AJY","CJY","PJY","SJY","EAD","ECD","RF","ECK","EHF",
        "PSF","RP","RME","CLP","IRS","ILS",
        "CZK","HUF",
        "RUR","ZAR","KWY","KRW","KRW2","KU","KJ",
        "TWD","TD",
        "THB","SGXTU",
        "USDKRW","USDINR","USDMXP","DEUA",
        "USDCAD","USDCHF","USDCNH","CNH2",
        "M6C","M6S",
        "AUDCAD","AUDJPY","CADJPY","CHFJPY","EURCAD","EURINR","EURMXP",
        "EURAUD","EURCHF","EURCZK","EURHUF","GBPCHF","GBPEUR","GBPINR","GBPJPY",
        "PLZEUR","HUFEUR","CNHEUR",
        "SARONA",
    }
    if s in CME_FX or exchange == "ICEUS":
        return ("14:00", tz,
                "CME/ICEUS FX futures — settlement window 13:59:30–14:00:00 CT")

    # ─── CME interest rates / STIR: 14:00 CT ─────────────────────────────────
    # Source: CME Chapter 460 (SOFR rulebook)
    CME_RATES = {
        "GE","SR3","SOFR3","SOFR1",
        "SONIA","SONIA1","SONIA3","SONIO.N","SONIA.N",
        "AMB30","AMT3M","AMT1S",
        "BB3M","BSBY",
        "LIBOR1","EM",
        "BAX","CRA",
        "CORRA",
        "IB",
        "CETES",
    }
    if s in CME_RATES:
        return ("14:00", tz,
                "CME/CDE STIR and overnight rate futures — "
                "settlement window 13:59:00–14:00:00 CT")

    # ─── CME livestock: 13:00 CT ─────────────────────────────────────────────
    # Settlement window: 12:59:30–13:00:00 CT.
    # Source: https://www.cmegroup.com/market-data/settlements/files/cme-livestock-futures-daily-settlement.pdf
    if s in {"HE","LE","GF","DA","MILK"}:
        return ("13:00", tz,
                "CME livestock futures (live cattle, lean hogs, feeder cattle) — "
                "settlement window 12:59:30–13:00:00 CT")

    # ─── CME dairy: 13:10 CT ─────────────────────────────────────────────────
    # Settlement window: 13:09:30–13:10:00 CT.
    if s in {"CSC","CB","GDK","NF","WHEY","DY","GDC"}:
        return ("13:10", tz,
                "CME dairy futures (cheese, butter, dry whey) — "
                "settlement window 13:09:30–13:10:00 CT")

    # ─── CME crypto: 15:00 CT ────────────────────────────────────────────────
    # Globex settlement window: 14:59:00–15:00:00 CT.
    # Note: CME CF reference rate published at 16:00 London is used for final/expiry settlement,
    # but daily settlement uses the Globex 15:00 CT window.
    if s in {"MBT","BTC","ETHUSDRR","MET","ETHEURRR","BTCEURRR","BRR","BRREUR"}:
        return ("15:00", tz,
                "CME crypto futures (micro Bitcoin, micro Ether) — "
                "Globex settlement window 14:59:00–15:00:00 CT")

    # ─── COMEX metals ─────────────────────────────────────────────────────────
    if exchange == "COMEX":
        if s in {"GC","MGC","QO","SGUF","SGC","GOLD","GOLD_MICRO",
                 "GDR","GDU","GOLD_CHINA","GOLD_CHINA_USD"}:
            return ("13:30", "America/New_York",
                    "COMEX gold (GC/MGC) — settlement window 13:29:00–13:30:00 ET; "
                    "LBMA PM gold fix reference")
        if s in {"SI","AGS","QI"}:
            # 12:24–12:25 CT = 13:25 ET (both EST and EDT — CT and ET differ by 1h all year)
            return ("13:25", "America/New_York",
                    "COMEX silver (SI) — settlement window 12:24:00–12:25:00 CT = 13:25 ET")
        if s in {"HG","QC","MHG"}:
            return ("13:00", "America/New_York",
                    "COMEX copper (HG) — settlement window 12:59:00–13:00:00 ET")
        if s in {"ALI","AH","ALUMINIUM"}:
            # ALI cash-settles to LME aluminium. LME Ring trading closes ~13:00 London
            # and the official LME settlement is published at ~17:00 London = ~12:00 ET.
            return ("12:00", "America/New_York",
                    "COMEX aluminium (ALI) — cash-settles to LME official settlement; "
                    "LME publishes official settlement ~17:00 London = ~12:00 ET")
        if s in {"TIO","IRON_CME"}:
            return ("13:00", "America/New_York", "COMEX iron ore — 13:00 ET")
        if s in {"HRC","STEEL"}:
            return ("13:00", "America/New_York", "COMEX US steel HRC — 13:00 ET")
        if s in {"UX","URANIUM"}:
            return ("13:00", "America/New_York", "COMEX uranium futures — 13:00 ET")
        return ("13:30", "America/New_York",
                "COMEX metals (default: gold settlement time)")

    # ─── NYMEX ────────────────────────────────────────────────────────────────
    if exchange == "NYMEX":
        if s in {"PL","PLAT"}:
            # 13:03–13:05 ET
            return ("13:05", "America/New_York",
                    "NYMEX platinum (PL) — settlement window 13:03:00–13:05:00 ET")
        if s in {"PA","PALLAD"}:
            # 12:58–13:00 ET
            return ("13:00", "America/New_York",
                    "NYMEX palladium (PA) — settlement window 12:58:00–13:00:00 ET")
        if s in {"LT","PIPELINE"}:
            return ("14:30", "America/New_York", "NYMEX pipeline gas basis — 14:30 ET")
        if s in {"AYV","MARS_ARGUS"}:
            return ("14:30", "America/New_York", "NYMEX MARS crude differential — 14:30 ET")
        # Brent Last (BZ): cash-settles to ICE Brent, which uses VWAP ending 19:30 London time.
        # 19:30 London = 14:30 ET in both winter (UTC+0) and summer (BST UTC+1 → 14:30 EDT).
        # So BZ is coincidentally same clock time as NYMEX energy, but follows ICE's schedule.
        if s in {"BZ"}:
            return ("14:30", "America/New_York",
                    "NYMEX Brent Last (BZ) — cash-settles to ICE Brent; "
                    "ICE VWAP window ends 19:30 London = 14:30 ET (both EST and EDT)")
        return ("14:30", "America/New_York",
                "NYMEX energy futures (CL, NG, RB, HO, QM, QG, HH) — "
                "settlement window 14:28:00–14:30:00 ET")

    # ─── NYBOT (ICE Futures US) ───────────────────────────────────────────────
    # Source: https://www.ice.com/publicdocs/futures_us/Settlement_Window.pdf (Feb 2026)
    if exchange == "NYBOT":
        if s in {"KC"}:
            return ("12:55", "America/New_York",
                    "ICE/NYBOT coffee 'C' (KC) — settlement window 12:53–12:55 ET")
        if s in {"CC","C"}:
            return ("11:50", "America/New_York",
                    "ICE/NYBOT cocoa (CC) — settlement window 11:48–11:50 ET")
        if s in {"CT","TT"}:
            return ("14:15", "America/New_York",
                    "ICE/NYBOT cotton No.2 (CT) — settlement window 14:14–14:15 ET")
        if s in {"SB","SUGAR11"}:
            return ("12:55", "America/New_York",
                    "ICE/NYBOT sugar No.11 (SB) — settlement window 12:53–12:55 ET")
        if s in {"SF","SUGAR16"}:
            return ("14:00", "America/New_York",
                    "ICE/NYBOT sugar No.16 domestic (SF) — 14:00 ET")
        if s in {"OJ"}:
            return ("13:30", "America/New_York",
                    "ICE/NYBOT FCOJ-A orange juice (OJ) — settlement window 13:29–13:30 ET")
        if s in {"DX"}:
            return ("15:00", "America/New_York",
                    "ICE/NYBOT US Dollar Index (DX) — settlement window 14:59–15:00 ET")
        if s in {"RS","CANOLA"}:
            return ("16:00", "America/New_York",
                    "ICE Canada canola — 16:00 ET")
        if s in {"NYFANG","FANG"}:
            return ("16:15", "America/Chicago", "ICE/NYBOT FANG index — 16:15 CT")
        if s in {"LRC30APR","LRJ30APR"}:
            return ("15:00", "America/New_York",
                    "ICE/NYBOT 30-yr mortgage rate futures — 15:00 ET")
        return ("14:00", "America/New_York", "ICE/NYBOT default")

    # ─── ICEUS FX ─────────────────────────────────────────────────────────────
    if exchange == "ICEUS":
        return ("14:00", "America/Chicago",
                "ICE Futures US FX — same settlement window as CME FX, 14:00 CT")

    # ─── Fallback ─────────────────────────────────────────────────────────────
    return ("16:15", tz, f"CME fallback (exchange={exchange}, symbol={s})")


CME_EXCHANGES = {"CME", "CBOT", "COMEX", "NYMEX", "NYBOT", "ICEUS"}
PER_SYMBOL_EXCHANGES = {"EUREX", "KSE", "OSE.JPN", "SGX"}


def get_settlement(row: pd.Series) -> tuple[str, str, str]:
    """Return (time_local, iana_tz, notes) for one IB config row."""
    exchange = str(row["IBExchange"]).strip()
    symbol   = str(row["IBSymbol"]).strip()

    if exchange in CME_EXCHANGES:
        return cme_settlement(symbol, exchange)

    if exchange == "EUREX":
        return eurex_settlement(symbol)

    if exchange == "KSE":
        return kse_settlement(symbol)

    if exchange == "OSE.JPN":
        return ose_settlement(symbol)

    if exchange == "SGX":
        return sgx_settlement(symbol)

    if exchange in EXCHANGE_RULES:
        t, tz, note = EXCHANGE_RULES[exchange]
        return (t, tz, note)

    return ("16:00", "UTC", f"UNKNOWN exchange {exchange} — defaulted to 16:00 UTC")


def build() -> pd.DataFrame:
    df = pd.read_csv(IB_CONFIG)

    records = []
    for _, row in df.iterrows():
        t_local, tz, notes = get_settlement(row)
        et_winter = local_to_et(t_local, tz, WINTER_DATE)
        et_summer = local_to_et(t_local, tz, SUMMER_DATE)

        h, m = map(int, t_local.split(":"))
        local_dt = datetime(WINTER_DATE.year, WINTER_DATE.month, WINTER_DATE.day,
                            h, m, tzinfo=ZoneInfo(tz))
        utc_dt   = local_dt.astimezone(ZoneInfo("UTC"))
        utc_str  = utc_dt.strftime("%H:%M")

        records.append({
            "instrument":            row["Instrument"],
            "ib_symbol":             row["IBSymbol"],
            "exchange":              row["IBExchange"],
            "ib_currency":           row["IBCurrency"],
            "settlement_time_local": t_local,
            "timezone":              tz,
            "settlement_time_utc":   utc_str,
            "settlement_time_est":   et_winter,
            "settlement_time_edt":   et_summer,
            "notes":                 notes,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("Data", exist_ok=True)
    out = build()
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Written {len(out)} instruments → {OUTPUT_FILE}")

    print("\nSettlement times by exchange (ET winter / summer):")
    summary = (
        out.groupby(["exchange", "settlement_time_est", "settlement_time_edt"])
           .size()
           .reset_index(name="n")
           .sort_values(["settlement_time_est", "exchange"])
    )
    for _, r in summary.iterrows():
        print(f"  {r['exchange']:12}  {r['settlement_time_est']} EST / "
              f"{r['settlement_time_edt']} EDT  ({r['n']} instruments)")

    unknown = out[out["notes"].str.startswith("UNKNOWN")]
    if not unknown.empty:
        print(f"\nWARNING: {len(unknown)} instruments with unknown exchange:")
        print(unknown[["instrument", "exchange"]].to_string(index=False))
