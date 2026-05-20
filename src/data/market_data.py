"""
Fetch market data and save to Parquet.

Sources:
  Helion dashboard JSON — price data (TRL weekly/daily, TRE slots, spot)
  opendata.swiss SFOE   — weekly hydro reservoir levels by canton

Output files:
  data/raw/prices/
    trl_weekly.parquet   — one row per week × direction
    trl_daily.parquet    — one row per 4h block × direction
    tre_slots.parquet    — one row per 15-min slot × direction
    spot_hourly.parquet  — one row per hour
  data/raw/market/
    reservoir_levels.parquet — one row per week, fill rate per canton (%)

Note on TRE marginal prices: None (no activation) is treated as 0 CHF/MWh.
No activation means zero clearing revenue — the correct target for bid price forecasting.

Note on timestamps: Swissgrid publishes in Swiss local time (CET/CEST).
All times are stored as UTC by converting from Europe/Zurich.
"""

import io
import json
import logging
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "prices"
MARKET_DIR = ROOT / "data" / "raw" / "market"
DATA_URL = "https://raw.githubusercontent.com/Helion-Energy-Solution/Market-Dashboard/master/data/data.json"
RESERVOIR_URL = "https://www.uvek-gis.admin.ch/BFE/ogd/17/ogd17_fuellungsgrad_speicherseen.csv"
TZ_LOCAL = "Europe/Zurich"


def fetch_data() -> dict:
    log.info("Fetching market data from GitHub ...")
    with urllib.request.urlopen(DATA_URL) as r:
        data = json.loads(r.read())
    log.info("  processedAt: %s", data.get("processedAt"))
    return data


def _local_to_utc(series: pd.Series) -> pd.Series:
    # ambiguous=False → fall-back hour treated as standard time (CET, UTC+1)
    return series.dt.tz_localize(TZ_LOCAL, ambiguous=False, nonexistent="shift_forward").dt.tz_convert("UTC")


def parse_trl_weekly(data: dict) -> pd.DataFrame:
    rows = []
    for item in data["trlWeekly"]:
        week_start = pd.Timestamp(item["date"])
        for direction in ("up", "down"):
            d = item[direction]
            rows.append({
                "week_start":     week_start,
                "direction":      direction,
                "offered_mw":     d["offered"],
                "awarded_mw":     d["awarded"],
                "marginal_chf":   d["marginal"],
                "median_bid_chf": d["medianBid"],
                "award_rate_pct": d["awardRate"],
            })
    df = pd.DataFrame(rows)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df.sort_values(["week_start", "direction"]).reset_index(drop=True)


def parse_trl_daily(data: dict) -> pd.DataFrame:
    rows = []
    for item in data["trlDaily"]:
        date = item["date"]
        for block in item["blocks"]:
            start_str, end_str = block["block"].split("-")
            block_start = pd.Timestamp(f"{date} {start_str}")
            for direction in ("up", "down"):
                if direction not in block:
                    continue
                d = block[direction]
                rows.append({
                    "block_start":    block_start,
                    "direction":      direction,
                    "offered_mw":     d["offered"],
                    "awarded_mw":     d["awarded"],
                    "marginal_chf":   d["marginal"],
                    "median_bid_chf": d["medianBid"],
                    "award_rate_pct": d["awardRate"],
                })
    df = pd.DataFrame(rows)
    df["block_start"] = _local_to_utc(df["block_start"])
    return df.sort_values(["block_start", "direction"]).reset_index(drop=True)


def parse_tre_slots(data: dict) -> pd.DataFrame:
    rows = []
    for item in data["treSlots"]:
        dt = pd.Timestamp(f"{item['d']} {item['s']}")
        rows.append({
            "slot_time":        dt,
            "direction":        "pos",
            "offered":          item["po"],
            "activated":        item["pa"],
            "marginal_chf":     item["pm"] if item["pm"] is not None else 0.0,
            "activation_rate":  round(item["pa"] / item["po"], 4) if item["po"] else 0.0,
        })
        rows.append({
            "slot_time":        dt,
            "direction":        "neg",
            "offered":          item["no"],
            "activated":        item["na"],
            "marginal_chf":     item["nm"] if item["nm"] is not None else 0.0,
            "activation_rate":  round(item["na"] / item["no"], 4) if item["no"] else 0.0,
        })
    df = pd.DataFrame(rows)
    df["slot_time"] = _local_to_utc(df["slot_time"])
    return df.sort_values(["slot_time", "direction"]).reset_index(drop=True)


def parse_spot_hourly(data: dict) -> pd.DataFrame:
    rows = []
    for item in data["spotHourly"]:
        date = item["date"]
        for h, price in enumerate(item["h"]):
            rows.append({
                "hour_time":    pd.Timestamp(f"{date} {h:02d}:00"),
                "price_eur_mwh": price,
            })
    df = pd.DataFrame(rows)
    df["hour_time"] = _local_to_utc(df["hour_time"])
    return df.sort_values("hour_time").reset_index(drop=True)


def parse_reservoir_levels() -> pd.DataFrame:
    log.info("Fetching reservoir levels from opendata.swiss ...")
    with urllib.request.urlopen(RESERVOIR_URL) as r:
        raw = r.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    df["date"] = pd.to_datetime(df["Datum"])

    cantons = ["Wallis", "Graubuenden", "Tessin", "TotalCH"]
    out = pd.DataFrame({"date": df["date"]})
    for c in cantons:
        content = df[f"{c}_speicherinhalt_gwh"]
        capacity = df[f"{c}_max_speicherinhalt_gwh"]
        out[f"{c.lower()}_gwh"] = content.values
        out[f"{c.lower()}_fill_pct"] = (content / capacity * 100).round(2).values

    return out.sort_values("date").reset_index(drop=True)


def save_prices(df: pd.DataFrame, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    log.info("  Saved %d rows → %s", len(df), path.name)


def save_market(df: pd.DataFrame, name: str):
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    path = MARKET_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    log.info("  Saved %d rows → %s", len(df), path.name)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    data = fetch_data()

    log.info("Parsing TRL Weekly ...")
    save_prices(parse_trl_weekly(data), "trl_weekly")

    log.info("Parsing TRL Daily ...")
    save_prices(parse_trl_daily(data), "trl_daily")

    log.info("Parsing TRE Slots ...")
    save_prices(parse_tre_slots(data), "tre_slots")

    log.info("Parsing Spot Hourly ...")
    save_prices(parse_spot_hourly(data), "spot_hourly")

    log.info("Parsing Reservoir Levels ...")
    save_market(parse_reservoir_levels(), "reservoir_levels")

    log.info("Done.")


if __name__ == "__main__":
    main()
