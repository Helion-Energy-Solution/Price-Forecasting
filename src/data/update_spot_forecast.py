"""
Download Volue Swiss day-ahead spot price forecast instances and save to parquet.

Curve  : pri ch spot merged €/mwh cet h f  (INSTANCES, hourly, CET, EUR/MWh)
Output : data/raw/market/spot_forecast_volue.parquet
Columns: issue_date (datetime64[us, UTC]), delivery_hour (datetime64[us, UTC]),
         price_eur_mwh (float64)

Access  : Volue subscription covers issue_dates from 2026-01-01 onward.
Batching: search_instances with with_data=True is called per calendar day to
          stay within the API size limit (~24 instances × 2160 rows each).

Run:
    python src/data/update_spot_forecast.py
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

ROOT       = Path(__file__).resolve().parents[2]
OUT        = ROOT / "data" / "raw" / "market" / "spot_forecast_volue.parquet"
CURVE_NAME = "pri ch spot merged €/mwh cet h f"
# Subscription access starts 2026-01-01 (accessRange.begin from the API).
ACCESS_START = pd.Timestamp("2026-01-01", tz="CET")


def _connect():
    import volue_insight_timeseries as vit
    return vit.Session(
        client_id=os.environ["VOLUE_CLIENT_ID"],
        client_secret=os.environ["VOLUE_CLIENT_SECRET"],
    )


def _last_stored_issue() -> pd.Timestamp:
    """Return latest issue_date already in parquet, or ACCESS_START if none."""
    if not OUT.exists():
        return ACCESS_START
    df = pd.read_parquet(OUT, columns=["issue_date"])
    return pd.Timestamp(df["issue_date"].max()).tz_convert("CET")


def _instance_to_df(instance) -> pd.DataFrame | None:
    """Convert a Volue TS instance to a tidy (issue_date, delivery_hour, price) DataFrame."""
    ts = instance.to_pandas()
    if ts is None or ts.empty:
        return None
    df = ts.to_frame("price_eur_mwh").reset_index()
    df.columns = ["delivery_hour", "price_eur_mwh"]
    df["issue_date"] = instance.issue_date
    return df[["issue_date", "delivery_hour", "price_eur_mwh"]]


def _fetch_day(curve, day_start: pd.Timestamp) -> list[pd.DataFrame]:
    """Fetch all instances issued on a single calendar day (CET) with data."""
    day_end = day_start + pd.Timedelta(days=1)
    try:
        instances = curve.search_instances(
            issue_date_from=day_start,
            issue_date_to=day_end,
            with_data=True,
        )
    except Exception as exc:
        log.warning("  Day %s: API error — %s", day_start.date(), exc)
        return []
    frames = [_instance_to_df(inst) for inst in instances]
    return [f for f in frames if f is not None]


def main():
    session = _connect()
    curve   = session.get_curve(name=CURVE_NAME)

    from_date = _last_stored_issue()
    to_date   = pd.Timestamp.now(tz="CET")

    if from_date >= to_date:
        log.info("Already up to date (latest issue: %s).", from_date)
        return

    log.info("Fetching instances  %s → %s  (daily batches) ...",
             from_date.date(), to_date.date())

    # Enumerate calendar days to fetch
    days = pd.date_range(
        from_date.normalize(),
        to_date.normalize(),
        freq="D",
        tz="CET",
    )

    all_frames: list[pd.DataFrame] = []
    for day in days:
        frames = _fetch_day(curve, day)
        if frames:
            all_frames.extend(frames)
            log.info("  %s: %d instance(s)", day.date(), len(frames))

    if not all_frames:
        log.info("No new data downloaded.")
        return

    new_df = pd.concat(all_frames, ignore_index=True)
    new_df["issue_date"]    = pd.to_datetime(new_df["issue_date"],    utc=True).astype("datetime64[us, UTC]")
    new_df["delivery_hour"] = pd.to_datetime(new_df["delivery_hour"], utc=True).astype("datetime64[us, UTC]")
    new_df["price_eur_mwh"] = new_df["price_eur_mwh"].astype("float64")

    if OUT.exists():
        existing = pd.read_parquet(OUT)
        combined = (
            pd.concat([existing, new_df], ignore_index=True)
            .drop_duplicates(subset=["issue_date", "delivery_hour"], keep="last")
        )
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df

    combined = combined.sort_values(["issue_date", "delivery_hour"]).reset_index(drop=True)
    combined.to_parquet(OUT, index=False)
    log.info(
        "Saved %d rows → %s  (latest issue: %s)",
        len(combined), OUT.name, combined["issue_date"].max(),
    )


if __name__ == "__main__":
    main()
