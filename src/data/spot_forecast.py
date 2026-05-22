"""
Fetch and cache Volue Insight spot price forecast data.

The spot curve on Volue is an INSTANCE curve — each forecast run is stored
separately, identified by its issue_date (when the forecast was published).

Use fetch_spot_forecast_history() to build the initial training cache, then
call it incrementally (overlap by a few days to catch late-arriving issues).
At inference time, call fetch_spot_forecast_history() for today or use
load_spot_forecast() if the cache is fresh enough.

Saved format (data/raw/market/spot_forecast_volue.parquet):
    issue_date (datetime64[us, UTC])    — forecast publication timestamp
    delivery_hour (datetime64[us, UTC]) — delivery hour the price applies to
    price_eur_mwh (float64)             — forecast spot price
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = ROOT / "data" / "raw" / "market" / "spot_forecast_volue.parquet"


def fetch_spot_forecast_history(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    area: str = "de",
    model: str = "ec00",
    config_file: str | Path = "~/.volue_config.ini",
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Download all Volue spot forecast runs with issue_date in [start_date, end_date].

    Appends to the existing cache file if it already exists (deduplicates by
    issue_date + delivery_hour after merging).

    Parameters
    ----------
    start_date, end_date : date range for forecast *issue* dates (not delivery dates)
    area  : Volue bidding zone code, e.g. "de", "fr", "nl"
    model : weather model suffix in the curve name, e.g. "ec00", "gfs00"
    config_file : path to a .ini file with [volue_insight] client_id / client_secret
    out_path : override the default cache path

    Returns
    -------
    Full cached DataFrame after the update.
    """
    import volue_insight_timeseries as vit  # noqa: PLC0415

    curve_name = f"pr {area} con {model} €/mwh cet h f"
    session = vit.Session(config_file=str(Path(config_file).expanduser()))
    curve = session.get_curve(name=curve_name)

    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)

    log.info(
        "Fetching Volue spot forecast runs %s → %s  curve='%s'",
        start_ts.date(), end_ts.date(), curve_name,
    )

    ts_list = curve.search_instances(
        issue_date_from=start_ts,
        issue_date_to=end_ts,
        with_data=True,
    )

    rows = []
    for ts in ts_list:
        issue_utc = pd.Timestamp(ts.issue_date).tz_localize("UTC") \
            if pd.Timestamp(ts.issue_date).tzinfo is None \
            else pd.Timestamp(ts.issue_date).tz_convert("UTC")
        series = ts.to_pandas()
        if series.index.tz is None:
            series.index = series.index.tz_localize(
                "Europe/Zurich", ambiguous="infer", nonexistent="shift_forward"
            )
        series.index = series.index.tz_convert("UTC")
        for delivery_hour, price in series.items():
            rows.append({
                "issue_date":    issue_utc,
                "delivery_hour": delivery_hour.floor("h"),
                "price_eur_mwh": float(price),
            })

    if not rows:
        log.warning("No forecast runs found for [%s, %s].", start_ts.date(), end_ts.date())
        new_df = pd.DataFrame(columns=["issue_date", "delivery_hour", "price_eur_mwh"])
    else:
        new_df = pd.DataFrame(rows)
        new_df["issue_date"]    = pd.to_datetime(new_df["issue_date"],    utc=True)
        new_df["delivery_hour"] = pd.to_datetime(new_df["delivery_hour"], utc=True)

    out = Path(out_path) if out_path else _DEFAULT_CACHE
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        existing = load_spot_forecast(out)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["issue_date", "delivery_hour"]) \
                           .sort_values(["issue_date", "delivery_hour"]) \
                           .reset_index(drop=True)
    else:
        combined = new_df.sort_values(["issue_date", "delivery_hour"]).reset_index(drop=True)

    combined.to_parquet(out, index=False)
    log.info(
        "  Saved %d rows (%d runs) → %s",
        len(combined), combined["issue_date"].nunique(), out.name,
    )
    return combined


def load_spot_forecast(path: Path | None = None) -> pd.DataFrame:
    """Load the cached Volue spot forecast Parquet file."""
    p = Path(path) if path else _DEFAULT_CACHE
    df = pd.read_parquet(p)
    df["issue_date"]    = pd.to_datetime(df["issue_date"],    utc=True)
    df["delivery_hour"] = pd.to_datetime(df["delivery_hour"], utc=True)
    return df
