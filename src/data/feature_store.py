"""
Build model-ready feature matrices from raw weather, reservoir, spot, and price data.

One feature matrix per model, saved to data/processed/features/.

Weather join logic (no data leakage):
  TRL Weekly  — forecast from preceding Tuesday 00z; steps covering delivery week
  TRL Daily   — forecast from D-1 00z; steps covering delivery day
  TRE         — most recent 00z available 1h before slot (00z assumed available ~09:00 UTC)

Output files:
  data/processed/features/
    trl_weekly_features.parquet
    trl_daily_features.parquet
    tre_features.parquet
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "data" / "processed" / "features"
PRICES_DIR = ROOT / "data" / "raw" / "prices"
MARKET_DIR = ROOT / "data" / "raw" / "market"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cos_zenith(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Cos(solar zenith angle) at Swiss centroid (46.8°N, 8.2°E), clipped to [0, 1]."""
    lat_rad = np.deg2rad(46.8)
    doy      = np.asarray(timestamps.day_of_year, dtype=float)
    hour_utc = np.asarray(timestamps.hour, dtype=float) + np.asarray(timestamps.minute, dtype=float) / 60
    decl     = np.deg2rad(23.45 * np.sin(np.deg2rad(360 / 365 * (doy - 81))))
    hour_angle = np.deg2rad(15 * (hour_utc + 8.2 / 15 - 12))
    cos_z = (np.sin(lat_rad) * np.sin(decl)
             + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle))
    return np.clip(cos_z, 0, None)


def load_weather_wide() -> pd.DataFrame:
    """Load weather_ensemble.parquet and pivot to wide format.

    Returns DataFrame with columns:
      init_time | valid_time | lead_hours |
      cloud_cover_mean | cloud_cover_std | ... |
      precip_rate_mmh_mean | ... |
      temp_2m_mean | ...
    """
    df = pd.read_parquet(FEATURES_DIR / "weather_ensemble.parquet")
    wide = df.pivot_table(
        index=["init_time", "valid_time", "lead_hours"],
        columns="variable",
        values=["mean", "std", "skew", "p10", "p90"],
        aggfunc="first",
    )
    wide.columns = [f"{var}_{stat}" for stat, var in wide.columns]
    wide = wide.reset_index().sort_values(["init_time", "valid_time"])
    wide["init_time"]  = _to_utc_us(wide["init_time"])
    wide["valid_time"] = _to_utc_us(wide["valid_time"])
    return wide


def _to_utc_us(series: pd.Series) -> pd.Series:
    """Normalize a datetime Series to datetime64[us, UTC]."""
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    else:
        series = series.dt.tz_convert("UTC")
    return series.astype("datetime64[us, UTC]")


def _init_time_for_bid(bid_times: pd.Series) -> pd.Series:
    """Return the 00z init_time that was available at each bid_time.

    ECMWF 00z results are published ~09:00 UTC.
    Most recent available init: floor(bid_time - 9h) to day boundary.
    """
    floored = (bid_times - pd.Timedelta(hours=9)).dt.floor("D")
    return _to_utc_us(floored)


def _tre_bid_time(slot_times: pd.Series) -> pd.Series:
    """Latest valid TRE bid submission time for each slot.

    Rules:
      - Must submit >= 1h before delivery
      - Submissions only on Mon–Fri 09:00–17:00 local (Europe/Zurich)

    Cases for candidate = slot_time - 1h:
      1. Workday, 09:00–17:00 → use candidate as-is
      2. Workday, >= 17:00    → clamp to same-day 17:00
      3. Workday, < 09:00 or weekend → previous workday 17:00
         (Mon early → Fri; Tue-Fri early → prev day; Sat → Fri; Sun → Fri)
    """
    local = slot_times.dt.tz_convert("Europe/Zurich")
    candidate = local - pd.Timedelta(hours=1)
    dow  = candidate.dt.dayofweek   # 0=Mon … 6=Sun
    hour = candidate.dt.hour

    is_workday = dow < 5
    in_window  = is_workday & (hour >= 9) & (hour < 17)
    too_late   = is_workday & (hour >= 17)
    # case 3: weekday before 09:00, or weekend
    prev_days  = pd.Series({0: 3, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2}, dtype=int)

    result = candidate.copy()
    result[too_late] = candidate[too_late].dt.normalize() + pd.Timedelta(hours=17)

    case3 = ~in_window & ~too_late
    if case3.any():
        days_back = dow[case3].map(prev_days)
        result[case3] = (
            candidate[case3].dt.normalize()
            - pd.to_timedelta(days_back.values, unit="D")
            + pd.Timedelta(hours=17)
        )

    return _to_utc_us(result)


def _weather_agg(weather: pd.DataFrame, init_times: pd.Series,
                 valid_start: pd.Series, valid_end: pd.Series,
                 weather_cols: list[str]) -> pd.DataFrame:
    """For each row, average weather features over valid_times in [valid_start, valid_end]
    for the given init_time. Returns a DataFrame aligned to the input index.

    Uses a merge then groupby-mean approach — efficient for large inputs.
    """
    # Reset index to avoid alignment issues; use Series (not .values) to preserve tz
    it = init_times.reset_index(drop=True)
    vs = valid_start.reset_index(drop=True)
    ve = valid_end.reset_index(drop=True)
    targets = pd.DataFrame({
        "row_idx":     range(len(it)),
        "init_time":   it,
        "valid_start": vs,
        "valid_end":   ve,
    })

    for col in ["init_time", "valid_start", "valid_end"]:
        targets[col] = _to_utc_us(targets[col])

    merged = targets.merge(weather[["init_time", "valid_time"] + weather_cols],
                           on="init_time", how="left")
    in_window = (merged["valid_time"] >= merged["valid_start"]) & \
                (merged["valid_time"] <= merged["valid_end"])
    merged = merged[in_window]

    agg = merged.groupby("row_idx")[weather_cols].mean()
    return agg.reindex(range(len(it)))


def _nearest_weather(weather: pd.DataFrame, init_times: pd.Series,
                     target_valid: pd.Series, weather_cols: list[str]) -> pd.DataFrame:
    """For each row, find the weather step whose valid_time is nearest to target_valid
    within the given init_time. Returns a DataFrame aligned to the input index.
    """
    it = init_times.reset_index(drop=True)
    tv = target_valid.reset_index(drop=True)
    targets = pd.DataFrame({
        "row_idx":      range(len(it)),
        "init_time":    it,
        "target_valid": tv,
    })

    for col in ["init_time", "target_valid"]:
        targets[col] = _to_utc_us(targets[col])

    merged = targets.merge(weather[["init_time", "valid_time"] + weather_cols],
                           on="init_time", how="left")
    merged["abs_diff"] = (merged["valid_time"] - merged["target_valid"]).abs()
    valid = merged.dropna(subset=["abs_diff"])
    if valid.empty:
        return pd.DataFrame(index=range(len(it)), columns=weather_cols, dtype=float)
    idx = valid.groupby("row_idx")["abs_diff"].idxmin()
    best = valid.loc[idx.values].set_index("row_idx")[weather_cols]
    return best.reindex(range(len(it)))


def _push_back_past_holidays(timestamps_utc: pd.Series, holiday_set: set, fallback_hour: int = 17) -> pd.Series:
    """Shift bid times that fall on Swiss public holidays back to the previous workday.

    Uses fallback_hour (local time) for the adjusted timestamp. Iterates in case
    consecutive days are also holidays (e.g. Ascension Thursday + company bridge Friday).
    """
    if not holiday_set:
        return timestamps_utc
    out = []
    for ts in timestamps_utc.dt.tz_convert("Europe/Zurich"):
        if ts.date() not in holiday_set:
            out.append(ts)
            continue
        d = ts - pd.Timedelta(days=1)
        while d.date() in holiday_set or d.dayofweek >= 5:
            d -= pd.Timedelta(days=1)
        out.append(d.normalize() + pd.Timedelta(hours=fallback_hour))
    return _to_utc_us(pd.Series(pd.DatetimeIndex(out), index=timestamps_utc.index))


def _reservoir_asof(reservoir: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Return the most recent reservoir reading available on or before each date."""
    res_cols = ["wallis_fill_pct", "graubuenden_fill_pct", "tessin_fill_pct", "totalch_fill_pct"]
    # Normalize reservoir dates to datetime64[us] (no tz — weekly resolution)
    res = reservoir.copy()
    res["date"] = pd.to_datetime(res["date"]).astype("datetime64[us]")
    res = res.sort_values("date")
    # Strip time and timezone from input dates for comparison
    dates_norm = pd.to_datetime(dates.dt.date).astype("datetime64[us]")
    result = pd.merge_asof(
        pd.DataFrame({"date": dates_norm.values, "row_idx": range(len(dates))}),
        res[["date"] + res_cols],
        on="date",
        direction="backward",
    )
    return result.set_index("row_idx")[res_cols]


# ---------------------------------------------------------------------------
# ENTSO-E load / generation forecasts (point-in-time)
# ---------------------------------------------------------------------------
#
# ENTSO-E returns each archived forecast as a flat series indexed by delivery
# time with no publish timestamp, so we model the publication deadline from the
# delivery date and only expose a value once that deadline has passed relative
# to the bid time — the same leakage guard as _spot_forecast_asof.
#
#   Day-ahead   (6.1.B load / 14.1.C gen / 14.1.D solar): published D-1.
#                Load by ~12:00 local; generation & solar by 18:00 Brussels.
#   Week-ahead  (6.1.C load, daily min/max): published the Friday before the
#                delivery ISO week (~10:00 local).
#
# Reachability (given bid lead times): TRE bids land close enough for day-ahead
# to cover the near slots (week-ahead fills the Fri→Mon gap); TRL Daily bids 2
# business days out so only week-ahead is ever published in time; TRL Weekly
# bids Tuesday, before even the week-ahead Friday publication, so neither is
# usable there.

ENTSOE_DA_PARQUET = MARKET_DIR / "entsoe_da.parquet"
ENTSOE_WK_PARQUET = MARKET_DIR / "entsoe_load_week.parquet"


def _load_entsoe_da() -> pd.DataFrame | None:
    """Hourly day-ahead forecasts indexed by delivery_time (UTC). None if absent."""
    if not ENTSOE_DA_PARQUET.exists():
        return None
    df = pd.read_parquet(ENTSOE_DA_PARQUET)
    df["delivery_time"] = _to_utc_us(pd.to_datetime(df["delivery_time"], utc=True))
    return df.set_index("delivery_time").sort_index()


def _load_entsoe_week() -> pd.DataFrame | None:
    """Daily week-ahead load min/max indexed by local delivery_date. None if absent."""
    if not ENTSOE_WK_PARQUET.exists():
        return None
    df = pd.read_parquet(ENTSOE_WK_PARQUET)
    df["delivery_date"] = pd.to_datetime(df["delivery_date"]).astype("datetime64[us]")
    return df.set_index("delivery_date").sort_index()


def _da_publication(delivery_hour_utc: pd.Series, pub_hour_local: int) -> pd.Series:
    """Publication deadline (UTC) of a day-ahead forecast: D-1 at pub_hour_local local."""
    local = delivery_hour_utc.dt.tz_convert("Europe/Zurich")
    pub_local = local.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=pub_hour_local)
    return _to_utc_us(pub_local)


def _entsoe_da_asof(da: pd.DataFrame | None, bid_times: pd.Series,
                    delivery_times: pd.Series, col: str, pub_hour_local: int) -> pd.Series:
    """Day-ahead value of `col` at each delivery hour, exposed only when its
    publication deadline (D-1) is at or before the bid time. NaN otherwise."""
    if da is None or col not in da.columns:
        return pd.Series(np.nan, index=bid_times.index)
    dh = _to_utc_us(delivery_times).dt.floor("h")
    vals = pd.Series(dh.map(da[col]).values, index=bid_times.index)
    avail = _da_publication(dh, pub_hour_local).values <= _to_utc_us(bid_times).values
    return vals.where(avail)


def _entsoe_week_asof(wk: pd.DataFrame | None, bid_times: pd.Series,
                      delivery_times: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Week-ahead (max, min) load for each delivery local-day, exposed only when
    the forecast (published the prior Friday ~10:00 local) precedes the bid."""
    nan = pd.Series(np.nan, index=bid_times.index)
    if wk is None:
        return nan, nan
    local = _to_utc_us(delivery_times).dt.tz_convert("Europe/Zurich")
    del_date = pd.to_datetime(local.dt.date).astype("datetime64[us]")
    maxv = pd.Series(del_date.map(wk["load_wk_max_mw"]).values, index=bid_times.index)
    minv = pd.Series(del_date.map(wk["load_wk_min_mw"]).values, index=bid_times.index)
    dow = pd.DatetimeIndex(del_date).dayofweek
    monday = del_date - pd.to_timedelta(dow, unit="D")
    fri_prev = monday - pd.Timedelta(days=3) + pd.Timedelta(hours=10)
    pub_utc = _to_utc_us(fri_prev.dt.tz_localize(
        "Europe/Zurich", ambiguous="infer", nonexistent="shift_forward"))
    avail = pub_utc.values <= _to_utc_us(bid_times).values
    return maxv.where(avail), minv.where(avail)


def _entsoe_tre_features(bid_times: pd.Series, slot_times: pd.Series) -> pd.DataFrame:
    """Full ENTSO-E feature block for TRE: day-ahead load/gen/solar (near slots)
    plus the week-ahead load envelope (Fri→Mon gap), all point-in-time safe."""
    da, wk = _load_entsoe_da(), _load_entsoe_week()
    load_da  = _entsoe_da_asof(da, bid_times, slot_times, "load_da_mw",  pub_hour_local=12)
    gen_da   = _entsoe_da_asof(da, bid_times, slot_times, "gen_da_mw",   pub_hour_local=18)
    solar_da = _entsoe_da_asof(da, bid_times, slot_times, "solar_da_mw", pub_hour_local=18)
    wind_da  = _entsoe_da_asof(da, bid_times, slot_times, "wind_da_mw",  pub_hour_local=18)
    wk_max, wk_min = _entsoe_week_asof(wk, bid_times, slot_times)
    return pd.DataFrame({
        "entsoe_load_da_mw":      load_da,
        "entsoe_gen_da_mw":       gen_da,
        "entsoe_solar_da_mw":     solar_da,
        "entsoe_net_load_da_mw":  load_da - solar_da - wind_da,
        "entsoe_load_wk_max_mw":  wk_max,
        "entsoe_load_wk_min_mw":  wk_min,
        "entsoe_load_wk_spread_mw": wk_max - wk_min,
    }, index=bid_times.index)


def _entsoe_week_features(bid_times: pd.Series, delivery_times: pd.Series) -> pd.DataFrame:
    """Week-ahead-only ENTSO-E block for TRL Daily (day-ahead never published in
    time for a 2-business-day-ahead bid)."""
    wk = _load_entsoe_week()
    wk_max, wk_min = _entsoe_week_asof(wk, bid_times, delivery_times)
    return pd.DataFrame({
        "entsoe_load_wk_max_mw":    wk_max,
        "entsoe_load_wk_min_mw":    wk_min,
        "entsoe_load_wk_spread_mw": wk_max - wk_min,
    }, index=bid_times.index)


def _price_lags(df: pd.DataFrame, time_col: str, price_col: str,
                lags: list[int], roll_windows: list[int] = (4, 12)) -> pd.DataFrame:
    """Compute integer-position lag features and rolling stats on sorted time series."""
    df = df.sort_values(time_col).reset_index(drop=True)
    out = {}
    for lag in lags:
        out[f"{price_col}_lag{lag}"] = df[price_col].shift(lag)
    for window in roll_windows:
        out[f"{price_col}_roll{window}_mean"] = (
            df[price_col].shift(1).rolling(window, min_periods=1).mean()
        )
        out[f"{price_col}_roll{window}_std"] = (
            df[price_col].shift(1).rolling(window, min_periods=2).std()
        )
    return pd.DataFrame(out)


def _price_lags_same_block(
    df: pd.DataFrame,
    time_col: str,
    price_col: str,
    roll_windows: list[int] = (7, 28),
    block_hours: int = 4,
) -> pd.DataFrame:
    """Rolling mean/std within the same block-of-day (time-of-day separated).

    window=7  → 7 past occurrences of the same block ≈ 7 days
    window=28 → 28 past occurrences of the same block ≈ 4 weeks

    Keeps the intraday shape signal separate from the price-level signal,
    so the model can learn that the 12-16 block premium scales with the regime.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    block_id = df[time_col].dt.hour // block_hours
    out = {}
    for window in roll_windows:
        shifted = df.groupby(block_id)[price_col].shift(1)
        out[f"{price_col}_sb_roll{window}_mean"] = (
            shifted.groupby(block_id)
            .rolling(window, min_periods=1)
            .mean()
            .droplevel(0)
            .reindex(df.index)
        )
        out[f"{price_col}_sb_roll{window}_std"] = (
            shifted.groupby(block_id)
            .rolling(window, min_periods=2)
            .std()
            .droplevel(0)
            .reindex(df.index)
        )
    return pd.DataFrame(out)


def _load_spot_forecast(n_revision_runs: int = 3) -> pd.DataFrame:
    """Load the Volue spot price forecast cache with revision statistics.

    Prepends realized spot prices (spot_hourly.parquet) as a static fallback for
    delivery hours not covered by Volue data (subscription starts 2026-01-01).
    Realized entries are given issue_date = delivery_hour − 30 days so the
    merge_asof lookup always finds them for any bid_time before delivery.

    Adds spot_fcst_std / spot_fcst_change for Volue-covered rows; NaN elsewhere
    (LightGBM handles NaN natively via its split logic).
    """
    volue_path    = MARKET_DIR / "spot_forecast_volue.parquet"
    realized_path = PRICES_DIR / "spot_hourly.parquet"

    if volue_path.exists():
        df = pd.read_parquet(volue_path)
        df["issue_date"]    = pd.to_datetime(df["issue_date"],    utc=True).astype("datetime64[us, UTC]")
        df["delivery_hour"] = pd.to_datetime(df["delivery_hour"], utc=True).astype("datetime64[us, UTC]")
    else:
        df = pd.DataFrame({
            "issue_date":    pd.Series(dtype="datetime64[us, UTC]"),
            "delivery_hour": pd.Series(dtype="datetime64[us, UTC]"),
            "price_eur_mwh": pd.Series(dtype="float64"),
        })

    volue_min = df["delivery_hour"].min() if len(df) else None

    if realized_path.exists():
        realized = pd.read_parquet(realized_path)
        realized["hour_time"] = pd.to_datetime(realized["hour_time"], utc=True).astype("datetime64[us, UTC]")
        if volue_min is not None:
            realized = realized[realized["hour_time"] < volue_min]
        if len(realized):
            realized_rows = pd.DataFrame({
                "issue_date":    _to_utc_us(realized["hour_time"] - pd.Timedelta(days=30)),
                "delivery_hour": _to_utc_us(realized["hour_time"]),
                "price_eur_mwh": realized["price_eur_mwh"].values,
            })
            df = pd.concat([realized_rows, df], ignore_index=True)

    df = df.sort_values(["delivery_hour", "issue_date"]).reset_index(drop=True)
    grp = df.groupby("delivery_hour")["price_eur_mwh"]
    price_mat = pd.concat(
        [df["price_eur_mwh"]] + [grp.shift(i).rename(f"_p{i}") for i in range(1, n_revision_runs)],
        axis=1,
    )
    df["spot_fcst_std"]    = price_mat.std(axis=1, ddof=1)
    df["spot_fcst_change"] = df["price_eur_mwh"] - grp.shift(n_revision_runs - 1)
    return df


def _spot_forecast_asof(
    spot_fcst: pd.DataFrame,
    bid_times: pd.Series,
    delivery_hours: pd.Series,
    return_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Point-in-time spot forecast lookup.

    For each (bid_time, delivery_hour) pair, returns return_cols from the latest
    forecast run issued at or before bid_time.  Returns NaN where no forecast
    exists before bid_time for that delivery hour.

    Uses vectorised numpy.searchsorted per delivery_hour group rather than
    merge_asof (which requires the 'on' column to be globally monotonic, which
    is impossible when the same delivery_hour appears across many issue_dates).
    """
    if return_cols is None:
        return_cols = ["price_eur_mwh"]

    fcst = (
        spot_fcst[["issue_date", "delivery_hour"] + return_cols]
        .sort_values(["delivery_hour", "issue_date"])
        .reset_index(drop=True)
    )

    # Integer views for fast searchsorted comparisons
    dh_f = fcst["delivery_hour"].to_numpy(dtype="int64")
    id_f = fcst["issue_date"].to_numpy(dtype="int64")
    val_arrs = {col: fcst[col].to_numpy() for col in return_cols}

    # Build delivery_hour → slice lookup (fcst is sorted by delivery_hour)
    unique_dh, first_idx = np.unique(dh_f, return_index=True)
    end_idx = np.append(first_idx[1:], len(dh_f))

    dh_to_slice: dict[int, tuple[int, int]] = {
        dh: (s, e) for dh, s, e in zip(unique_dh, first_idx, end_idx)
    }

    bt_q = bid_times.to_numpy(dtype="int64")
    dh_q = delivery_hours.to_numpy(dtype="int64")
    n = len(bt_q)

    out_vals = {col: np.full(n, np.nan) for col in return_cols}

    # Vectorised per-group searchsorted
    for dh_val, (s, e) in dh_to_slice.items():
        mask = dh_q == dh_val
        if not mask.any():
            continue
        grp_ids = id_f[s:e]
        bt_grp  = bt_q[mask]
        idx = np.searchsorted(grp_ids, bt_grp, side="right") - 1
        valid = idx >= 0
        for col in return_cols:
            vals = out_vals[col]
            rows = np.where(mask)[0]
            vals[rows[valid]] = val_arrs[col][s + idx[valid]]

    result = pd.DataFrame(out_vals, index=bid_times.index)
    return result


def _spot_week_features(
    spot_fcst: pd.DataFrame,
    bid_times: pd.Series,
    week_starts: pd.Series,
) -> pd.DataFrame:
    """Weekly spot price forecast aggregates for TRL Weekly.

    Fetches Volue forecasts for all 168 hours of the delivery week as-of bid_time,
    then aggregates to six features. Peak = 08:00–19:59 Europe/Zurich.
    Daily spread uses UTC day buckets (offset // 24) — DST error < 1h, negligible.
    """
    n, n_hours   = len(bid_times), 168
    hour_offsets = np.tile(np.arange(n_hours), n)
    row_idx      = np.repeat(np.arange(n), n_hours)

    bid_times_rep  = _to_utc_us(bid_times.iloc[row_idx].reset_index(drop=True))
    delivery_hours = _to_utc_us(
        week_starts.iloc[row_idx].reset_index(drop=True)
        + pd.to_timedelta(hour_offsets, unit="h")
    )

    prices = _spot_forecast_asof(
        spot_fcst, bid_times_rep, delivery_hours, return_cols=["price_eur_mwh"]
    )["price_eur_mwh"].values.reshape(n, n_hours)

    local_hour = pd.DatetimeIndex(delivery_hours).tz_convert("Europe/Zurich").hour
    is_peak    = ((local_hour >= 8) & (local_hour < 20)).reshape(n, n_hours)
    day_bucket = (hour_offsets // 24).reshape(n, n_hours)

    daily_spreads = np.full((n, 7), np.nan)
    for d in range(7):
        day_p = np.where(day_bucket == d, prices, np.nan)
        daily_spreads[:, d] = np.nanmax(day_p, axis=1) - np.nanmin(day_p, axis=1)

    return pd.DataFrame({
        "spot_baseload_mean":     np.nanmean(prices, axis=1),
        "spot_peakload_mean":     np.nanmean(np.where(is_peak, prices, np.nan), axis=1),
        "spot_max":               np.nanmax(prices, axis=1),
        "spot_min":               np.nanmin(prices, axis=1),
        "spot_daily_spread_mean": np.nanmean(daily_spreads, axis=1),
        "spot_neg_hours":         np.nansum(prices < 0, axis=1).astype(float),
    }, index=bid_times.index)


# ---------------------------------------------------------------------------
# Swiss public holidays
# ---------------------------------------------------------------------------

def _easter(year: int) -> date:
    """Gregorian Easter Sunday — Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _swiss_holiday_set(years) -> set:
    """Dates of Swiss public holidays observed across the majority of cantons."""
    holidays: set = set()
    for year in years:
        easter = _easter(year)
        holidays |= {
            date(year, 1, 1),                        # New Year's Day
            date(year, 1, 2),                        # Berchtoldstag
            easter - timedelta(days=2),              # Good Friday
            easter + timedelta(days=1),              # Easter Monday
            date(year, 5, 1),                        # Labour Day
            easter + timedelta(days=39),             # Ascension Day
            easter + timedelta(days=50),             # Whit Monday
            date(year, 8, 1),                        # Swiss National Day
            date(year, 12, 25),                      # Christmas Day
            date(year, 12, 26),                      # St. Stephen's Day
        }
    return holidays


# ---------------------------------------------------------------------------
# TRL Weekly
# ---------------------------------------------------------------------------

def build_trl_weekly_features() -> pd.DataFrame:
    log.info("Building TRL Weekly features ...")
    prices = pd.read_parquet(PRICES_DIR / "trl_weekly.parquet")
    weather = load_weather_wide()
    reservoir = pd.read_parquet(MARKET_DIR / "reservoir_levels.parquet")
    spot_fcst = _load_spot_forecast()

    prices["week_start"] = pd.to_datetime(prices["week_start"], utc=True)
    week_start = prices["week_start"]

    # Auction: Tuesday before delivery week (week_start - 6 days)
    bid_times = week_start - pd.Timedelta(days=6)
    init_times = _init_time_for_bid(bid_times)

    # Delivery period: full week
    valid_start = week_start
    valid_end = week_start + pd.Timedelta(days=7)

    weather_cols = [c for c in weather.columns
                    if c not in ("init_time", "valid_time", "lead_hours")]
    w = _weather_agg(weather, init_times, valid_start, valid_end, weather_cols)

    res = _reservoir_asof(reservoir, bid_times)
    spot_week = _spot_week_features(spot_fcst, bid_times, week_start)

    # Calendar
    years_w = set(week_start.dt.year) | set((week_start + pd.Timedelta(days=6)).dt.year)
    holiday_set_w = _swiss_holiday_set(years_w)
    n_holidays_in_week = [
        sum(1 for i in range(7) if (d + timedelta(days=i)) in holiday_set_w)
        for d in week_start.dt.date
    ]
    cal = pd.DataFrame({
        "week_of_year":       week_start.dt.isocalendar().week.astype(int),
        "month":              week_start.dt.month,
        "year":               week_start.dt.year,
        "n_holidays_in_week": n_holidays_in_week,
    }, index=prices.index)

    # Price lags (per direction separately, then re-align to original index)
    lag_parts = []
    for direction in ("up", "down"):
        mask = prices["direction"] == direction
        sub = prices[mask].copy()
        lag_df = _price_lags(sub, "week_start", "marginal_chf", lags=[1, 4, 52], roll_windows=[4, 12])
        lag_df.index = sub.index
        lag_parts.append(lag_df)
    price_lags = pd.concat(lag_parts).reindex(prices.index)

    s1_num_cols = [c for c in ["s1_awarded_mw", "s1_marginal_chf", "s1_vwap_chf"]
                   if c in prices.columns]
    s1_cols = (["s1_is_active"] if "s1_is_active" in prices.columns else []) + s1_num_cols
    if s1_num_cols:
        # Fill with 0 when s1_is_active=0 so dropna in run_backtest doesn't eliminate UP rows
        # or non-S1 DOWN weeks. LightGBM uses s1_is_active as a gate before interpreting these.
        for col in s1_num_cols:
            prices[col] = prices[col].where(prices.get("s1_is_active", pd.Series(0, index=prices.index)) == 1, 0.0)

    features = pd.concat([
        prices[["week_start", "direction", "marginal_chf",
                "offered_mw", "awarded_mw", "award_rate_pct"] + s1_cols],
        cal,
        w.set_index(prices.index),
        res.set_index(prices.index),
        spot_week.set_index(prices.index),
        price_lags,
    ], axis=1)

    features = features.sort_values(["week_start", "direction"]).reset_index(drop=True)
    out = FEATURES_DIR / "trl_weekly_features.parquet"
    features.to_parquet(out, index=False)
    log.info("  Saved %d rows → %s", len(features), out.name)
    return features


# ---------------------------------------------------------------------------
# TRL Daily
# ---------------------------------------------------------------------------

def build_trl_daily_features() -> pd.DataFrame:
    log.info("Building TRL Daily features ...")
    prices = pd.read_parquet(PRICES_DIR / "trl_daily.parquet")
    weather = load_weather_wide()
    reservoir = pd.read_parquet(MARKET_DIR / "reservoir_levels.parquet")
    trl_weekly_raw = pd.read_parquet(PRICES_DIR / "trl_weekly.parquet")

    prices["block_start"] = pd.to_datetime(prices["block_start"], utc=True)
    block_start = prices["block_start"]
    block_end = block_start + pd.Timedelta(hours=4)

    # Bid: gate closure 14:00 local, 2 business days before delivery.
    # mo→we, tu→th, we→fr, th→sa+su, fr→mo+tu
    # days_back: calendar days from delivery date back to bid date
    _local_delivery = block_start.dt.tz_convert("Europe/Zurich")
    _dow = _local_delivery.dt.dayofweek  # 0=Mon ... 6=Sun
    _days_back_map = {0: 3, 1: 4, 2: 2, 3: 2, 4: 2, 5: 2, 6: 3}
    _days_back = _dow.map(_days_back_map)
    _bid_local = _local_delivery.dt.normalize() - pd.to_timedelta(_days_back, unit="D") + pd.Timedelta(hours=14)
    bid_times = _to_utc_us(_bid_local)
    # Push back if bid day is a Swiss public holiday (e.g. Good Friday → Thursday)
    _bid_years = set(bid_times.dt.tz_convert("Europe/Zurich").dt.year)
    bid_times = _push_back_past_holidays(bid_times, _swiss_holiday_set(_bid_years), fallback_hour=14)
    init_times = _init_time_for_bid(bid_times)

    weather_cols = [c for c in weather.columns
                    if c not in ("init_time", "valid_time", "lead_hours")
                    and "precip" not in c]  # no precipitation for TRL Daily
    block_mid = block_start + pd.Timedelta(hours=2)
    w = _nearest_weather(weather, init_times, block_mid, weather_cols)

    res = _reservoir_asof(reservoir, block_start)

    # Spot price: Volue forecast for each of the 4 block hours, averaged.
    # The delivery-day DA price is never published at TRL Daily bid time
    # (gate closes 2 business days ahead; DA clears only 1 day before delivery).
    spot_fcst = _load_spot_forecast()
    _SPOT_COLS = ["price_eur_mwh", "spot_fcst_std", "spot_fcst_change"]
    _hour_fcsts = [
        _spot_forecast_asof(spot_fcst, bid_times,
                            (block_start + pd.Timedelta(hours=h)).dt.floor("h"),
                            return_cols=_SPOT_COLS)
        for h in range(4)
    ]
    spot_vals             = pd.concat([d["price_eur_mwh"]    for d in _hour_fcsts], axis=1).mean(axis=1).values
    spot_fcst_std_vals    = pd.concat([d["spot_fcst_std"]    for d in _hour_fcsts], axis=1).mean(axis=1).values
    spot_fcst_change_vals = pd.concat([d["spot_fcst_change"] for d in _hour_fcsts], axis=1).mean(axis=1).values

    # cos_zenith: mean over the 4h block (at 15-min intervals)
    def block_cos_zenith(bs, be):
        ts = pd.date_range(bs, be, freq="15min", inclusive="left")
        return float(cos_zenith(ts).mean())

    czn = np.array([block_cos_zenith(bs, be)
                    for bs, be in zip(block_start, block_end)])

    # Calendar
    local = block_start.dt.tz_convert("Europe/Zurich")
    holiday_set_d = _swiss_holiday_set(set(local.dt.year))
    is_holiday_d = [int(d in holiday_set_d) for d in local.dt.date]
    ssrd_proxy_d     = czn * (1.0 - w["cloud_cover_mean"].values)
    ssrd_proxy_unc_d = czn * w["cloud_cover_std"].values
    cal = pd.DataFrame({
        "block_of_day":   (local.dt.hour // 4).astype(int),
        "day_of_week":    local.dt.dayofweek,
        "month":          local.dt.month,
        "is_weekend":     (local.dt.dayofweek >= 5).astype(int),
        "is_thursday":    (local.dt.dayofweek == 3).astype(int),  # Thu bid covers Sat+Sun
        "is_friday":      (local.dt.dayofweek == 4).astype(int),  # Fri bid covers Mon+Tue
        "is_holiday":     is_holiday_d,
        "days_ahead":     (
            (block_start.dt.tz_convert("Europe/Zurich").dt.normalize()
             - bid_times.dt.tz_convert("Europe/Zurich").dt.normalize()).dt.days.values
        ),
        "cos_zenith":     czn,
        "ssrd_proxy":     ssrd_proxy_d,
        "ssrd_proxy_unc": ssrd_proxy_unc_d,
        "spot_fcst_std":    spot_fcst_std_vals,
        "spot_fcst_change": spot_fcst_change_vals,
        "spot_eur_mwh":     spot_vals,
    }, index=prices.index)

    # TRL Weekly auction results for the delivery week — always known at TRL Daily bid time
    # (TRL Weekly auction closes Tuesday of prior week; TRL Daily bids ≥2 days ahead).
    trl_weekly_raw["week_start"] = _to_utc_us(pd.to_datetime(trl_weekly_raw["week_start"], utc=True))
    # 0-fill S1 awarded volume for non-S1 weeks so no NaN propagates into TRL Daily features.
    if "s1_awarded_mw" in trl_weekly_raw.columns:
        s1_active = trl_weekly_raw.get("s1_is_active", pd.Series(0, index=trl_weekly_raw.index))
        trl_weekly_raw["s1_awarded_mw"] = trl_weekly_raw["s1_awarded_mw"].where(s1_active == 1, 0.0)

    pivot_vals = ["marginal_chf", "vwap_chf", "awarded_mw"]
    if "s1_awarded_mw" in trl_weekly_raw.columns:
        pivot_vals.append("s1_awarded_mw")
    trl_weekly_wide = (
        trl_weekly_raw.pivot_table(index="week_start", columns="direction",
                                   values=pivot_vals, aggfunc="first")
    )
    # Flatten MultiIndex columns: (value, direction) → trl_weekly_{direction}_{short_name}
    _name_map = {"marginal_chf": "chf", "vwap_chf": "vwap_chf",
                 "awarded_mw": "awarded_mw", "s1_awarded_mw": "s1_awarded_mw"}
    trl_weekly_wide.columns = [
        f"trl_weekly_{dir_}_{_name_map[val]}"
        for val, dir_ in trl_weekly_wide.columns
    ]
    trl_weekly_wide = trl_weekly_wide.reset_index()

    block_week_start = _to_utc_us(
        (block_start - pd.to_timedelta(block_start.dt.dayofweek, unit="D")).dt.normalize()
    )
    _weekly_cols = [c for c in trl_weekly_wide.columns if c != "week_start"]
    weekly_vals = (
        pd.DataFrame({"week_start": block_week_start})
        .merge(trl_weekly_wide, on="week_start", how="left")
        [_weekly_cols]
    )

    # Price lags
    lag_parts    = []
    sb_lag_parts = []
    for direction in ("up", "down"):
        mask = prices["direction"] == direction
        sub = prices[mask].copy()
        lag_df = _price_lags(sub, "block_start", "marginal_chf", lags=[6, 42], roll_windows=[42, 180])
        lag_df.index = sub.index
        lag_parts.append(lag_df)
        sb_lag_df = _price_lags_same_block(sub, "block_start", "marginal_chf", roll_windows=[7, 28])
        sb_lag_df.index = sub.index
        sb_lag_parts.append(sb_lag_df)
    price_lags    = pd.concat(lag_parts).reindex(prices.index)
    sb_price_lags = pd.concat(sb_lag_parts).reindex(prices.index)

    # ENTSO-E CH load forecast (point-in-time): week-ahead only — the day-ahead
    # forecast is published D-1, after a TRL Daily bid (2 business days ahead).
    entsoe = _entsoe_week_features(bid_times, block_start)

    features = pd.concat([
        prices[["block_start", "direction", "marginal_chf",
                "offered_mw", "awarded_mw", "award_rate_pct"]],
        cal,
        w.set_index(prices.index),
        res.set_index(prices.index),
        weekly_vals.set_index(prices.index),
        price_lags,
        sb_price_lags,
        entsoe,
    ], axis=1)

    features = features.sort_values(["block_start", "direction"]).reset_index(drop=True)
    out = FEATURES_DIR / "trl_daily_features.parquet"
    features.to_parquet(out, index=False)
    log.info("  Saved %d rows → %s", len(features), out.name)
    return features


# ---------------------------------------------------------------------------
# TRE
# ---------------------------------------------------------------------------

def build_tre_features() -> pd.DataFrame:
    log.info("Building TRE features ...")
    prices = pd.read_parquet(PRICES_DIR / "tre_slots.parquet")
    weather = load_weather_wide()
    reservoir = pd.read_parquet(MARKET_DIR / "reservoir_levels.parquet")
    spot = pd.read_parquet(PRICES_DIR / "spot_hourly.parquet")
    trl_weekly_raw = pd.read_parquet(PRICES_DIR / "trl_weekly.parquet")

    prices["slot_time"] = pd.to_datetime(prices["slot_time"], utc=True)
    slot_time = prices["slot_time"]

    # Latest valid bid time: workday 09:00–17:00, >= 1h before delivery
    # (Fri 17:00 covers Sat/Sun/Mon-00:00-09:45; next window opens Mon 09:00)
    # Holiday set computed here (before bid_time) so we can push back bid days that land on holidays
    _slot_years = set(slot_time.dt.tz_convert("Europe/Zurich").dt.year)
    _tre_holiday_set = _swiss_holiday_set(_slot_years | {y - 1 for y in _slot_years})
    bid_time = _tre_bid_time(slot_time)
    bid_time = _push_back_past_holidays(bid_time, _tre_holiday_set, fallback_hour=17)
    init_times = _init_time_for_bid(bid_time)
    lead_hours = (slot_time - init_times).dt.total_seconds() / 3600
    hours_until_delivery = (slot_time - bid_time).dt.total_seconds() / 3600

    weather_cols = [c for c in weather.columns
                    if c not in ("init_time", "valid_time", "lead_hours")
                    and "precip" not in c]
    w = _nearest_weather(weather, init_times, slot_time, weather_cols)

    res = _reservoir_asof(reservoir, slot_time)

    # Spot price: actual DA when published before bid_time; Volue forecast otherwise.
    # DA auction for day D clears ~12:00 local on D-1; use 13:00 as safe cutoff.
    spot["hour_time"] = pd.to_datetime(spot["hour_time"], utc=True)
    spot_indexed = spot.drop_duplicates("hour_time").set_index("hour_time")["price_eur_mwh"]

    bid_local  = bid_time.dt.tz_convert("Europe/Zurich")
    slot_local = slot_time.dt.tz_convert("Europe/Zurich")
    da_pub_local = slot_local.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=13)
    da_pub_local = da_pub_local.dt.tz_localize(None).dt.tz_localize(
        "Europe/Zurich", ambiguous="infer", nonexistent="shift_forward"
    )
    da_known = da_pub_local <= bid_local

    slot_hour = slot_time.dt.floor("h")
    spot_fcst = _load_spot_forecast()
    _SPOT_COLS = ["price_eur_mwh", "spot_fcst_std", "spot_fcst_change"]
    fcst_df = _spot_forecast_asof(spot_fcst, bid_time, slot_hour, return_cols=_SPOT_COLS)
    spot_vals = pd.Series(
        np.where(da_known, slot_hour.map(spot_indexed), fcst_df["price_eur_mwh"].values),
        index=prices.index,
    )
    spot_fcst_std_vals    = fcst_df["spot_fcst_std"].values
    spot_fcst_change_vals = fcst_df["spot_fcst_change"].values

    # TRL Weekly prices — auction clears on Tuesday prior week, always known at TRE bid time.
    # Align each 15-min slot to its ISO week start (Monday 00:00 UTC) then merge.
    trl_weekly_raw["week_start"] = _to_utc_us(pd.to_datetime(trl_weekly_raw["week_start"], utc=True))
    trl_weekly_wide = (
        trl_weekly_raw.pivot_table(index="week_start", columns="direction",
                                   values="marginal_chf", aggfunc="first")
        .rename(columns={"up": "trl_weekly_up_chf", "down": "trl_weekly_down_chf"})
        .reset_index()
    )
    slot_week_start = _to_utc_us(
        (slot_time - pd.to_timedelta(slot_time.dt.dayofweek, unit="D")).dt.normalize()
    )
    weekly_vals = (
        pd.DataFrame({"week_start": slot_week_start})
        .merge(trl_weekly_wide, on="week_start", how="left")
        [["trl_weekly_up_chf", "trl_weekly_down_chf"]]
    )

    # Calendar + cos_zenith
    local = slot_time.dt.tz_convert("Europe/Zurich")
    czn = cos_zenith(pd.DatetimeIndex(slot_time))
    holiday_set_t = _swiss_holiday_set(set(local.dt.year))
    is_holiday_t = [int(d in holiday_set_t) for d in local.dt.date]
    ssrd_proxy_t     = czn * (1.0 - w["cloud_cover_mean"].values)
    ssrd_proxy_unc_t = czn * w["cloud_cover_std"].values

    cal = pd.DataFrame({
        "quarter_of_hour":      (local.dt.minute // 15).astype(int),
        "hour_of_day":          local.dt.hour,
        "day_of_week":          local.dt.dayofweek,
        "month":                local.dt.month,
        "is_weekend":           (local.dt.dayofweek >= 5).astype(int),
        "is_friday":            (local.dt.dayofweek == 4).astype(int),
        "is_holiday":           is_holiday_t,
        "lead_hours":           lead_hours.values,
        "hours_until_delivery": hours_until_delivery.values,
        "cos_zenith":           czn,
        "ssrd_proxy":           ssrd_proxy_t,
        "ssrd_proxy_unc":       ssrd_proxy_unc_t,
        "spot_is_realized":     da_known.astype(int).values,
        "spot_fcst_std":        spot_fcst_std_vals,
        "spot_fcst_change":     spot_fcst_change_vals,
        "spot_eur_mwh":         spot_vals.values,
    }, index=prices.index)

    # Price lags — no exact-slot lags (too noisy at 15-min resolution).
    # lag96h: mean of the 4 slots in the same clock-hour yesterday (smoother same-time anchor).
    # Rolling 24h and 7d stats capture regime level and volatility.
    lag_parts = []
    for direction in ("pos", "neg"):
        mask = prices["direction"] == direction
        sub = prices[mask].sort_values("slot_time")  # keep original index
        lag_df = _price_lags(sub, "slot_time", "marginal_chf", lags=[], roll_windows=[96, 672])
        # _price_lags resets index internally; restore original (sorted) index
        lag_df.index = sub.index
        lag_df["marginal_chf_lag96h"] = (
            sub["marginal_chf"].shift(96).rolling(4, min_periods=1).mean().values
        )
        lag_parts.append(lag_df)
    price_lags = pd.concat(lag_parts).reindex(prices.index)

    # ENTSO-E CH load/generation forecasts (point-in-time): day-ahead + week-ahead
    entsoe = _entsoe_tre_features(bid_time, slot_time)

    features = pd.concat([
        prices[["slot_time", "direction", "marginal_chf",
                "offered", "activated", "activation_rate"]],
        cal,
        w.set_index(prices.index),
        res.set_index(prices.index),
        price_lags,
        weekly_vals.set_index(prices.index),
        entsoe,
    ], axis=1)

    features = features.sort_values(["slot_time", "direction"]).reset_index(drop=True)
    out = FEATURES_DIR / "tre_features.parquet"
    features.to_parquet(out, index=False)
    log.info("  Saved %d rows → %s", len(features), out.name)
    return features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    build_trl_weekly_features()
    build_trl_daily_features()
    build_tre_features()
    log.info("Done.")


if __name__ == "__main__":
    main()
