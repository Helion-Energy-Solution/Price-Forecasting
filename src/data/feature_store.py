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

    features = pd.concat([
        prices[["week_start", "direction", "marginal_chf",
                "offered_mw", "awarded_mw", "award_rate_pct"]],
        cal,
        w.set_index(prices.index),
        res.set_index(prices.index),
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
    spot = pd.read_parquet(PRICES_DIR / "spot_hourly.parquet")
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

    # Spot price: average of day-ahead hours covering the 4h block
    spot["hour_time"] = pd.to_datetime(spot["hour_time"], utc=True)
    # Assign each block's hours via floor then merge
    prices["_hour0"] = block_start.dt.floor("h")
    prices["_hour1"] = (block_start + pd.Timedelta(hours=1)).dt.floor("h")
    prices["_hour2"] = (block_start + pd.Timedelta(hours=2)).dt.floor("h")
    prices["_hour3"] = (block_start + pd.Timedelta(hours=3)).dt.floor("h")
    spot_map = spot.drop_duplicates("hour_time").set_index("hour_time")["price_eur_mwh"]
    spot_vals = (prices[["_hour0","_hour1","_hour2","_hour3"]]
                 .apply(lambda col: col.map(spot_map))
                 .mean(axis=1).values)
    prices.drop(columns=["_hour0","_hour1","_hour2","_hour3"], inplace=True)

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
        "spot_eur_mwh":   spot_vals,
    }, index=prices.index)

    # TRL Weekly price for the delivery week — always known at TRL Daily bid time
    # (TRL Weekly auction closes Tuesday of prior week; TRL Daily bids ≥2 days ahead).
    trl_weekly_raw["week_start"] = _to_utc_us(pd.to_datetime(trl_weekly_raw["week_start"], utc=True))
    trl_weekly_wide = (
        trl_weekly_raw.pivot_table(index="week_start", columns="direction",
                                   values="marginal_chf", aggfunc="first")
        .rename(columns={"up": "trl_weekly_up_chf", "down": "trl_weekly_down_chf"})
        .reset_index()
    )
    block_week_start = _to_utc_us(
        (block_start - pd.to_timedelta(block_start.dt.dayofweek, unit="D")).dt.normalize()
    )
    weekly_vals = (
        pd.DataFrame({"week_start": block_week_start})
        .merge(trl_weekly_wide, on="week_start", how="left")
        [["trl_weekly_up_chf", "trl_weekly_down_chf"]]
    )

    # Price lags
    lag_parts = []
    for direction in ("up", "down"):
        mask = prices["direction"] == direction
        sub = prices[mask].copy()
        lag_df = _price_lags(sub, "block_start", "marginal_chf", lags=[6, 42], roll_windows=[42, 180])
        lag_df.index = sub.index
        lag_parts.append(lag_df)
    price_lags = pd.concat(lag_parts).reindex(prices.index)

    features = pd.concat([
        prices[["block_start", "direction", "marginal_chf",
                "offered_mw", "awarded_mw", "award_rate_pct"]],
        cal,
        w.set_index(prices.index),
        res.set_index(prices.index),
        weekly_vals.set_index(prices.index),
        price_lags,
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

    # Spot price: day-ahead price for the delivery hour, but only if that DA auction
    # result was published before gate closure (bid_time).
    # DA auction for day D clears ~12:00 local on day D-1; use 13:00 to be safe.
    # For Sunday delivery and Monday-pre-09:00 (gate = Friday 17:00), Sunday/Monday
    # DA is not yet published — fall back to same hour of day on bid_time's date.
    spot["hour_time"] = pd.to_datetime(spot["hour_time"], utc=True)
    spot_indexed = spot.drop_duplicates("hour_time").set_index("hour_time")["price_eur_mwh"]

    bid_local   = bid_time.dt.tz_convert("Europe/Zurich")
    slot_local  = slot_time.dt.tz_convert("Europe/Zurich")
    # DA publish time: 13:00 local on the day before delivery
    da_pub_local = slot_local.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=13)
    da_pub_local = da_pub_local.dt.tz_localize(None).dt.tz_localize("Europe/Zurich", ambiguous="infer",
                                                                      nonexistent="shift_forward")
    da_known = da_pub_local <= bid_local

    # Where DA is known: use actual slot hour; else: same hour-of-day on bid_time date
    slot_hour   = slot_time.dt.floor("h")
    bid_samedayhour = (
        bid_local.dt.normalize() + pd.to_timedelta(slot_local.dt.hour, unit="h")
    ).dt.tz_localize(None).dt.tz_localize("Europe/Zurich", ambiguous="infer",
                                           nonexistent="shift_forward")
    bid_samedayhour_utc = bid_samedayhour.dt.tz_convert("UTC").dt.floor("h")

    spot_vals = pd.Series(
        slot_hour.where(da_known, bid_samedayhour_utc).map(spot_indexed).values,
        index=prices.index,
    )

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

    features = pd.concat([
        prices[["slot_time", "direction", "marginal_chf",
                "offered", "activated", "activation_rate"]],
        cal,
        w.set_index(prices.index),
        res.set_index(prices.index),
        price_lags,
        weekly_vals.set_index(prices.index),
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
