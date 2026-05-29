"""
Inference pipeline for Swiss balancing market bid prices.

Downloads today's ECMWF ENS 00z forecast, builds features for upcoming
delivery slots, runs the trained models, applies the pay-as-bid strategy,
and writes JSON forecasts to output/forecasts/.

Output files
------------
output/forecasts/trl_weekly_latest.json
output/forecasts/trl_daily_latest.json
output/forecasts/tre_latest.json

Weather note
------------
Uses ECMWF Open Data (no auth, no embargo). Downloads temp_2m and cloud_cover
only — precipitation is not available from single-step Open Data files.
TRL Weekly precipitation features (precip_rate_mmh_*) will be NaN; LightGBM
routes those rows through its NaN branch, causing a small quality reduction
on TRL Weekly only.

S1 note
-------
TRL Weekly down S1 features default to s1_is_active=0. If the S1 auction has
cleared for the upcoming delivery week, pass --s1-price and --s1-volume.

Usage
-----
python src/pipeline/inference.py
python src/pipeline/inference.py --now "2026-05-26T09:00:00+02:00"
python src/pipeline/inference.py --s1-price 1450.0 --s1-volume 120.0
"""

import argparse
import json
import logging
import pickle
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.feature_store import (
    _to_utc_us, _init_time_for_bid, _weather_agg, _nearest_weather,
    _reservoir_asof, _load_spot_forecast, _spot_forecast_asof,
    _spot_week_features, _price_lags, _price_lags_same_block,
    _push_back_past_holidays, _swiss_holiday_set, _tre_bid_time, cos_zenith,
    _entsoe_tre_features, _entsoe_week_features,
)
from src.data.ecds_download import process_opendata_run
from src.models.trl_weekly_model import FEATURE_COLS_BY_DIRECTION as TW_FC
from src.models.trl_daily_model import FEATURE_COLS_BY_DIRECTION as TD_FC
from src.models.tre_model import FEATURE_COLS as TRE_FC
from src.pipeline.bid_strategy import _opt_bid_pos, _opt_bid_neg, _predict_pos, _predict_neg_extreme

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = ROOT / "models"
PRICES_DIR = ROOT / "data" / "raw" / "prices"
MARKET_DIR = ROOT / "data" / "raw" / "market"
OUTPUT_DIR = ROOT / "output" / "forecasts"
CONFIG_PATH = ROOT / "config" / "config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Weather download and pivot
# ---------------------------------------------------------------------------

def _weather_wide_from_df(rows_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot raw weather rows (from process_opendata_run) to wide format."""
    if rows_df.empty:
        return rows_df
    rows_df = rows_df.copy()
    rows_df["init_time"]  = pd.to_datetime(rows_df["init_time"],  utc=True).astype("datetime64[us, UTC]")
    rows_df["valid_time"] = pd.to_datetime(rows_df["valid_time"], utc=True).astype("datetime64[us, UTC]")
    wide = rows_df.pivot_table(
        index=["init_time", "valid_time", "lead_hours"],
        columns="variable",
        values=["mean", "std", "skew", "p10", "p90"],
        aggfunc="first",
    )
    wide.columns = [f"{var}_{stat}" for stat, var in wide.columns]
    return wide.reset_index().sort_values(["init_time", "valid_time"])


def download_inference_weather(run_date: date, run_hour: int = 0) -> pd.DataFrame:
    """Download the most recent available ENS from ECMWF Open Data.

    Tries run_date 00z first, then steps back through recent 12z/00z runs
    (up to 3 days) before falling back to weather_ensemble.parquet.
    """
    cfg = _load_cfg()

    # Most-recent-first: today 00z, yesterday 12z, yesterday 00z, ...
    candidates = [(run_date, 0)]
    for days_back in range(1, 4):
        d = run_date - timedelta(days=days_back)
        candidates += [(d, 12), (d, 0)]

    for d, h in candidates:
        log.info("Downloading ECMWF Open Data ENS %s %02dz ...", d, h)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rows_df = process_opendata_run(cfg, d, h)
            if rows_df.empty:
                log.warning("  No rows returned — trying next candidate")
                continue
            wide = _weather_wide_from_df(rows_df)
            log.info("  %d steps, %d wide rows", rows_df["lead_hours"].nunique(), len(wide))
            return wide
        except Exception as exc:
            log.warning("  Failed (%s) — trying next candidate", exc)

    log.warning("All ECMWF candidates failed — using weather_ensemble.parquet fallback")
    parquet = ROOT / "data" / "processed" / "features" / "weather_ensemble.parquet"
    if not parquet.exists():
        raise RuntimeError("No weather_ensemble.parquet fallback available.")
    from src.data.feature_store import load_weather_wide
    full = load_weather_wide()
    latest = full["init_time"].max()
    log.info("  Fallback: using init_time=%s", latest)
    return full[full["init_time"] == latest].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Delivery-period helpers
# ---------------------------------------------------------------------------

def _next_weekly_delivery(now: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (week_start, bid_time) for the next TRL Weekly delivery.

    Bid gate closes Tuesday 14:00 Europe/Zurich.
    week_start = bid_day + 6 days (Monday of delivery week).
    bid_time returned as midnight UTC of the Tuesday bid day — consistent with
    how feature_store.py computes init_time via _init_time_for_bid.
    """
    local = now.tz_convert("Europe/Zurich")
    days_to_tue = (1 - local.weekday()) % 7          # 0 if today is Tuesday
    bid_local   = local.normalize() + pd.Timedelta(days=days_to_tue)
    gate_local  = bid_local + pd.Timedelta(hours=14)
    if local >= gate_local:
        bid_local += pd.Timedelta(days=7)             # gate already closed; go to next week
    week_start_local = bid_local + pd.Timedelta(days=6)
    bid_utc        = _to_utc_us(pd.Series([bid_local.tz_convert("UTC")])).iloc[0]
    week_start_utc = _to_utc_us(pd.Series([week_start_local.normalize().tz_convert("UTC")])).iloc[0]
    return week_start_utc, bid_utc


def _trl_daily_future_blocks(now: pd.Timestamp, lookahead_days: int = 7) -> pd.DataFrame:
    """Return biddable TRL Daily delivery blocks within the next lookahead_days.

    Columns: block_start, bid_time (both UTC).
    """
    _days_back = {0: 3, 1: 4, 2: 2, 3: 2, 4: 2, 5: 2, 6: 3}
    local_now = now.tz_convert("Europe/Zurich")
    block_starts_local = [
        local_now.normalize() + pd.Timedelta(days=d, hours=h)
        for d in range(lookahead_days + 1)
        for h in range(0, 24, 4)
    ]
    block_series = _to_utc_us(pd.Series([b.tz_convert("UTC") for b in block_starts_local]))
    local_del    = block_series.dt.tz_convert("Europe/Zurich")
    days_back    = local_del.dt.dayofweek.map(_days_back)
    bid_local    = (local_del.dt.normalize()
                   - pd.to_timedelta(days_back, unit="D")
                   + pd.Timedelta(hours=14))
    bid_times    = _to_utc_us(bid_local)
    years        = set(bid_times.dt.tz_convert("Europe/Zurich").dt.year)
    bid_times    = _push_back_past_holidays(bid_times, _swiss_holiday_set(years), fallback_hour=14)
    mask         = bid_times >= now
    return pd.DataFrame({
        "block_start": block_series[mask].reset_index(drop=True),
        "bid_time":    bid_times[mask].reset_index(drop=True),
    })


def _tre_future_slots(now: pd.Timestamp) -> pd.DataFrame:
    """Return biddable TRE 15-min slots from now until end of the current bidding window.

    Window closes Friday 17:00 local; slots through Monday 10:00 local are
    included to cover the Friday→Monday gap.
    Columns: slot_time, bid_time (both UTC).
    """
    local_now = now.tz_convert("Europe/Zurich")
    days_to_fri = (4 - local_now.weekday()) % 7
    fri_close   = local_now.normalize() + pd.Timedelta(days=days_to_fri, hours=17)
    if local_now >= fri_close:
        fri_close += pd.Timedelta(days=7)
    window_end  = fri_close.normalize() + pd.Timedelta(days=3, hours=10)  # Monday 10:00

    slot_start_local = (now + pd.Timedelta(hours=1)).tz_convert("Europe/Zurich").ceil("15min")
    slot_times_local = pd.date_range(slot_start_local, window_end, freq="15min")
    slot_times = _to_utc_us(pd.Series(slot_times_local.tz_convert("UTC")))

    bid_times = _tre_bid_time(slot_times)
    years     = set(bid_times.dt.tz_convert("Europe/Zurich").dt.year)
    bid_times = _push_back_past_holidays(bid_times, _swiss_holiday_set(years), fallback_hour=17)
    mask      = bid_times >= now
    return pd.DataFrame({
        "slot_time": slot_times[mask].reset_index(drop=True),
        "bid_time":  bid_times[mask].reset_index(drop=True),
    })


# ---------------------------------------------------------------------------
# Lag feature helpers (dummy-row approach)
# ---------------------------------------------------------------------------

def _weekly_lags(direction: str, future_week_start: pd.Timestamp) -> pd.Series:
    """Compute TRL Weekly lag features for one future delivery week.

    Appends a dummy NaN row and runs _price_lags so the future row inherits
    its lags from history.
    """
    prices = pd.read_parquet(PRICES_DIR / "trl_weekly.parquet")
    prices["week_start"] = _to_utc_us(pd.to_datetime(prices["week_start"], utc=True))
    sub = prices[prices["direction"] == direction][["week_start", "marginal_chf"]].copy()
    sub = pd.concat(
        [sub, pd.DataFrame({"week_start": [future_week_start], "marginal_chf": [np.nan]})],
        ignore_index=True,
    ).sort_values("week_start").reset_index(drop=True)
    return _price_lags(sub, "week_start", "marginal_chf", lags=[1, 4, 52], roll_windows=[4, 12]).iloc[-1]


def _daily_lags(direction: str, future_blocks: pd.Series) -> pd.DataFrame:
    """Compute TRL Daily lag + same-block-rolling features for all future blocks at once.

    Keeps uncleared (NaN-price) history so shift/rolling windows stay calendar-aligned,
    exactly as feature_store builds them at training time. Future rows are identified by
    block_start membership (not isna), since uncleared history is also NaN.
    """
    prices = pd.read_parquet(PRICES_DIR / "trl_daily.parquet")
    prices["block_start"] = _to_utc_us(pd.to_datetime(prices["block_start"], utc=True))
    sub = prices[prices["direction"] == direction][["block_start", "marginal_chf"]].copy()
    future_set = set(future_blocks)
    sub = sub[~sub["block_start"].isin(future_set)]                       # avoid duplicate boundary rows
    future_df = pd.DataFrame({"block_start": list(future_set), "marginal_chf": np.nan})
    combined  = pd.concat([sub, future_df], ignore_index=True).sort_values("block_start").reset_index(drop=True)
    lags    = _price_lags(combined, "block_start", "marginal_chf", lags=[6, 42], roll_windows=[42, 180])
    sb_lags = _price_lags_same_block(combined, "block_start", "marginal_chf", roll_windows=[7, 28])
    is_future = combined["block_start"].isin(future_set).values
    return pd.concat([lags[is_future], sb_lags[is_future]], axis=1).reset_index(drop=True)


def _tre_lags(direction: str, future_slots: pd.Series) -> pd.DataFrame:
    """Compute TRE lag features for all future slots at once.

    Keeps unactivated (NaN-price) history so shift/rolling windows stay calendar-aligned,
    exactly as feature_store builds them at training time. Future rows are identified by
    slot_time membership (not isna), since unactivated history is also NaN.
    """
    prices = pd.read_parquet(PRICES_DIR / "tre_slots.parquet")
    prices["slot_time"] = _to_utc_us(pd.to_datetime(prices["slot_time"], utc=True))
    sub = prices[prices["direction"] == direction][["slot_time", "marginal_chf"]].copy()
    future_set = set(future_slots)
    sub = sub[~sub["slot_time"].isin(future_set)]                         # avoid duplicate boundary rows
    future_df = pd.DataFrame({"slot_time": list(future_set), "marginal_chf": np.nan})
    combined  = pd.concat([sub, future_df], ignore_index=True).sort_values("slot_time").reset_index(drop=True)
    lags = _price_lags(combined, "slot_time", "marginal_chf", lags=[], roll_windows=[96, 672])
    lags["marginal_chf_lag96h"] = combined["marginal_chf"].shift(96).rolling(4, min_periods=1).mean()
    is_future = combined["slot_time"].isin(future_set).values
    return lags[is_future].reset_index(drop=True)


# ---------------------------------------------------------------------------
# TRL Weekly inference
# ---------------------------------------------------------------------------

def infer_trl_weekly(
    weather: pd.DataFrame,
    now: pd.Timestamp,
    cfg: dict,
    s1_price: float | None = None,
    s1_volume: float | None = None,
) -> dict:
    week_start, bid_time = _next_weekly_delivery(now)
    log.info("[TRL Weekly] delivery_week=%s  bid_deadline=%s", week_start.date(), bid_time.date())

    quantiles  = cfg["training"]["quantiles"]
    init_time  = _init_time_for_bid(pd.Series([bid_time]))
    valid_start = pd.Series([week_start])
    valid_end   = pd.Series([week_start + pd.Timedelta(days=7)])

    weather_cols = [c for c in weather.columns if c not in ("init_time", "valid_time", "lead_hours")]
    w   = _weather_agg(weather, init_time, valid_start, valid_end, weather_cols)
    res = _reservoir_asof(pd.read_parquet(MARKET_DIR / "reservoir_levels.parquet"), pd.Series([bid_time]))
    spot_week = _spot_week_features(_load_spot_forecast(), pd.Series([bid_time]), pd.Series([week_start]))

    local_ws = week_start.tz_convert("Europe/Zurich")
    years    = {local_ws.year, (local_ws + pd.Timedelta(days=6)).year}
    holiday_set = _swiss_holiday_set(years)
    cal = {
        "week_of_year":       int(local_ws.isocalendar()[1]),
        "month":              int(local_ws.month),
        "year":               int(local_ws.year),
        "n_holidays_in_week": sum(1 for i in range(7) if (local_ws.date() + timedelta(days=i)) in holiday_set),
    }
    s1_is_active = 1 if (s1_price is not None and s1_volume is not None) else 0
    s1_feats = {
        "s1_is_active":    s1_is_active,
        "s1_awarded_mw":   float(s1_volume) if s1_volume else 0.0,
        "s1_marginal_chf": float(s1_price)  if s1_price  else 0.0,
        "s1_vwap_chf":     float(s1_price)  if s1_price  else 0.0,
    }

    gate_utc = (bid_time.tz_convert("Europe/Zurich") + pd.Timedelta(hours=14)).tz_convert("UTC")
    output = {
        "delivery_week_start": week_start.tz_convert("Europe/Zurich").strftime("%Y-%m-%d"),
        "bid_deadline":        gate_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for direction in ("up", "down"):
        fc   = TW_FC[direction]
        lags = _weekly_lags(direction, week_start)

        row = {**cal}
        for col in weather_cols:
            row[col] = float(w[col].iloc[0]) if col in w.columns else np.nan
        for col in ["wallis_fill_pct", "graubuenden_fill_pct", "tessin_fill_pct", "totalch_fill_pct"]:
            row[col] = float(res[col].iloc[0]) if col in res.columns else np.nan
        for col in spot_week.columns:
            row[col] = float(spot_week[col].iloc[0])
        for col, val in lags.items():
            row[col] = float(val)
        if direction == "down":
            row.update(s1_feats)

        X = pd.DataFrame([{c: row.get(c, np.nan) for c in fc}])
        with open(MODELS_DIR / "trl_weekly" / f"trl_weekly_{direction}.pkl", "rb") as f:
            models = pickle.load(f)

        q_preds = np.array([models[q].predict(X)[0] for q in quantiles])
        q_vals  = {f"q{int(q*100):02d}": round(float(p), 2) for q, p in zip(quantiles, q_preds)}
        opt_bid = _opt_bid_pos(q_preds, quantiles, opp_cost=0.0)
        output[direction] = {"quantiles": q_vals, "optimal_bid": round(opt_bid, 2)}
        log.info("  %s  q50=%.1f  opt_bid=%.1f", direction, q_vals["q50"], opt_bid)

    return output


# ---------------------------------------------------------------------------
# TRL Daily inference
# ---------------------------------------------------------------------------

def infer_trl_daily(weather: pd.DataFrame, now: pd.Timestamp, cfg: dict) -> list[dict]:
    quantiles = cfg["training"]["quantiles"]
    blocks_df = _trl_daily_future_blocks(now)
    if blocks_df.empty:
        log.warning("[TRL Daily] no biddable blocks found")
        return []

    n = len(blocks_df)
    log.info("[TRL Daily] %d blocks (%s → %s)", n,
             blocks_df["block_start"].iloc[0].date(), blocks_df["block_start"].iloc[-1].date())

    block_start = blocks_df["block_start"]
    bid_times   = blocks_df["bid_time"]
    block_end   = block_start + pd.Timedelta(hours=4)
    init_times  = _init_time_for_bid(bid_times)

    weather_cols = [c for c in weather.columns
                    if c not in ("init_time", "valid_time", "lead_hours") and "precip" not in c]
    w   = _nearest_weather(weather, init_times, block_start + pd.Timedelta(hours=2), weather_cols)
    res = _reservoir_asof(pd.read_parquet(MARKET_DIR / "reservoir_levels.parquet"), block_start)

    spot_fcst  = _load_spot_forecast()
    _SPOT_COLS = ["price_eur_mwh", "spot_fcst_std", "spot_fcst_change"]
    hour_fcsts = [
        _spot_forecast_asof(spot_fcst, bid_times,
                            (block_start + pd.Timedelta(hours=h)).dt.floor("h"),
                            return_cols=_SPOT_COLS)
        for h in range(4)
    ]
    spot_vals = pd.concat([d["price_eur_mwh"]    for d in hour_fcsts], axis=1).mean(axis=1)
    spot_std  = pd.concat([d["spot_fcst_std"]    for d in hour_fcsts], axis=1).mean(axis=1)
    spot_chg  = pd.concat([d["spot_fcst_change"] for d in hour_fcsts], axis=1).mean(axis=1)

    czn = np.array([
        cos_zenith(pd.date_range(bs, be, freq="15min", inclusive="left")).mean()
        for bs, be in zip(block_start, block_end)
    ])
    cloud_mean = w["cloud_cover_mean"].values if "cloud_cover_mean" in w.columns else np.zeros(n)
    cloud_std  = w["cloud_cover_std"].values  if "cloud_cover_std"  in w.columns else np.zeros(n)

    local      = block_start.dt.tz_convert("Europe/Zurich")
    years      = set(local.dt.year)
    holiday_set = _swiss_holiday_set(years)
    days_ahead = (
        (local.dt.normalize() - bid_times.dt.tz_convert("Europe/Zurich").dt.normalize()).dt.days.values
    )
    cal_df = pd.DataFrame({
        "block_of_day":   (local.dt.hour // 4).astype(int).values,
        "day_of_week":    local.dt.dayofweek.values,
        "month":          local.dt.month.values,
        "is_weekend":     (local.dt.dayofweek >= 5).astype(int).values,
        "is_thursday":    (local.dt.dayofweek == 3).astype(int).values,
        "is_friday":      (local.dt.dayofweek == 4).astype(int).values,
        "is_holiday":     [int(d in holiday_set) for d in local.dt.date],
        "days_ahead":     days_ahead,
        "cos_zenith":     czn,
        "ssrd_proxy":     czn * (1.0 - cloud_mean),
        "ssrd_proxy_unc": czn * cloud_std,
        "spot_eur_mwh":     spot_vals.values,
        "spot_fcst_std":    spot_std.values,
        "spot_fcst_change": spot_chg.values,
    })

    # TRL Weekly context: as-of join using most recent known weekly clearing
    trl_weekly_raw = pd.read_parquet(PRICES_DIR / "trl_weekly.parquet")
    trl_weekly_raw["week_start"] = _to_utc_us(pd.to_datetime(trl_weekly_raw["week_start"], utc=True))
    if "s1_awarded_mw" in trl_weekly_raw.columns:
        s1_active = trl_weekly_raw.get("s1_is_active", pd.Series(0, index=trl_weekly_raw.index))
        trl_weekly_raw["s1_awarded_mw"] = trl_weekly_raw["s1_awarded_mw"].where(s1_active == 1, 0.0)
    pivot_vals = ["marginal_chf", "vwap_chf", "awarded_mw"] + (
        ["s1_awarded_mw"] if "s1_awarded_mw" in trl_weekly_raw.columns else []
    )
    trl_wide = trl_weekly_raw.pivot_table(
        index="week_start", columns="direction", values=pivot_vals, aggfunc="first"
    )
    _nm = {"marginal_chf": "chf", "vwap_chf": "vwap_chf", "awarded_mw": "awarded_mw", "s1_awarded_mw": "s1_awarded_mw"}
    trl_wide.columns = [f"trl_weekly_{d}_{_nm[v]}" for v, d in trl_wide.columns]
    trl_wide = trl_wide.reset_index().sort_values("week_start")

    block_week_start = _to_utc_us(
        (block_start - pd.to_timedelta(block_start.dt.dayofweek, unit="D")).dt.normalize()
    )
    weekly_cols = [c for c in trl_wide.columns if c != "week_start"]
    weekly_vals = pd.merge_asof(
        pd.DataFrame({"week_start": block_week_start.reset_index(drop=True)}).reset_index(names="row_idx"),
        trl_wide, on="week_start", direction="backward",
    ).set_index("row_idx")[weekly_cols]

    # Precompute lags for all blocks (one parquet read per direction)
    lag_frames = {d: _daily_lags(d, block_start.reset_index(drop=True)) for d in ("up", "down")}
    res_cols = ["wallis_fill_pct", "graubuenden_fill_pct", "tessin_fill_pct", "totalch_fill_pct"]

    # ENTSO-E CH week-ahead load forecast (point-in-time)
    entsoe = _entsoe_week_features(bid_times, block_start)

    # Load models
    bundles = {}
    for direction in ("up", "down"):
        with open(MODELS_DIR / "trl_daily" / f"trl_daily_{direction}.pkl", "rb") as f:
            bundles[direction] = pickle.load(f)

    # Batch predict per direction for efficiency
    X_dir = {}
    for direction in ("up", "down"):
        fc = TD_FC[direction]
        rows = []
        for i in range(n):
            row = {}
            for col in cal_df.columns:
                row[col] = float(cal_df[col].iloc[i])
            for col in weather_cols:
                row[col] = float(w[col].iloc[i]) if col in w.columns else np.nan
            for col in res_cols:
                row[col] = float(res[col].iloc[i]) if col in res.columns else np.nan
            for col in weekly_cols:
                row[col] = float(weekly_vals[col].iloc[i]) if col in weekly_vals.columns else np.nan
            for col in lag_frames[direction].columns:
                row[col] = float(lag_frames[direction][col].iloc[i])
            for col in entsoe.columns:
                row[col] = float(entsoe[col].iloc[i])
            rows.append({c: row.get(c, np.nan) for c in fc})
        X_dir[direction] = pd.DataFrame(rows)

    preds = {}
    for direction in ("up", "down"):
        preds[direction] = np.column_stack([
            bundles[direction][q].predict(X_dir[direction]) for q in quantiles
        ])  # shape (n, n_q)

    output_blocks = []
    for i in range(n):
        row = blocks_df.iloc[i]
        block_out = {
            "block_start":  row["block_start"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "block_end":    (row["block_start"] + pd.Timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bid_deadline": row["bid_time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        for direction in ("up", "down"):
            q_preds = preds[direction][i]
            q_vals  = {f"q{int(q*100):02d}": round(float(p), 2) for q, p in zip(quantiles, q_preds)}
            opt_bid = _opt_bid_pos(q_preds, quantiles, opp_cost=0.0)
            block_out[direction] = {"quantiles": q_vals, "optimal_bid": round(opt_bid, 2)}
        output_blocks.append(block_out)

    log.info("  Forecasted %d blocks", n)
    return output_blocks


# ---------------------------------------------------------------------------
# TRE inference
# ---------------------------------------------------------------------------

def infer_tre(weather: pd.DataFrame, now: pd.Timestamp, cfg: dict) -> list[dict]:
    quantiles = cfg["training"]["quantiles"]
    slots_df  = _tre_future_slots(now)
    if slots_df.empty:
        log.warning("[TRE] no biddable slots found")
        return []

    n = len(slots_df)
    log.info("[TRE] %d slots (%s → %s)", n,
             slots_df["slot_time"].iloc[0].strftime("%Y-%m-%d %H:%M"),
             slots_df["slot_time"].iloc[-1].strftime("%Y-%m-%d %H:%M"))

    slot_time  = slots_df["slot_time"]
    bid_time   = slots_df["bid_time"]
    init_times = _init_time_for_bid(bid_time)
    lead_hours = (slot_time - init_times).dt.total_seconds() / 3600
    hours_until = (slot_time - bid_time).dt.total_seconds() / 3600

    weather_cols = [c for c in weather.columns
                    if c not in ("init_time", "valid_time", "lead_hours") and "precip" not in c]
    w   = _nearest_weather(weather, init_times, slot_time, weather_cols)
    res = _reservoir_asof(pd.read_parquet(MARKET_DIR / "reservoir_levels.parquet"), slot_time)

    # Spot: realized DA if known, Volue forecast otherwise
    spot_raw = pd.read_parquet(PRICES_DIR / "spot_hourly.parquet")
    spot_raw["hour_time"] = pd.to_datetime(spot_raw["hour_time"], utc=True)
    spot_indexed = spot_raw.drop_duplicates("hour_time").set_index("hour_time")["price_eur_mwh"]

    spot_fcst  = _load_spot_forecast()
    slot_hour  = slot_time.dt.floor("h")
    slot_local = slot_time.dt.tz_convert("Europe/Zurich")
    bid_local  = bid_time.dt.tz_convert("Europe/Zurich")
    da_pub     = (slot_local.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=13))
    da_pub     = da_pub.dt.tz_localize(None).dt.tz_localize(
        "Europe/Zurich", ambiguous="infer", nonexistent="shift_forward"
    )
    da_known  = da_pub <= bid_local
    _SPOT_COLS = ["price_eur_mwh", "spot_fcst_std", "spot_fcst_change"]
    fcst_df    = _spot_forecast_asof(spot_fcst, bid_time, slot_hour, return_cols=_SPOT_COLS)
    spot_vals  = pd.Series(
        np.where(da_known, slot_hour.map(spot_indexed), fcst_df["price_eur_mwh"].values),
        index=slots_df.index,
    )

    # TRL Weekly context (most recent known clearing, as-of)
    trl_weekly_raw = pd.read_parquet(PRICES_DIR / "trl_weekly.parquet")
    trl_weekly_raw["week_start"] = _to_utc_us(pd.to_datetime(trl_weekly_raw["week_start"], utc=True))
    trl_wide = (
        trl_weekly_raw.pivot_table(index="week_start", columns="direction",
                                   values="marginal_chf", aggfunc="first")
        .rename(columns={"up": "trl_weekly_up_chf", "down": "trl_weekly_down_chf"})
        .reset_index().sort_values("week_start")
    )
    slot_week_start = _to_utc_us(
        (slot_time - pd.to_timedelta(slot_time.dt.dayofweek, unit="D")).dt.normalize()
    )
    weekly_vals = pd.merge_asof(
        pd.DataFrame({"week_start": slot_week_start.reset_index(drop=True)}).reset_index(names="row_idx"),
        trl_wide, on="week_start", direction="backward",
    ).set_index("row_idx")[["trl_weekly_up_chf", "trl_weekly_down_chf"]]

    local = slot_time.dt.tz_convert("Europe/Zurich")
    years = set(local.dt.year)
    holiday_set = _swiss_holiday_set(years)
    czn        = cos_zenith(pd.DatetimeIndex(slot_time))
    cloud_mean = w["cloud_cover_mean"].values if "cloud_cover_mean" in w.columns else np.zeros(n)
    cloud_std  = w["cloud_cover_std"].values  if "cloud_cover_std"  in w.columns else np.zeros(n)

    cal_df = pd.DataFrame({
        "quarter_of_hour":      (local.dt.minute // 15).astype(int).values,
        "hour_of_day":          local.dt.hour.values,
        "day_of_week":          local.dt.dayofweek.values,
        "month":                local.dt.month.values,
        "is_weekend":           (local.dt.dayofweek >= 5).astype(int).values,
        "is_friday":            (local.dt.dayofweek == 4).astype(int).values,
        "is_holiday":           [int(d in holiday_set) for d in local.dt.date],
        "lead_hours":           lead_hours.values,
        "hours_until_delivery": hours_until.values,
        "cos_zenith":           czn,
        "ssrd_proxy":           czn * (1.0 - cloud_mean),
        "ssrd_proxy_unc":       czn * cloud_std,
        "spot_is_realized":     da_known.astype(int).values,
        "spot_fcst_std":        fcst_df["spot_fcst_std"].values,
        "spot_fcst_change":     fcst_df["spot_fcst_change"].values,
        "spot_eur_mwh":         spot_vals.values,
    })

    res_cols = ["wallis_fill_pct", "graubuenden_fill_pct", "tessin_fill_pct", "totalch_fill_pct"]

    # ENTSO-E CH load/generation forecasts (point-in-time): day-ahead + week-ahead
    entsoe = _entsoe_tre_features(bid_time, slot_time)

    # Precompute lags and batch feature matrices per direction
    lag_frames = {d: _tre_lags(d, slot_time.reset_index(drop=True)) for d in ("pos", "neg")}

    X_dir = {}
    for direction in ("pos", "neg"):
        rows = []
        for i in range(n):
            row = {}
            for col in cal_df.columns:
                row[col] = float(cal_df[col].iloc[i])
            for col in weather_cols:
                row[col] = float(w[col].iloc[i]) if col in w.columns else np.nan
            for col in res_cols:
                row[col] = float(res[col].iloc[i]) if col in res.columns else np.nan
            row["trl_weekly_up_chf"]   = float(weekly_vals["trl_weekly_up_chf"].iloc[i])
            row["trl_weekly_down_chf"] = float(weekly_vals["trl_weekly_down_chf"].iloc[i])
            for col in lag_frames[direction].columns:
                row[col] = float(lag_frames[direction][col].iloc[i])
            for col in entsoe.columns:
                row[col] = float(entsoe[col].iloc[i])
            rows.append({c: row.get(c, np.nan) for c in TRE_FC})
        X_dir[direction] = pd.DataFrame(rows)

    bundles = {}
    for direction in ("pos", "neg"):
        with open(MODELS_DIR / "tre" / f"tre_{direction}.pkl", "rb") as f:
            bundles[direction] = pickle.load(f)

    # neg display: classifier picks regime; normal model when p_extreme < 0.5,
    #              extreme model when p_extreme >= 0.5. Conditional quantiles, no blending.
    # neg bid:     always extreme-regime (correct for pay-as-bid curtailment strategy)
    # p_extreme:   classifier output — context for interpreting the displayed quantiles
    preds_pos     = _predict_pos(bundles["pos"], X_dir["pos"], quantiles)
    p_extreme_neg = bundles["neg"]["clf"].predict_proba(X_dir["neg"])[:, 1]
    preds_neg_normal = np.column_stack([bundles["neg"]["normal"][q].predict(X_dir["neg"]) for q in quantiles])
    preds_neg_bid    = _predict_neg_extreme(bundles["neg"], X_dir["neg"], quantiles)
    is_extreme       = (p_extreme_neg >= 0.5)[:, np.newaxis]
    preds_neg_disp   = np.where(is_extreme, preds_neg_bid, preds_neg_normal)

    output_slots = []
    for i in range(n):
        row = slots_df.iloc[i]
        slot_out = {
            "slot_time":    row["slot_time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bid_deadline": row["bid_time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        q_pos          = preds_pos[i]
        q_neg_display  = preds_neg_disp[i]
        q_neg_bid      = preds_neg_bid[i]
        slot_out["pos"] = {
            "quantiles":  {f"q{int(q*100):02d}": round(float(p), 2) for q, p in zip(quantiles, q_pos)},
            "optimal_bid": round(_opt_bid_pos(q_pos, quantiles, opp_cost=0.0), 2),
        }
        slot_out["neg"] = {
            "quantiles":   {f"q{int(q*100):02d}": round(float(p), 2) for q, p in zip(quantiles, q_neg_display)},
            "p_extreme":   round(float(p_extreme_neg[i]), 4),
            "optimal_bid": round(_opt_bid_neg(q_neg_bid, quantiles, opp_cost=200.0), 2),
        }
        output_slots.append(slot_out)

    log.info("  Forecasted %d slots", n)
    return output_slots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run inference for all three balancing markets")
    parser.add_argument("--now",       default=None,
                        help="Override current time (ISO 8601, e.g. '2026-05-26T09:00:00+02:00')")
    parser.add_argument("--s1-price",  type=float, default=None,
                        help="S1 marginal price CHF/MW for TRL Weekly down (omit if inactive)")
    parser.add_argument("--s1-volume", type=float, default=None,
                        help="S1 awarded volume MW for TRL Weekly down (omit if inactive)")
    args = parser.parse_args()

    now = (pd.Timestamp.now(tz="UTC") if args.now is None
           else pd.Timestamp(args.now).tz_convert("UTC"))
    log.info("Reference time: %s", now.isoformat())

    cfg      = _load_cfg()
    run_date = now.tz_convert("Europe/Zurich").date()
    weather  = download_inference_weather(run_date, run_hour=0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== TRL Weekly ===")
    weekly_out = infer_trl_weekly(weather, now, cfg, args.s1_price, args.s1_volume)
    (OUTPUT_DIR / "trl_weekly_latest.json").write_text(json.dumps(
        {"generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "weather_run":  f"{run_date} 00z",
         **weekly_out}, indent=2))

    log.info("=== TRL Daily ===")
    daily_blocks = infer_trl_daily(weather, now, cfg)
    (OUTPUT_DIR / "trl_daily_latest.json").write_text(json.dumps(
        {"generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "weather_run":  f"{run_date} 00z",
         "blocks":       daily_blocks}, indent=2))

    log.info("=== TRE ===")
    tre_slots = infer_tre(weather, now, cfg)
    (OUTPUT_DIR / "tre_latest.json").write_text(json.dumps(
        {"generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
         "weather_run":  f"{run_date} 00z",
         "slots":        tre_slots}, indent=2))

    log.info("Done — outputs written to %s", OUTPUT_DIR)
    log.info("  trl_weekly_latest.json  1 week, 2 directions")
    log.info("  trl_daily_latest.json   %d blocks", len(daily_blocks))
    log.info("  tre_latest.json         %d slots",  len(tre_slots))


if __name__ == "__main__":
    main()
