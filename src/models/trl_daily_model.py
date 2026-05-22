"""
TRL Daily quantile model — LightGBM with pinball loss.

One model per direction (up/down) × quantile.
Train/val split: config training.val_start.

Usage
-----
python src/models/trl_daily_model.py
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features" / "trl_daily_features.parquet"
MODELS_DIR = ROOT / "models" / "trl_daily"
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.where(e >= 0, q * e, (q - 1) * e).mean())


FEATURE_COLS = [
    "block_of_day", "day_of_week", "month", "is_weekend",
    "is_thursday", "is_friday", "is_holiday", "days_ahead",
    "cos_zenith", "ssrd_proxy", "ssrd_proxy_unc",
    "spot_fcst_std", "spot_fcst_change", "spot_eur_mwh",
    "cloud_cover_mean", "cloud_cover_std", "cloud_cover_skew", "cloud_cover_p10", "cloud_cover_p90",
    "temp_2m_mean", "temp_2m_std", "temp_2m_skew", "temp_2m_p10", "temp_2m_p90",
    "wallis_fill_pct", "graubuenden_fill_pct", "tessin_fill_pct", "totalch_fill_pct",
    "marginal_chf_lag6", "marginal_chf_lag42",
    "marginal_chf_roll42_mean", "marginal_chf_roll42_std",
    "marginal_chf_roll180_mean", "marginal_chf_roll180_std",
    "marginal_chf_sb_roll7_mean", "marginal_chf_sb_roll7_std",
    "marginal_chf_sb_roll28_mean", "marginal_chf_sb_roll28_std",
    "trl_weekly_up_chf", "trl_weekly_up_vwap_chf", "trl_weekly_up_awarded_mw",
    "trl_weekly_down_chf", "trl_weekly_down_vwap_chf", "trl_weekly_down_awarded_mw",
]

# S1 awarded volume only for down — constant 0 for up (no anticipated auction exists).
FEATURE_COLS_BY_DIRECTION = {
    "up":   FEATURE_COLS,
    "down": FEATURE_COLS + ["trl_weekly_down_s1_awarded_mw"],
}


def train():
    cfg = load_config()
    quantiles = cfg["training"]["quantiles"]
    val_start = pd.Timestamp(cfg["training"]["val_start"], tz="UTC")
    train_start = pd.Timestamp(cfg["models"]["trl_daily"]["train_start"], tz="UTC")

    df = pd.read_parquet(FEATURES_PATH)
    df["block_start"] = pd.to_datetime(df["block_start"], utc=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for direction in ("up", "down"):
        fc = FEATURE_COLS_BY_DIRECTION[direction]
        sub = df[df["direction"] == direction].copy()
        sub = sub[sub["block_start"] >= train_start].dropna(subset=["marginal_chf"])

        train_mask = sub["block_start"] < val_start
        val_mask   = sub["block_start"] >= val_start

        X_train = sub.loc[train_mask, fc]
        y_train = sub.loc[train_mask, "marginal_chf"]
        X_val   = sub.loc[val_mask,   fc]
        y_val   = sub.loc[val_mask,   "marginal_chf"]

        log.info("[trl_daily/%s] train=%d  val=%d", direction, len(X_train), len(X_val))

        models = {}
        for q in quantiles:
            params = {
                "objective":    "quantile",
                "alpha":        q,
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves":   63,
                "min_child_samples": 20,
                "subsample":    0.8,
                "colsample_bytree": 0.8,
                "verbose":      -1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            pb = pinball(y_val.values, model.predict(X_val), q)
            pb_norm = pb / float(y_val.mean())
            log.info("  q=%.2f  pinball=%.4f (norm=%.4f)  best_iter=%d", q, pb, pb_norm, model.best_iteration_)
            results.append({"direction": direction, "quantile": q, "pinball_kpi": pb, "pinball_kpi_norm": pb_norm})
            models[q] = model

        out = MODELS_DIR / f"trl_daily_{direction}.pkl"
        with open(out, "wb") as f:
            pickle.dump(models, f)
        log.info("  Saved → %s", out.name)

    summary = pd.DataFrame(results)
    log.info("\n%s", summary.to_string(index=False))

    payload = {
        "timestamp":     datetime.now().strftime("%Y%m%d_%H%M%S"),
        "kpi_val_start": str(val_start.date()),
        "results":       results,
    }
    pinball_path = MODELS_DIR / "pinball_latest.json"
    pinball_path.write_text(json.dumps(payload, indent=2))
    log.info("  Pinball metrics → %s", pinball_path.name)

    return summary


if __name__ == "__main__":
    train()
