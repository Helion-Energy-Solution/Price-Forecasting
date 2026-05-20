"""
TRL Weekly quantile model — LightGBM with pinball loss.

One model per direction (up/down) × quantile.

Three-way split
---------------
  train   : train_start → es_val_start          (grows tree count via CV)
  es_val  : es_val_start → kpi_val_start        (~52 weeks, monitoring only)
  kpi_val : kpi_val_start → present             (official pinball / revenue KPIs)

n_estimators is selected per-quantile via time-series CV on the training set.
No early-stopping callback on the final fit — avoids noise from the ~5-sample
KPI window and removes the need to sacrifice recent data for stopping signal.

Usage
-----
python src/models/trl_weekly_model.py
"""

import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features" / "trl_weekly_features.parquet"
MODELS_DIR = ROOT / "models" / "trl_weekly"
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.where(e >= 0, q * e, (q - 1) * e).mean())


FEATURE_COLS = [
    "week_of_year", "month", "year", "n_holidays_in_week",
    "cloud_cover_mean", "cloud_cover_std", "cloud_cover_skew", "cloud_cover_p10", "cloud_cover_p90",
    "precip_rate_mmh_mean", "precip_rate_mmh_std", "precip_rate_mmh_skew", "precip_rate_mmh_p10", "precip_rate_mmh_p90",
    "temp_2m_mean", "temp_2m_std", "temp_2m_skew", "temp_2m_p10", "temp_2m_p90",
    "wallis_fill_pct", "graubuenden_fill_pct", "tessin_fill_pct", "totalch_fill_pct",
    "marginal_chf_lag1", "marginal_chf_lag4", "marginal_chf_lag52",
    "marginal_chf_roll4_mean", "marginal_chf_roll4_std",
    "marginal_chf_roll12_mean", "marginal_chf_roll12_std",
]

_BASE_PARAMS = dict(
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
)


def _cv_n_estimators(X: pd.DataFrame, y: pd.Series, q: float, n_splits: int = 3) -> int:
    """
    Time-series CV within the training set to choose n_estimators for quantile q.

    Uses 3 expanding-window folds (largest fold covers ~75 % of training data).
    Early stopping inside each fold provides the stopping signal; we average the
    best_iteration across folds and add a small buffer so the final fit is not
    under-trained relative to the full training set.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_iters = []
    for tr_idx, va_idx in tscv.split(X):
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=800,
            **_BASE_PARAMS,
        )
        m.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        best_iters.append(m.best_iteration_)

    # +20 % buffer: final model trains on more data → needs slightly more trees
    n_est = max(int(np.mean(best_iters) * 1.2), 30)
    log.info("    q=%.2f  CV best_iters=%s  → n_estimators=%d", q, best_iters, n_est)
    return n_est


def train():
    cfg = load_config()
    quantiles  = cfg["training"]["quantiles"]
    model_cfg  = cfg["models"]["trl_weekly"]

    kpi_val_start = pd.Timestamp(model_cfg.get("val_start") or cfg["training"]["val_start"], tz="UTC")
    es_val_start  = pd.Timestamp(model_cfg.get("es_val_start") or kpi_val_start, tz="UTC")
    train_start   = pd.Timestamp(model_cfg["train_start"], tz="UTC")

    df = pd.read_parquet(FEATURES_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"], utc=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for direction in ("up", "down"):
        sub = df[df["direction"] == direction].copy()
        sub = sub[sub["week_start"] >= train_start].dropna(subset=["marginal_chf"])

        train_mask   = sub["week_start"] < es_val_start
        es_val_mask  = (sub["week_start"] >= es_val_start) & (sub["week_start"] < kpi_val_start)
        kpi_val_mask = sub["week_start"] >= kpi_val_start

        X_train = sub.loc[train_mask,   FEATURE_COLS]
        y_train = sub.loc[train_mask,   "marginal_chf"]
        X_es    = sub.loc[es_val_mask,  FEATURE_COLS]
        y_es    = sub.loc[es_val_mask,  "marginal_chf"]
        X_kpi   = sub.loc[kpi_val_mask, FEATURE_COLS]
        y_kpi   = sub.loc[kpi_val_mask, "marginal_chf"]

        log.info(
            "[trl_weekly/%s] train=%d  es_val=%d  kpi_val=%d",
            direction, len(X_train), len(X_es), len(X_kpi),
        )

        models = {}
        for q in quantiles:
            n_est = _cv_n_estimators(X_train, y_train, q)

            model = lgb.LGBMRegressor(
                objective="quantile", alpha=q,
                n_estimators=n_est,
                **_BASE_PARAMS,
            )
            model.fit(X_train, y_train)

            pb_es  = pinball(y_es.values,  model.predict(X_es),  q) if len(y_es)  else float("nan")
            pb_kpi = pinball(y_kpi.values, model.predict(X_kpi), q) if len(y_kpi) else float("nan")
            log.info(
                "  q=%.2f  n_est=%d  pinball_es=%.4f  pinball_kpi=%.4f",
                q, n_est, pb_es, pb_kpi,
            )
            results.append({
                "direction": direction, "quantile": q,
                "n_estimators": n_est, "pinball_es": pb_es, "pinball_kpi": pb_kpi,
            })
            models[q] = model

        out = MODELS_DIR / f"trl_weekly_{direction}.pkl"
        with open(out, "wb") as f:
            pickle.dump(models, f)
        log.info("  Saved → %s", out.name)

    summary = pd.DataFrame(results)
    log.info("\n%s", summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    train()
