"""
TRE two-stage quantile model.

Stage 1 — Binary classifier: is_extreme
  pos direction: price > 300 CHF/MWh
  neg direction: price < -200 CHF/MWh

Stage 2a — Normal-regime quantile model (trained on non-extreme subset only)
Stage 2b — Extreme-regime quantile model (trained on extreme subset only)

Prediction for quantile q:
  p_ext  = classifier P(extreme)
  pred(q) = (1 - p_ext) * normal_model[q](X) + p_ext * extreme_model[q](X)

Models saved per direction as a dict:
  {"clf": LGBMClassifier, "normal": {q: LGBMRegressor}, "extreme": {q: LGBMRegressor}}

Usage
-----
python src/models/tre_model.py
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
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features" / "tre_features.parquet"
MODELS_DIR = ROOT / "models" / "tre"
CONFIG_PATH = ROOT / "config" / "config.yaml"

def _load_thresholds() -> dict:
    with open(CONFIG_PATH) as f:
        c = yaml.safe_load(f)
    tre = c["models"]["tre"]
    return {"pos": tre.get("extreme_threshold_pos", 300.0),
            "neg": tre.get("extreme_threshold_neg", -200.0)}

THRESHOLDS = _load_thresholds()

FEATURE_COLS = [
    "quarter_of_hour", "hour_of_day", "day_of_week", "month",
    "is_weekend", "is_friday", "is_holiday",
    "lead_hours", "hours_until_delivery",
    "cos_zenith", "ssrd_proxy", "ssrd_proxy_unc",
    "spot_is_realized", "spot_fcst_std", "spot_fcst_change", "spot_eur_mwh",
    "cloud_cover_mean", "cloud_cover_std", "cloud_cover_skew", "cloud_cover_p10", "cloud_cover_p90",
    "temp_2m_mean", "temp_2m_std", "temp_2m_skew", "temp_2m_p10", "temp_2m_p90",
    "wallis_fill_pct", "graubuenden_fill_pct", "tessin_fill_pct", "totalch_fill_pct",
    "marginal_chf_lag96h",
    "marginal_chf_roll96_mean", "marginal_chf_roll96_std",
    "marginal_chf_roll672_mean", "marginal_chf_roll672_std",
    "trl_weekly_up_chf", "trl_weekly_down_chf",
]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = y_true - y_pred
    return float(np.where(e >= 0, q * e, (q - 1) * e).mean())


def _extreme_mask(prices: pd.Series, direction: str) -> pd.Series:
    t = THRESHOLDS[direction]
    return prices > t if direction == "pos" else prices < t


def train():
    cfg = load_config()
    quantiles = cfg["training"]["quantiles"]
    val_start  = pd.Timestamp(cfg["training"]["val_start"], tz="UTC")
    train_start = pd.Timestamp(cfg["models"]["tre"]["train_start"], tz="UTC")

    df = pd.read_parquet(FEATURES_PATH)
    df["slot_time"] = pd.to_datetime(df["slot_time"], utc=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for direction in ("pos", "neg"):
        sub = df[df["direction"] == direction].copy()
        sub = sub[sub["slot_time"] >= train_start].dropna(subset=["marginal_chf"] + FEATURE_COLS)

        is_ext = _extreme_mask(sub["marginal_chf"], direction)

        train_mask = sub["slot_time"] < val_start
        val_mask   = sub["slot_time"] >= val_start

        X_train = sub.loc[train_mask, FEATURE_COLS]
        y_train = sub.loc[train_mask, "marginal_chf"]
        X_val   = sub.loc[val_mask,   FEATURE_COLS]
        y_val   = sub.loc[val_mask,   "marginal_chf"]
        ext_train = is_ext[train_mask].astype(int)
        ext_val   = is_ext[val_mask].astype(int)

        n_ext_tr = ext_train.sum()
        n_ext_va = ext_val.sum()
        log.info("[tre/%s] train=%d (extreme=%d, %.1f%%)  val=%d (extreme=%d, %.1f%%)",
                 direction, len(X_train), n_ext_tr, 100 * n_ext_tr / len(X_train),
                 len(X_val), n_ext_va, 100 * n_ext_va / len(X_val))

        # ------------------------------------------------------------------ #
        # Stage 1 — Classifier                                                #
        # ------------------------------------------------------------------ #
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=max((len(ext_train) - n_ext_tr) / max(n_ext_tr, 1), 1),
            verbose=-1,
        )
        clf.fit(X_train, ext_train,
                eval_set=[(X_val, ext_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        p_ext_val = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(ext_val, p_ext_val)
        log.info("  Classifier  AUC=%.4f  best_iter=%d", auc, clf.best_iteration_)
        clf_auc = float(auc)

        # ------------------------------------------------------------------ #
        # Stage 2 — Regime-specific quantile models                           #
        # ------------------------------------------------------------------ #
        norm_tr = train_mask & ~is_ext
        extr_tr = train_mask &  is_ext
        norm_va = val_mask   & ~is_ext
        extr_va = val_mask   &  is_ext

        X_tr_n, y_tr_n = sub.loc[norm_tr, FEATURE_COLS], sub.loc[norm_tr, "marginal_chf"]
        X_va_n, y_va_n = sub.loc[norm_va, FEATURE_COLS], sub.loc[norm_va, "marginal_chf"]
        X_tr_e, y_tr_e = sub.loc[extr_tr, FEATURE_COLS], sub.loc[extr_tr, "marginal_chf"]
        X_va_e, y_va_e = sub.loc[extr_va, FEATURE_COLS], sub.loc[extr_va, "marginal_chf"]

        log.info("  Normal  train=%d val=%d  |  Extreme  train=%d val=%d",
                 len(X_tr_n), len(X_va_n), len(X_tr_e), len(X_va_e))

        normal_models  = {}
        extreme_models = {}

        for q in quantiles:
            # Normal regime — full regularisation
            m_n = lgb.LGBMRegressor(
                objective="quantile", alpha=q,
                n_estimators=500, learning_rate=0.05,
                num_leaves=63, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8, verbose=-1,
            )
            m_n.fit(X_tr_n, y_tr_n,
                    eval_set=[(X_va_n, y_va_n)],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            normal_models[q] = m_n

            # Extreme regime — more regularised (fewer samples)
            eval_e = (X_va_e, y_va_e) if len(X_va_e) >= 5 else (X_tr_e, y_tr_e)
            m_e = lgb.LGBMRegressor(
                objective="quantile", alpha=q,
                n_estimators=300, learning_rate=0.05,
                num_leaves=15, min_child_samples=5,
                subsample=0.8, colsample_bytree=0.8, verbose=-1,
            )
            m_e.fit(X_tr_e, y_tr_e,
                    eval_set=[eval_e],
                    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
            extreme_models[q] = m_e

        # ------------------------------------------------------------------ #
        # Evaluate combined on validation set                                 #
        # ------------------------------------------------------------------ #
        p_ext = clf.predict_proba(X_val)[:, 1]
        for q in quantiles:
            pred = (1 - p_ext) * normal_models[q].predict(X_val) \
                 + p_ext       * extreme_models[q].predict(X_val)
            pb = pinball(y_val.values, pred, q)
            pb_norm = pb / abs(float(y_val.mean()))
            log.info("  q=%.2f  pinball=%.4f (norm=%.4f)", q, pb, pb_norm)
            results.append({"direction": direction, "quantile": q, "pinball_kpi": pb, "pinball_kpi_norm": pb_norm, "clf_auc": clf_auc})

        # ------------------------------------------------------------------ #
        # Save                                                                #
        # ------------------------------------------------------------------ #
        out = MODELS_DIR / f"tre_{direction}.pkl"
        with open(out, "wb") as f:
            pickle.dump({"clf": clf, "normal": normal_models, "extreme": extreme_models}, f)
        log.info("  Saved -> %s", out.name)

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
