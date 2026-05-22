"""
SHAP beeswarm feature importance for TRL Weekly, TRL Daily, and TRE models.

Loads the q=0.50 model for each direction and plots SHAP values on the
validation set (2026-01-01 onwards). Saves figures to notebooks/figures/.

Usage
-----
python notebooks/04_shap_importance.py
"""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — saves to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "notebooks" / "figures"
FEATURES_DIR = ROOT / "data" / "processed" / "features"
MODELS_DIR = ROOT / "models"
CONFIG_PATH = ROOT / "config" / "config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

VAL_START = pd.Timestamp(cfg["training"]["val_start"], tz="UTC")
QUANTILE = 0.5

PRETTY = {
    # calendar
    "week_of_year": "Week of year",
    "month": "Month",
    "year": "Year",
    "block_of_day": "Block of day",
    "day_of_week": "Day of week",
    "quarter_of_hour": "Quarter of hour",
    "hour_of_day": "Hour of day",
    "is_weekend": "Is weekend",
    "is_thursday": "Is Thursday",
    "is_friday": "Is Friday",
    "days_ahead": "Days ahead",
    "lead_hours": "Lead hours",
    "hours_until_delivery": "Hours until delivery",
    # weather
    "temp_2m_mean": "Temp 2m (mean)",
    "temp_2m_std": "Temp 2m (spread)",
    "temp_2m_skew": "Temp 2m (skew)",
    "temp_2m_p10": "Temp 2m (p10)",
    "temp_2m_p90": "Temp 2m (p90)",
    "cloud_cover_mean": "Cloud cover (mean)",
    "cloud_cover_std": "Cloud cover (spread)",
    "cloud_cover_skew": "Cloud cover (skew)",
    "cloud_cover_p10": "Cloud cover (p10)",
    "cloud_cover_p90": "Cloud cover (p90)",
    "precip_rate_mmh_mean": "Precip rate (mean)",
    "precip_rate_mmh_std": "Precip rate (spread)",
    "precip_rate_mmh_skew": "Precip rate (skew)",
    "precip_rate_mmh_p10": "Precip rate (p10)",
    "precip_rate_mmh_p90": "Precip rate (p90)",
    "cos_zenith": "Solar zenith (cos)",
    "spot_eur_mwh": "Spot price (EUR/MWh)",
    # reservoir
    "wallis_fill_pct": "Reservoir Valais (%)",
    "graubuenden_fill_pct": "Reservoir Graubünden (%)",
    "tessin_fill_pct": "Reservoir Ticino (%)",
    "totalch_fill_pct": "Reservoir CH total (%)",
    # price lags
    "marginal_chf_lag1": "Price lag 1w",
    "marginal_chf_lag4": "Price lag 4w",
    "marginal_chf_lag52": "Price lag 52w",
    "marginal_chf_lag6": "Price lag 6 blocks (24h)",
    "marginal_chf_lag42": "Price lag 42 blocks (7d)",
    "marginal_chf_lag96h": "Same-hour-yesterday price (avg)",
    "trl_weekly_up_chf": "TRL Weekly up price (CHF/MW)",
    "trl_weekly_down_chf": "TRL Weekly down price (CHF/MW)",
    "marginal_chf_roll4_mean": "Rolling 4w mean",
    "marginal_chf_roll4_std": "Rolling 4w std",
    "marginal_chf_roll12_mean": "Rolling 12w mean",
    "marginal_chf_roll12_std": "Rolling 12w std",
    "marginal_chf_roll42_mean": "Rolling 7d mean",
    "marginal_chf_roll42_std": "Rolling 7d std",
    "marginal_chf_roll180_mean": "Rolling 30d mean",
    "marginal_chf_roll180_std": "Rolling 30d std",
    "marginal_chf_roll96_mean": "Rolling 24h mean",
    "marginal_chf_roll96_std": "Rolling 24h std",
    "marginal_chf_roll672_mean": "Rolling 7d mean",
    "marginal_chf_roll672_std": "Rolling 7d std",
}


def load_model(pkl_path: Path) -> object:
    with open(pkl_path, "rb") as f:
        models = pickle.load(f)
    # two-stage model: {"clf": ..., "normal": {q: m}, "extreme": {q: m}}
    if isinstance(models, dict) and "normal" in models:
        return models["normal"][QUANTILE]
    return models[QUANTILE]


def shap_beeswarm(ax, shap_values, X, feature_names, title, max_features=15):
    """Draw a horizontal beeswarm on the given axes."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-max_features:]
    sv = shap_values[:, top_idx]
    fv = X[:, top_idx]
    labels = [PRETTY.get(feature_names[i], feature_names[i]) for i in top_idx]

    # Normalise feature values to [0,1] for colouring
    fv_norm = fv.copy().astype(float)
    for j in range(fv_norm.shape[1]):
        col = fv_norm[:, j]
        rng = col.max() - col.min()
        fv_norm[:, j] = (col - col.min()) / rng if rng > 0 else 0.5

    cmap = plt.cm.RdBu_r
    y_pos = np.arange(max_features)

    # Jitter vertically for beeswarm effect
    rng = np.random.default_rng(42)
    for j, y in enumerate(y_pos):
        sv_col = sv[:, j]
        fv_col = fv_norm[:, j]
        jitter = rng.uniform(-0.35, 0.35, size=len(sv_col))
        sc = ax.scatter(sv_col, y + jitter, c=fv_col, cmap=cmap,
                        vmin=0, vmax=1, alpha=0.5, s=8, linewidths=0)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("SHAP value (impact on price forecast)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Feature value\n(low → high)", fontsize=8)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])


# ---------------------------------------------------------------------------
# TRL Weekly
# ---------------------------------------------------------------------------

def plot_trl_weekly():
    df = pd.read_parquet(FEATURES_DIR / "trl_weekly_features.parquet")
    df["week_start"] = pd.to_datetime(df["week_start"], utc=True)

    from src.models.trl_weekly_model import FEATURE_COLS_BY_DIRECTION
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("TRL Weekly — SHAP feature importance (q=0.50, validation 2026)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, direction in zip(axes, ("up", "down")):
        fc = FEATURE_COLS_BY_DIRECTION[direction]
        sub = df[(df["direction"] == direction) & (df["week_start"] >= VAL_START)]
        sub = sub.dropna(subset=["marginal_chf"] + fc)
        X = sub[fc]

        model = load_model(MODELS_DIR / "trl_weekly" / f"trl_weekly_{direction}.pkl")
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)

        shap_beeswarm(ax, sv, X.values, fc,
                      title=f"Direction: {direction.upper()}")

    plt.tight_layout()
    out = FIGURES_DIR / "shap_trl_weekly.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out.name}")


# ---------------------------------------------------------------------------
# TRL Daily
# ---------------------------------------------------------------------------

def plot_trl_daily():
    df = pd.read_parquet(FEATURES_DIR / "trl_daily_features.parquet")
    df["block_start"] = pd.to_datetime(df["block_start"], utc=True)

    from src.models.trl_daily_model import FEATURE_COLS_BY_DIRECTION
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("TRL Daily — SHAP feature importance (q=0.50, validation 2026)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, direction in zip(axes, ("up", "down")):
        fc = FEATURE_COLS_BY_DIRECTION[direction]
        sub = df[(df["direction"] == direction) & (df["block_start"] >= VAL_START)]
        sub = sub.dropna(subset=["marginal_chf"] + fc)
        X = sub[fc]

        model = load_model(MODELS_DIR / "trl_daily" / f"trl_daily_{direction}.pkl")
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)

        shap_beeswarm(ax, sv, X.values, fc,
                      title=f"Direction: {direction.upper()}")

    plt.tight_layout()
    out = FIGURES_DIR / "shap_trl_daily.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out.name}")


# ---------------------------------------------------------------------------
# TRE
# ---------------------------------------------------------------------------

def plot_tre():
    df = pd.read_parquet(FEATURES_DIR / "tre_features.parquet")
    df["slot_time"] = pd.to_datetime(df["slot_time"], utc=True)

    from src.models.tre_model import FEATURE_COLS, THRESHOLDS

    # ------------------------------------------------------------------ #
    # Figure 1 — Normal-regime quantile model (q=0.50)                    #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("TRE — Normal regime SHAP (q=0.50, validation May 2026, sample 5k)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, direction in zip(axes, ("pos", "neg")):
        sub = df[(df["direction"] == direction) & (df["slot_time"] >= VAL_START)]
        sub = sub.dropna(subset=["marginal_chf"] + FEATURE_COLS)
        if len(sub) > 5000:
            sub = sub.sample(5000, random_state=42)
        X = sub[FEATURE_COLS].values

        model = load_model(MODELS_DIR / "tre" / f"tre_{direction}.pkl")
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)

        shap_beeswarm(ax, sv, X, FEATURE_COLS,
                      title=f"Direction: {direction.upper()}")

    plt.tight_layout()
    out = FIGURES_DIR / "shap_tre_normal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out.name}")

    # ------------------------------------------------------------------ #
    # Figure 2 — Classifier: P(extreme price)                             #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("TRE — Classifier SHAP: P(extreme price), full training set",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, direction in zip(axes, ("pos", "neg")):
        import pickle
        with open(MODELS_DIR / "tre" / f"tre_{direction}.pkl", "rb") as f:
            bundle = pickle.load(f)
        clf = bundle["clf"]

        # Use full training set (val has very few extremes; train has ~1700/2600)
        sub = df[(df["direction"] == direction) & (df["slot_time"] < VAL_START)]
        sub = sub.dropna(subset=["marginal_chf"] + FEATURE_COLS)
        if len(sub) > 5000:
            # Stratified sample: keep all extremes + random sample of normals
            t = THRESHOLDS[direction]
            is_ext = (sub["marginal_chf"] > t) if direction == "pos" else (sub["marginal_chf"] < t)
            extremes = sub[is_ext]
            normals  = sub[~is_ext].sample(min(5000 - len(extremes), len(sub[~is_ext])),
                                           random_state=42)
            sub = pd.concat([extremes, normals])

        X = sub[FEATURE_COLS].values
        t = THRESHOLDS[direction]
        is_ext = (sub["marginal_chf"] > t) if direction == "pos" else (sub["marginal_chf"] < t)

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X)
        # TreeExplainer on classifier returns [shap_neg, shap_pos]; take positive class
        if isinstance(sv, list):
            sv = sv[1]

        # Colour dots by actual class (extreme=red, normal=blue) rather than feature value
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[-15:]
        sv_top   = sv[:, top_idx]
        fv_top   = X[:, top_idx].astype(float)
        labels   = [PRETTY.get(FEATURE_COLS[i], FEATURE_COLS[i]) for i in top_idx]

        # Normalise feature values for colouring
        fv_norm = fv_top.copy()
        for j in range(fv_norm.shape[1]):
            col = fv_norm[:, j]
            rng = col.max() - col.min()
            fv_norm[:, j] = (col - col.min()) / rng if rng > 0 else 0.5

        cmap   = plt.cm.RdBu_r
        y_pos  = np.arange(15)
        rng_   = np.random.default_rng(42)

        for j, y in enumerate(y_pos):
            sv_col = sv_top[:, j]
            fv_col = fv_norm[:, j]
            jitter = rng_.uniform(-0.35, 0.35, size=len(sv_col))
            ax.scatter(sv_col, y + jitter, c=fv_col, cmap=cmap,
                       vmin=0, vmax=1, alpha=0.5, s=8, linewidths=0)

        n_ext = int(is_ext.sum())
        n_tot = len(sub)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("SHAP value (pushes toward P(extreme))", fontsize=9)
        thresh_label = f"> {THRESHOLDS[direction]:.0f}" if direction == "pos" else f"< {THRESHOLDS[direction]:.0f}"
        ax.set_title(f"Direction: {direction.upper()}  ({n_ext}/{n_tot} extreme, price {thresh_label} CHF/MWh)",
                     fontsize=11, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label("Feature value\n(low -> high)", fontsize=8)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])

    plt.tight_layout()
    out = FIGURES_DIR / "shap_tre_classifier.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out.name}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))

    print("TRL Weekly ...")
    plot_trl_weekly()
    print("TRL Daily ...")
    plot_trl_daily()
    print("TRE ...")
    plot_tre()
    print("Done. Figures saved to notebooks/figures/")
    print("  shap_tre_normal.png    — normal-regime quantile model (q=0.50)")
    print("  shap_tre_classifier.png — classifier: what drives P(extreme price)")
