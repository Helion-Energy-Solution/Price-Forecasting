"""
Revenue-maximising bid strategy for pay-as-bid Swiss balancing markets.

All three markets (TRL Weekly, TRL Daily, TRE) use pay-as-bid clearing:
you earn your own submitted bid price if selected, not the clearing price.

Positive-price markets (TRL Weekly/Daily up & down, TRE pos)
------------------------------------------------------------
Selected when bid b <= clearing price P_c.
Bidding at the q-quantile of the predicted distribution => P(selected) = 1 - q.
Expected profit per slot = (price[q] - opportunity_cost) * (1 - q).
Optimal q = argmax over available quantiles.

TRE neg (negative prices, CHF/MWh, e.g. -300)
----------------------------------------------
TSO pays providers to curtail; cheapest providers (least-negative bids) are
selected first. Selected when bid b >= clearing price P_c.
Bidding at extreme-regime quantile q => P(selected | extreme event) = q.
Expected profit given extreme = (|price[q]| - opportunity_cost) * q.
Optimal q = argmax over available quantiles.
Opportunity cost = foregone PV spot revenue (~200 CHF/MWh).

The extreme-regime model (trained on P_c < threshold, e.g. -200 CHF/MWh)
is used for bid-level optimisation; the full blended model is not used here
because normal-regime prices lie above the profitability threshold.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _load_bundle(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _predict_pos(bundle: dict, X: np.ndarray, quantiles: list) -> np.ndarray:
    """Blended two-stage (TRE) or direct quantile predictions (TRL). Shape (n, len(quantiles))."""
    if "normal" in bundle:
        p = bundle["clf"].predict_proba(X)[:, 1]
        return np.column_stack(
            [(1 - p) * bundle["normal"][q].predict(X) + p * bundle["extreme"][q].predict(X)
             for q in quantiles]
        )
    return np.column_stack([bundle[q].predict(X) for q in quantiles])


def _predict_neg_extreme(bundle: dict, X: np.ndarray, quantiles: list) -> np.ndarray:
    """Extreme-regime-only predictions for TRE neg (prices conditional on P_c < threshold)."""
    return np.column_stack([bundle["extreme"][q].predict(X) for q in quantiles])


def _opt_bid_pos(prices: np.ndarray, quantiles: list, opp_cost: float) -> float:
    """argmax_q  (price[q] - opp_cost) * (1 - q). Falls back to median."""
    best_e, best_bid = 0.0, prices[len(prices) // 2]
    for q, price in zip(quantiles, prices):
        net = price - opp_cost
        if net <= 0:
            continue
        e = net * (1.0 - q)
        if e > best_e:
            best_e, best_bid = e, price
    return float(best_bid)


def _opt_bid_neg(prices: np.ndarray, quantiles: list, opp_cost: float) -> float:
    """argmax_q  (|price[q]| - opp_cost) * q. Falls back to most aggressive quantile."""
    best_e, best_bid = 0.0, float(np.min(prices))
    for q, price in zip(quantiles, prices):
        net = abs(price) - opp_cost
        if net <= 0:
            continue
        e = net * q
        if e > best_e:
            best_e, best_bid = e, price
    return float(best_bid)


def run_backtest(
    features_parquet: str | Path,
    time_col: str,
    direction: str,
    feature_cols: list,
    model_pkl: str | Path,
    quantiles: list,
    val_start: str,
    clearing_col: str = "marginal_chf",
    opp_cost: float = 0.0,
    mode: str = "pos_market",   # "pos_market" | "neg_market"
) -> pd.DataFrame:
    """
    Backtest optimal pay-as-bid strategy on the validation slice.

    Returns a per-slot DataFrame with:
      time_col, clearing_price, optimal_bid, selected, pnl_chf,
      median_bid, median_selected, median_pnl_chf, oracle_pnl_chf

    oracle_pnl_chf: perfect-foresight P&L (bid at clearing price, always selected).
    Provides an upper bound; capture% = avg(pnl_chf) / avg(oracle_pnl_chf).
    """
    df = pd.read_parquet(features_parquet)
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    # Drop only on the clearing price — LightGBM predicts through feature NaN, so
    # dropping on feature_cols would silently exclude biddable slots (far-horizon
    # ENTSO-E NaN, ~88%-NaN spot revision stats) and bias the backtest to easy rows.
    sub = df[
        (df[time_col] >= pd.Timestamp(val_start, tz="UTC")) &
        (df["direction"] == direction)
    ].dropna(subset=[clearing_col]).copy()

    if sub.empty:
        return pd.DataFrame()

    bundle = _load_bundle(Path(model_pkl))
    X = sub[feature_cols]  # keep DataFrame so feature names match what models were trained on

    if mode == "neg_market":
        q_arr = _predict_neg_extreme(bundle, X, quantiles)   # (n, n_q), prices < threshold
    else:
        q_arr = _predict_pos(bundle, X, quantiles)           # (n, n_q)

    # Optimal bids
    mid = len(quantiles) // 2  # index of the median quantile
    opt_bids = np.empty(len(sub))
    for i in range(len(sub)):
        if mode == "neg_market":
            opt_bids[i] = _opt_bid_neg(q_arr[i], quantiles, opp_cost)
        else:
            opt_bids[i] = _opt_bid_pos(q_arr[i], quantiles, opp_cost)

    med_bids = q_arr[:, mid]
    clearing = sub[clearing_col].values

    if mode == "neg_market":
        opt_sel    = opt_bids >= clearing
        med_sel    = med_bids >= clearing
        opt_pnl    = np.where(opt_sel,  np.abs(opt_bids) - opp_cost, 0.0)
        med_pnl    = np.where(med_sel,  np.abs(med_bids) - opp_cost, 0.0)
        oracle_pnl = np.maximum(np.abs(clearing) - opp_cost, 0.0)
    else:
        opt_sel    = opt_bids <= clearing
        med_sel    = med_bids <= clearing
        opt_pnl    = np.where(opt_sel,  opt_bids - opp_cost, 0.0)
        med_pnl    = np.where(med_sel,  med_bids - opp_cost, 0.0)
        oracle_pnl = np.maximum(clearing - opp_cost, 0.0)

    result = sub[[time_col, clearing_col]].copy()
    result.columns = [time_col, "clearing_price"]
    result["optimal_bid"]     = opt_bids
    result["selected"]        = opt_sel
    result["pnl_chf"]         = opt_pnl
    result["median_bid"]      = med_bids
    result["median_selected"] = med_sel
    result["median_pnl_chf"]  = med_pnl
    result["oracle_pnl_chf"]  = oracle_pnl

    return result.reset_index(drop=True)
