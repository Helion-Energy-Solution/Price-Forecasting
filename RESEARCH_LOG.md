# Price Forecasting — Research Log

Chronological record of pipeline changes, design decisions, assumptions, and experiment results.
Add a new entry under **Experiments** each time you retrain and run the backtest.

---

## Pipeline Overview (current)

| Component | Description |
|---|---|
| Weather | ECMWF ENS via ECDS (TIGGE), 50 members, 3 variables (t2m, tp, tcc) |
| Prices | TRL Weekly, TRL Daily, TRE from Swissgrid / Helion dashboard |
| Reservoir | Weekly fill % by region (Wallis, Graubünden, Tessin, CH total) |
| Spot | EPEX day-ahead EUR/MWh, hourly |
| Feature store | `src/data/feature_store.py` — aligns all sources on UTC index |
| Models | LightGBM quantile regression, one per market × direction |
| Bid strategy | `src/pipeline/bid_strategy.py` — pay-as-bid revenue optimiser |

### Models

| Model | File | Type | Quantiles | Val method |
|---|---|---|---|---|
| TRL Weekly | `trl_weekly_model.py` | LightGBM quantile | q10–q90 | CV n_estimators + 3-way split |
| TRL Daily | `trl_daily_model.py` | LightGBM quantile | q10–q90 | Early stopping on val set |
| TRE (two-stage) | `tre_model.py` | Classifier + 2× quantile | q10–q90 | Early stopping on val set |

---

## Key Assumptions

| # | Assumption | Justification |
|---|---|---|
| 1 | All three markets are pay-as-bid | Confirmed from Swissgrid documentation and objective.md |
| 2 | Objective is revenue maximisation, not price prediction error | Bidding at optimal quantile is more valuable than minimising pinball loss |
| 3 | TRE neg opportunity cost = 200 CHF/MWh | Foregone PV spot revenue from curtailment; only bids ≤ −200 are profitable |
| 4 | SSRD (param 169) not available in TIGGE | Replaced with cos_zenith (geometry) × cloud_cover (ENS ensemble) |
| 5 | P(selected | bid=q) ≈ 1 − q for positive markets | Pay-as-bid: selected when bid ≤ clearing; bidding at quantile q means clearing exceeds bid in (1−q) of cases |
| 6 | P(selected | bid=q, extreme event) ≈ q for TRE neg | Extreme-regime model gives conditional quantiles; TSO selects least-negative (cheapest) bids first |
| 7 | TRL Weekly price for delivery week is always known at TRL Daily bid time | TRL Weekly clears Tuesday prior week; TRL Daily bids ≥ 2 business days ahead |
| 8 | As-of join for reservoir levels (no lookahead) | Most recent weekly reading at or before bid date |
| 9 | TIGGE 48-hour embargo | END_DATE = today − 2 days; ecds_parallel_launch.py computes remaining dates automatically |

---

## Design Decisions & Questions

### Why separate TRE positive and negative into two directions?
TRE neg prices can reach −300 CHF/MWh. Positive and negative regimes have different drivers and the optimal bidding formula differs (`|price[q]| − 200` vs `price[q]`). A single model would try to fit both tails simultaneously.

### Why use the extreme-regime model for TRE neg bidding (not the blended model)?
The blended model mixes normal-regime predictions (prices near 0) into the quantile output. For bidding into the negative extreme, only prices below −200 are relevant — using the extreme-regime model directly gives cleaner conditional quantiles for the selection probability calculation.

### Why not train directly on revenue loss (end-to-end)?
- Requires differentiating through the indicator function P(selected) — needs smooth approximation
- High variance with small val sets (especially TRL Weekly with ~7 samples)
- Post-hoc bid optimisation over quantile outputs is already close to optimal when quantile calibration is good
- Can revisit when val sets are larger

### Why cross-validation for TRL Weekly n_estimators instead of early stopping?
TRL Weekly has ~7 recent val weeks — too few for reliable early stopping signal (LightGBM was stopping at best_iter=1 for high quantiles, severely undertraining). TimeSeriesSplit CV within the training set gives stable estimates. A 3-fold CV with the training data gives ~29 weeks per val fold.

### Does ERA5 SSRD make sense as a replacement for cos_zenith?
No — ERA5 is a reanalysis (hindcast), not a forecast. Training on ERA5 SSRD would create a training/inference distribution mismatch (inference uses ECMWF ENS). The better approach is an explicit `ssrd_proxy = cos_zenith × (1 − cloud_cover_mean)` feature, which approximates what SSRD measures using data already available at both training and inference time.

### Why use block-specific (same-block-of-day) rolling features instead of a flat rolling average?
TRL Daily prices follow a strong intraday pattern (the 12–16h block typically peaks 2–3× higher than the 00–04h block), but the *amplitude* of these oscillations varies across weeks. A flat `roll42_mean` mixes all 6 blocks of the day, so the rolling average reflects the cross-block average level rather than "what block X typically costs". A `_price_lags_same_block()` feature groups by `block_of_day` before rolling, so the look-back window contains only same-block observations — it captures both the level for that block and how that level has been trending. The same pattern applies to any market where there is a strong periodic component in the target (e.g., intraday for TRE).

### Why use `FEATURE_COLS_BY_DIRECTION` for TRL Daily (and not a single shared list)?
TRL Weekly S1 features (anticipated-auction results) are structurally absent for direction=up in every week — they only exist for the down auction in Feb–May. Including them in the UP feature list as constant-zero columns wastes tree splits and can introduce spurious interactions. The `FEATURE_COLS_BY_DIRECTION` pattern (UP uses the base list; DOWN adds direction-specific columns) keeps each model lean and ensures no constant features enter training. The same principle applies whenever a feature is structurally 0 or NaN for an entire sub-group.

### Why fill S1 numeric columns with 0.0 instead of NaN for non-S1 rows?
`run_backtest` calls `dropna(subset=feature_cols)` before scoring. If `s1_marginal_chf` is NaN for all UP rows and non-S1 DOWN weeks, all those rows are silently dropped — leaving only ~14 S1 rows per year in the backtest. The fix: fill with 0.0 when `s1_is_active=0`. LightGBM will first split on `s1_is_active` and route 0-filled rows to the inactive branch, so the zero is semantically correct (not noise injection). The same pattern applies to any future feature that is structurally absent for a subset of rows: gate with a binary flag, fill numeric columns with a neutral value.

---

## Experiments

### How to read the table
- `capture_%` = our P&L ÷ oracle P&L, where oracle = perfect-foresight bid at clearing price
- `opt_pnl/slot` = average CHF per MW per delivery slot (including unselected slots = 0)
- Val set sizes: TRL Weekly ≈ 7 weeks, TRL Daily ≈ 89 blocks, TRE ≈ 1288 slots
- TRL Weekly results are high-variance due to small val set — treat ±10 pp moves with caution

---

### EXP-001 — Baseline with early stopping on full val set
**Date:** 2026-05-15  
**Changes from prior:** First backtest run with pay-as-bid strategy layer  
**Val period:** TRL Weekly: 2026-04-06+, TRL Daily/TRE: 2026-05-01+  
**Notable:** TRL Weekly used early stopping on the 5-week val set → best_iter=1 for q0.75, q0.90 (severe undertraining)

| Market | capture_% | opt_select_% | opt_pnl/slot | oracle_pnl/slot |
|---|---|---|---|---|
| TRL Weekly up | 73.1 | 85.7 | 419.11 | 573.67 |
| TRL Weekly down | 37.8 | 42.9 | 411.64 | 1089.39 |
| TRL Daily up | 34.5 | 52.8 | 5.76 | 16.71 |
| TRL Daily down | 43.7 | 43.8 | 24.29 | 55.59 |
| TRE pos | 35.7 | 28.5 | 33.92 | 95.08 |
| TRE neg | 85.9 | 0.9 | 0.90 | 1.04 |

---

### EXP-002 — TRL Weekly: CV n_estimators + 3-way split
**Date:** 2026-05-15  
**Changes from EXP-001:**
- TRL Weekly: 3-fold TimeSeriesSplit CV within training set to determine n_estimators per quantile
- TRL Weekly: 3-way split — train (<2025-04-07), es_val (2025-04-07 to 2026-04-06, monitoring only), kpi_val (≥2026-04-06)
- No early stopping on final TRL Weekly fit — CV-derived n_estimators controls complexity
- TRL Daily and TRE unchanged

**Motivation:** Eliminate best_iter=1 for high quantiles caused by noisy 5-week stopping signal  
**Result:** n_estimators now 85–289 (up from 1); q0.75 and q0.90 now properly trained

| Market | capture_% | opt_select_% | opt_pnl/slot | oracle_pnl/slot |
|---|---|---|---|---|
| TRL Weekly up | **80.6** | 100.0 | 462.39 | 573.67 |
| TRL Weekly down | **48.0** | 57.1 | 522.90 | 1089.39 |
| TRL Daily up | 34.5 | 52.8 | 5.76 | 16.71 |
| TRL Daily down | 43.7 | 43.8 | 24.29 | 55.59 |
| TRE pos | 35.7 | 28.5 | 33.92 | 95.08 |
| TRE neg | 85.9 | 0.9 | 0.90 | 1.04 |

---

### EXP-003 — TRL Weekly price as feature for TRL Daily
**Date:** 2026-05-16  
**Changes from EXP-002:**
- Feature store: `trl_weekly_up_chf` and `trl_weekly_down_chf` added to TRL Daily features
- Joined on delivery week start (Monday); always known at TRL Daily bid time (no lookahead)
- TRL Daily retrained; TRL Weekly and TRE unchanged

| Market | capture_% | opt_select_% | opt_pnl/slot | oracle_pnl/slot |
|---|---|---|---|---|
| TRL Weekly up | 80.5 | 100.0 | 461.58 | 573.67 |
| TRL Weekly down | 48.0 | 57.1 | 522.49 | 1089.39 |
| TRL Daily up | 34.6 | 50.6 | 5.79 | 16.71 |
| TRL Daily down | **49.0** | 41.6 | 27.21 | 55.59 |
| TRE pos | 35.7 | 28.5 | 33.92 | 95.08 |
| TRE neg | 85.9 | 0.9 | 0.90 | 1.04 |

---

### EXP-004 — Swiss public holiday feature
**Date:** 2026-05-19  
**Changes from EXP-003:**
- Feature store: `is_holiday` (0/1) added to TRL Daily and TRE features
- Feature store: `n_holidays_in_week` (0–3) added to TRL Weekly features
- 10 holidays: New Year's Day, Berchtoldstag, Good Friday, Easter Monday, Labour Day, Ascension Day, Whit Monday, Swiss National Day, Christmas Day, St. Stephen's Day
- All three models retrained

| Market | capture_% | opt_select_% | opt_pnl/slot | oracle_pnl/slot | Δ capture vs EXP-003 |
|---|---|---|---|---|---|
| TRL Weekly up | **83.9** | 100.0 | 481.33 | 573.67 | +3.4 pp ✓ |
| TRL Weekly down | 19.3 | 14.3 | 209.93 | 1089.39 | −28.7 pp ⚠ noisy (7 slots) |
| TRL Daily up | 35.5 | 48.3 | 5.93 | 16.71 | +0.9 pp ✓ |
| TRL Daily down | **52.5** | 46.1 | 29.20 | 55.59 | +3.5 pp ✓ |
| TRE pos | 34.2 | 26.1 | 32.56 | 95.08 | −1.5 pp ≈ |
| TRE neg | 53.8 | 0.5 | 0.56 | 1.04 | −32.1 pp ⚠ noisy (11 extreme slots) |

**Notes:**
- TRL Weekly down and TRE neg regressions are almost certainly noise — val sets are 7 and 11 extreme-event slots respectively; a single mispriced slot swings capture% by >10 pp
- TRL Daily down improvement (+3.5 pp) is more reliable given 89 val slots

---

### EXP-005 — ssrd_proxy interaction features for TRL Daily and TRE
**Date:** 2026-05-19
**Changes from EXP-004:**
- Feature store: `ssrd_proxy = cos_zenith × (1 − cloud_cover_mean)` added to TRL Daily and TRE features
- Feature store: `ssrd_proxy_unc = cos_zenith × cloud_cover_std` added to TRL Daily and TRE features
- Both computed after weather join, before the `cal` DataFrame; no change to TRL Weekly
- TRL Daily and TRE retrained; TRL Weekly unchanged (pkl reused from EXP-004)

**Motivation:** LightGBM needs two consecutive splits to learn the cos_zenith × cloud_cover interaction. An explicit product gives it in one split and also correctly forces ssrd_proxy to zero at night regardless of cloud cover.

⚠ **Val sets grew vs EXP-004** — TRL Daily 89→113 slots, TRE 1288→1624 slots (more May data ingested). Oracle P&L also shifted (TRL Daily up: 16.71→13.84, down: 55.59→68.97), so Δ capture is not directly comparable to prior experiments. Treat deltas with caution.

| Market | capture_% | opt_select_% | opt_pnl/slot | oracle_pnl/slot | Δ capture vs EXP-004 |
|---|---|---|---|---|---|
| TRL Weekly up | 83.9 | 100.0 | 481.33 | 573.67 | 0.0 pp (model unchanged) |
| TRL Weekly down | 19.3 | 14.3 | 209.93 | 1089.39 | 0.0 pp (model unchanged) |
| TRL Daily up | 32.8 | 47.8 | 4.54 | 13.84 | −2.7 pp ⚠ val set changed |
| TRL Daily down | 46.2 | 34.2 | 31.89 | 68.97 | −6.3 pp ⚠ val set changed |
| TRE pos | **34.4** | 23.8 | 28.97 | 84.34 | +0.2 pp ≈ val set changed |
| TRE neg | **73.5** | 0.6 | 0.61 | 0.83 | +19.7 pp ⚠ noisy (few extreme slots) |

**Notes:**
- TRL Daily regressions cannot be attributed to ssrd_proxy with confidence — the val set shifted to include 3 additional weeks of May with lower up-prices (oracle 13.84 vs 16.71) and higher down-prices (oracle 68.97 vs 55.59). The new weeks may simply be harder to forecast.
- TRE pos is essentially flat; ssrd_proxy added no measurable improvement at this sample size.
- TRE neg swing (+19.7 pp) is almost certainly noise — extreme-event count in the extended val set is very small.
- To properly evaluate ssrd_proxy: fix val period to EXP-004 dates and retrain, or wait until the val set is larger and the distribution stabilises.

---

### EXP-006 — Volue spot forecast features + weekly spot aggregates + holiday bid push-back
**Date:** 2026-05-22  
**Changes from EXP-005:**
- **New data source:** `src/data/update_spot_forecast.py` — downloads Volue `pri ch spot merged €/mwh cet h f` (INSTANCES curve) via the `volue-insight-timeseries` API, batched by calendar day (`with_data=True` per day to stay within API size limit). Subscription access starts 2026-01-01; covers ~24 hourly issues per day.
- **Realized-price fallback:** `_load_spot_forecast()` now blends Volue data (2026+) with realized DA prices from `spot_hourly.parquet` for delivery hours before 2026-01-01. Realized entries get a synthetic `issue_date = delivery_hour − 30 days` so the point-in-time lookup always finds them for any bid time. `spot_fcst_std` / `spot_fcst_change` remain NaN for 2022-2025 rows (no multi-run history), accepted as LightGBM handles NaN natively.
- **`_spot_forecast_asof()` rewritten:** Replaced `pd.merge_asof(by=…)` (which requires globally monotonic `on` column, incompatible with grouped delivery_hour data) with vectorized `numpy.searchsorted` per delivery_hour group.
- **TRL Weekly — 6 weekly spot aggregates:** `spot_baseload_mean`, `spot_peakload_mean`, `spot_max`, `spot_min`, `spot_daily_spread_mean`, `spot_neg_hours`. Computed via `_spot_week_features()`: expands all 168 delivery hours for the week, does one batched as-of lookup, then aggregates. Peak = 08:00–19:59 Europe/Zurich.
- **TRL Daily — 3 spot features:** `spot_eur_mwh` (Volue forecast averaged over the 4-hour block, or realized price fallback), `spot_fcst_std`, `spot_fcst_change`.
- **TRE — 4 spot features:** `spot_eur_mwh` (actual DA price when DA auction has cleared before bid_time; Volue forecast otherwise), `spot_is_realized` (1 = DA price known, 0 = Volue forecast), `spot_fcst_std`, `spot_fcst_change`.
- **Holiday bid-day push-back (TRL Daily and TRE):** If the computed bid day falls on a Swiss public holiday, iterates backward to the previous non-holiday workday. Affects `days_ahead` and `init_time` derivation.
- All three models retrained.

**Motivation:** DA spot price is the strongest observable proxy for balancing price level. Volue forecasts give the model what would actually be known at bid time (for 2026+ data). Realized prices as fallback for 2022-2025 introduce mild lookahead bias but keep the feature populated throughout training; the signal content is similar (DA forecasts correlate ~90% with realised prices). The weekly aggregates translate the hourly spot signal into a shape directly usable by the TRL Weekly model.

⚠ **Val sets grew again vs EXP-005** — TRL Weekly 7→8 slots, TRL Daily 113→131 slots, TRE 1624→1960 slots. Oracle P&L also shifted. Δ capture is indicative, not controlled.

| Market | capture_% | opt_select_% | opt_pnl/slot | oracle_pnl/slot | Δ capture vs EXP-005 |
|---|---|---|---|---|---|
| TRL Weekly up | 75.2 | 87.5 | 413.15 | 549.22 | −8.7 pp ⚠ noisy (8 slots) |
| TRL Weekly down | 25.5 | 25.0 | 285.89 | 1120.16 | +6.2 pp ⚠ noisy (8 slots) |
| TRL Daily up | 35.8 | 45.0 | 4.38 | 12.24 | +3.0 pp ✓ |
| TRL Daily down | **51.1** | 38.1 | 39.48 | 77.22 | +4.9 pp ✓ |
| TRE pos | **47.0** | 37.0 | 39.20 | 83.34 | +12.6 pp ✓ reliable |
| TRE neg | 32.2 | 0.2 | 0.22 | 0.69 | −41.3 pp ⚠ very few extreme slots |

**Notes:**
- **TRE pos +12.6 pp** is the most credible result — 1960 val slots, economically sensible (DA spot price is well known to correlate with balancing price level), and the improvement is large enough to survive val-set composition changes. This is the clearest positive signal in the experiment.
- **TRL Daily down +4.9 pp** (126 val slots) is also credible. Spot price gives the model regime context for the 2-day-ahead block bid.
- **TRL Daily up +3.0 pp** (131 slots) is directionally positive and consistent with TRL Daily down.
- **TRL Weekly figures** (8 slots) cannot be interpreted — one slot difference equals ~12 pp swing. The -8.7 pp drop for up is almost certainly noise, not a feature regression.
- **TRE neg -41.3 pp**: extreme-event slot count in the val set is very small; this is noise.
- `spot_fcst_std` and `spot_fcst_change` are ~90% NaN in training (2022-2025 rows use realized price with no revision history). These features are effectively inactive for most training data; their long-run value depends on continued daily Volue downloads. Run `update_spot_forecast.py` daily to grow the revision signal.

---

### EXP-007 — S1 (anticipated auction) features for TRL Weekly down
**Date:** 2026-05-22  
**Changes from EXP-006:**
- **`src/data/market_data.py`**: `parse_trl_weekly()` now extracts the `anticipated` JSON key into four new columns: `s1_is_active` (int 0/1), `s1_awarded_mw`, `s1_marginal_chf`, `s1_vwap_chf`. For direction=up and non-S1 down weeks the numeric columns are `None`; `s1_is_active=0`.
- **`src/data/feature_store.py`**: `build_trl_weekly_features()` includes S1 columns in the feature concat. Numeric S1 columns are filled with `0.0` where `s1_is_active=0` (prevents `dropna` in `run_backtest` from eliminating UP rows and non-S1 DOWN weeks; LightGBM gates on `s1_is_active` before interpreting the values).
- **`src/models/trl_weekly_model.py`**: `FEATURE_COLS` extended with `"s1_is_active"`, `"s1_awarded_mw"`, `"s1_marginal_chf"`, `"s1_vwap_chf"`.
- TRL Weekly retrained (both directions); TRL Daily and TRE unchanged.

**Motivation:** The S1 (anticipated) auction clears before the regular Tuesday TRL Weekly Down auction in Feb–May each year. Its results (clearing price, VWAP, awarded volume) are fully observable at regular-auction bid time — no lookahead. Historically, S1 marginal prices are ~2× higher than the regular auction price, reflecting the TSO's liquidity premium for early procurement. The remaining volume procured in the regular auction is correlated with S1 volume and price: knowing the S1 result directly constrains the residual demand for the regular auction. LightGBM can learn this relationship through the `s1_is_active` gate.

⚠ **Val set composition shifted vs EXP-006** — TRE dropped from 1960 → 1912 slots after the full feature-store rebuild. The small TRL Daily and TRE drifts below are not attributable to model changes (those models were not retrained).

| Market | capture_% | opt_select_% | opt_pnl/slot | oracle_pnl/slot | Δ capture vs EXP-006 |
|---|---|---|---|---|---|
| TRL Weekly up | 75.2 | 87.5 | 413.15 | 549.22 | 0.0 pp (s1 features constant 0 for up) |
| TRL Weekly down | **29.7** | 25.0 | 332.58 | 1120.16 | +4.2 pp ⚠ noisy (8 slots) |
| TRL Daily up | 35.8 | 45.0 | 4.38 | 12.24 | 0.0 pp (model unchanged) |
| TRL Daily down | 51.1 | 38.1 | 39.48 | 77.22 | 0.0 pp (model unchanged) |
| TRE pos | 45.9 | 36.2 | 38.67 | 84.22 | −1.1 pp (model unchanged, val set shift) |
| TRE neg | 37.2 | 0.8 | 10.03 | 26.98 | +5.0 pp (model unchanged, val set shift) |

**Notes:**
- **TRL Weekly up = 0.0 pp** as expected: `s1_is_active=0` for all up rows and the numeric S1 features are filled with 0.0 — these are constant, contributing nothing to splits. The model is equivalent to EXP-006.
- **TRL Weekly down +4.2 pp** is directionally positive. The S1 clearing price is a strong leading indicator (TSO procures S1 at ~2× the regular price to pre-empt liquidity shortfalls; what remains shapes the regular auction). However, 8 val slots means one misfired slot = ~12 pp swing — treat with caution.
- **TRL Daily and TRE drifts** (−1.1 pp / +5.0 pp) are val-set composition noise from the feature-store rebuild; those models were not changed.
- **TRE neg oracle_pnl/slot** shifted from 0.69 → 26.98 CHF/slot, indicating the val set now captures more high-value extreme events; capture% swings are dominated by this composition change, not model quality.
- The S1 gating design (0-fill + `s1_is_active` flag) is correct: LightGBM will learn to split on `s1_is_active` first and only interpret the S1 price features on the positive branch. Non-S1 rows receive a safe 0 that does not distort the scale.

---

### EXP-008 — TRL Daily same-block rolling features + TRL Weekly VWAP/volume + direction-specific splits
**Date:** 2026-05-22  
**Changes from EXP-007:**
- **`src/data/feature_store.py`**: `_price_lags_same_block()` added — groups TRL Daily price history by `block_of_day` (0–5) and computes within-block rolling mean/std at 7-day and 28-day windows. Previous `roll42_mean`/`roll180_mean` averaged all 6 blocks together, washing out the intraday shape signal.
- **`src/data/feature_store.py`**: TRL Weekly join extended via `pivot_table(aggfunc="first")` to extract `vwap_chf` and `awarded_mw` per direction. Produces `trl_weekly_up_vwap_chf`, `trl_weekly_up_awarded_mw`, `trl_weekly_down_vwap_chf`, `trl_weekly_down_awarded_mw`, `trl_weekly_down_s1_awarded_mw`.
- **`trl_daily_model.py`**: `FEATURE_COLS` extended with 4 same-block rolling features (`marginal_chf_sb_roll7_mean/std`, `marginal_chf_sb_roll28_mean/std`) and 4 TRL Weekly context columns (`up/down_vwap_chf`, `up/down_awarded_mw`). `FEATURE_COLS_BY_DIRECTION` added: DOWN additionally receives `trl_weekly_down_s1_awarded_mw`.
- **`trl_weekly_model.py`**: `FEATURE_COLS_BY_DIRECTION` added — UP no longer receives the 4 S1 feature columns (constant 0 for UP direction; removing them eliminates wasted splits without changing predictions).
- **All models**: Normalised pinball `pb / |mean(y_val)|` added to training metrics and `pinball_latest.json`. TRE neg denominator uses `abs()` to ensure metric is always positive.
- All three models retrained.

**Motivation:** Historical TRL Daily prices show strong block-of-day patterns (the 12–16 block peaks massively) but the amplitude varies week-to-week. A rolling average mixing all blocks confounds the level signal with the intraday shape. Block-specific rolling recovers both simultaneously. TRL Weekly VWAP and awarded volume give the daily model regime context beyond just the marginal price: high weekly volume signals high TSO demand and tends to compress margins in the daily auction.

| Market | capture_% | opt_select_% | opt_pnl/slot | oracle_pnl/slot | Δ capture vs EXP-007 |
|---|---|---|---|---|---|
| TRL Weekly up | 75.2 | 87.5 | 413.15 | 549.22 | 0.0 pp (S1 removal had no effect — those features were always 0) |
| TRL Weekly down | 29.7 | 25.0 | 332.58 | 1120.16 | 0.0 pp (DOWN feature set unchanged) |
| TRL Daily up | 28.7 | 40.5 | 3.51 | 12.24 | **−7.1 pp ⚠ regression** |
| TRL Daily down | 50.7 | 54.8 | 39.13 | 77.22 | −0.4 pp ≈ |
| TRE pos | 46.5 | 35.9 | 39.15 | 84.22 | +0.6 pp ≈ (stochastic retraining noise; features unchanged) |
| TRE neg | 27.4 | 0.3 | 7.40 | 26.98 | −9.8 pp ⚠ noisy (few extreme slots; features unchanged) |

**Pinball metrics** (first run with `pinball_latest.json` tracking; no prior baseline for comparison):

TRL Weekly training uses three segments: train / es_val (~52 weeks, fixed 2025-04-07→2026-04-06) / kpi_val (2026-04-06+, ~8 weeks). The `es` row reports pinball on the fixed 52-week window — large enough to be stable and directly comparable across experiments. The `kpi` row reports pinball on the 8 most-recent weeks — the official performance window, but too small to compare reliably (one week ≈ 12 pp swing). TRL Daily and TRE each have a single val window, so one row per direction. `clf_auc` is the ROC AUC of TRE's Stage 1 binary classifier (P(extreme price)); blank for TRL Weekly and TRL Daily, which are pure quantile models.

| Model | direction | window | q10 (norm) | q25 (norm) | q50 (norm) | q75 (norm) | q90 (norm) | clf_auc |
|---|---|---|---|---|---|---|---|---|
| TRL Weekly | up | es (fixed ~52w) | 18.6 (0.070) | 24.9 (0.093) | 41.7 (0.156) | 45.6 (0.170) | 44.2 (0.165) | — |
| TRL Weekly | up | kpi (8 slots) | 25.6 (0.047) | 27.4 (0.050) | 33.4 (0.061) | 34.4 (0.063) | 37.2 (0.068) | — |
| TRL Weekly | down | es (fixed ~52w) | 168.8 (0.161) | 267.1 (0.254) | 345.7 (0.329) | 234.0 (0.223) | 140.9 (0.134) | — |
| TRL Weekly | down | kpi (8 slots) | 78.0 (0.070) | 114.4 (0.102) | 250.8 (0.224) | 505.2 (0.451) | 297.5 (0.266) | — |
| TRL Daily | up | kpi (131 slots) | 0.82 (0.072) | 1.89 (0.167) | 3.16 (0.279) | 4.12 (0.364) | 3.93 (0.347) | — |
| TRL Daily | down | kpi (126 slots) | 5.92 (0.070) | 10.82 (0.127) | 14.49 (0.171) | 15.16 (0.178) | 12.70 (0.149) | — |
| TRE | pos | kpi (1912 slots) | 10.5 (0.124) | 24.5 (0.291) | 39.7 (0.471) | 40.7 (0.484) | 30.2 (0.359) | 0.712 |
| TRE | neg | kpi (1912 slots) | 35.9 (3.05†) | 52.1 (4.42†) | 50.2 (4.26†) | 39.9 (3.39†) | 21.9 (1.86†) | 0.764 |

**Notes:**
- **TRL Daily up −7.1 pp** is the most concerning result. The val set (131 slots) and oracle (12.24) are identical to EXP-007 so this is a direct comparison. Possible causes: (a) same-block rolling introduces NaN for early rows within each block group (first 7–28 observations per block), though LightGBM handles NaN; (b) VWAP/volume features for UP add noise (volume procured weekly does not strongly predict the UP daily price level); (c) statistical variance over 131 slots. Warrants investigation — see Q8.
- **TRL Daily down −0.4 pp** despite a clear pinball improvement at q50 (session summary: prior ≈ 23–24 → now 14.49). The optimal bid quantile may have shifted such that selection rate changes offset the better calibration.
- **† TRE neg normalised pinball (>1) is not comparable to other models.** The denominator `|mean(y_val)|` ≈ 12 CHF/MWh (most neg prices are small, e.g. −3 to −15 CHF), but the loss is driven by extreme events at −100 to −300 CHF/MWh, producing errors far larger than the mean. Use absolute pinball (36–52 CHF/MWh) and classifier AUC (0.764) as the meaningful TRE neg metrics.
- **TRL Weekly kpi down q75 = 0.451** is notably worse than the es window (0.223). The 8 kpi weeks (Apr–May 2026) appear to have had higher-than-usual down prices, causing systematic underprediction at the 75th quantile. Eight observations — treat with caution.
- This is the baseline run for `pinball_latest.json` tracking; future experiments can compare directly.

---

## Open Questions

| # | Question | Status |
|---|---|---|
| 1 | Add `ssrd_proxy = cos_zenith × (1 − cloud_cover_mean)` as explicit interaction feature? | ✅ Done — EXP-005 |
| 2 | Apply same CV n_estimators approach to TRL Daily and TRE for consistency? | Open |
| 3 | Add ACE / system load features to TRE pos (was 34% capture)? | Partially addressed — spot price raised TRE pos to ~46–47% in EXP-006/007; further gains from load/ACE still possible |
| 4 | Extend val window for TRL Weekly down to reduce noise in capture% estimates? | Open |
| 5 | Build inference pipeline (`src/pipeline/inference.py`) for live bidding? | Open |
| 6 | Conformal prediction wrapper (MAPIE) for coverage-guaranteed intervals? | Planned |
| 7 | Run `update_spot_forecast.py` daily to grow the Volue revision signal (spot_fcst_std / spot_fcst_change currently ~90% NaN in training) | Open — schedule or add to data update notebook cell |
| 8 | TRL Daily up regression in EXP-008 (−7.1 pp): is it caused by same-block rolling NaN fill behaviour, VWAP/volume noise for UP, or statistical variance? Consider ablation: retrain with only same-block rolling (no VWAP/volume) vs only VWAP/volume to isolate the cause. | Open |
