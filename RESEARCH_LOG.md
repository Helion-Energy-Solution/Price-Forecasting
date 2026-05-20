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

## Open Questions

| # | Question | Status |
|---|---|---|
| 1 | Add `ssrd_proxy = cos_zenith × (1 − cloud_cover_mean)` as explicit interaction feature? | Open |
| 2 | Apply same CV n_estimators approach to TRL Daily and TRE for consistency? | Open |
| 3 | Add ACE / system load features to TRE pos (currently 34% capture — biggest gap)? | Open |
| 4 | Extend val window for TRL Weekly down to reduce noise in capture% estimates? | Open |
| 5 | Build inference pipeline (`src/pipeline/inference.py`) for live bidding? | Open |
| 6 | Conformal prediction wrapper (MAPIE) for coverage-guaranteed intervals? | Planned |
