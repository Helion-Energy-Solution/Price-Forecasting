# Swiss Balancing Energy Price Forecasting Pipeline

Forecast marginal prices for three Swiss balancing markets to support optimal bid price decisions. We always participate in all three markets — the forecast drives what price to bid, not whether to participate.

---

## Markets and bidding structure

### TRL Weekly — capacity auction (weekly granularity)
- **What:** Reserve capacity (MW) for the full following week (Monday–Sunday)
- **When bids are submitted:** Tuesday of the prior week
- **Forecast horizon:** 6–12 days ahead (bids submitted Tuesday, delivery starts following Monday)
- **Price granularity:** One marginal price per week
- **Clearing:** Pay-as-bid — you earn your submitted bid price (CHF/MW), not the clearing price


### TRL Daily — capacity auction (4-hour block granularity)
- **What:** Reserve capacity (MW) for 4-hour blocks on one or two future delivery days
- **When bids are submitted:** 2 business days ahead, gate closure at **14:00 local time**
- **Bidding calendar:**
  - Monday → Wednesday delivery
  - Tuesday → Thursday delivery
  - Wednesday → Friday delivery
  - **Thursday → Saturday + Sunday delivery** (2 days, no weekend gate closure)
  - **Friday → Monday + Tuesday delivery** (2 days, no weekend gate closure)
- **Forecast horizon:** 2–5 days ahead depending on bid day. Maximum 12 blocks (48h) for Thursday and Friday bids
- **Price granularity:** One marginal price per 4-hour block
- **Clearing:** Pay-as-bid — you earn your submitted bid price (CHF/MW), not the clearing price

### TRE — energy auction (15-minute granularity)
- **What:** Activate energy (MWh) in real time for 15-minute settlement slots
- **When bids are submitted:** Workdays Mon–Fri, 09:00–17:00 local, at least 1 hour before delivery
- **Forecast horizon:** As short as 1h (intraday weekday) up to ~65h (Friday 17:00 → Monday 10:00)
- **Weekend rule:** No submission on Saturday or Sunday. Friday 17:00 (last submission of the week) must cover all slots until Monday 10:00 — the first slot reachable after Monday's window opens at 09:00
- **Price granularity:** One marginal price per 15-minute slot
- **Clearing:** Pay-as-bid — you earn your submitted bid price (CHF/MWh), not the clearing price
- **TRE neg opportunity cost:** ~200 CHF/MWh (foregone PV spot revenue from curtailment). Only bids ≤ −200 CHF/MWh are economically justified for a PV asset.

---

## Model structure

Three separate models, one per market, because bidding lead times and relevant input features differ substantially:

```
ECMWF ENS (Open Data)      Historical prices         Market data
  ensemble mean             TRL weekly · daily        Spot prices (EPEX)
  ensemble spread               TRE (15-min)               │
  cos_zenith (det.)                │                       │
        │                          └──────────┬────────────┘
        │                                     │
        ▼                                     ▼
  Weather features                    Feature store
  (agg. over CH domain)         lags · calendar · rolling stats
        │                                     │
        └──────────────────┬──────────────────┘
                           │
          ┌────────────────┼──────────────────┐
          ▼                ▼                  ▼
   TRL Weekly          TRL Daily           TRE
   LightGBM            LightGBM           LightGBM
   quantile            quantile           quantile
   (6–12 day           (1–3 day           (1–60 hour
    horizon)            horizon)           horizon)
          │                │                  │
          ▼                ▼                  │
   Conformal          Conformal     uses TRL Daily forecast
   wrapper            wrapper       as input feature
                                           │
                                           ▼
                                       Conformal
                                       wrapper
```

---

## Repository structure

```
Price Forecasting/
├── data/
│   ├── raw/
│   │   ├── ecmwf/          # ECMWF Open Data GRIB2 downloads (temp, deleted after extraction)
│   │   ├── prices/         # TRL weekly, TRL daily, TRE price parquets
│   │   └── market/         # Volue spot forecast, reservoir levels, ENTSO-E forecasts
│   └── processed/
│       ├── features/       # Parquet feature store
│       └── targets/        # Aligned price targets
├── logs/                   # Task Scheduler run logs
├── models/                 # Trained model artefacts
├── notebooks/
├── output/
│   └── forecasts/          # JSON outputs from inference.py
├── src/
│   ├── data/
│   │   ├── ecds_download.py     # ECMWF historical ENS download (training only)
│   │   ├── weather_features.py  # GRIB2 feature extraction
│   │   ├── feature_store.py     # Align weather, market, price on common UTC index
│   │   ├── refresh_prices.py    # Daily price refresh from Swissgrid
│   │   └── entsoe_download.py   # ENTSO-E CH load/generation forecast download
│   ├── models/
│   │   ├── trl_weekly_model.py
│   │   ├── trl_daily_model.py
│   │   ├── tre_model.py
│   │   └── conformal.py
│   └── pipeline/
│       ├── train.py
│       └── inference.py
├── config/
│   └── config.yaml
├── .github/workflows/
│   └── daily_inference.yml  # CI entry point (daily at 10:00 UTC)
└── requirements.txt
```

---

## Environment setup

Virtual environment: `C:\Users\ThijsAntoniedeBoer\OneDrive - HELION\Dokumente\python-projects\standard_env\`

eccodes C library required for GRIB2 reading — DLLs placed in `standard_env\Scripts\`. `ECCODES_PYTHON_USE_FINDLIBS=1` set as a permanent user environment variable. `ecds_download.py` also sets this at import time for self-containment.

**`requirements.txt`**

```
# Data retrieval
cdsapi                  # ECDS/CDS API client (historical TIGGE downloads)
ecmwf-opendata          # ECMWF Open Data client (real-time ENS, no auth)
cfgrib                  # GRIB2 parsing
eccodes                 # GRIB2 backend (C library must be on PATH)
entsoe-py               # ENTSO-E Transparency Platform (CH load/generation forecasts)

# Data handling
pandas
numpy
pyarrow                 # Parquet feature store
xarray
scipy

# Modelling
lightgbm
scikit-learn
mapie                   # Conformal prediction wrapper

# Utilities
pyyaml
tqdm
python-dotenv
```

---

## Data sources

### Weather — ECMWF ENS

**Training:** TIGGE historical ENS archive via ECMWF Data Store (ECDS), `ecds-test.ecmwf.int`. Credentials in `~/.cdsapirc`. Download script: `src/data/ecds_download.py`.

**Inference:** ECMWF Open Data (no auth, last ~100 days). Falls back to latest `weather_ensemble.parquet` if download fails.

**Variables:**

| Variable | ECMWF param ID | Notes |
|---|---|---|
| 2m temperature | 167 | Load proxy |
| Total precipitation | 228 | Hydro inflow signal |
| Total cloud cover | 164 | Solar attenuation proxy |

Note: Surface solar radiation (param 169) is not in TIGGE. Replaced by deterministic `cos_zenith` computed from day-of-year, time-of-day, and Swiss latitude (46.8°N).

**Spatial aggregation:** Area-weighted mean over Swiss domain `[48, 5, 45, 11]` (N/W/S/E).

**Ensemble features extracted per variable per step:** `mean`, `std`, `skew`, `p10`, `p90`

**Step resolution:**

| Steps | Resolution | Relevant for |
|---|---|---|
| 0–72h | 6-hourly | TRE, TRL Daily |
| 72–168h | 24-hourly | TRL Daily (Friday), TRL Weekly |
| 168–360h | 48-hourly | TRL Weekly |

---

### Prices — TRL and TRE

Source: Swissgrid public tenders page (same source as the Market Dashboard website).

Prices in the CSV files are EUR-denominated; the `marginal_chf` column name is a legacy label — values are EUR.

**Daily refresh:** `src/data/refresh_prices.py` downloads the current-period CSVs directly from Swissgrid, parses them, and appends new rows to the parquets. Run automatically by the daily GitHub Actions workflow before inference. Training also uses these same parquets — run `refresh_prices.py` manually before retraining to include the latest data.

**Parquet schemas:**

`data/raw/prices/tre_slots.parquet`
```
slot_time (datetime64[us, UTC]) | direction (pos/neg) | offered | activated | marginal_chf | activation_rate
```

`data/raw/prices/trl_daily.parquet`
```
block_start (datetime64[us, UTC]) | direction (up/down) | offered_mw | awarded_mw | marginal_chf | median_bid_chf | award_rate_pct
```

`data/raw/prices/trl_weekly.parquet`
```
week_start (datetime64[us]) | direction (up/down) | offered_mw | awarded_mw | marginal_chf | median_bid_chf | vwap_chf | award_rate_pct | s1_is_active | s1_awarded_mw | s1_marginal_chf | s1_vwap_chf
```

---

### Load & generation forecasts — ENTSO-E

Source: ENTSO-E Transparency Platform (CH = `10YCH-SWISSGRIDZ`), via `entsoe-py`. Archived back to 2015; we backfill from 2023-01-01. Requires a free API token in `.env` as `ENTSOE_API_TOKEN` (request "Restful API access" from `transparency@entsoe.eu`).

**Refresh:** `src/data/entsoe_download.py` — incremental refresh by default (fetches a forward window to today+8d to catch newly published day-ahead/week-ahead forecasts); `--start 2023-01-01` for a full historical backfill. Run automatically by the daily GitHub Actions workflow (step 4) before inference; run manually before retraining.

**Data items pulled:**

| Item | ENTSO-E code | `process_type` | Resolution |
|---|---|---|---|
| Total load forecast, day-ahead | 6.1.B | A01 | hourly |
| Aggregated generation forecast, day-ahead | 14.1.C | A01 | hourly |
| Wind & solar generation forecast, day-ahead | 14.1.D | A01 | hourly |
| Total load forecast, week-ahead | 6.1.C | A31 | daily (max+min) |

**Parquet schemas:**

`data/raw/market/entsoe_da.parquet`
```
delivery_time (datetime64[us, UTC]) | load_da_mw | gen_da_mw | solar_da_mw | wind_da_mw
```

`data/raw/market/entsoe_load_week.parquet`
```
delivery_date (datetime64[us], local midnight) | load_wk_max_mw | load_wk_min_mw
```

**Point-in-time / leakage:** ENTSO-E returns each archived forecast as a flat series with no publish timestamp. `feature_store.py` models the publication deadline from the delivery date — day-ahead load D-1 ~12:00 local, day-ahead gen/solar D-1 18:00, week-ahead the prior Friday ~10:00 local — and exposes a value only once that deadline ≤ bid time (mirrors `_spot_forecast_asof`).

**Reachability by market** (driven by bid lead times):
- **TRE** — bids land close, so day-ahead load/gen/solar cover the near slots; week-ahead load fills the Friday→Monday gap. Full feature set.
- **TRL Daily** — bids 2 business days ahead, before the D-1 day-ahead publication → only **week-ahead load** is available.
- **TRL Weekly** — bids Tuesday, before the week-ahead Friday publication → **no** ENTSO-E forecast is leakage-safe; not used.

---

### Market data

| Variable | Granularity | Source | Status |
|---|---|---|---|
| Spot prices | Hourly | EPEX realized + Volue forecast | **implemented** (`spot_forecast_volue.parquet`, `spot_hourly.parquet`) |
| Hydro reservoir fill | Weekly | Swissgrid / BFE | **implemented** (`reservoir_levels.parquet`) |
| Load & generation forecast | Hourly / daily | ENTSO-E (see above) | **implemented** (`entsoe_da.parquet`, `entsoe_load_week.parquet`) |
| System load (actual) | 15-min | Swissgrid transparency | not yet wired in |
| Grid imbalance / ACE | 15-min | Swissgrid transparency | not yet wired in |

---

## Feature engineering

### Calendar and deterministic features
- Hour of day, quarter-hour of day, day of week, month, season
- `cos_zenith` at Swiss centroid (46.8°N, 8.2°E) — intraday solar potential proxy, computed at 15-min resolution
- Swiss public holiday indicator
- "Friday pre-weekend" flag (extended bidding horizon day)

### Price lag features (TRL Weekly)
- Lag 1 week, lag 4 weeks, lag 52 weeks
- Rolling mean and std: 4-week, 12-week window

### Price lag features (TRL Daily)
- Lag 1 block (4h), lag 6 blocks (24h), lag 42 blocks (7 days)
- Rolling mean and std: 7-day, 30-day window

### Price lag features (TRE)
- Lag 1 step (15min), lag 4 (1h), lag 96 (24h), lag 672 (1 week)
- Rolling mean and std: 24h, 7-day window
- TRL Daily forecast for the enclosing 4h block (cross-model feature)

### Weather features (from ECMWF ENS, aggregated over CH)
- Per variable: mean, std, skew, p10, p90
- Lead-time-adjusted spread: spread at 24h vs 72h lead and their ratio
- Assigned to models by relevant horizon bands (see step resolution table above)

### Market features (implemented)
- Spot prices: point-in-time Volue forecast (realized EPEX fallback), forecast revision std/change, weekly aggregates for TRL Weekly
- Hydro reservoir fill % (Wallis, Graubünden, Tessin, total CH), as-of bid time
- Cross-model: TRL Weekly clearing prices fed to TRL Daily and TRE

### ENTSO-E forecast features (implemented, point-in-time)
- **TRE:** `entsoe_load_da_mw`, `entsoe_gen_da_mw`, `entsoe_solar_da_mw`, `entsoe_net_load_da_mw` (load − solar − wind), plus week-ahead `entsoe_load_wk_max_mw` / `_min_mw` / `_spread_mw`
- **TRL Daily:** week-ahead `entsoe_load_wk_max_mw` / `_min_mw` / `_spread_mw` only
- Far-horizon rows where the day-ahead forecast was not yet published fall back to week-ahead (load) or NaN (gen/solar), which LightGBM routes through its NaN branch

### Not yet wired in
- Actual system load, RES generation, ACE, accepted volumes (sources identified; not in the feature store)

---

## Modelling approach

The primary objective is **revenue maximisation**, not price prediction error minimisation.
Models output quantile distributions; a bid-strategy layer converts the quantile forecasts into the revenue-maximising bid for each slot.

### Bid strategy (pay-as-bid)

In a pay-as-bid auction, bidding at quantile `q` of the predicted price distribution implies:
- **Positive-price markets** (TRL up/down, TRE pos): P(selected) = 1 − q. Optimal bid = argmax_q (price[q] − opp_cost) × (1 − q).
- **TRE neg** (negative prices): P(selected | extreme event) = q. Optimal bid = argmax_q (|price[q]| − 200) × q, using the extreme-regime model exclusively.

The KPI reported alongside pinball loss is **capture%** = backtest P&L / oracle P&L, where oracle = perfect-foresight bid at the clearing price.

### TRL Weekly model
- **Algorithm:** LightGBM with quantile loss
- **Train start:** 2023-01-02; early-stop val from 2025-04-07; final val from 2026-04-06
- **Horizon:** Single-step (one price per week)
- **Input weather:** Steps 144–360h (6–15 days ahead, 24h/48h resolution)
- **Target quantiles:** q10, q25, q50, q75, q90
- **Spot feature:** Yes — weekly aggregates (baseload mean, peakload mean, max, min, daily spread mean, neg hours) from Volue forecast

### TRL Daily model
- **Algorithm:** LightGBM with quantile loss
- **Train start:** 2023-01-01
- **Horizon:** Direct multi-output — 12 blocks × 4h = 48h
- **Input weather:** Steps 0–120h (6h resolution)
- **Target quantiles:** q10, q25, q50, q75, q90
- **Spot feature:** Yes
- **ENTSO-E feature:** Week-ahead CH load forecast (max/min/spread) — day-ahead is published after the 2-business-day-ahead bid, so not usable here

### TRE model (two-stage)
- **Train start:** 2023-01-01
- **Stage 1:** LGBMClassifier — P(extreme price): pos > 300 EUR/MWh, neg < −200 EUR/MWh
- **Stage 2a:** Normal-regime quantile model (prices within thresholds)
- **Stage 2b:** Extreme-regime quantile model (prices beyond thresholds)
- **Prediction:** blended as (1 − p_ext) × normal[q] + p_ext × extreme[q]
- **Bidding:** TRE neg uses the extreme-regime model directly; TRE pos uses the blended model
- **Input weather:** Steps 0–96h (6h resolution) + cos_zenith at 15-min resolution
- **Additional feature:** TRL Daily quantile forecast for the enclosing 4h block
- **Target quantiles:** q10, q25, q50, q75, q90
- **Spot feature:** Yes
- **ENTSO-E feature:** Day-ahead CH load/generation/solar + net load (near slots) and week-ahead load (Fri→Mon gap), point-in-time gated

### Conformal prediction wrapper
All three models wrapped with MAPIE's `TimeSeriesSplit`-compatible conformal wrapper for coverage-guaranteed prediction intervals.

---

## Config file

`config/config.yaml` — current values:

```yaml
domain:
  area: [48, 5, 45, 11]   # N W S E — Swiss bounding box

ecmwf:
  dataset: tigge
  origin: ecmf
  type: pf
  grid: "0.5/0.5"
  param_ids: [167, 228, 164]   # t2m, tp, tcc
  members: 50
  steps: [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 96, 120, 144, 168, 216, 264, 312, 360]
  run_times: ["00:00", "12:00"]

training:
  end_date: "2026-05-20"    # update after each price refresh before retraining
  val_start: "2026-05-01"
  quantiles: [0.1, 0.25, 0.5, 0.75, 0.9]

models:
  trl_weekly:
    train_start: "2023-01-02"
    es_val_start: "2025-04-07"   # early-stopping validation split
    val_start: "2026-04-06"
    horizon_weeks: 1
    weather_steps_min: 144
    weather_steps_max: 360
    spot_feature: true
  trl_daily:
    train_start: "2023-01-01"
    horizon_blocks: 12
    block_hours: 4
    weather_steps_max: 120
    spot_feature: true
  tre:
    train_start: "2023-01-01"
    horizon_steps: 260
    step_minutes: 15
    weather_steps_max: 96
    spot_feature: true
    extreme_threshold_pos: 300.0
    extreme_threshold_neg: -200.0

paths:
  raw_ecmwf: data/raw/ecmwf/
  raw_prices: data/raw/prices/
  raw_market: data/raw/market/
  features: data/processed/features/
  targets: data/processed/targets/
  models: models/
```

---

## Daily automation

Runs as a **GitHub Actions workflow** — `.github/workflows/daily_inference.yml`, scheduled at **10:00 UTC** (ECMWF 00z fully published by then) and triggerable manually via `workflow_dispatch`. (The former local `push_forecasts.ps1` / Windows Task Scheduler path has been retired — everything runs in CI now.)

```
1  refresh_prices.py        — download TRE + SRL&TRL CSVs from Swissgrid, append new rows
2  market_data.py           — refresh realized spot + reservoir levels
3  update_spot_forecast.py  — refresh Volue spot forecast        (secrets: VOLUE_CLIENT_ID/SECRET)
4  entsoe_download.py       — refresh CH load/generation forecasts (secret: ENTSOE_API_TOKEN;
                              non-fatal — skips cleanly if token/data missing)
5  inference.py             — download today's ECMWF Open Data 00z, run all three models,
                              write JSON to output/forecasts/
6  commit data parquets     — git add -f data/raw/ + weather_ensemble.parquet, commit+push to this repo
7  push forecasts           — clone Market-Dashboard (secret: MARKET_DASHBOARD_PAT),
                              copy 3 JSONs to data/forecasts/, commit+push → Vercel redeploys
```

**Required GitHub secrets:** `VOLUE_CLIENT_ID`, `VOLUE_CLIENT_SECRET`, `ENTSOE_API_TOKEN`, `MARKET_DASHBOARD_PAT`.

### Market Dashboard integration (cross-repo data flow)

Two repos with split responsibilities:
- **Price Forecasting (this repo)** — owns models + forecasts. Produces the three `*_latest.json` and pushes them to the dashboard.
- **Market-Dashboard** — owns realized prices, the forecast archive, the backtest join, and the public site (Vercel).
  - GitHub: `https://github.com/Helion-Energy-Solution/Market-Dashboard`
  - Local clone: `C:\Users\ThijsAntoniedeBoer\OneDrive - HELION\Dokumente\Market Dashboard`

```
THIS REPO (daily_inference.yml, 10:00 UTC)
  inference.py → output/forecasts/{trl_weekly,trl_daily,tre}_latest.json
       │  (step 7: clone Market-Dashboard, copy the 3 JSONs to data/forecasts/, commit+push)
       ▼
MARKET-DASHBOARD (update.yml, 01:00 + 10:00 UTC → patch_data.py → build_backtest.py)
  patch_data.py        — pull realized TRE/TRL/spot from Swissgrid → data/data.json
  build_backtest.py:
    archive_forecasts()    — append each *_latest.json as one line to
                             data/forecasts/history/{...}_history.jsonl
                             (one record per run, deduped by generated_at)
    build_*_backtest()     — for each slot/block/week, pick the most recent archived
                             forecast with generated_at < bid_deadline, join it with the
                             realized price → data/forecasts/backtest_*.json
                             + embed under data.json["backtests"]
       ▼
  index.html (Vercel)  — reads data/data.json: live forecast (*_latest.json), realized
                         prices, and the forecast-vs-realized backtests
```

**Key invariants (learned the hard way — violating them silently breaks the backtest):**
- **Point-in-time archival.** `*_latest.json` holds only *future biddable* slots and is overwritten every run. A slot's forecast must be archived into `*_history.jsonl` **while it is still in the future**, or it can never enter the backtest — the realized price will appear in the price overview but the backtest row will be permanently absent. (The forecast-history mechanism began ~2026-05-28; earlier slots needed git-history recovery.)
- **TRE archives the full window.** `archive_forecasts()` keeps all slots for TRE (`slot_limit = None`); the Fri→Mon window is ~260 slots (a former 96-slot/24h cap dropped weekend forecasts).
- **The backtest shows a market as soon as it *clears*, not when it delivers.** TRL Weekly (and Daily) capacity auctions clear days before delivery; `build_trl_weekly_backtest` includes any week whose clearing price is published, not only already-delivered weeks.
- **The dashboard owns realized data; this repo owns forecasts/models.** Don't commit `data/raw/**` locally (see the pre-commit hook below) — the dashboard's `patch_data.py` is the source of truth for realized prices.

**Retraining workflow:**
1. Run `python src/data/refresh_prices.py` to pull latest prices into parquets
2. Run `python src/data/entsoe_download.py` to pull latest CH load/generation forecasts
3. Update `training.end_date` (and optionally `val_start`) in `config/config.yaml`
4. Rebuild features: `python src/data/feature_store.py`
5. Re-run training notebooks / `src/pipeline/train.py`

**Data ownership & the pre-commit hook**

The daily Action is the **sole committer of refreshed data**: `data/raw/**` and
`data/processed/features/weather_ensemble.parquet`. A local notebook rerun also
rewrites those (binary) parquets; if they get committed locally they collide with
the Action's daily commit (binary parquets can't be merged → manual conflict every
time). To prevent that, `.githooks/pre-commit` unstages `data/raw/` and
`data/processed/features/` from **local** commits (it's skipped in CI, so the Action
still commits data normally). Your models, code, notebook, RESEARCH_LOG, and
backtest results commit as usual.

- Enable once per clone: `git config core.hooksPath .githooks`
- Force a deliberate local data commit (e.g. a one-time backfill): `git commit --no-verify`
- If you retrain locally, commit only `models/**` + `config.yaml` + `RESEARCH_LOG.md`; let the next Action run refresh the data parquets.
