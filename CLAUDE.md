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
ECMWF ENS (ECDS)          Historical prices         Market data
  ensemble mean            TRL weekly · daily        Load · RES · ACE
  ensemble spread              TRE (15-min)               │
  cos_zenith (det.)               │                       │
        │                         └──────────┬────────────┘
        │                                    │
        ▼                                    ▼
  Weather features                   Feature store
  (agg. over CH domain)        lags · calendar · rolling stats
        │                                    │
        └─────────────────┬──────────────────┘
                          │
          ┌───────────────┼──────────────────┐
          ▼               ▼                  ▼
   TRL Weekly         TRL Daily           TRE
   LightGBM           LightGBM           LightGBM
   quantile           quantile           quantile
   (6–12 day          (1–3 day           (1–60 hour
    horizon)           horizon)           horizon)
          │               │                  │
          ▼               ▼                  │
   Conformal         Conformal     uses TRL Daily forecast
   wrapper           wrapper       as input feature
                                          │
                                          ▼
                                      Conformal
                                      wrapper
```

---

## Repository structure

```
balancing-price-forecast/
├── data/
│   ├── raw/
│   │   ├── ecmwf/          # ECDS GRIB2 downloads (temp, deleted after extraction)
│   │   ├── prices/         # TRL weekly, TRL daily, TRE historical prices
│   │   └── market/         # Load, RES generation, ACE, volumes
│   └── processed/
│       ├── features/       # Parquet feature store
│       └── targets/        # Aligned price targets
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_baseline.ipynb
├── src/
│   ├── data/
│   │   ├── ecds_download.py     # ECMWF ENS download + Swiss-domain extraction
│   │   ├── weather_features.py  # Standalone GRIB2 feature extraction
│   │   └── feature_store.py     # Align weather, market, price on common UTC index
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
├── requirements.txt
└── README.md
```

---

## Environment setup

Virtual environment: `C:\Users\ThijsAntoniedeBoer\OneDrive - HELION\Dokumente\python-projects\standard_env\`

eccodes C library required for GRIB2 reading — DLLs placed in `standard_env\Scripts\`. Set `ECCODES_PYTHON_USE_FINDLIBS=1` as a permanent user environment variable (already done). `ecds_download.py` also sets this at import time for self-containment.

**`requirements.txt`**

```
# Data retrieval
cdsapi                  # ECDS/CDS API client
ecmwf-opendata          # ECMWF open data client (real-time ENS)
cfgrib                  # GRIB2 parsing
eccodes                 # GRIB2 backend (C library must be on PATH)

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

### Weather — ECMWF ENS via ECDS

TIGGE historical ENS archive via **ECMWF Data Store (ECDS)**, `ecds-test.ecmwf.int`. Operational forecasts (inference) from ECMWF Open Data (no auth, last ~100 days).

**Variables retrieved from ECMWF ENS:**

| Variable | ECMWF param ID | Notes |
|---|---|---|
| 2m temperature | 167 | Load proxy |
| Total precipitation | 228 | Hydro inflow signal |
| Total cloud cover | 164 | Solar attenuation proxy |

Note: Surface solar radiation downward (param 169) is **not available in TIGGE**. It is replaced by a deterministic `cos_zenith` feature computed from day-of-year, time-of-day, and Swiss latitude (46.8°N) — this captures the potential insolation envelope. Cloud cover (ensemble) × cos_zenith (deterministic) together recover the key solar signal.

**Spatial aggregation:** Area-weighted mean over the Swiss domain `[48, 5, 45, 11]` (N/W/S/E), collapsed to a single time series per variable. TRL and TRE prices are uniform across Switzerland; fine spatial resolution adds no value.

**Ensemble features extracted per variable per step:**

- `mean` — best estimate
- `std` — ensemble spread (forecast uncertainty signal)
- `skew` — directional asymmetry
- `p10`, `p90` — tail probabilities

**Step resolution and model relevance:**

| Steps | Resolution | Relevant for |
|---|---|---|
| 0–72h | 6-hourly | TRE, TRL Daily |
| 72–168h | 24-hourly | TRL Daily (Friday), TRL Weekly |
| 168–360h | 48-hourly | TRL Weekly |

**ECDS credentials:** `~/.cdsapirc` → `url: https://ecds-test.ecmwf.int/api`. TIGGE licence accepted. Download script: `src/data/ecds_download.py`.

---

### Prices — TRL and TRE

Source: Swissgrid transparency platform or internal data store.

**TRL Weekly** — one row per week:
```
week_start_utc | marginal_price_chf | accepted_volume_mw
```

**TRL Daily** — one row per 4-hour block:
```
datetime_utc (block start) | marginal_price_chf | accepted_volume_mw
```

**TRE** — one row per 15-minute slot:
```
datetime_utc | marginal_price_chf | accepted_volume_mw
```

Store as Parquet in `data/raw/prices/`.

---

### Market data

| Variable | Granularity | Source |
|---|---|---|
| System load | 15-min | Swissgrid transparency |
| RES generation (solar, wind) | 15-min | Swissgrid transparency |
| Grid imbalance / ACE | 15-min | Swissgrid transparency |
| Cross-border flows | Hourly | ENTSO-E transparency |

ENTSO-E data via `entsoe-py`:
```bash
pip install entsoe-py
```

---

## Feature engineering plan

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
- Lead-time-adjusted spread: spread at 24h vs 72h lead and their ratio (uncertainty trajectory)
- Assigned to models by relevant horizon bands (see step resolution table above)

### Market features
- Current system load and 24h lag
- Solar generation, 24h lag
- ACE (area control error) — recent 1h mean and std (TRE only)
- Accepted TRL/TRE volumes, lagged

---

## Modelling approach

The primary objective is **revenue maximisation**, not price prediction error minimisation.
Models output quantile distributions; a separate bid-strategy layer (`src/pipeline/bid_strategy.py`)
converts the quantile forecasts into the revenue-maximising bid for each slot.

### Bid strategy (pay-as-bid)

In a pay-as-bid auction, bidding at quantile `q` of the predicted price distribution implies:
- **Positive-price markets** (TRL up/down, TRE pos): P(selected) = 1 − q. Optimal bid = argmax_q (price[q] − opp_cost) × (1 − q).
- **TRE neg** (negative prices): P(selected | extreme event) = q. Optimal bid = argmax_q (|price[q]| − 200) × q, using the extreme-regime model exclusively.

The KPI reported alongside pinball loss is **capture%** = backtest P&L / oracle P&L, where oracle = perfect-foresight bid at the clearing price.

### TRL Weekly model
- **Algorithm:** LightGBM with quantile loss
- **Horizon:** Single-step (one price per week)
- **Input weather:** 6–12 day ENS ensemble features (24h/48h steps)
- **Target quantiles:** q10, q25, q50, q75, q90

### TRL Daily model
- **Algorithm:** LightGBM with quantile loss
- **Horizon:** Direct multi-output — 6 blocks × 4h = 24h (extends to 18 blocks on Fridays)
- **Input weather:** 1–3 day ENS ensemble features (6h steps)
- **Target quantiles:** q10, q25, q50, q75, q90

### TRE model (two-stage)
- **Stage 1:** LGBMClassifier — P(extreme price): pos > 300 CHF/MWh, neg < −200 CHF/MWh
- **Stage 2a:** Normal-regime quantile model (prices within thresholds)
- **Stage 2b:** Extreme-regime quantile model (prices beyond thresholds)
- **Prediction:** blended as (1 − p_ext) × normal[q] + p_ext × extreme[q]
- **Bidding:** TRE neg uses the extreme-regime model directly; TRE pos uses the blended model
- **Input weather:** 0–3 day ENS ensemble features (6h steps) + cos_zenith at 15-min resolution
- **Additional feature:** TRL Daily quantile forecast for the enclosing 4h block
- **Target quantiles:** q10, q25, q50, q75, q90

### Conformal prediction wrapper
Wrap all three models with MAPIE's `TimeSeriesSplit`-compatible conformal wrapper for coverage-guaranteed prediction intervals:

```python
from mapie.time_series import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap
```

---

## Config file

**`config/config.yaml`**

```yaml
domain:
  area: [48, 5, 45, 11]   # N W S E — Swiss bounding box

ecmwf:
  dataset: tigge
  origin: ecmf
  type: pf
  grid: "0.5/0.5"
  param_ids: [167, 228, 164]   # t2m, tp, tcc (ssrd not in TIGGE)
  members: 50
  steps: [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 96, 120, 144, 168, 216, 264, 312, 360]
  run_times: ["00:00", "12:00"]

training:
  start_date: "2018-01-01"
  end_date: "2024-12-31"
  val_start: "2024-01-01"
  quantiles: [0.1, 0.25, 0.5, 0.75, 0.9]

models:
  trl_weekly:
    horizon_weeks: 1
    weather_steps_min: 144    # 6 days ahead
    weather_steps_max: 360    # 15 days ahead
  trl_daily:
    horizon_blocks: 18        # 6 normal + 18 for Friday (covers Mon)
    block_hours: 4
    weather_steps_max: 72     # 3 days ahead
  tre:
    horizon_steps: 240        # 1h normal; up to 60h Friday
    step_minutes: 15
    weather_steps_max: 72     # 3 days ahead

paths:
  raw_ecmwf: data/raw/ecmwf/
  raw_prices: data/raw/prices/
  raw_market: data/raw/market/
  features: data/processed/features/
  targets: data/processed/targets/
  models: models/
```

---

## Next steps
