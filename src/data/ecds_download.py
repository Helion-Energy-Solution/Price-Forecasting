"""
Download ECMWF ENS forecasts and extract ensemble features for Switzerland.

Two backends:

  --source ecds       Historical ENS via ECMWF Data Store (auth required).
                      Register at https://ecds-test.ecmwf.int and accept TIGGE licence.
                      ~/.cdsapirc must contain the ECDS url/key.
                      Returns one GRIB file per run (all steps + params, area-subsetted).

  --source opendata   Real-time ENS via ECMWF Open Data (no auth, last ~100 days).
                      Serves global GRIB2 per step; we extract Swiss points and delete.

Spatial extraction:
  Temperature, cloud cover  → mean over 4 city grid points (Geneva, Bern, Zürich, Lugano)
  Precipitation             → mean over 8 hydro-canton grid cells (Valais, GR, Ticino)

Precipitation handling:
  ECMWF tp is accumulated from T=0. ECDS (multi-step file) differences consecutive
  steps to get interval totals, then divides by interval hours → mm/h rate.
  Open Data (single-step files) skips precipitation entirely — TRL Daily and TRE
  (the only use cases for Open Data) do not use precipitation as a feature.

Output: data/processed/features/weather_ensemble.parquet
Schema: init_time | valid_time | lead_hours | variable | mean | std | skew | p10 | p90

Usage
-----
python src/data/ecds_download.py --source ecds --date 2023-06-01
python src/data/ecds_download.py --source ecds --start 2022-01-01 --end 2024-12-31
python src/data/ecds_download.py --source opendata --date 2026-04-29
"""

import os
os.environ.setdefault("ECCODES_PYTHON_USE_FINDLIBS", "1")

# Guard against re-spawned child processes on Windows.
# The launcher passes ECDS_WORKER_ACTIVE=child via CreateProcess env so the
# python.exe stub propagates it to the real interpreter. Any further children
# spawned by libraries (numpy/cdsapi/cfgrib) inherit it and exit immediately.
# "worker" → legitimate process launched by ecds_parallel_launch.py; continue but
#             flip flag so any sub-processes it spawns (via numpy/cdsapi/cfgrib) exit.
# "child"  → spawned BY a worker; exit immediately before doing any download work.
# unset    → direct one-off invocation (e.g. --date 2023-06-01); run normally.
_ecds_flag = os.environ.get("ECDS_WORKER_ACTIVE", "")
if _ecds_flag == "child":
    os._exit(0)
os.environ["ECDS_WORKER_ACTIVE"] = "child"

import argparse
import logging
import sys
import tempfile
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
OUTPUT_PARQUET = ROOT / "data" / "processed" / "features" / "weather_ensemble.parquet"

# ---------------------------------------------------------------------------
# Spatial extraction points (0.5° ECMWF grid)
# ---------------------------------------------------------------------------

# Temperature and cloud cover — population/load centres
# Geneva (46.0, 6.0), Bern (47.0, 7.5), Zürich (47.5, 8.5), Lugano (46.0, 9.0)
CITY_LATS = [46.0, 47.0, 47.5, 46.0]
CITY_LONS = [ 6.0,  7.5,  8.5,  9.0]

# Precipitation — Alpine hydro catchment cantons
# Valais: (46.0,7.0) (46.0,7.5) (46.0,8.0)
# Graubünden: (46.5,9.0) (46.5,9.5) (47.0,9.5)
# Ticino: (46.0,8.5) (46.0,9.0)
HYDRO_LATS = [46.0, 46.0, 46.0, 46.5, 46.5, 47.0, 46.0, 46.0]
HYDRO_LONS = [ 7.0,  7.5,  8.0,  9.0,  9.5,  9.5,  8.5,  9.0]

SHORTNAME_TO_LABEL = {
    "2t":  "temp_2m",
    "t2m": "temp_2m",
    "tcc": "cloud_cover",
    "tp":  "precip_rate_mmh",  # stored as mm/h after interval differencing
}
CITY_LABELS  = {"temp_2m", "cloud_cover"}
HYDRO_LABELS = {"precip_rate_mmh"}

# Open Data params — no precipitation (single-step files make differencing complex
# and TRL Daily / TRE do not use precipitation as a feature)
OPENDATA_PARAMS = ["2t", "tcc"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def ensemble_stats(members: np.ndarray) -> dict:
    from scipy.stats import skew as scipy_skew
    members = np.atleast_1d(members.ravel())
    return {
        "mean": float(np.mean(members)),
        "std":  float(np.std(members, ddof=1)) if len(members) > 1 else 0.0,
        "skew": float(scipy_skew(members)) if len(members) > 2 else 0.0,
        "p10":  float(np.percentile(members, 10)),
        "p90":  float(np.percentile(members, 90)),
    }


def _select_points(da, lats: list, lons: list):
    """Select specific lat/lon grid points and return mean across them.

    Uses nearest-neighbour selection to handle floating-point grid alignment.
    """
    import xarray as xr
    lat_idx = xr.DataArray(lats, dims="pt")
    lon_idx = xr.DataArray(lons, dims="pt")
    return da.sel(latitude=lat_idx, longitude=lon_idx, method="nearest").mean(dim="pt")


# ---------------------------------------------------------------------------
# ECDS extraction — one file per run, all steps + params, pre-subsetted.
# Dims: (number=50, step=N, latitude, longitude)
# ---------------------------------------------------------------------------

def extract_ecds(grib_path: Path, init_time: pd.Timestamp) -> list[dict]:
    import cfgrib

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        datasets = cfgrib.open_datasets(str(grib_path), indexpath=None)

    for ds in datasets:
        if not ds.data_vars:
            continue
        var_name = list(ds.data_vars)[0]
        label = SHORTNAME_TO_LABEL.get(var_name)
        if label is None:
            log.debug("  Unknown variable %s — skipping", var_name)
            continue
        if "number" not in ds[var_name].dims or "step" not in ds[var_name].dims:
            ds.close()
            continue

        da = ds[var_name]

        if label in CITY_LABELS:
            spatial_mean = _select_points(da, CITY_LATS, CITY_LONS)
        elif label in HYDRO_LABELS:
            spatial_mean = _select_points(da, HYDRO_LATS, HYDRO_LONS)
        else:
            ds.close()
            continue
        # spatial_mean shape: (number, step)

        step_ns = da["step"].values

        for s_idx in range(len(step_ns)):
            lead_h = int(pd.Timedelta(step_ns[s_idx]).total_seconds() // 3600)
            valid_time = init_time + pd.Timedelta(hours=lead_h)
            members = spatial_mean.isel(step=s_idx).values  # (n_members,)

            if label in HYDRO_LABELS:
                # Difference consecutive accumulated steps → interval total → mm/h rate
                prev_h = int(pd.Timedelta(step_ns[s_idx - 1]).total_seconds() // 3600) if s_idx > 0 else 0
                interval_h = lead_h - prev_h
                if s_idx > 0:
                    members = members - spatial_mean.isel(step=s_idx - 1).values
                members = members / max(interval_h, 1)

            rows.append({
                "init_time":  init_time,
                "valid_time": valid_time,
                "lead_hours": lead_h,
                "variable":   label,
                **ensemble_stats(members),
            })

        ds.close()

    return rows


# ---------------------------------------------------------------------------
# Open Data extraction — one file per step, global grid.
# Temperature and cloud cover only (no precipitation — see module docstring).
# Dims: (number, latitude, longitude)
# ---------------------------------------------------------------------------

def extract_opendata_step(grib_path: Path, init_time: pd.Timestamp, step_hours: int) -> list[dict]:
    import xarray as xr

    rows = []
    valid_time = init_time + pd.Timedelta(hours=step_hours)

    for short_name in OPENDATA_PARAMS:
        label = SHORTNAME_TO_LABEL.get(short_name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = xr.open_dataset(
                    grib_path,
                    engine="cfgrib",
                    backend_kwargs={"filter_by_keys": {"shortName": short_name}},
                    indexpath=None,
                )
        except Exception as e:
            log.debug("  Cannot read %s: %s", short_name, e)
            continue

        data_vars = list(ds.data_vars)
        if not data_vars:
            ds.close()
            continue
        var_name = data_vars[0]
        if label is None:
            label = SHORTNAME_TO_LABEL.get(var_name)
        if label is None or "number" not in ds[var_name].dims:
            ds.close()
            continue

        members = _select_points(ds[var_name], CITY_LATS, CITY_LONS).values  # (n_members,)

        rows.append({
            "init_time":  init_time,
            "valid_time": valid_time,
            "lead_hours": step_hours,
            "variable":   label,
            **ensemble_stats(members),
        })
        ds.close()

    return rows


# ---------------------------------------------------------------------------
# ECDS source — all steps in one request
# ---------------------------------------------------------------------------

def process_ecds_run(cfg: dict, run_date: date, run_hour: int) -> pd.DataFrame:
    import cdsapi

    init_time = pd.Timestamp(f"{run_date} {run_hour:02d}:00", tz="UTC")
    ecmwf = cfg["ecmwf"]

    request = {
        "class": "ti",
        "dataset": "tigge",
        "date": run_date.strftime("%Y-%m-%d"),
        "expver": "prod",
        "grid": ecmwf["grid"],
        "levtype": "sfc",
        "number": list(range(1, ecmwf["members"] + 1)),
        "origin": ecmwf["origin"],
        "param": "/".join(str(p) for p in ecmwf["param_ids"]),
        "step": "/".join(str(s) for s in ecmwf["steps"]),
        "time": f"{run_hour:02d}:00",
        "type": ecmwf["type"],
        "area": cfg["domain"]["area"],
        "format": "grib2",
    }

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        log.info("[ecds] %s %02dz — submitting ...", run_date, run_hour)
        client = cdsapi.Client()
        client.retrieve("tigge", request, str(tmp_path))
        size_mb = tmp_path.stat().st_size / 1e6
        log.info("  Downloaded %.1f MB  extracting ...", size_mb)
        rows = extract_ecds(tmp_path, init_time)
        log.info("  Extracted %d feature rows", len(rows))
    except Exception as e:
        log.error("ECDS request failed: %s", e)
        raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Open Data source — one step at a time
# ---------------------------------------------------------------------------

def process_opendata_run(cfg: dict, run_date: date, run_hour: int) -> pd.DataFrame:
    from ecmwf.opendata import Client

    init_time = pd.Timestamp(f"{run_date} {run_hour:02d}:00", tz="UTC")
    steps = cfg["ecmwf"]["steps"]
    source = os.environ.get("ECMWF_OPENDATA_SOURCE", "ecmwf")
    client = Client(source=source)
    all_rows = []

    log.info("[opendata] %s %02dz — %d steps", run_date, run_hour, len(steps))

    for step in steps:
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                client.retrieve(
                    date=int(run_date.strftime("%Y%m%d")),
                    time=run_hour,
                    step=[step],
                    stream="enfo",
                    type="pf",
                    param=OPENDATA_PARAMS,
                    levtype="sfc",
                    target=str(tmp_path),
                )
            size_mb = tmp_path.stat().st_size / 1e6
            log.info("  step=%3dh → %.0f MB  extracting ...", step, size_mb)
            rows = extract_opendata_step(tmp_path, init_time, step)
            all_rows.extend(rows)
            log.info("  step=%3dh → %d rows", step, len(rows))
        except Exception as e:
            log.warning("  step=%dh failed: %s", step, e)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Parquet output — append and deduplicate
# ---------------------------------------------------------------------------

def save_features(df: pd.DataFrame, output: Path = None):
    if df.empty:
        log.warning("No features to save.")
        return
    out_path = output or OUTPUT_PARQUET
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.drop_duplicates(
            subset=["init_time", "valid_time", "variable"], keep="last", inplace=True
        )
    else:
        combined = df

    combined.sort_values(["init_time", "valid_time", "variable"], inplace=True)
    # Write to a temp file then atomically replace to avoid partial-write corruption
    tmp_path = out_path.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    log.info("Saved %d rows → %s", len(combined), out_path.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["opendata", "ecds"], required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Single date YYYY-MM-DD")
    group.add_argument("--start", help="Start of range YYYY-MM-DD")
    parser.add_argument("--end", help="End of range (default: today)")
    parser.add_argument("--run", choices=["00", "12", "both"], default="both")
    parser.add_argument("--output", default=None, help="Override output parquet path")
    args = parser.parse_args()

    cfg = load_config()
    run_hours = {"00": [0], "12": [12], "both": [0, 12]}[args.run]
    process_fn = process_opendata_run if args.source == "opendata" else process_ecds_run

    if args.date:
        dates = [datetime.strptime(args.date, "%Y-%m-%d").date()]
    else:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
        dates = list(date_range(start, end))

    log.info("Source=%s | %d date(s) × %d run(s)", args.source, len(dates), len(run_hours))

    failed = []
    for d in dates:
        for h in run_hours:
            try:
                df = process_fn(cfg, d, h)
                save_features(df, output=Path(args.output) if args.output else None)
            except Exception:
                log.exception("Failed: %s %02dz", d, h)
                failed.append((d, h))

    if failed:
        log.error("Failed runs: %s", failed)
        sys.exit(1)
    log.info("Done.")


if __name__ == "__main__":
    main()
