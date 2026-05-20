"""
Extract ensemble weather features from GRIB2 files and write to Parquet.

For each ECMWF ENS GRIB2 file the script:
  1. Reads all members with cfgrib via xarray.
  2. Subsets to the Swiss bounding box [45–48°N, 5–11°E].
  3. Computes area-weighted mean across the Swiss grid cells.
  4. Computes ensemble statistics: mean, std, skew, p10, p90.
  5. Stacks into a long-format DataFrame indexed by (init_time, valid_time, variable).
  6. Appends to a Parquet file in data/processed/features/.

Usage
-----
# Single file:
python src/data/weather_features.py data/raw/ecmwf/2026/04/ens_20260428_00z.grib2

# All files in a folder:
python src/data/weather_features.py data/raw/ecmwf/2026/04/

# Process everything under data/raw/ecmwf/:
python src/data/weather_features.py
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "data" / "processed" / "features" / "weather_ensemble.parquet"

# Swiss bounding box: N=48, W=5, S=45, E=11
CH_LAT_SLICE = slice(48, 45)   # cfgrib stores lat descending
CH_LON_SLICE = slice(5, 11)

# ECMWF short name → friendly column prefix
PARAM_NAMES = {
    "2t":   "temp_2m",
    "ssrd": "ssrd",
    "tp":   "precip_total",
    "tcc":  "cloud_cover",
}


def area_weights(lats: np.ndarray) -> np.ndarray:
    """Cosine-latitude weights for area-weighted mean."""
    w = np.cos(np.deg2rad(lats))
    return w / w.sum()


def ensemble_stats(arr: np.ndarray) -> dict:
    """Compute ensemble statistics over the member axis (axis 0)."""
    from scipy.stats import skew as scipy_skew

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr, ddof=1)),
            "skew": float(scipy_skew(arr)),
            "p10":  float(np.percentile(arr, 10)),
            "p90":  float(np.percentile(arr, 90)),
        }


def process_file(grib_path: Path) -> pd.DataFrame | None:
    """Read one GRIB2 file and return a feature DataFrame."""
    log.info("Processing %s", grib_path.name)
    rows = []

    for short_name, col_prefix in PARAM_NAMES.items():
        try:
            ds = xr.open_dataset(
                grib_path,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": {"shortName": short_name}},
                indexpath=None,
            )
        except Exception as e:
            log.warning("  Could not read %s from %s: %s", short_name, grib_path.name, e)
            continue

        # Identify the data variable (cfgrib names it after the short name or GRIB param)
        data_vars = [v for v in ds.data_vars if v not in ("latitude", "longitude")]
        if not data_vars:
            log.warning("  No data variable found for %s", short_name)
            continue
        var = data_vars[0]

        # Subset to Swiss domain
        ds_ch = ds.sel(latitude=CH_LAT_SLICE, longitude=CH_LON_SLICE)
        if ds_ch[var].size == 0:
            log.warning("  Swiss subsetting returned empty array for %s", short_name)
            continue

        lats = ds_ch["latitude"].values
        weights = area_weights(lats)  # shape: (n_lat,)

        # Dims expected: (number, step, latitude, longitude)
        da = ds_ch[var]
        init_time = pd.Timestamp(ds.attrs.get("time", None) or ds["time"].values.flat[0])

        step_dim = "step" if "step" in da.dims else None
        if step_dim is None:
            log.warning("  No step dimension for %s", short_name)
            continue

        for step_idx in range(da.sizes["step"]):
            step_da = da.isel(step=step_idx)
            valid_time = init_time + pd.Timedelta(step_da["step"].values.item(), unit="ns")

            # Weighted spatial mean per member → shape: (n_members,)
            # da dims after isel(step): (number, latitude, longitude)
            spatial_mean_per_member = (
                (step_da * weights[np.newaxis, :, np.newaxis])
                .sum(dim="latitude")
                .mean(dim="longitude")
                .values
            )

            stats = ensemble_stats(spatial_mean_per_member)
            row = {
                "init_time": init_time,
                "valid_time": valid_time,
                "lead_hours": int(step_da["step"].values.item() / 3.6e12),  # ns → hours
                "variable": col_prefix,
                **stats,
            }
            rows.append(row)

        ds.close()

    if not rows:
        return None

    return pd.DataFrame(rows)


def save_features(df: pd.DataFrame):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        # Deduplicate: drop rows that already exist for the same (init_time, valid_time, variable)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["init_time", "valid_time", "variable"], keep="last")
        combined.sort_values(["init_time", "valid_time", "variable"], inplace=True)
        combined.to_parquet(OUTPUT_PATH, index=False)
        log.info("Updated %s (%d rows total)", OUTPUT_PATH.name, len(combined))
    else:
        df.sort_values(["init_time", "valid_time", "variable"], inplace=True)
        df.to_parquet(OUTPUT_PATH, index=False)
        log.info("Created %s (%d rows)", OUTPUT_PATH.name, len(df))


def collect_grib_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(target.rglob("*.grib2"))
    # Default: everything under raw/ecmwf
    return sorted((ROOT / "data" / "raw" / "ecmwf").rglob("*.grib2"))


def main():
    parser = argparse.ArgumentParser(description="Extract ensemble features from GRIB2 files.")
    parser.add_argument("path", nargs="?", help="GRIB2 file or folder (default: all data/raw/ecmwf/**/*.grib2)")
    args = parser.parse_args()

    target = Path(args.path) if args.path else ROOT / "data" / "raw" / "ecmwf"
    files = collect_grib_files(target)

    if not files:
        log.error("No .grib2 files found under %s", target)
        return

    log.info("Found %d GRIB2 file(s) to process", len(files))

    all_dfs = []
    for f in files:
        df = process_file(f)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        log.error("No features extracted.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    save_features(combined)
    log.info("Done.")


if __name__ == "__main__":
    main()
