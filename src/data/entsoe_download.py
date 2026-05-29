"""
entsoe_download.py — Download Swiss (CH) load & generation forecasts from the
ENTSO-E Transparency Platform and sync them to parquet.

Two products, both archived back to 2015 and used as model features:

  Day-ahead (hourly), saved to data/raw/market/entsoe_da.parquet
    load_da_mw   — Total Load day-ahead forecast        (6.1.B, A01)
    gen_da_mw    — Aggregated generation day-ahead       (14.1.C, A01)
    solar_da_mw  — Solar generation day-ahead            (14.1.D, A01)
    wind_da_mw   — Wind onshore generation day-ahead     (14.1.D, A01) — negligible in CH

  Week-ahead (daily min/max), saved to data/raw/market/entsoe_load_week.parquet
    load_wk_max_mw / load_wk_min_mw — Week-ahead total load forecast (6.1.C, A31)

Point-in-time note
------------------
ENTSO-E returns each archived forecast as a flat series indexed by *delivery*
time, with no publish timestamp. feature_store.py models the publication
deadline from the delivery date (DA load: D-1; DA gen/solar: D-1 18:00;
week-ahead: the prior Friday) and only exposes a value once that deadline has
passed relative to the bid time — so these parquets store the raw archived
forecast and leakage is handled downstream.

Auth
----
Requires ENTSOE_API_TOKEN in the environment or .env (free; request "Restful
API access" from transparency@entsoe.eu). Falls back to existing parquets if
the token is missing or a download fails.

Usage
-----
python src/data/entsoe_download.py                       # incremental refresh
python src/data/entsoe_download.py --start 2023-01-01    # historical backfill
python src/data/entsoe_download.py --start 2023-01-01 --end 2026-05-29
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MARKET_DIR = ROOT / "data" / "raw" / "market"
DA_PARQUET = MARKET_DIR / "entsoe_da.parquet"
WK_PARQUET = MARKET_DIR / "entsoe_load_week.parquet"

CH = "CH"                       # entsoe-py resolves to 10YCH-SWISSGRIDZ
TZ = "Europe/Brussels"         # ENTSO-E publication time zone
BACKFILL_START = "2023-01-01"  # matches the earliest model train_start


# ---------------------------------------------------------------------------
# Token / client
# ---------------------------------------------------------------------------

def _get_token() -> str | None:
    tok = os.environ.get("ENTSOE_API_TOKEN")
    if tok:
        return tok.strip()
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("ENTSOE_API_TOKEN") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _client():
    from entsoe import EntsoePandasClient
    tok = _get_token()
    if not tok:
        raise RuntimeError(
            "ENTSOE_API_TOKEN not found in environment or .env — "
            "request 'Restful API access' from transparency@entsoe.eu."
        )
    return EntsoePandasClient(api_key=tok)


def _to_utc_us(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalise a tz-aware index to datetime64[us, UTC] (codebase convention)."""
    return idx.tz_convert("UTC").as_unit("us")


# ---------------------------------------------------------------------------
# Per-endpoint fetch (chunked by year to stay within request limits)
# ---------------------------------------------------------------------------

def _year_chunks(start: pd.Timestamp, end: pd.Timestamp):
    cur = start
    while cur < end:
        nxt = min(pd.Timestamp(year=cur.year + 1, month=1, day=1, tz=TZ), end)
        yield cur, nxt
        cur = nxt


def _fetch_series(client, fn_name: str, start: pd.Timestamp, end: pd.Timestamp, **kw) -> pd.DataFrame:
    """Call a client query method over yearly chunks; return a concatenated frame.

    Each chunk is wrapped in try/except so a single missing period does not abort
    the whole backfill. Returns an empty frame if every chunk fails.
    """
    fn = getattr(client, fn_name)
    parts = []
    for cs, ce in _year_chunks(start, end):
        try:
            r = fn(CH, start=cs, end=ce, **kw)
            if r is not None and len(r):
                parts.append(r.to_frame() if isinstance(r, pd.Series) else r)
        except Exception as exc:  # noqa: BLE001 — one bad period shouldn't kill the run
            log.warning("  %s %s–%s failed: %s", fn_name, cs.date(), ce.date(), str(exc)[:120])
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def fetch_day_ahead(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Hourly day-ahead frame: load_da_mw, gen_da_mw, solar_da_mw, wind_da_mw."""
    load = _fetch_series(client, "query_load_forecast", start, end, process_type="A01")
    gen = _fetch_series(client, "query_generation_forecast", start, end, process_type="A01")
    ws = _fetch_series(client, "query_wind_and_solar_forecast", start, end, process_type="A01")

    cols = {}
    if not load.empty:
        col = "Forecasted Load" if "Forecasted Load" in load.columns else load.columns[0]
        cols["load_da_mw"] = load[col]
    if not gen.empty:
        cols["gen_da_mw"] = gen.iloc[:, 0]
    if not ws.empty:
        if "Solar" in ws.columns:
            cols["solar_da_mw"] = ws["Solar"]
        for wcol in ("Wind Onshore", "Wind Offshore"):
            if wcol in ws.columns:
                cols["wind_da_mw"] = cols.get("wind_da_mw", 0) + ws[wcol]

    if not cols:
        return pd.DataFrame()
    da = pd.concat(cols, axis=1)
    da.index = _to_utc_us(da.index)
    da.index.name = "delivery_time"
    return da.reset_index()


def fetch_week_ahead(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Daily week-ahead frame: load_wk_max_mw, load_wk_min_mw, keyed by local date."""
    wk = _fetch_series(client, "query_load_forecast", start, end, process_type="A31")
    if wk.empty:
        return pd.DataFrame()
    rename = {"Max Forecasted Load": "load_wk_max_mw", "Min Forecasted Load": "load_wk_min_mw"}
    wk = wk.rename(columns=rename)
    keep = [c for c in ("load_wk_max_mw", "load_wk_min_mw") if c in wk.columns]
    # Index is one row per delivery day at local midnight — store as the naive local date
    # so feature_store can match on the slot/block's local calendar day directly.
    local_date = wk.index.tz_convert(TZ).normalize().tz_localize(None).as_unit("us")
    out = wk[keep].copy()
    out.insert(0, "delivery_date", local_date)
    out = out.drop_duplicates("delivery_date", keep="last").sort_values("delivery_date")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sync helpers
# ---------------------------------------------------------------------------

def _merge_save(new: pd.DataFrame, parquet: Path, key: str) -> int:
    """Upsert new rows into an existing parquet on `key`; return rows added/updated."""
    if new.empty:
        return 0
    if parquet.exists():
        old = pd.read_parquet(parquet)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
        added = len(combined) - len(old)
    else:
        combined = new.sort_values(key).reset_index(drop=True)
        added = len(combined)
    combined.to_parquet(parquet, index=False)
    return added


def _refresh_window(parquet: Path, key: str) -> pd.Timestamp:
    """Start the incremental window a few days before the last stored row so newly
    published forward forecasts (DA for tomorrow, this week's week-ahead) are caught."""
    if parquet.exists():
        last = pd.read_parquet(parquet, columns=[key])[key].max()
        start_local = pd.Timestamp(last).tz_localize(None) if pd.Timestamp(last).tzinfo is None \
            else pd.Timestamp(last).tz_convert(TZ)
        start = pd.Timestamp(start_local).tz_localize(TZ) if pd.Timestamp(start_local).tzinfo is None \
            else start_local
        return (start - pd.Timedelta(days=3)).normalize()
    return pd.Timestamp(BACKFILL_START, tz=TZ)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    parser = argparse.ArgumentParser(description="Download CH ENTSO-E load/generation forecasts")
    parser.add_argument("--start", default=None, help="Backfill start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD); default = today + 8 days")
    args = parser.parse_args()

    MARKET_DIR.mkdir(parents=True, exist_ok=True)

    # Forward horizon: DA forecasts for tomorrow and week-ahead are published ahead of delivery.
    end = (pd.Timestamp(args.end, tz=TZ) if args.end
           else (pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=8)))

    try:
        client = _client()
    except Exception as exc:  # noqa: BLE001
        log.warning("ENTSO-E download skipped: %s", exc)
        log.warning("Feature build will reuse existing parquets if present.")
        return

    da_start = pd.Timestamp(args.start, tz=TZ) if args.start else _refresh_window(DA_PARQUET, "delivery_time")
    wk_start = pd.Timestamp(args.start, tz=TZ) if args.start else _refresh_window(WK_PARQUET, "delivery_date")

    log.info("Day-ahead   %s → %s", da_start.date(), end.date())
    da = fetch_day_ahead(client, da_start, end)
    n_da = _merge_save(da, DA_PARQUET, "delivery_time")
    if not da.empty:
        log.info("  cols=%s  rows fetched=%d  +%d to parquet",
                 [c for c in da.columns if c != "delivery_time"], len(da), n_da)

    log.info("Week-ahead  %s → %s", wk_start.date(), end.date())
    wk = fetch_week_ahead(client, wk_start, end)
    n_wk = _merge_save(wk, WK_PARQUET, "delivery_date")
    if not wk.empty:
        log.info("  rows fetched=%d  +%d to parquet", len(wk), n_wk)

    log.info("Done: entsoe_da +%d, entsoe_load_week +%d", n_da, n_wk)


if __name__ == "__main__":
    main()
