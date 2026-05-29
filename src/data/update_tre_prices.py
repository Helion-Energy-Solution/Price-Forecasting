"""
Download and parse TRE auction results from Swissgrid, append to tre_slots.parquet.

Source: https://www.swissgrid.ch/en/home/customers/topics/ancillary-services/tenders.html
Files:  YYYY-MM-TRE-Ergebnis.csv (or .csv.zip)

CSV schema (semicolon-delimited):
  Ausschreibung ; Von ; Bis ; Produkt ; Angebotene Menge ; Einheit ;
  Abgerufene Menge ; Einheit ; Preis ; Einheit ; Status

Parquet output schema:
  slot_time (datetime64[us, UTC]) | direction (pos/neg) |
  offered (int64 MW) | activated (int64 MW) |
  marginal_chf (float64 EUR/MWh, named for consistency with existing data) |
  activation_rate (float64)

Marginal price logic:
  For each (slot, direction): max(Preis where Abgerufene Menge > 0).
  If no activation: NaN — these slots are dropped during model training
  (dropna on marginal_chf) so the models learn the conditional price given
  activation, not a zero-inflated distribution.

Usage
-----
python src/data/update_tre_prices.py
"""

import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT       = Path(__file__).resolve().parents[2]
OUTPUT     = ROOT / "data" / "raw" / "prices" / "tre_slots.parquet"

BASE_URL    = "https://www.swissgrid.ch"
TENDERS_URL = BASE_URL + "/en/home/customers/topics/ancillary-services/tenders.html"
TRE_PATTERN = re.compile(r"/dam/jcr:[a-f0-9\-]+/(\d{4}-\d{2}-TRE-Ergebnis\.csv(?:\.zip)?)")


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _find_tre_links() -> list[tuple[str, int, int, str]]:
    """Return [(filename, year, month, url)] for all TRE files on the tenders page."""
    html = _http_get(TENDERS_URL).decode("utf-8", errors="replace")
    results = []
    seen = set()
    for m in TRE_PATTERN.finditer(html):
        filename = m.group(1)
        if filename in seen:
            continue
        seen.add(filename)
        dm = re.match(r"(\d{4})-(\d{2})-TRE", filename)
        if dm:
            url = BASE_URL + m.group(0)
            results.append((filename, int(dm.group(1)), int(dm.group(2)), url))
    return sorted(results, key=lambda x: (x[1], x[2]))


def _download_csv_bytes(url: str, filename: str) -> bytes:
    data = _http_get(url, timeout=120)
    if filename.endswith(".zip") or data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV in ZIP: {filename}")
            data = zf.read(csv_names[0])
    return data


def parse_tre_csv(raw: bytes) -> pd.DataFrame:
    """Parse raw CSV bytes into a per-slot DataFrame matching tre_slots.parquet schema."""
    text = raw.decode("latin-1", errors="replace")
    df = pd.read_csv(
        io.StringIO(text),
        sep=";",
        header=0,
        names=["tender", "von", "bis", "product", "offered_mw", "unit1",
               "activated_mw", "unit2", "price", "unit3", "status"],
        dtype={"offered_mw": float, "activated_mw": float, "price": float},
        on_bad_lines="skip",
    )

    # Drop header-repeat rows and rows with unparseable prices
    df = df.dropna(subset=["tender", "von", "price"])
    df = df[df["tender"].str.match(r"TRE_\d{2}_\d{2}_\d{2}", na=False)]

    # Extract date from tender ID: TRE_YY_MM_DD
    def tender_to_date(t):
        m = re.match(r"TRE_(\d{2})_(\d{2})_(\d{2})", str(t))
        if not m:
            return pd.NaT
        return pd.Timestamp(f"20{m.group(1)}-{m.group(2)}-{m.group(3)}")

    df["date"] = df["tender"].map(tender_to_date)
    df = df.dropna(subset=["date"])

    # Parse Von time — handle both "HH:MM" and stray "HH:MM:SS" variants
    def parse_time(t):
        t = str(t).strip()
        parts = t.split(":")
        try:
            return pd.Timedelta(hours=int(parts[0]), minutes=int(parts[1]))
        except Exception:
            return pd.NaT

    df["von_td"] = df["von"].map(parse_time)
    df = df.dropna(subset=["von_td"])

    # Slot timestamp in Swiss local time → UTC
    naive_local = df["date"] + df["von_td"]
    local_ts = pd.DatetimeIndex(naive_local).tz_localize(
        "Europe/Zurich", ambiguous="infer", nonexistent="shift_forward"
    )
    df["slot_time"] = local_ts.tz_convert("UTC").astype("datetime64[us, UTC]")

    # Direction from product suffix
    df["direction"] = df["product"].apply(
        lambda p: "pos" if str(p).endswith("+") else ("neg" if str(p).endswith("-") else None)
    )
    df = df.dropna(subset=["direction"])
    df["offered_mw"]   = pd.to_numeric(df["offered_mw"],   errors="coerce").fillna(0).astype(int)
    df["activated_mw"] = pd.to_numeric(df["activated_mw"], errors="coerce").fillna(0).astype(int)
    df["price"]        = pd.to_numeric(df["price"],        errors="coerce")

    # Aggregate per (slot_time, direction)
    def agg_slot(g):
        offered   = int(g["offered_mw"].sum())
        activated = int(g["activated_mw"].sum())
        called    = g[g["activated_mw"] > 0]["price"]
        marginal  = float(called.max()) if len(called) > 0 else float("nan")  # NaN = no activation (dropna'd in training)
        rate      = activated / offered if offered > 0 else 0.0
        return pd.Series({
            "offered":         offered,
            "activated":       activated,
            "marginal_chf":    marginal,
            "activation_rate": rate,
        })

    agg = (
        df.groupby(["slot_time", "direction"], observed=True)
          .apply(agg_slot, include_groups=False)
          .reset_index()
    )
    agg["offered"]   = agg["offered"].astype("int64")
    agg["activated"] = agg["activated"].astype("int64")
    return agg[["slot_time", "direction", "offered", "activated",
                "marginal_chf", "activation_rate"]]


def update():
    links = _find_tre_links()
    print(f"Found {len(links)} TRE file(s) on Swissgrid:")
    for fn, yr, mo, url in links:
        print(f"  {fn}")

    # Determine which months are already fully covered in the parquet
    existing_months = set()
    if OUTPUT.exists():
        ex = pd.read_parquet(OUTPUT, columns=["slot_time"])
        ex["slot_time"] = pd.to_datetime(ex["slot_time"], utc=True)
        for ts in ex["slot_time"]:
            local = ts.tz_convert("Europe/Zurich")
            existing_months.add((local.year, local.month))

    new_frames = []
    latest_month = max((y, m) for _, y, m, _ in links)
    for fn, yr, mo, url in links:
        # Always re-download the latest month — Swissgrid publishes it incrementally.
        is_latest = (yr, mo) == latest_month
        if (yr, mo) in existing_months and not is_latest:
            print(f"  Skipping {fn} (already in parquet)")
            continue

        print(f"  Downloading {fn} ...", end=" ", flush=True)
        raw = _download_csv_bytes(url, fn)
        print(f"{len(raw)//1024} KB  parsing ...", end=" ", flush=True)
        parsed = parse_tre_csv(raw)
        print(f"{len(parsed)} rows")
        new_frames.append(parsed)

    if not new_frames:
        print("Nothing new to add.")
        return

    new_data = pd.concat(new_frames, ignore_index=True)

    if OUTPUT.exists():
        existing = pd.read_parquet(OUTPUT)
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    combined.drop_duplicates(subset=["slot_time", "direction"], keep="last", inplace=True)
    combined.sort_values(["slot_time", "direction"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    tmp = OUTPUT.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp, index=False)
    import os; os.replace(tmp, OUTPUT)

    print(f"\nSaved {len(combined)} rows to {OUTPUT.name}")
    print(f"Latest slot: {combined['slot_time'].max()}")


if __name__ == "__main__":
    update()
