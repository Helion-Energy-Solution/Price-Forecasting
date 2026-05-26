"""
refresh_prices.py — Download fresh Swissgrid CSVs and sync price parquets.

Fetches the current-period CSV files directly from Swissgrid (same source as
the Market Dashboard GitHub Actions workflow), then appends any new rows to
the Price Forecasting parquet files so lag features use up-to-date prices.

Run before inference.py (handled automatically by push_forecasts.ps1).
"""

import csv
import io
import re
import datetime
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo

FORECAST_DIR = Path(__file__).resolve().parents[2]

TRE_PARQUET   = FORECAST_DIR / "data" / "raw" / "prices" / "tre_slots.parquet"
TRL_D_PARQUET = FORECAST_DIR / "data" / "raw" / "prices" / "trl_daily.parquet"
TRL_W_PARQUET = FORECAST_DIR / "data" / "raw" / "prices" / "trl_weekly.parquet"

ZURICH = ZoneInfo("Europe/Zurich")

SWISSGRID_TENDERS = "https://www.swissgrid.ch/en/home/customers/topics/ancillary-services/tenders.html"
SWISSGRID_BASE    = "https://www.swissgrid.ch"


# ── Download helpers ───────────────────────────────────────────────────────────

def _http_get(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _extract_if_zip(data: bytes) -> bytes:
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("ZIP contains no CSV files")
            return zf.read(csv_names[0])
    return data


def _find_csv_links(html: str) -> dict[str, str]:
    """Return {filename: full_url} for every .csv link on the tenders page."""
    pattern = r'(/dam/jcr:[a-f0-9\-]+/([^"\'>\s]+\.csv))'
    links: dict[str, str] = {}
    for path, filename in re.findall(pattern, html):
        if filename not in links:
            links[filename] = SWISSGRID_BASE + path
    return links


def download_current_csvs(dest_dir: Path, parquet_last_date: datetime.date) -> None:
    """Download TRE and SRL&TRL CSVs needed to fill gaps since parquet_last_date."""
    today = datetime.date.today()

    (dest_dir / "TRE").mkdir(parents=True, exist_ok=True)
    (dest_dir / "SRL&TRL").mkdir(parents=True, exist_ok=True)

    print("  Fetching Swissgrid tenders page...")
    html  = _http_get(SWISSGRID_TENDERS).decode("utf-8", errors="replace")
    links = _find_csv_links(html)

    # SRL&TRL: one annual file — always current year
    fn_trl = f"{today.year}-PRL-SRL-TRL-Ergebnis.csv"
    if fn_trl in links:
        print(f"  Downloading {fn_trl}...", end=" ", flush=True)
        raw = _extract_if_zip(_http_get(links[fn_trl]))
        (dest_dir / "SRL&TRL" / fn_trl).write_bytes(raw)
        print(f"{len(raw)//1024} KB")
    else:
        print(f"  WARNING: {fn_trl} not found on page")

    # TRE: download any months that could have data newer than parquet
    # (Swissgrid keeps several months on the tenders page)
    tre_pattern = re.compile(r"^(\d{4})-(\d{2})-TRE-Ergebnis\.csv$")
    for filename, url in sorted(links.items()):
        m = tre_pattern.match(filename)
        if not m:
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        month_start = datetime.date(y, mo, 1)
        # Download if this month overlaps with [parquet_last_date, today]
        month_end = (datetime.date(y, mo + 1, 1) if mo < 12
                     else datetime.date(y + 1, 1, 1)) - datetime.timedelta(days=1)
        if month_end < parquet_last_date or month_start > today:
            continue
        print(f"  Downloading {filename}...", end=" ", flush=True)
        try:
            raw = _extract_if_zip(_http_get(url))
            (dest_dir / "TRE" / filename).write_bytes(raw)
            print(f"{len(raw)//1024} KB")
        except Exception as e:
            print(f"failed: {e}")


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _parse_num(s: str) -> float:
    if not s:
        return 0.0
    try:
        return float(s.replace(",", ".").strip())
    except ValueError:
        return 0.0


def _median(prices: list[float]) -> float:
    if not prices:
        return float("nan")
    s = sorted(prices)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


# ── TRE ───────────────────────────────────────────────────────────────────────

def refresh_tre(tre_dir: Path) -> int:
    existing = pd.read_parquet(TRE_PARQUET)
    last_ts  = existing["slot_time"].max()
    print(f"  TRE: parquet last slot = {last_ts}")

    slot_map: dict = {}   # (date_str, slot_from, direction) -> aggregates

    for path in sorted(tre_dir.rglob("*-TRE-Ergebnis.csv")):
        with open(path, encoding="latin-1", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                auction   = row[0].strip()
                slot_from = row[1].strip() if len(row) > 1 else ""
                product   = row[3].strip() if len(row) > 3 else ""
                offered   = _parse_num(row[4]) if len(row) > 4 else 0.0
                activated = _parse_num(row[6]) if len(row) > 6 else 0.0
                price     = _parse_num(row[8]) if len(row) > 8 else 0.0
                status    = row[10].strip() if len(row) > 10 else ""

                m = re.match(r"^TRE_(\d{2})_(\d{2})_(\d+)$", auction)
                if not m:
                    continue
                yy, mm, dd = int(m.group(1)), m.group(2), int(m.group(3))
                date_str  = f"{2000+yy}-{mm}-{dd:02d}"

                is_pos    = "sa+" in product or "da+" in product
                direction = "pos" if is_pos else "neg"

                key = (date_str, slot_from, direction)
                if key not in slot_map:
                    slot_map[key] = {"offered": 0.0, "activated": 0.0, "marginal": None}
                r = slot_map[key]
                r["offered"] += offered
                if status == "aktiviert" and activated > 0:
                    r["activated"] += activated
                    if is_pos:
                        if r["marginal"] is None or price > r["marginal"]:
                            r["marginal"] = price
                    else:
                        if r["marginal"] is None or price < r["marginal"]:
                            r["marginal"] = price

    new_rows = []
    for (date_str, slot_from, direction), r in sorted(slot_map.items()):
        if not slot_from or ":" not in slot_from:
            continue
        try:
            h, mn = map(int, slot_from.split(":"))
            y, mo, d = map(int, date_str.split("-"))
            local_dt = datetime.datetime(y, mo, d, h, mn, tzinfo=ZURICH)
            utc_ts   = pd.Timestamp(local_dt).tz_convert("UTC")
        except Exception:
            continue

        if utc_ts <= last_ts:
            continue

        offered   = r["offered"]
        activated = r["activated"]
        marginal  = r["marginal"] if r["marginal"] is not None else 0.0
        act_rate  = round(activated / offered, 4) if offered > 0 else 0.0

        new_rows.append({
            "slot_time":       utc_ts,
            "direction":       direction,
            "offered":         int(round(offered)),
            "activated":       int(round(activated)),
            "marginal_chf":    marginal,
            "activation_rate": act_rate,
        })

    if not new_rows:
        print("  TRE: already up to date")
        return 0

    new_df   = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["slot_time", "direction"]).reset_index(drop=True)
    combined.to_parquet(TRE_PARQUET, index=False)
    print(f"  TRE: +{len(new_rows)} rows  new last = {new_df['slot_time'].max()}")
    return len(new_rows)


# ── TRL shared CSV parse ───────────────────────────────────────────────────────

def _parse_trl_csvs(trl_dir: Path) -> tuple[dict, dict, dict]:
    """Return (daily_map, weekly_map, weekly_s1_map).

    Keys:
      daily_map:     (date_str, block_label, direction)
      weekly_map:    (year, iso_week, direction)
      weekly_s1_map: (year, iso_week, direction)
    Values: {offered, awarded, max_price, total_cost, bid_prices}
    """
    daily_map:     dict = {}
    weekly_map:    dict = {}
    weekly_s1_map: dict = {}

    def _entry():
        return {"offered": 0.0, "awarded": 0.0, "max_price": 0.0,
                "total_cost": 0.0, "bid_prices": []}

    def _block_label(desc: str) -> str | None:
        m = re.search(r"(\d{2}:\d{2})\s+bis\s+(\d{2}:\d{2})", desc)
        return f"{m.group(1)}-{m.group(2)}" if m else None

    def _direction(auction: str, desc: str) -> str:
        if auction.startswith("TRL+"):
            return "up"
        if auction.startswith("TRL-"):
            return "down"
        if "DOWN" in desc:
            return "down"
        return "up"

    for path in sorted(trl_dir.glob("*-PRL-SRL-TRL-Ergebnis.csv")):
        with open(path, encoding="latin-1", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                auction   = row[0].strip()
                desc      = row[1].strip() if len(row) > 1 else ""
                offered   = _parse_num(row[2]) if len(row) > 2 else 0.0
                awarded   = _parse_num(row[4]) if len(row) > 4 else 0.0
                cap_price = _parse_num(row[6]) if len(row) > 6 else 0.0
                costs     = _parse_num(row[8]) if len(row) > 8 else 0.0

                if not auction.startswith("TRL"):
                    continue

                direction = _direction(auction, desc)

                # Daily: TRL[+-]?_YY_MM_DD
                m = re.match(r"^TRL[+-]?_(\d{2})_(\d{2})_(\d+)$", auction)
                if m:
                    yy, mm, dd = int(m.group(1)), m.group(2), int(m.group(3))
                    date_str = f"{2000+yy}-{mm}-{dd:02d}"
                    block    = _block_label(desc)
                    if block is None:
                        continue
                    key = (date_str, block, direction)
                    if key not in daily_map:
                        daily_map[key] = _entry()
                    r = daily_map[key]
                    r["offered"] += offered
                    if awarded > 0:
                        r["awarded"]    += awarded
                        r["total_cost"] += costs
                        if cap_price > r["max_price"]:
                            r["max_price"] = cap_price
                        if cap_price > 0:
                            r["bid_prices"].append(cap_price)
                    continue

                # Weekly regular: TRL[+-]?_YY_KWnn
                m = re.match(r"^TRL[+-]?_(\d{2})_(KW\d+)$", auction)
                if m:
                    year = 2000 + int(m.group(1))
                    kw   = int(re.search(r"\d+", m.group(2)).group())
                    key  = (year, kw, direction)
                    if key not in weekly_map:
                        weekly_map[key] = _entry()
                    r = weekly_map[key]
                    r["offered"] += offered
                    if awarded > 0:
                        r["awarded"]    += awarded
                        r["total_cost"] += costs
                        if cap_price > r["max_price"]:
                            r["max_price"] = cap_price
                        if cap_price > 0:
                            r["bid_prices"].append(cap_price)
                    continue

                # Weekly S1 (anticipated): TRL[+-]?_YY_KWnn_S1
                m = re.match(r"^TRL[+-]?_(\d{2})_(KW\d+)_S1$", auction)
                if m:
                    year = 2000 + int(m.group(1))
                    kw   = int(re.search(r"\d+", m.group(2)).group())
                    key  = (year, kw, direction)
                    if key not in weekly_s1_map:
                        weekly_s1_map[key] = _entry()
                    r = weekly_s1_map[key]
                    r["offered"] += offered
                    if awarded > 0:
                        r["awarded"]    += awarded
                        r["total_cost"] += costs
                        if cap_price > r["max_price"]:
                            r["max_price"] = cap_price
                        if cap_price > 0:
                            r["bid_prices"].append(cap_price)

    return daily_map, weekly_map, weekly_s1_map


# ── TRL Daily ─────────────────────────────────────────────────────────────────

def refresh_trl_daily(daily_map: dict) -> int:
    existing = pd.read_parquet(TRL_D_PARQUET)
    last_ts  = existing["block_start"].max()
    print(f"  TRL Daily: parquet last block = {last_ts}")

    new_rows = []
    for (date_str, block, direction), r in sorted(daily_map.items()):
        block_start_local = block.split("-")[0]   # "HH:MM"
        try:
            h, mn = map(int, block_start_local.split(":"))
            y, mo, d = map(int, date_str.split("-"))
            local_dt = datetime.datetime(y, mo, d, h, mn, tzinfo=ZURICH)
            utc_ts   = pd.Timestamp(local_dt).tz_convert("UTC")
        except Exception:
            continue

        if utc_ts <= last_ts:
            continue

        offered   = r["offered"]
        awarded   = r["awarded"]
        marginal  = r["max_price"] if r["max_price"] > 0 else float("nan")
        med_bid   = _median(r["bid_prices"])
        award_pct = round(awarded / offered * 100, 1) if offered > 0 else 0.0

        new_rows.append({
            "block_start":    utc_ts,
            "direction":      direction,
            "offered_mw":     int(round(offered)),
            "awarded_mw":     int(round(awarded)),
            "marginal_chf":   round(marginal, 1) if marginal == marginal else float("nan"),
            "median_bid_chf": round(med_bid, 1)  if med_bid  == med_bid  else float("nan"),
            "award_rate_pct": award_pct,
        })

    if not new_rows:
        print("  TRL Daily: already up to date")
        return 0

    new_df   = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["block_start", "direction"]).reset_index(drop=True)
    combined.to_parquet(TRL_D_PARQUET, index=False)
    print(f"  TRL Daily: +{len(new_rows)} rows  new last = {new_df['block_start'].max()}")
    return len(new_rows)


# ── TRL Weekly ────────────────────────────────────────────────────────────────

def refresh_trl_weekly(weekly_map: dict, weekly_s1_map: dict) -> int:
    existing     = pd.read_parquet(TRL_W_PARQUET)
    last_week_ts = existing["week_start"].max()
    print(f"  TRL Weekly: parquet last week = {last_week_ts}")

    all_yw   = {(y, kw) for (y, kw, _) in weekly_map}
    new_rows = []

    for (year, kw) in sorted(all_yw):
        week_start_ts = pd.Timestamp(datetime.date.fromisocalendar(year, kw, 1))
        if week_start_ts <= last_week_ts:
            continue

        for direction in ("up", "down"):
            r = weekly_map.get((year, kw, direction))
            if r is None:
                continue

            offered   = r["offered"]
            awarded   = r["awarded"]
            marginal  = r["max_price"] if r["max_price"] > 0 else float("nan")
            med_bid   = _median(r["bid_prices"])
            vwap      = r["total_cost"] / awarded if awarded > 0 else float("nan")
            award_pct = round(awarded / offered * 100, 1) if offered > 0 else 0.0

            s1 = weekly_s1_map.get((year, kw, direction))
            s1_active   = 1 if s1 and s1["awarded"] > 0 else 0
            s1_awarded  = s1["awarded"]   if s1 else float("nan")
            s1_marginal = s1["max_price"] if (s1 and s1["max_price"] > 0) else float("nan")
            s1_vwap     = (s1["total_cost"] / s1["awarded"]
                           if (s1 and s1["awarded"] > 0) else float("nan"))

            new_rows.append({
                "week_start":      week_start_ts,
                "direction":       direction,
                "offered_mw":      int(round(offered)),
                "awarded_mw":      int(round(awarded)),
                "marginal_chf":    round(marginal, 1) if marginal == marginal else float("nan"),
                "median_bid_chf":  round(med_bid, 1)  if med_bid  == med_bid  else float("nan"),
                "vwap_chf":        round(vwap, 1)     if vwap     == vwap     else float("nan"),
                "award_rate_pct":  award_pct,
                "s1_is_active":    s1_active,
                "s1_awarded_mw":   s1_awarded,
                "s1_marginal_chf": s1_marginal if s1_marginal == s1_marginal else float("nan"),
                "s1_vwap_chf":     s1_vwap     if s1_vwap     == s1_vwap     else float("nan"),
            })

    if not new_rows:
        print("  TRL Weekly: already up to date")
        return 0

    new_df   = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["week_start", "direction"]).reset_index(drop=True)
    combined.to_parquet(TRL_W_PARQUET, index=False)
    print(f"  TRL Weekly: +{len(new_rows)} rows  new last = {new_df['week_start'].max()}")
    return len(new_rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Determine how far back to look based on least-current parquet
    tre_last = pd.read_parquet(TRE_PARQUET)["slot_time"].max()
    parquet_last_date = tre_last.date() - datetime.timedelta(days=1)

    print("Refreshing price parquets from Swissgrid...")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        print("[Download]")
        try:
            download_current_csvs(tmp_dir, parquet_last_date)
        except Exception as e:
            print(f"  ERROR during download: {e}")
            print("  Skipping price refresh — inference will use existing parquets.")
            return
        print()

        print("[TRE]")
        n_tre = refresh_tre(tmp_dir / "TRE")
        print()

        print("[TRL — parsing CSVs]")
        daily_map, weekly_map, weekly_s1_map = _parse_trl_csvs(tmp_dir / "SRL&TRL")
        print()

        print("[TRL Daily]")
        n_trl_d = refresh_trl_daily(daily_map)
        print()

        print("[TRL Weekly]")
        n_trl_w = refresh_trl_weekly(weekly_map, weekly_s1_map)
        print()

    total = n_tre + n_trl_d + n_trl_w
    if total == 0:
        print("All parquets are already up to date.")
    else:
        print(f"Done: TRE +{n_tre}, TRL Daily +{n_trl_d}, TRL Weekly +{n_trl_w}")


if __name__ == "__main__":
    main()
