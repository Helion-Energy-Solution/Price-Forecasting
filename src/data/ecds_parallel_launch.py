"""
Parallel ECDS bulk downloader.

Each worker handles a non-overlapping date range and writes to its own
chunk parquet — no race conditions. Use --merge to combine into the main
weather_ensemble.parquet once all workers are done (or periodically).

Usage
-----
  python src/data/ecds_parallel_launch.py --workers 6          # launch 6 workers
  python src/data/ecds_parallel_launch.py --status             # progress per chunk
  python src/data/ecds_parallel_launch.py --merge              # combine chunks
"""

import argparse
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

ROOT        = Path(__file__).resolve().parents[2]
ECDS_SCRIPT = ROOT / "src" / "data" / "ecds_download.py"
MAIN_PARQUET = ROOT / "data" / "processed" / "features" / "weather_ensemble.parquet"
CHUNKS_DIR   = ROOT / "data" / "processed" / "features" / "chunks"
LOG_DIR      = ROOT / "logs"

START_DATE = date(2022, 1, 1)
with open(ROOT / "config" / "config.yaml") as _f:
    END_DATE = date.fromisoformat(yaml.safe_load(_f)["training"]["end_date"])

# Isolate workers from Ctrl+C signals in this terminal
_NEW_PROC_GROUP = 0x00000200


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _all_dates():
    d = START_DATE
    while d <= END_DATE:
        yield d
        d += timedelta(days=1)


def _completed_dates() -> set:
    """Dates already present in main parquet or any chunk parquet."""
    paths = [MAIN_PARQUET] + sorted(CHUNKS_DIR.glob("weather_ensemble_chunk_*.parquet"))
    done = set()
    for p in paths:
        if p.exists():
            df = pd.read_parquet(p, columns=["init_time"])
            for t in df["init_time"].unique():
                done.add(pd.Timestamp(t).date())
    return done


def _remaining_dates() -> list:
    done = _completed_dates()
    return [d for d in _all_dates() if d not in done]


def _split(dates: list, n: int) -> list:
    """Split into n roughly equal non-overlapping chunks."""
    k, m = divmod(len(dates), n)
    chunks = []
    i = 0
    for c in range(n):
        size = k + (1 if c < m else 0)
        chunks.append(dates[i:i + size])
        i += size
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def cmd_launch(n_workers: int):
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    remaining = _remaining_dates()
    total = len(list(_all_dates()))
    done  = total - len(remaining)
    print(f"Total: {total}  |  Done: {done}  |  Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing left to download.")
        return

    chunks = _split(remaining, n_workers)
    print(f"Splitting {len(remaining)} dates across {len(chunks)} workers\n")

    for i, chunk in enumerate(chunks):
        out_path = CHUNKS_DIR / f"weather_ensemble_chunk_{i:02d}.parquet"
        log_path = LOG_DIR / f"ecds_chunk_{i:02d}.log"

        cmd = [
            sys.executable, str(ECDS_SCRIPT),
            "--source", "ecds",
            "--start",  chunk[0].strftime("%Y-%m-%d"),
            "--end",    chunk[-1].strftime("%Y-%m-%d"),
            "--run",    "00",
            "--output", str(out_path),
        ]

        # Pass flag in CreateProcess env so the python.exe launcher stub propagates
        # it to the real interpreter it spawns, which then kills any further children.
        worker_env = os.environ.copy()
        worker_env["ECDS_WORKER_ACTIVE"] = "worker"

        with open(log_path, "a") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=log_f,
                cwd=str(ROOT),
                env=worker_env,
                creationflags=_NEW_PROC_GROUP,
            )
        print(
            f"  Worker {i}: PID {proc.pid} | "
            f"{chunk[0]} to {chunk[-1]} ({len(chunk)} dates) | "
            f"log: logs/ecds_chunk_{i:02d}.log"
        )

    print(f"\nAll {len(chunks)} workers launched. Monitor with: python src/data/ecds_parallel_launch.py --status")


def cmd_status():
    done_total = _completed_dates()
    total = len(list(_all_dates()))
    print(f"Overall: {len(done_total)} / {total} dates done ({100*len(done_total)/total:.1f}%)\n")

    # Main parquet
    if MAIN_PARQUET.exists():
        df = pd.read_parquet(MAIN_PARQUET, columns=["init_time"])
        n = len(df["init_time"].unique())
        print(f"  main parquet : {n} dates")

    # Chunk parquets
    for p in sorted(CHUNKS_DIR.glob("weather_ensemble_chunk_*.parquet")):
        df = pd.read_parquet(p, columns=["init_time"])
        dates = sorted(set(pd.Timestamp(t).date() for t in df["init_time"].unique()))
        if dates:
            print(f"  {p.name}: {len(dates)} dates  ({dates[0]} to {dates[-1]})")

    # Check log tails for in-progress activity
    print()
    for log_path in sorted(LOG_DIR.glob("ecds_chunk_*.log")):
        lines = log_path.read_text(errors="replace").splitlines()
        last = next((l for l in reversed(lines) if l.strip()), "")
        print(f"  {log_path.name}: {last[-120:]}")


def cmd_merge():
    """Combine all chunk parquets + main parquet into main parquet."""
    paths = [MAIN_PARQUET] + sorted(CHUNKS_DIR.glob("weather_ensemble_chunk_*.parquet"))
    dfs = [pd.read_parquet(p) for p in paths if p.exists()]
    if not dfs:
        print("No parquet files found.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined.drop_duplicates(subset=["init_time", "valid_time", "variable"], keep="last", inplace=True)
    combined.sort_values(["init_time", "valid_time", "variable"], inplace=True)
    MAIN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(MAIN_PARQUET, index=False)

    init_dates = combined["init_time"].nunique()
    print(f"Merged {len(dfs)} files -> {len(combined)} rows, {init_dates} unique init_times -> {MAIN_PARQUET.name}")

    # Remove chunk files after successful merge
    for p in CHUNKS_DIR.glob("weather_ensemble_chunk_*.parquet"):
        p.unlink()
    print("Chunk parquets removed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--workers", type=int, metavar="N", help="Launch N parallel workers")
    group.add_argument("--status",  action="store_true",   help="Show per-chunk progress")
    group.add_argument("--merge",   action="store_true",   help="Merge chunks into main parquet")
    args = parser.parse_args()

    if args.workers:
        cmd_launch(args.workers)
    elif args.status:
        cmd_status()
    elif args.merge:
        cmd_merge()


if __name__ == "__main__":
    main()
