"""
Connectivity check for both ECMWF data sources.

Run:
    python src/data/check_cds.py
"""

import logging
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def check_opendata():
    """Test ECMWF Open Data (no auth, real-time)."""
    log.info("=== ECMWF Open Data (real-time, no auth) ===")
    try:
        from ecmwf.opendata import Client
    except ImportError:
        log.error("ecmwf-opendata not installed. Run: pip install ecmwf-opendata")
        return False

    # Use yesterday so the run is definitely published
    test_date = date.today() - timedelta(days=1)

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        client = Client(source="ecmwf")
        client.retrieve(
            date=int(test_date.strftime("%Y%m%d")),
            time=0,
            step=[24],
            stream="enfo",
            type="pf",
            param=["2t"],          # 2m temperature only
            levtype="sfc",
            number=[1],            # single member
            area=[48, 5, 45, 11],  # Swiss bounding box
            target=str(tmp_path),
        )
        size_kb = tmp_path.stat().st_size / 1024
        log.info("SUCCESS — Open Data: %.1f KB received for %s 00z", size_kb, test_date)
        tmp_path.unlink()
        return True
    except Exception as e:
        log.error("Open Data request failed: %s", e)
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def check_ecds():
    """Test ECMWF Data Store (historical, requires ~/.cdsapirc with ECDS credentials)."""
    log.info("=== ECMWF Data Store / ECDS (historical, auth required) ===")
    try:
        import cdsapi
    except ImportError:
        log.error("cdsapi not installed. Run: pip install 'cdsapi>=0.7.0'")
        return False

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        client = cdsapi.Client()
        client.retrieve(
            "tigge",
            {
                "class": "ti",
                "dataset": "tigge",
                "date": "2023-06-01",
                "expver": "prod",
                "grid": "0.5/0.5",
                "levtype": "sfc",
                "number": "1",
                "origin": "ecmf",
                "param": "167",
                "step": "24",
                "time": "00:00",
                "type": "pf",
                "area": [48, 5, 45, 11],
                "format": "grib2",
            },
            str(tmp_path),
        )
        size_kb = tmp_path.stat().st_size / 1024
        log.info("SUCCESS — ECDS: %.1f KB received", size_kb)
        tmp_path.unlink()
        return True
    except Exception as e:
        log.error("ECDS request failed: %s", e)
        log.info("To fix: register at https://cds.ecmwf.int and update ~/.cdsapirc with ECDS credentials.")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


if __name__ == "__main__":
    ok_opendata = check_opendata()
    ok_ecds = check_ecds()

    print()
    print("Summary:")
    print(f"  Open Data (real-time, inference): {'OK' if ok_opendata else 'FAILED'}")
    print(f"  ECDS (historical, training):      {'OK' if ok_ecds else 'FAILED — see above'}")

    if not ok_opendata:
        sys.exit(1)
