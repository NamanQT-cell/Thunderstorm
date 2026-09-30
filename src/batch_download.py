"""
BATCH DOWNLOADER — multiple dates in one run
===============================================
Downloads ERA5 (NWP) and IMERG (satellite/rainfall) for a LIST of dates
instead of one file at a time. This is what actually builds you a real
training dataset instead of a single test snapshot.

WHAT YOU NEED TO DO:
1. Make sure CDS_API_KEY is already in your .env (you have this working).
2. Add your NASA Earthdata login to the same .env file:
       EARTHDATA_USERNAME=your-earthdata-username
       EARTHDATA_PASSWORD=your-earthdata-password
   (This is the SAME account you registered with earlier at
   urs.earthdata.nasa.gov -- no new signup needed.)
3. pip install earthaccess --break-system-packages
   (This is NASA's own official library for exactly this kind of
   automated, authenticated bulk download -- much easier than manually
   clicking through the GES DISC website for every date.)
4. Edit the DATES list near the bottom of this file to whichever days
   you want (mix some known storm days with some calm days for good
   training balance, if you know your region's storm history).
5. Run: python batch_download.py

WHY BOTH STORM AND NON-STORM DAYS MATTER:
Your model needs to learn what "no thunderstorm" looks like too, not
just storm days -- otherwise it can't distinguish signal from normal
weather. Aim for roughly half storm days, half calm days if possible.
"""

import os
import time
from config import NWP_RAW_DIR, SATELLITE_RAW_DIR, RADAR_RAW_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, \
    CDS_API_KEY, EARTHDATA_USERNAME, EARTHDATA_PASSWORD

try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    CDSAPI_AVAILABLE = False

try:
    import earthaccess
    EARTHACCESS_AVAILABLE = True
except ImportError:
    EARTHACCESS_AVAILABLE = False


def call_with_timeout(func, timeout_seconds, *args, **kwargs):
    """
    Runs func in a DAEMON thread and gives up after timeout_seconds.
    Returns (finished, result_or_exception).

    Why not ThreadPoolExecutor? Its `with` block waits for the worker to
    finish on exit, so a hung call still blocks the script even after the
    "timeout" fires. A daemon thread can simply be abandoned: the script
    moves on, and the stuck thread is killed when Python exits.
    """
    import threading
    box = {}

    def runner():
        try:
            box["result"] = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout_seconds)
    if t.is_alive():
        return False, None
    if "error" in box:
        return True, box["error"]
    return True, box["result"]


def download_era5_for_date(date, hour="12:00"):
    """Download one ERA5 file for a given date (reuses the same logic as ingest_nwp.py)."""
    target = os.path.join(NWP_RAW_DIR, f"era5_{date}.nc")
    if os.path.exists(target):
        print(f"  [skip] ERA5 {date} already downloaded.")
        return target

    client = cdsapi.Client(key=CDS_API_KEY, url="https://cds.climate.copernicus.eu/api")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "convective_available_potential_energy",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "total_column_water_vapour",
            ],
            "year": date.split("-")[0],
            "month": date.split("-")[1],
            "day": date.split("-")[2],
            "time": hour,
            "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],  # N, W, S, E
            "format": "netcdf",
        },
        target,
    )
    print(f"  [done] Saved ERA5: {target}")
    return target


def download_imerg_for_date(date, hour="12"):
    """
    Download one IMERG half-hourly file for a given date/hour using
    earthaccess (handles NASA Earthdata login automatically).
    """
    if not EARTHACCESS_AVAILABLE:
        print("  [!] earthaccess not installed. Run: pip install earthaccess --break-system-packages")
        return None

    start_time = f"{date} {hour}:00:00"
    end_time = f"{date} {hour}:59:59"

    def do_search():
        return earthaccess.search_data(
            short_name="GPM_3IMERGHH",
            version="07",
            temporal=(start_time, end_time),
            bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
        )

    finished, results = call_with_timeout(do_search, 45)
    if not finished:
        print(f"  [!] IMERG search timed out for {date} -- skipping")
        return None
    if isinstance(results, Exception):
        print(f"  [!] IMERG search failed for {date}: {results}")
        return None
    if not results:
        print(f"  [!] No IMERG granule found for {date} {hour}:00")
        return None

    finished, downloaded = call_with_timeout(earthaccess.download, 180, results[:1], SATELLITE_RAW_DIR)
    if not finished:
        print(f"  [!] IMERG download timed out for {date} (>180s) -- skipping")
        return None
    if isinstance(downloaded, Exception):
        print(f"  [!] IMERG download failed for {date}: {downloaded}")
        return None
    print(f"  [done] Saved IMERG: {downloaded}")
    return downloaded


def download_gpm_dpr_for_date(date, hour="12"):
    """
    Downloads a real GPM DPR (Dual-frequency Precipitation Radar) granule --
    this IS genuine spaceborne RADAR data (Ku/Ka-band precipitation radar
    aboard the GPM Core satellite), not a ground radar like IMD's network,
    but a real, physically-measured radar reflectivity product. It's freely
    accessible via the same NASA Earthdata account already used for IMERG --
    no separate approval needed, unlike IMD's ground radar network.

    This is what lets you honestly claim a real "radar" data source in your
    problem statement, without waiting on IMD.

    Searches the WHOLE DAY (not just one hour) since the satellite only
    passes over any given small region a couple of times a day -- a single
    hour is too narrow and misses almost everything. Uses a hard timeout
    (via concurrent.futures) on the search itself so a slow/unresponsive
    API call can never hang the whole batch download again.
    """
    if not EARTHACCESS_AVAILABLE:
        print("  [!] earthaccess not installed. Run: pip install earthaccess --break-system-packages")
        return None

    def do_search():
        return earthaccess.search_data(
            short_name="GPM_2ADPR",
            version="07",
            temporal=(f"{date} 00:00:00", f"{date} 23:59:59"),
            bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
            count=1,
        )

    finished, results = call_with_timeout(do_search, 30)
    if not finished:
        print(f"  [!] GPM DPR search timed out for {date} (>30s) -- skipping radar for this date")
        return None
    if isinstance(results, Exception):
        print(f"  [!] GPM DPR search failed for {date}: {results}")
        return None
    if not results:
        print(f"  [!] No GPM DPR granule found for {date} (no satellite pass over this region that day) -- skipping radar")
        return None

    # radar files are large -- tell the user what is happening instead of going silent
    print(f"  [..] Downloading GPM DPR radar file for {date} (large file, up to 5 min)...")
    finished, downloaded = call_with_timeout(earthaccess.download, 300, results[:1], RADAR_RAW_DIR)
    if not finished:
        print(f"  [!] GPM DPR download timed out for {date} (>300s) -- skipping and moving on")
        return None
    if isinstance(downloaded, Exception):
        print(f"  [!] GPM DPR download failed for {date}: {downloaded}")
        return None
    print(f"  [done] Saved GPM DPR (real radar): {downloaded}")
    return downloaded


def run_batch_download(dates, hour="12", enable_radar=False):
    """
    Loop over a list of 'YYYY-MM-DD' dates and download ERA5 + IMERG for
    each (and GPM DPR radar too, only if enable_radar=True -- off by
    default since it's slower/less reliable and was found to occasionally
    hang; run it as a separate, smaller batch later if you want to try it).
    """
    print(f"Starting batch download for {len(dates)} date(s)...\n")

    if EARTHACCESS_AVAILABLE and (EARTHDATA_USERNAME and EARTHDATA_PASSWORD):
        print("Logging into NASA Earthdata once for this whole run...")
        earthaccess.login(strategy="environment")
        print("Logged in.\n")

    for date in dates:
        print(f"=== {date} ===")

        if CDS_API_KEY:
            try:
                download_era5_for_date(date, hour=f"{hour}:00")
            except Exception as e:
                print(f"  [error] ERA5 download failed for {date}: {e}")
        else:
            print("  [!] No CDS_API_KEY set -- skipping ERA5.")

        if EARTHDATA_USERNAME and EARTHDATA_PASSWORD:
            try:
                download_imerg_for_date(date, hour=hour)
            except Exception as e:
                print(f"  [error] IMERG download failed for {date}: {e}")

            if enable_radar:
                try:
                    download_gpm_dpr_for_date(date, hour=hour)
                except Exception as e:
                    print(f"  [error] GPM DPR (radar) download failed for {date}: {e}")
        else:
            print("  [!] No Earthdata credentials set -- skipping IMERG and radar.")

        time.sleep(2)  # be polite to the APIs between requests
        print()

    print("Batch download complete. Check data/raw/nwp/, data/raw/satellite/, and data/raw/radar/.")


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # >>> YOU INSERT HERE: your list of dates. Mix storm days (check IMD's
    #     past severe weather bulletins or news archives for your region)
    #     with ordinary calm days for balanced training data.
    # ---------------------------------------------------------------------
    DATES = [
        # --- 2023 monsoon/pre-monsoon (storm-likely) ---
        "2023-07-15",  # already have this one
        "2023-06-05",
        "2023-06-18",
        "2023-07-02",
        "2023-07-22",
        "2023-08-08",
        "2023-08-19",
        "2023-09-03",
        "2023-09-14",
        # --- 2023 off-season (calm days, for balance) ---
        "2023-01-10",
        "2023-11-25",
        "2023-12-15",

        # --- 2024 monsoon/pre-monsoon (storm-likely) ---
        "2024-05-30",
        "2024-06-12",
        "2024-06-25",
        "2024-07-08",
        "2024-07-19",
        "2024-08-01",
        "2024-08-14",
        "2024-08-27",
        "2024-09-09",
        "2024-09-20",
        # --- 2024 off-season (calm days) ---
        "2024-01-20",
        "2024-02-14",
        "2024-11-10",
        "2024-12-05",

        # --- 2025 monsoon/pre-monsoon (storm-likely) ---
        "2025-05-21",  # already have this one
        "2025-05-24",  # already have this one
        "2025-06-29",  # already have this one
        "2025-06-15",
        "2025-07-05",
        "2025-07-18",
        "2025-08-12",  # already have this one
        "2025-08-25",
        "2025-09-02",  # already have this one
        "2025-09-05",  # already have this one
        # --- 2025 off-season (calm days) ---
        "2025-02-20",  # already have this one
        "2025-02-22",  # already have this one
        "2025-01-15",
    ]

    run_batch_download(DATES, hour="12", enable_radar=True)  # radar enabled -- confirmed ~40% coverage for this region