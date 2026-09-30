"""
RADAR DOWNLOAD DIAGNOSTIC
============================
Tests GPM DPR (real satellite radar) download for ONE date only, with a
hard timeout on each step, so we can see exactly where it's getting stuck
instead of it hanging forever with no output.

Run: python test_radar_download.py
"""

import concurrent.futures
import time
from config import RADAR_RAW_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, \
    EARTHDATA_USERNAME, EARTHDATA_PASSWORD

try:
    import earthaccess
    EARTHACCESS_AVAILABLE = True
except ImportError:
    EARTHACCESS_AVAILABLE = False


def run_with_timeout(func, timeout_seconds, *args, **kwargs):
    """Runs func in a separate thread and aborts if it takes too long."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            print(f"  [TIMEOUT] This step took longer than {timeout_seconds}s -- aborting it.")
            return "TIMEOUT"


def main():
    if not EARTHACCESS_AVAILABLE:
        print("earthaccess not installed. Run: pip install earthaccess --break-system-packages")
        return

    if not (EARTHDATA_USERNAME and EARTHDATA_PASSWORD):
        print("No Earthdata credentials found in .env -- add EARTHDATA_USERNAME and EARTHDATA_PASSWORD.")
        return

    test_date = "2025-07-05"  # a monsoon date already in your DATES list
    test_hour = "12"

    print(f"Testing GPM DPR download for {test_date} {test_hour}:00, region "
          f"lat[{LAT_MIN},{LAT_MAX}] lon[{LON_MIN},{LON_MAX}]\n")

    print("Step 1/3: Logging into NASA Earthdata...")
    t0 = time.time()
    login_result = run_with_timeout(earthaccess.login, 20, strategy="environment")
    print(f"  -> took {time.time()-t0:.1f}s, result: {login_result}\n")
    if login_result == "TIMEOUT":
        print("Login itself is hanging -- this points to a network/credentials issue, not the search.")
        return

    print("Step 2/3: Searching for a GPM DPR granule (whole day, since a single")
    print("hour is too narrow for a small region)...")
    t0 = time.time()

    def do_search():
        return earthaccess.search_data(
            short_name="GPM_2ADPR",
            version="07",
            temporal=(f"{test_date} 00:00:00", f"{test_date} 23:59:59"),
            bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
            count=1,
        )

    results = run_with_timeout(do_search, 30)
    print(f"  -> took {time.time()-t0:.1f}s")

    if results == "TIMEOUT":
        print("The SEARCH itself is what's hanging. This is likely a CMR/earthaccess API")
        print("slowness issue, not something wrong with your code or credentials.")
        return

    if not results:
        print("  -> No granule found for this exact hour (normal -- satellite coverage is sparse")
        print("     for a small region). This is NOT a hang, just no match. Try a different")
        print("     hour or date, or accept that radar coverage will be spotty for this region.")
        return

    print(f"  -> Found {len(results)} granule(s)\n")

    print("Step 3/3: Downloading the first granule...")
    t0 = time.time()
    downloaded = run_with_timeout(earthaccess.download, 60, results[:1], RADAR_RAW_DIR)
    print(f"  -> took {time.time()-t0:.1f}s")

    if downloaded == "TIMEOUT":
        print("The DOWNLOAD step is what's hanging -- likely a slow/unstable connection to NASA's servers.")
        return

    print(f"\nSUCCESS: {downloaded}")


def scan_date_range(start_test_date="2025-07-01", num_days=15):
    """
    Checks a range of consecutive days to see what fraction actually have
    GPM DPR coverage for this region -- one missed date is expected and
    normal, but this tells us the REAL hit rate so you can decide if
    pursuing GPM DPR radar is worth it at all for this small a region.
    """
    from datetime import datetime, timedelta

    if not EARTHACCESS_AVAILABLE:
        print("earthaccess not installed. Run: pip install earthaccess --break-system-packages")
        return
    if not (EARTHDATA_USERNAME and EARTHDATA_PASSWORD):
        print("No Earthdata credentials found in .env")
        return

    print("Logging in once for this whole scan...")
    run_with_timeout(earthaccess.login, 20, strategy="environment")

    start = datetime.strptime(start_test_date, "%Y-%m-%d")
    hits = 0

    print(f"\nScanning {num_days} consecutive days starting {start_test_date} "
          f"for region lat[{LAT_MIN},{LAT_MAX}] lon[{LON_MIN},{LON_MAX}]...\n")

    for i in range(num_days):
        d = (start + timedelta(days=i)).strftime("%Y-%m-%d")

        def do_search():
            return earthaccess.search_data(
                short_name="GPM_2ADPR",
                version="07",
                temporal=(f"{d} 00:00:00", f"{d} 23:59:59"),
                bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
                count=1,
            )

        result = run_with_timeout(do_search, 20)
        if result == "TIMEOUT":
            print(f"  {d}: [timeout]")
            continue
        found = bool(result)
        if found:
            hits += 1
        print(f"  {d}: {'FOUND' if found else 'no coverage'}")

    print(f"\n--- Result: {hits}/{num_days} days had GPM DPR coverage "
          f"({hits/num_days*100:.0f}%) ---")
    if hits == 0:
        print("Zero hits across the whole range strongly suggests this small region")
        print("gets essentially no GPM DPR passes -- not worth pursuing further.")
    elif hits / num_days < 0.15:
        print("Very low hit rate -- radar coverage would be too sparse to rely on")
        print("for a meaningful number of your training dates.")
    else:
        print("Reasonable hit rate -- worth enabling enable_radar=True in batch_download.py.")


if __name__ == "__main__":
    main()
    print("\n\n" + "="*60)
    print("Now scanning a wider date range to check the REAL hit rate...")
    print("="*60 + "\n")
    scan_date_range()