"""
NWP / MODEL DATA INGESTION
============================
Handles CAPE, CIN, wind shear, precipitable water from ERA5 (open, no
approval needed) or NCMRWF (needs institutional access).

WHAT YOU NEED TO DO:
1. Go to https://cds.climate.copernicus.eu and create a free account.
2. Get your API key from your CDS profile page.
3. Create a file named ".env" in the project root (NOT committed/shared)
   with the line:
       CDS_API_KEY=your-actual-key-here
4. pip install cdsapi --break-system-packages
5. Run this script's `download_era5_sample()` function once to pull a
   small test file for your region/date range.
"""

import os
from config import NWP_RAW_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, CDS_API_KEY

try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    CDSAPI_AVAILABLE = False


def download_era5_sample(date="2023-07-15", hour="12:00"):
    """
    Downloads a single ERA5 pressure-level file (CAPE, wind components)
    for your region on a given date/time -- good for a first pipeline test.
    >>> YOU INSERT HERE: change `date` and `hour` to a real thunderstorm day
        in your study period once you pick one from your lightning records.
    """
    if not CDSAPI_AVAILABLE:
        print("Run: pip install cdsapi --break-system-packages")
        return None
    if not CDS_API_KEY:
        print("[!] CDS_API_KEY not set. See docstring above for setup steps.")
        return None

    client = cdsapi.Client(key=CDS_API_KEY, url="https://cds.climate.copernicus.eu/api")
    target = os.path.join(NWP_RAW_DIR, f"era5_{date}.nc")

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
    print(f"Saved: {target}")
    return target


def generate_synthetic_nwp_features():
    """TEMPORARY placeholder scalar features for pipeline testing."""
    return {
        "CAPE_J_per_kg": 1800.0,   # typical strong pre-monsoon instability value
        "wind_shear_ms": 12.0,
        "precipitable_water_mm": 55.0,
    }


def load_real_nwp_frame(filepath, shape=None):
    """
    Loads a real downloaded ERA5 file (as produced by download_era5_sample())
    and returns a dict of 2D arrays -- one per variable -- cropped to the
    region already set in config.py, for the FIRST time step in the file.

    Real variable names confirmed from an actual downloaded file:
        cape, u10, v10, tcwv  -- all shaped (valid_time, latitude, longitude)

    Returns a dict: {"CAPE": 2D array, "wind_shear": 2D array, "PWAT": 2D array}
    """
    import numpy as np
    if not XARRAY_AVAILABLE_CHECK():
        raise ImportError("Run: pip install xarray netCDF4 --break-system-packages")

    import xarray as xr
    ds = xr.open_dataset(filepath)

    # Take the first time step in the file (adjust index if you download
    # multiple hours/dates in one file later)
    cape = ds["cape"].isel(valid_time=0).values
    u10 = ds["u10"].isel(valid_time=0).values
    v10 = ds["v10"].isel(valid_time=0).values
    tcwv = ds["tcwv"].isel(valid_time=0).values

    # wind shear approximated here as simple wind speed magnitude at 10m
    # (a true shear calculation needs two pressure levels -- fine for now)
    wind_speed = np.sqrt(u10 ** 2 + v10 ** 2)

    return {
        "CAPE": cape,
        "wind_shear": wind_speed,
        "PWAT": tcwv,
    }


def XARRAY_AVAILABLE_CHECK():
    try:
        import xarray  # noqa
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    if CDS_API_KEY:
        download_era5_sample()
    else:
        print("No CDS_API_KEY found -- printing synthetic NWP features instead.")
        print(generate_synthetic_nwp_features())
