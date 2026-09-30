"""
SATELLITE DATA INGESTION
=========================
Handles INSAT-3D/3DR (via MOSDAC) or, in OPEN mode, NASA GPM IMERG rainfall
+ any open geostationary brightness-temperature product.

WHAT YOU NEED TO DO:
1. Register at https://mosdac.gov.in (free) if using IMD mode.
2. For OPEN mode, register at https://urs.earthdata.nasa.gov (free, instant)
   to get IMERG access.
3. Download sample files manually first (a handful of .h5/.nc4 files) and
   place them in: data/raw/satellite/
4. Run this script to load + standardize them into xarray format.

You do NOT need to write scraping code for the portals -- both MOSDAC and
NASA Earthdata require you to manually download from their web UI or use
their official order/API tool with your own credentials. Automating login
programmatically is against most portals' terms of use, so manual/batch
download via their official tools is the correct approach here.
"""

import os
import glob
import numpy as np

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

from config import SATELLITE_RAW_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX


def list_available_files(pattern="*.nc*"):
    """List whatever satellite files you've placed in the raw folder."""
    files = sorted(glob.glob(os.path.join(SATELLITE_RAW_DIR, pattern)))
    if not files:
        print(f"[!] No files found in {SATELLITE_RAW_DIR}")
        print("    >>> YOU INSERT HERE: download sample .nc/.h5 files and place them there.")
    return files


def load_insat_brightness_temp(filepath):
    """
    Load INSAT-3D/3DR L1B brightness temperature file.
    >>> YOU INSERT HERE: adjust variable name below once you have a real file
        (open it once with `xr.open_dataset(filepath)` and print `.data_vars`
        to see the exact variable name -- it varies by product version).
    """
    if not XARRAY_AVAILABLE:
        raise ImportError("Run: pip install xarray netCDF4 --break-system-packages")
    ds = xr.open_dataset(filepath)
    # Placeholder variable name -- REPLACE after inspecting a real file:
    var_name = "IMG_TIR1"  # >>> YOU INSERT HERE: real variable name from your file
    if var_name not in ds.data_vars:
        print(f"[!] Variable '{var_name}' not found. Available vars: {list(ds.data_vars)}")
        return None
    return ds[var_name]


def load_imerg_rainfall(filepath):
    """Load a GPM IMERG half-hourly rainfall file (OPEN-mode fallback)."""
    if not XARRAY_AVAILABLE:
        raise ImportError("Run: pip install xarray netCDF4 --break-system-packages")
    ds = xr.open_dataset(filepath, group="Grid") if filepath.endswith(".HDF5") else xr.open_dataset(filepath)
    return ds


def crop_to_region(data_array):
    """Crop any lat/lon-indexed DataArray to our study region (see config.py)."""
    return data_array.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX)
    )


def generate_synthetic_satellite_frame(shape=(64, 64)):
    """
    TEMPORARY placeholder so the rest of the pipeline can run before you
    have real files. Produces a fake brightness-temperature field with a
    cold (convective) blob -- use this to test the pipeline end-to-end.
    Delete/ignore once real data is plugged in.
    """
    x, y = np.meshgrid(np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0]))
    blob = np.exp(-((x - 0.2) ** 2 + (y + 0.1) ** 2) / 0.05)
    brightness_temp = 280 - 80 * blob  # cold cloud top ~200K in the storm core
    return brightness_temp


def load_real_imerg_frame(filepath):
    """
    Loads a real downloaded IMERG half-hourly HDF5 file (as saved from
    GES DISC) and returns the precipitation field as a 2D numpy array,
    cropped to the region set in config.py.

    Real variable name confirmed from an actual downloaded file:
        precipitation -- shaped (time, lon, lat), inside the 'Grid' group.

    NOTE: IMERG stores dimensions as (time, lon, lat) -- note lon BEFORE
    lat, which is the opposite of most other products -- so we transpose
    it to the more usual (lat, lon) order before returning.
    """
    if not XARRAY_AVAILABLE:
        raise ImportError("Run: pip install xarray netCDF4 h5netcdf --break-system-packages")

    ds = xr.open_dataset(filepath, group="Grid")
    precip = ds["precipitation"].isel(time=0)  # first (only) time step in the file
    precip = crop_to_region(precip.rename({"lat": "lat", "lon": "lon"}))
    precip_2d = precip.transpose("lat", "lon").values  # -> (lat, lon) order
    return precip_2d


if __name__ == "__main__":
    files = list_available_files()
    if not files:
        print("Using synthetic placeholder frame instead (for pipeline testing).")
        frame = generate_synthetic_satellite_frame()
        print("Synthetic frame shape:", frame.shape, "| min/max (K):", frame.min(), frame.max())
    else:
        print(f"Found {len(files)} file(s). Loading first one as a test...")
        result = load_insat_brightness_temp(files[0])
        print(result)
