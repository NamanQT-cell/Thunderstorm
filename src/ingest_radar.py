"""
RADAR DATA INGESTION
======================
Handles IMD Doppler Weather Radar volume scans (NetCDF/HDF5).

WHAT YOU NEED TO DO:
1. Once IMD access is approved, download sample radar volume files
   (usually .nc or vendor-specific binary) and place them in:
   data/raw/radar/
2. Install `wradlib` (pip install wradlib) -- purpose-built for exactly
   this kind of radar reprojection (polar -> Cartesian grid).
3. If IMD access does NOT arrive in time, skip this module entirely and
   rely on satellite + NWP features only (see README's fallback section).
   The rest of the pipeline (feature_engineering.py, model.py) is written
   to work fine with radar features left out.
"""

import os
import glob
import numpy as np

from config import RADAR_RAW_DIR, DATA_SOURCE_MODE

try:
    import wradlib as wrl
    WRADLIB_AVAILABLE = True
except ImportError:
    WRADLIB_AVAILABLE = False


def list_available_files(pattern="*.nc*"):
    files = sorted(glob.glob(os.path.join(RADAR_RAW_DIR, pattern)))
    if not files:
        print(f"[!] No radar files found in {RADAR_RAW_DIR}")
        print("    >>> YOU INSERT HERE: place IMD radar volume files here once available.")
    return files


def load_and_reproject_radar(filepath):
    """
    Load a radar volume scan and reproject from polar (range, azimuth) to
    a Cartesian lat/lon grid.
    >>> YOU INSERT HERE: exact loading call depends on IMD's file format --
        confirm the format (ODIM_H5 is IMD's typical format) and use:
            raw = wrl.io.read_opera_hdf5(filepath)
        then follow wradlib's georeferencing example to get lat/lon/reflectivity.
    """
    if not WRADLIB_AVAILABLE:
        raise ImportError("Run: pip install wradlib --break-system-packages")

    raw = wrl.io.read_opera_hdf5(filepath)  # >>> YOU INSERT HERE: confirm this matches your file format
    return raw


def generate_synthetic_radar_frame(shape=(64, 64)):
    """
    TEMPORARY placeholder reflectivity field (dBZ) for pipeline testing
    before real radar access is available.
    """
    x, y = np.meshgrid(np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0]))
    storm_core = np.exp(-((x - 0.2) ** 2 + (y + 0.1) ** 2) / 0.03)
    reflectivity_dbz = 55 * storm_core  # up to ~55 dBZ in storm core
    return reflectivity_dbz


if __name__ == "__main__":
    if DATA_SOURCE_MODE == "OPEN":
        print("DATA_SOURCE_MODE = OPEN -> radar not required. Using synthetic frame for testing.")
        frame = generate_synthetic_radar_frame()
        print("Synthetic radar frame shape:", frame.shape, "| max dBZ:", frame.max())
    else:
        files = list_available_files()
        if files:
            data = load_and_reproject_radar(files[0])
            print(data)
        else:
            print("No radar files yet -- using synthetic frame.")
            frame = generate_synthetic_radar_frame()
