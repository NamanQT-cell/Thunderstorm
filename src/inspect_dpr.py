"""
INSPECT A GPM DPR (SPACEBORNE RADAR) FILE
===========================================
Before writing the real radar loader we need to know three things about
these ~770 MB files (same "look first, then code" step we did for ERA5
and IMERG):

  1. What the reflectivity / latitude / longitude / time variables are
     actually called inside the file.
  2. How much of your Punjab-Haryana-Delhi box the radar swath covers
     (a satellite radar sees a narrow strip, not the whole region).
  3. WHEN the satellite passed over the region (it will NOT be 12:00 like
     your ERA5 / IMERG files).

Run (from the src folder):
    python inspect_dpr.py
or on a specific file:
    python inspect_dpr.py ..\\data\\raw\\radar\\NAME_OF_FILE.HDF5

Then paste the whole output back. Reading only a few small arrays from the
file, so this takes a few seconds even though the file is big.
"""

import glob
import os
import sys

import numpy as np

from config import RADAR_RAW_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

try:
    import h5py
except ImportError:
    h5py = None

KEYWORDS = ("zfactor", "precipratenearsurface", "latitude", "longitude",
            "scantime", "flagprecip", "heightstormtop")
FILL_LIMIT = -900  # DPR uses about -9999.9 for "no data"


def region_summary(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """Which radar pixels fall inside the study box? Returns (mask, n_pixels, scan_indices)."""
    inside = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    scans = np.where(inside.any(axis=1))[0]
    return inside, int(inside.sum()), scans


def find_dataset(f, group_names, suffix):
    """First existing dataset path like 'FS/Latitude', trying each swath group in turn."""
    for g in group_names:
        path = f"{g}/{suffix}"
        try:
            obj = f[path]
        except KeyError:
            continue
        if hasattr(obj, "shape"):
            return path
    return None


def list_candidate_variables(f, limit=45):
    found = []

    def visitor(name, obj):
        if hasattr(obj, "shape") and any(k in name.lower() for k in KEYWORDS):
            found.append((name, tuple(obj.shape), str(obj.dtype)))

    f.visititems(visitor)
    print(f"Variables whose names look relevant ({len(found)} found, showing up to {limit}):")
    for name, shape, dtype in found[:limit]:
        print(f"   {name:55s} shape={shape}  {dtype}")
    return found


def read_scan_time(f, group, scan_index):
    """UTC 'YYYY-MM-DD HH:MM:SS' of one scan, or None if the time variables are not where expected."""
    try:
        parts = {}
        for key in ("Year", "Month", "DayOfMonth", "Hour", "Minute", "Second"):
            parts[key] = int(f[f"{group}/ScanTime/{key}"][scan_index])
        return (f"{parts['Year']:04d}-{parts['Month']:02d}-{parts['DayOfMonth']:02d} "
                f"{parts['Hour']:02d}:{parts['Minute']:02d}:{parts['Second']:02d}")
    except Exception:  # noqa: BLE001
        return None


def inspect_file(f):
    print("Top-level groups:", list(f.keys()), "\n")
    list_candidate_variables(f)

    groups = [g for g in ("FS", "NS", "HS") if g in f]
    lat_path = find_dataset(f, groups, "Latitude")
    lon_path = find_dataset(f, groups, "Longitude")
    if not (lat_path and lon_path):
        print("\n[!] Could not find Latitude/Longitude under FS/NS/HS -- paste this output back "
              "and I will adapt the loader to the real layout.")
        return

    group = lat_path.split("/")[0]
    lat = f[lat_path][:]
    lon = f[lon_path][:]
    print(f"\nUsing swath group '{group}': lat/lon arrays are {lat.shape} (scans x rays)")

    inside, n_in, scans = region_summary(lat, lon, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    box = f"lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}"
    print(f"Study box ({box}): {n_in} radar pixels inside, in {len(scans)} scan lines")
    if n_in == 0:
        print("  -> the file's orbit segment does not actually cross your box; nothing to extract.")
        return

    t0 = read_scan_time(f, group, int(scans[0]))
    t1 = read_scan_time(f, group, int(scans[-1]))
    print(f"Satellite is over the box from {t0} to {t1} UTC (add 5h30m for Indian time)")
    print(f"Coverage: {n_in} pixels of the ~{lat.shape[1]}-ray swath (a narrow strip, not the whole box)")

    for suffix in ("SLV/zFactorFinalNearSurface", "SLV/precipRateNearSurface"):
        path = f"{group}/{suffix}"
        try:
            ds = f[path]
        except KeyError:
            print(f"\n[!] {path} not found (see the variable list above for what exists).")
            continue
        vals = ds[:][inside]
        vals = vals[vals > FILL_LIMIT]
        units = ds.attrs.get("units", b"?") if hasattr(ds, "attrs") else "?"
        if isinstance(units, bytes):
            units = units.decode(errors="ignore")
        if vals.size:
            print(f"\n{path}  shape={tuple(ds.shape)}  units={units}")
            print(f"   valid values inside box: {vals.size}  min={vals.min():.1f}  "
                  f"max={vals.max():.1f}  mean={vals.mean():.1f}")
        else:
            print(f"\n{path}: no valid values inside the box (no rain seen there on this pass).")


def main():
    if h5py is None:
        print("h5py is not installed. Run: pip install h5py --break-system-packages")
        return

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = sorted(glob.glob(os.path.join(RADAR_RAW_DIR, "*.HDF5")))
        if not files:
            print(f"No .HDF5 radar files found in {RADAR_RAW_DIR}")
            return
        path = files[-1]

    print(f"Inspecting: {os.path.basename(path)}\n")
    with h5py.File(path, "r") as f:
        inspect_file(f)


if __name__ == "__main__":
    main()