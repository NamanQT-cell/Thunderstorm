"""
BUILD REAL TRAINING DATASET
==============================
Loops through every date you've downloaded (ERA5 + IMERG pairs) and
builds one merged, normalized training example per date, saving them
as .npy files in data/processed/. This is the real dataset model.py
will train on next.

WHAT YOU NEED TO DO:
Nothing to edit for a first run -- this script automatically finds
every ERA5 file in data/raw/nwp/ and matches it to the IMERG file for
the same date in data/raw/satellite/. Just run:
    python build_dataset.py

If a date's ERA5 file has no matching IMERG file (or vice versa), that
date is skipped with a warning -- not an error -- so partial downloads
don't break the whole run.
"""

import os
import re
import glob
import numpy as np

from config import NWP_RAW_DIR, SATELLITE_RAW_DIR, PROCESSED_DIR
from ingest_nwp import load_real_nwp_frame
from ingest_satellite import load_real_imerg_frame
from ingest_lightning import generate_cape_informed_lightning_frame  # CAPE-informed proxy label (see ingest_lightning.py)
from ingest_radar import generate_synthetic_radar_frame            # still synthetic -- see note below
from labels import make_binary_label
from preprocessing import resize_to_shape, normalize_channels

TARGET_SHAPE = (64, 64)


def find_date_pairs():
    """
    Matches ERA5 files (era5_YYYY-MM-DD.nc) to IMERG files for the same
    date (3B-HHR.MS.MRG.3IMERG.YYYYMMDD-*.HDF5) and returns a list of
    (date_str, era5_path, imerg_path) tuples.
    """
    era5_files = glob.glob(os.path.join(NWP_RAW_DIR, "era5_*.nc"))
    pairs = []

    for era5_path in era5_files:
        match = re.search(r"era5_(\d{4}-\d{2}-\d{2})\.nc", os.path.basename(era5_path))
        if not match:
            continue
        date_str = match.group(1)              # e.g. "2025-02-25"
        date_compact = date_str.replace("-", "")  # e.g. "20250225"

        imerg_candidates = glob.glob(
            os.path.join(SATELLITE_RAW_DIR, f"*3IMERG.{date_compact}-*.HDF5")
        )
        if not imerg_candidates:
            print(f"[!] Skipping {date_str}: no matching IMERG file found.")
            continue

        pairs.append((date_str, era5_path, imerg_candidates[0]))

    return pairs


def build_one_example(date_str, era5_path, imerg_path, shape=TARGET_SHAPE):
    """Builds one merged, normalized (channels, H, W) array + one label grid."""
    satellite_channel = resize_to_shape(load_real_imerg_frame(imerg_path), shape)
    nwp_dict = load_real_nwp_frame(era5_path)
    nwp_channels = [resize_to_shape(v, shape) for v in nwp_dict.values()]

    # CAPE-informed proxy label: ties simulated lightning to the REAL CAPE
    # field for this date instead of a fixed pattern (see ingest_lightning.py
    # for full reasoning). Replace with a real lightning loader once IMD/ISRO
    # or Blitzortung data is available.
    real_cape_grid = nwp_dict["CAPE"]
    lightning_channel = generate_cape_informed_lightning_frame(real_cape_grid, date_str, shape)

    # radar is still fully synthetic -- replace once IMD radar access arrives
    radar_channel = generate_synthetic_radar_frame(shape)

    channels = [satellite_channel, lightning_channel, radar_channel] + nwp_channels
    merged = np.stack(channels, axis=0)
    merged = normalize_channels(merged)

    label = make_binary_label(lightning_channel)  # CAPE-informed proxy label until real lightning data arrives
    return merged, label


def build_full_dataset():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pairs = find_date_pairs()

    if not pairs:
        print("No matched ERA5+IMERG date pairs found. Check your data/raw/ folders.")
        return

    print(f"Found {len(pairs)} matched date(s). Building dataset...\n")

    X_list, y_list, dates_used = [], [], []

    for date_str, era5_path, imerg_path in pairs:
        try:
            merged, label = build_one_example(date_str, era5_path, imerg_path)
            X_list.append(merged)
            y_list.append(label)
            dates_used.append(date_str)
            print(f"  [ok] {date_str} -> merged shape {merged.shape}")
        except Exception as e:
            print(f"  [error] {date_str} failed: {e}")

    if not X_list:
        print("No examples were successfully built.")
        return

    X = np.stack(X_list)  # shape: (N, channels, H, W)
    y = np.stack(y_list)  # shape: (N, H, W)

    np.save(os.path.join(PROCESSED_DIR, "X_real.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "y_real.npy"), y)

    with open(os.path.join(PROCESSED_DIR, "dates_used.txt"), "w") as f:
        f.write("\n".join(dates_used))

    print(f"\nSaved dataset: X_real.npy {X.shape}, y_real.npy {y.shape}")
    print(f"Dates used written to dates_used.txt")


if __name__ == "__main__":
    build_full_dataset()
