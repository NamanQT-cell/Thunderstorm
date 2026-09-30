"""
LIGHTNING DATA INGESTION
==========================
Handles IMD/ISRO lightning network data, OR (OPEN mode fallback) data
downloaded manually from Blitzortung.org or a WWLLN research request.

WHAT YOU NEED TO DO:
1. IMD mode: request lightning strike CSV exports (lat, lon, timestamp,
   polarity) from IMD/ISRO and place files in data/raw/lightning/
2. OPEN mode: sign up at https://www.blitzortung.org (community network,
   has India coverage) or request WWLLN academic access
   (http://wwlln.net) and place the exported CSV files in the same folder.
3. Expected CSV columns (adjust the column names below to match your file):
   timestamp, latitude, longitude, current_ka (optional), type (CG/IC)
"""

import os
import glob
import numpy as np
import pandas as pd

from config import LIGHTNING_RAW_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_RESOLUTION_KM


def list_available_files(pattern="*.csv"):
    files = sorted(glob.glob(os.path.join(LIGHTNING_RAW_DIR, pattern)))
    if not files:
        print(f"[!] No lightning files found in {LIGHTNING_RAW_DIR}")
        print("    >>> YOU INSERT HERE: place your lightning CSV exports here.")
    return files


def load_lightning_csv(filepath):
    """
    >>> YOU INSERT HERE: rename these columns to match your actual file's headers.
    Run `pd.read_csv(filepath).head()` first to check real column names.
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        "timestamp": "time",       # >>> YOU INSERT HERE if your column is named differently
        "latitude": "lat",         # >>> YOU INSERT HERE
        "longitude": "lon",        # >>> YOU INSERT HERE
    })
    df["time"] = pd.to_datetime(df["time"])
    df = df[(df.lat >= LAT_MIN) & (df.lat <= LAT_MAX) & (df.lon >= LON_MIN) & (df.lon <= LON_MAX)]
    return df


def strikes_to_grid_density(df, grid_shape=(64, 64)):
    """Convert point strikes into a gridded strike-count map for one time window."""
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, grid_shape[0] + 1)
    lon_bins = np.linspace(LON_MIN, LON_MAX, grid_shape[1] + 1)
    density, _, _ = np.histogram2d(df["lat"], df["lon"], bins=[lat_bins, lon_bins])
    return density


def generate_synthetic_lightning_frame(shape=(64, 64)):
    """TEMPORARY synthetic strike-density grid for pipeline testing."""
    density = np.zeros(shape)
    # scatter a handful of fake strikes near the synthetic storm core used
    # in ingest_radar.py / ingest_satellite.py so labels line up spatially
    rng = np.random.default_rng(42)
    core_r, core_c = int(shape[0] * 0.6), int(shape[1] * 0.4)
    for _ in range(15):
        r = np.clip(core_r + rng.integers(-3, 3), 0, shape[0] - 1)
        c = np.clip(core_c + rng.integers(-3, 3), 0, shape[1] - 1)
        density[r, c] += 1
    return density


def generate_cape_informed_lightning_frame(cape_grid, date_str, shape=(64, 64)):
    """
    IMPROVED PLACEHOLDER LABEL, used until real lightning data (IMD/ISRO or
    Blitzortung) is available. Instead of an identical fixed pattern for
    every date (the old bug), this ties the synthetic strike density to
    the REAL CAPE field for that date:
      - Higher CAPE (more atmospheric instability) -> more simulated strikes
      - Strikes are placed near the grid cell(s) with the highest CAPE,
        not a fixed location
      - The random pattern itself is seeded from the date string, so each
        date gets a different, but reproducible, pattern

    This is still not real lightning -- document this clearly in your
    report as a "CAPE-informed proxy label used pending institutional
    lightning data access" -- but it's a physically-grounded placeholder
    instead of an arbitrary constant one, so the model actually learns
    a relationship between atmospheric conditions and the label.
    """
    # seed the RNG deterministically from the date so each date differs
    # but the same date always reproduces the same result
    seed = int(date_str.replace("-", ""))
    rng = np.random.default_rng(seed)

    # resize cape_grid to match shape if needed (cheap nearest-neighbor)
    if cape_grid.shape != shape:
        row_idx = np.linspace(0, cape_grid.shape[0] - 1, shape[0]).round().astype(int)
        col_idx = np.linspace(0, cape_grid.shape[1] - 1, shape[1]).round().astype(int)
        cape_resized = cape_grid[row_idx][:, col_idx]
    else:
        cape_resized = cape_grid

    # normalize CAPE for spatial placement (relative pattern within this field)
    cape_min, cape_max = cape_resized.min(), cape_resized.max()
    cape_norm = (cape_resized - cape_min) / (cape_max - cape_min + 1e-6)

    # normalize the ABSOLUTE CAPE level against typical real-world bounds
    # (roughly 0 J/kg = stable, 4000+ J/kg = extreme instability) so that a
    # genuinely high-CAPE day produces more strikes than a genuinely calm
    # day -- using cape_resized.min()/max() here instead would cancel out
    # the actual magnitude and only reflect the spatial pattern, which was
    # the bug in the first version of this function.
    typical_cape_ceiling = 4000.0
    mean_absolute_cape = float(cape_resized.mean())
    intensity_factor = np.clip(mean_absolute_cape / typical_cape_ceiling, 0, 1)

    # CRITICAL: below this intensity, treat the date as a genuine CALM DAY
    # with ZERO simulated strikes. Without this cutoff, every date got at
    # least 2 strikes regardless of how low CAPE was, meaning EVERY date
    # counted as "a storm happened" -- there was no negative class at all
    # for the model to learn from, which is why region-level training was
    # stuck outputting a constant, meaningless probability. This threshold
    # is a rough, adjustable placeholder (not a real meteorological
    # threshold) purely to introduce genuine calm-day variety.
    calm_day_cutoff = 0.20
    if intensity_factor < calm_day_cutoff:
        return np.zeros(shape)  # no simulated storm at all this date

    # more instability (higher absolute CAPE) -> more simulated strikes, roughly 2-45
    num_strikes = int(2 + intensity_factor * 43)

    # turn the CAPE field into strike-placement probabilities (higher CAPE
    # cells are more likely to get a simulated strike)
    flat_probs = cape_norm.flatten()
    flat_probs = flat_probs / flat_probs.sum()

    density = np.zeros(shape)
    chosen_indices = rng.choice(len(flat_probs), size=num_strikes, p=flat_probs)
    for idx in chosen_indices:
        r, c = np.unravel_index(idx, shape)
        density[r, c] += 1

    return density


if __name__ == "__main__":
    files = list_available_files()
    if files:
        df = load_lightning_csv(files[0])
        print(df.head())
        grid = strikes_to_grid_density(df)
        print("Grid strike density shape:", grid.shape, "| total strikes:", grid.sum())
    else:
        print("Using synthetic lightning frame for testing.")
        grid = generate_synthetic_lightning_frame()
        print("Synthetic grid shape:", grid.shape, "| total strikes:", grid.sum())