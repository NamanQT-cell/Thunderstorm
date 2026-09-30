"""
CENTRAL CONFIG FILE
====================
This is the ONLY file you should need to edit to point the project at
your real data. Every other script imports paths/settings from here.
"""

import os

# ---------------------------------------------------------------------
# 0. LOAD .env FILE AUTOMATICALLY (so you don't need to `export` manually)
#    Put a file named ".env" in the project root (one level above src/)
#    with lines like: CDS_API_KEY=your-key-here
# ---------------------------------------------------------------------
def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

_load_env_file()

# ---------------------------------------------------------------------
# 1. PROJECT REGION (edit to your study area)
# ---------------------------------------------------------------------
REGION_NAME = "Punjab-Haryana-Delhi"
LAT_MIN, LAT_MAX = 28.0, 32.5      # >>> YOU INSERT HERE: your region's bounding box
LON_MIN, LON_MAX = 74.0, 78.0      # >>> YOU INSERT HERE
GRID_RESOLUTION_KM = 4             # spatial grid cell size

# ---------------------------------------------------------------------
# 2. DATA SOURCE MODE
# ---------------------------------------------------------------------
# Set to "IMD" once your IMD/ISRO access is approved.
# Set to "OPEN" to use the open-data fallback (ERA5 + INSAT open + IMERG + Blitzortung/WWLLN).
DATA_SOURCE_MODE = "OPEN"          # >>> YOU INSERT HERE: "IMD" or "OPEN"

# ---------------------------------------------------------------------
# 3. RAW DATA FOLDERS (put downloaded files here, or point to your own path)
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RADAR_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "radar")
SATELLITE_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "satellite")
LIGHTNING_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "lightning")
NWP_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "nwp")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

# ---------------------------------------------------------------------
# 4. API KEYS / CREDENTIALS
#    Never hardcode secrets in code you'll upload/share (e.g. GitHub).
#    Put them in a local ".env" file (same folder as this config) with lines like:
#       CDS_API_KEY=your-key-here
#       MOSDAC_USERNAME=your-username
#       MOSDAC_PASSWORD=your-password
#    This file reads them from environment variables so nothing is exposed.
# ---------------------------------------------------------------------
CDS_API_KEY = os.environ.get("CDS_API_KEY", "")            # >>> YOU INSERT HERE (.env file, not this line)
MOSDAC_USERNAME = os.environ.get("MOSDAC_USERNAME", "")     # >>> YOU INSERT HERE
MOSDAC_PASSWORD = os.environ.get("MOSDAC_PASSWORD", "")     # >>> YOU INSERT HERE
IMD_API_KEY = os.environ.get("IMD_API_KEY", "")             # >>> YOU INSERT HERE (once approved)
EARTHDATA_USERNAME = os.environ.get("EARTHDATA_USERNAME", "")  # >>> YOU INSERT HERE (.env file) -- your urs.earthdata.nasa.gov login
EARTHDATA_PASSWORD = os.environ.get("EARTHDATA_PASSWORD", "")  # >>> YOU INSERT HERE (.env file)

# ---------------------------------------------------------------------
# 5. TIME SETTINGS
# ---------------------------------------------------------------------
TIME_STEP_MINUTES = 30             # matches INSAT-3D/IMERG cadence
LEAD_TIMES_MINUTES = [30, 60, 90]  # forecast horizons to predict
LOOKBACK_STEPS = 6                 # how many past frames feed the model (6 x 30min = 3 hours history)

# ---------------------------------------------------------------------
# 6. LABEL THRESHOLD (what counts as a "thunderstorm event")
# ---------------------------------------------------------------------
LIGHTNING_DENSITY_THRESHOLD = 1    # >=1 strike per grid cell per time step counts as positive label
