# Thunderstorm & Lightning Nowcasting — Project Guide

## What's already built for you

The entire pipeline skeleton is done and **runs right now on synthetic
(fake) data**, so you can test it immediately without waiting for any
data access approval:

```
thunderstorm-nowcasting/
├── src/
│   ├── config.py            <- EDIT THIS FIRST (region, mode, paths)
│   ├── ingest_satellite.py  <- INSAT-3D / IMERG loader
│   ├── ingest_radar.py      <- IMD radar loader (optional, see fallback)
│   ├── ingest_lightning.py  <- Lightning strike loader
│   ├── ingest_nwp.py        <- ERA5/NCMRWF (CAPE, shear, etc.) loader
│   ├── preprocessing.py     <- MERGES all sources into one array (this
│   │                           answers your "how does merging work" question)
│   ├── labels.py            <- Turns lightning data into training labels
│   └── model.py             <- CNN model + training loop
├── data/raw/{radar,satellite,lightning,nwp}/  <- put downloaded files here
├── data/processed/          <- merged arrays get saved here
├── outputs/models/          <- trained model files go here
└── requirements.txt
```

## Step-by-step: what to do, in order

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### Step 2 — Run the pipeline on synthetic data (do this TODAY)
This proves the whole thing works before you touch any real data:
```bash
cd src
python preprocessing.py   # should print a merged array shape
python labels.py          # should print a label grid
python model.py           # should train for 10 epochs and print loss decreasing
```
If all three run without errors, your pipeline architecture is correct.

### Step 3 — Register for data access (do this in parallel, starting TODAY)
| Source | Where | Approval time |
|---|---|---|
| INSAT-3D/3DR satellite | mosdac.gov.in | Usually fast (days) |
| ERA5 model data | cds.climate.copernicus.eu | Instant |
| GPM IMERG rainfall | urs.earthdata.nasa.gov | Instant |
| IMD radar | IMD data portal / institutional request | **Can be slow (weeks)** |
| IMD/ISRO lightning | Institutional request | **Can be slow (weeks)** |
| Blitzortung lightning (fallback) | blitzortung.org | Fast |

### Step 4 — Once you have real files
1. Drop files into the matching `data/raw/<source>/` folder.
2. Open the matching `ingest_*.py` file — every spot you need to edit is
   marked with `>>> YOU INSERT HERE`.
3. In `preprocessing.py`, swap the `generate_synthetic_*` calls for the
   real loader functions (also marked).
4. Re-run Step 2's three commands — they should now use real data.

### Step 5 — Scale up training
Once single-frame training works, loop `preprocessing.py` + `labels.py`
over your full time series, save the arrays as `.npy` files in
`data/processed/`, and update `model.py`'s `train_on_synthetic_data()`
to load those instead (also marked with a comment showing exactly where).

### Step 6 — Build the dashboard
Once you have a trained model, I can help you build a Streamlit dashboard
showing the map + probability heatmap + time chart discussed earlier —
just ask when you're ready for that step.

## If IMD radar/lightning access doesn't arrive in time

`config.py` has a `DATA_SOURCE_MODE` setting:
```python
DATA_SOURCE_MODE = "OPEN"   # already set this way by default
```
With this setting, `ingest_radar.py` is skipped automatically and the
model trains on satellite + NWP + open-lightning data only — a
legitimate, defensible approach you can explicitly document in your
report as "operating in open-data mode due to institutional data access
timelines," with IMD integration noted as future work. No code changes
needed to switch later if access does arrive — just flip this one setting
and drop real files into the radar folder.

## Where numbers/diagrams will eventually come from
- `model.py`'s output is a probability grid (numbers, 0–1 per cell)
- `preprocessing.py`'s merged frames ARE the satellite/radar images
- Lightning strikes from `ingest_lightning.py` become dot markers
- All of this feeds into the dashboard (Step 6) as one combined view
