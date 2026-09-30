"""
JSON API FOR THE REACT FRONTEND
==================================
This file is NEW and purely additive: it does not change a single line
of preprocessing.py, model.py, labels.py, ingest_*.py or config.py. It
imports them exactly the way dashboard.py already does, and exposes the
same logic as JSON endpoints instead of a Streamlit page, so the React
app in ../frontend can consume it.

MODEL CONNECTION
----------------
On startup, this file looks for a trained checkpoint at
outputs/models/model.pt (the file model.py's save_model() produces).

  - If it's there: real inference runs, using ThunderstormCNN from
    model.py, and every response is tagged "usedRealModel": true.
  - If it's not there yet: responses fall back to the exact same
    illustrative formula dashboard.py uses (clearly marked
    "usedRealModel": false), so the frontend works today on synthetic
    data and needs ZERO changes once you drop a real model.pt in place.

HOW TO RUN
----------
    cd src
    pip install fastapi "uvicorn[standard]" --break-system-packages   # one-time
    uvicorn api:app --reload --port 8000

Then point the frontend at it -- see frontend/.env.example.
"""

import os
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from preprocessing import build_merged_frame, normalize_channels, USE_REAL_DATA
from ingest_lightning import generate_synthetic_lightning_frame
from labels import make_binary_label
from config import REGION_NAME, LEAD_TIMES_MINUTES, MODEL_DIR

app = FastAPI(title="Vidyut Nowcasting API")

# Wide-open CORS for local dev. Tighten this to your real frontend origin
# before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL_PATH = os.path.join(MODEL_DIR, "model.pt")
_REGION_MODEL_PATH = os.path.join(MODEL_DIR, "model_region.pt")
_model_cache = {"model": None, "tried": False}
_region_model_cache = {"model": None, "tried": False}


def _try_load_model(in_channels):
    """Load the trained checkpoint once and cache it. Never raises -- if
    anything about the checkpoint or torch is unavailable, we just fall
    back to the demo formula below, same as dashboard.py's own comment
    says to do manually."""
    if _model_cache["tried"]:
        return _model_cache["model"]
    _model_cache["tried"] = True
    if not os.path.exists(_MODEL_PATH):
        return None
    try:
        import torch
        from model import ThunderstormCNN
        m = ThunderstormCNN(in_channels=in_channels)
        m.load_state_dict(torch.load(_MODEL_PATH, map_location="cpu"))
        m.eval()
        _model_cache["model"] = m
        print(f"[api] loaded trained model from {_MODEL_PATH}")
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a soft fallback
        print(f"[api] could not load {_MODEL_PATH}, using demo formula instead: {exc}")
    return _model_cache["model"]


def _predict(merged, satellite_channel, lightning_channel):
    """Real model output if a checkpoint is available, else the same
    illustrative formula dashboard.py uses for the synthetic demo."""
    model = _try_load_model(in_channels=merged.shape[0])
    if model is not None:
        import torch
        with torch.no_grad():
            x = torch.tensor(merged).unsqueeze(0).float()
            prob = model(x).squeeze().numpy()
        return prob, True

    prob = np.clip(
        satellite_channel * 0.6
        + lightning_channel * 0.4
        + np.random.normal(0, 0.05, satellite_channel.shape),
        0,
        1,
    )
    return prob, False


def _try_load_region_model(in_channels):
    """
    Loads the REGION-LEVEL model (single storm-likelihood probability for
    the whole area, trained on real ERA5+IMERG data -- see model.py's
    ThunderstormRegionCNN docstring). This is the more statistically
    reliable prediction at small dataset sizes, shown alongside the
    per-pixel map for context.
    """
    if _region_model_cache["tried"]:
        return _region_model_cache["model"]
    _region_model_cache["tried"] = True
    if not os.path.exists(_REGION_MODEL_PATH):
        return None
    try:
        import torch
        from model import ThunderstormRegionCNN
        m = ThunderstormRegionCNN(in_channels=in_channels)
        m.load_state_dict(torch.load(_REGION_MODEL_PATH, map_location="cpu"))
        m.eval()
        _region_model_cache["model"] = m
        print(f"[api] loaded trained region model from {_REGION_MODEL_PATH}")
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a soft fallback
        print(f"[api] could not load {_REGION_MODEL_PATH}: {exc}")
    return _region_model_cache["model"]


def _predict_region(merged):
    """Real region-level model output if a checkpoint is available, else None."""
    model = _try_load_region_model(in_channels=merged.shape[0])
    if model is None:
        return None, False
    import torch
    with torch.no_grad():
        x = torch.tensor(merged).unsqueeze(0).float()
        prob = float(model(x).squeeze().item())
    return prob, True


@app.get("/api/config")
def get_config():
    return {
        "region": REGION_NAME,
        "leadTimes": LEAD_TIMES_MINUTES,
        "usingRealModel": os.path.exists(_MODEL_PATH),
        "usingRegionModel": os.path.exists(_REGION_MODEL_PATH),
        "dataMode": (
            "Real satellite (IMERG) + NWP (ERA5), synthetic radar/lightning"
            if USE_REAL_DATA
            else "Fully synthetic demo data"
        ),
    }


@app.get("/api/frame")
def get_frame(
    time_step: int = Query(0, ge=-180, le=0, description="Minutes from now, e.g. -90"),
    lead: int = Query(60, description="Forecast lead time in minutes, for the trend readout"),
    grid: int = Query(64, description="Grid resolution, e.g. 32/64/96"),
):
    """Mirrors dashboard.py's per-slider-move computation, as JSON."""
    np.random.seed(abs(time_step) + grid)
    shape = (grid, grid)
    merged = normalize_channels(build_merged_frame(shape=shape))
    lightning_density = generate_synthetic_lightning_frame(shape=shape)
    make_binary_label(lightning_density)  # computed for parity with dashboard.py; not returned

    satellite_channel = merged[0]
    lightning_channel = merged[1]
    prob_map, used_real_model = _predict(merged, satellite_channel, lightning_channel)
    region_prob, used_region_model = _predict_region(merged)

    max_prob = float(prob_map.max())
    mean_prob = float(prob_map.mean())
    lead_time_probs = {
        str(lt): round(float(np.clip(max_prob * (1 - lt / 200), 0, 1)) * 100, 1)
        for lt in LEAD_TIMES_MINUTES
    }
    strike_rows, strike_cols = np.where(lightning_density > 0)

    return {
        "usedRealModel": used_real_model,
        "satellite": satellite_channel.tolist(),
        "probability": prob_map.tolist(),
        "strikes": [[int(c), int(r)] for r, c in zip(strike_rows, strike_cols)],
        "maxProb": max_prob,
        "meanProb": mean_prob,
        "totalStrikes": int(lightning_density.sum()),
        "leadTimeProbs": lead_time_probs,
        "regionProbability": region_prob,       # None if model_region.pt not trained yet
        "usedRegionModel": used_region_model,
    }


@app.get("/api/trend")
def get_trend(grid: int = Query(64)):
    """Mirrors dashboard.py's 3-hour trend chart."""
    shape = (grid, grid)
    history_minutes = list(range(-180, 31, 30))
    history_probs = []
    for m in history_minutes:
        np.random.seed(abs(m) + grid)
        frame = normalize_channels(build_merged_frame(shape=shape))
        history_probs.append(
            float(np.clip(frame[0].max() * 0.7 + np.random.normal(0, 0.05), 0, 1))
        )
    return {"minutes": history_minutes, "maxProbability": history_probs}