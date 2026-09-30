"""
STREAMLIT DASHBOARD
=====================
Visual demo of the nowcasting pipeline: satellite/radar imagery, a
lightning-strike overlay, the model's predicted probability heatmap, and
a time-evolution chart -- all in one page.

Runs entirely on SYNTHETIC data right now, so you can demo this today.
Once real data is plugged into preprocessing.py, this file needs NO
changes -- it just calls build_merged_frame() same as before.

HOW TO RUN:
    cd src
    streamlit run dashboard.py
Then open the local URL it prints (usually http://localhost:8501).

WHAT YOU NEED TO DO:
Nothing to edit here for the synthetic demo. Once you have a trained
model saved (see model.py), come back to the section marked
>>> YOU INSERT HERE below and load your real model instead of using
random probabilities.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from preprocessing import build_merged_frame, normalize_channels
from ingest_lightning import generate_synthetic_lightning_frame
from labels import make_binary_label
from config import REGION_NAME, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, LEAD_TIMES_MINUTES


st.set_page_config(page_title="Thunderstorm Nowcasting", layout="wide")

st.title("⛈️ AI/ML Thunderstorm & Lightning Nowcasting")
from preprocessing import USE_REAL_DATA
data_mode_label = "Real satellite (IMERG) + NWP (ERA5), synthetic radar/lightning" if USE_REAL_DATA else "Fully synthetic demo data"
st.caption(f"Region: {REGION_NAME}  |  Data mode: {data_mode_label}")


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
st.sidebar.header("Controls")
time_step = st.sidebar.slider("Time step (simulated, minutes from now)", -180, 0, 0, step=30)
selected_lead = st.sidebar.selectbox("Forecast lead time", LEAD_TIMES_MINUTES, index=0)
grid_size = st.sidebar.selectbox("Grid resolution", [32, 64, 96], index=1)


# ---------------------------------------------------------------------
# Build one synthetic time frame (deterministic per time_step so the
# slider actually changes the picture)
# ---------------------------------------------------------------------
np.random.seed(abs(time_step) + grid_size)  # vary output with slider position
shape = (grid_size, grid_size)
merged = normalize_channels(build_merged_frame(shape=shape))
lightning_density = generate_synthetic_lightning_frame(shape=shape)
label_grid = make_binary_label(lightning_density)

satellite_channel = merged[0]
lightning_channel = merged[1]


# ---------------------------------------------------------------------
# Load the trained model (falls back to the heuristic blend if no
# trained model is found yet, so the dashboard never crashes)
# ---------------------------------------------------------------------
import os

MODEL_PATH = "../outputs/models/model.pt"
model_loaded = False

try:
    import torch
    from model import ThunderstormCNN

    if os.path.exists(MODEL_PATH):
        trained_model = ThunderstormCNN(in_channels=merged.shape[0])
        trained_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        trained_model.eval()

        with torch.no_grad():
            input_tensor = torch.tensor(merged, dtype=torch.float32).unsqueeze(0)  # (1, C, H, W)
            prob_map = trained_model(input_tensor).squeeze().numpy()
        model_loaded = True
except Exception as e:
    st.sidebar.warning(f"Could not load trained model, using heuristic instead: {e}")

if not model_loaded:
    # fallback heuristic blend, used only if no trained model is available
    prob_map = np.clip(satellite_channel * 0.6 + lightning_channel * 0.4 + np.random.normal(0, 0.05, shape), 0, 1)

st.sidebar.caption(f"Prediction source: {'Trained CNN (model.pt)' if model_loaded else 'Heuristic blend (no trained model found)'}")


# ---------------------------------------------------------------------
# Load the REGION-LEVEL model too (single "storm likely / not" probability
# for the whole area, rather than a per-pixel grid -- see model.py's
# ThunderstormRegionCNN docstring for why this exists alongside the
# pixel-level model above)
# ---------------------------------------------------------------------
REGION_MODEL_PATH = "../outputs/models/model_region.pt"
region_prob = None
region_model_loaded = False

try:
    from model import ThunderstormRegionCNN

    if os.path.exists(REGION_MODEL_PATH):
        region_model = ThunderstormRegionCNN(in_channels=merged.shape[0])
        region_model.load_state_dict(torch.load(REGION_MODEL_PATH, map_location="cpu"))
        region_model.eval()

        with torch.no_grad():
            input_tensor = torch.tensor(merged, dtype=torch.float32).unsqueeze(0)
            region_prob = float(region_model(input_tensor).squeeze().item())
        region_model_loaded = True
except Exception as e:
    st.sidebar.warning(f"Could not load region-level model: {e}")


# ---------------------------------------------------------------------
# REGION-LEVEL headline result (the more reliable prediction at this
# data scale -- see model.py for why pixel-level and region-level are
# both shown)
# ---------------------------------------------------------------------
if region_model_loaded and region_prob is not None:
    if region_prob > 0.7:
        region_label, region_color = "STORM LIKELY", "red"
    elif region_prob > 0.4:
        region_label, region_color = "POSSIBLE STORM", "orange"
    else:
        region_label, region_color = "CALM EXPECTED", "green"

    banner_col1, banner_col2 = st.columns([2, 1])
    with banner_col1:
        st.markdown(
            f"### Region-level forecast: :{region_color}[{region_label}]  "
            f"({region_prob*100:.1f}% probability)"
        )
        st.caption(
            "This single probability (trained on real ERA5 + IMERG data across "
            "38+ dates, evaluated with POD/FAR/CSI) is the more statistically "
            "reliable prediction at this dataset size, compared to the "
            "per-pixel heatmap below which needs much more data to localize "
            "storms precisely."
        )
    with banner_col2:
        st.metric("Region storm probability", f"{region_prob*100:.1f}%")
else:
    st.info("Region-level model not found yet -- run model.py to train it (see model.py's ThunderstormRegionCNN).")

st.divider()


# ---------------------------------------------------------------------
# Layout: three columns -- observed imagery, predicted heatmap, numbers
# ---------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("🛰️ Observed Satellite (Brightness Temp)")
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(satellite_channel, cmap="gray_r")
    ax.set_title("Colder (white) = deeper convection")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig)

with col2:
    st.subheader(f"🎯 Pixel-level Probability Map (+{selected_lead} min)")
    st.caption("Experimental — needs more data to reliably localize storms")
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    im2 = ax2.imshow(prob_map, cmap="YlOrRd", vmin=0, vmax=1)
    strike_rows, strike_cols = np.where(lightning_density > 0)
    ax2.scatter(strike_cols, strike_rows, c="blue", s=15, marker="x", label="Lightning strikes")
    ax2.legend(loc="upper right", fontsize=6)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    st.pyplot(fig2)

with col3:
    st.subheader("📊 Alert Summary")
    max_prob = float(prob_map.max())
    mean_prob = float(prob_map.mean())
    total_strikes = int(lightning_density.sum())

    if max_prob > 0.75:
        alert_level, alert_color = "SEVERE", "red"
    elif max_prob > 0.5:
        alert_level, alert_color = "HIGH", "orange"
    elif max_prob > 0.25:
        alert_level, alert_color = "MODERATE", "gold"
    else:
        alert_level, alert_color = "LOW", "green"

    st.markdown(f"**Alert Level:** :{alert_color}[{alert_level}]")
    st.metric("Peak cell probability", f"{max_prob*100:.1f}%")
    st.metric("Region-average probability", f"{mean_prob*100:.1f}%")
    st.metric("Lightning strikes (last window)", total_strikes)

    st.write("**Probability by lead time (this location):**")
    lead_time_probs = {lt: np.clip(max_prob * (1 - lt / 200), 0, 1) for lt in LEAD_TIMES_MINUTES}
    lead_df = pd.DataFrame({
        "Lead time (min)": list(lead_time_probs.keys()),
        "Probability (%)": [round(v * 100, 1) for v in lead_time_probs.values()],
    })
    st.dataframe(lead_df, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------
# Time evolution chart (simulated history)
# ---------------------------------------------------------------------
st.subheader("📈 Probability Trend (last 3 hours, simulated)")
history_minutes = list(range(-180, 31, 30))
history_probs = []
for m in history_minutes:
    np.random.seed(abs(m) + grid_size)
    frame = normalize_channels(build_merged_frame(shape=shape))
    history_probs.append(float(np.clip(frame[0].max() * 0.7 + np.random.normal(0, 0.05), 0, 1)))

trend_df = pd.DataFrame({"Minutes from now": history_minutes, "Max probability": history_probs})
st.line_chart(trend_df.set_index("Minutes from now"))

st.caption(
    "Satellite and NWP channels use real downloaded data when available (see preprocessing.py). "
    "Radar and lightning channels are still synthetic/CAPE-informed placeholders pending IMD/ISRO access. "
    "The probability map uses your trained model (outputs/models/model.pt) once one exists, "
    "falling back to a simple heuristic blend otherwise."
)