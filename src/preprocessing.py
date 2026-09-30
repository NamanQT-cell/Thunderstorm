"""
CO-REGISTRATION / MERGING MODULE
===================================
This is the step you asked about specifically: "merging all the data
from different sources." Here's what merging actually means here:

Each source (radar, satellite, lightning, NWP) has different resolution,
projection, and update frequency. This module stacks them into ONE
common array per time step: shape (num_channels, height, width).

REAL-DATA MODE (added once you had real ERA5 + IMERG files):
Set USE_REAL_DATA = True below and point REAL_NWP_FILE / REAL_SATELLITE_FILE
at your actual downloaded files. Radar and lightning still use synthetic
placeholders until IMD/lightning access comes through -- that's fine, the
merge logic handles a mix of real and synthetic channels without any
special-casing.
"""

import numpy as np

from ingest_satellite import generate_synthetic_satellite_frame, load_real_imerg_frame
from ingest_radar import generate_synthetic_radar_frame
from ingest_lightning import generate_synthetic_lightning_frame
from ingest_nwp import generate_synthetic_nwp_features, load_real_nwp_frame

# ---------------------------------------------------------------------
# >>> YOU INSERT HERE: flip this to True once you have real files ready,
#     and set the two paths below to your actual downloaded filenames.
# ---------------------------------------------------------------------
USE_REAL_DATA = True
REAL_NWP_FILE = "../data/raw/nwp/era5_2023-07-15.nc"
REAL_SATELLITE_FILE = "../data/raw/satellite/3B-HHR.MS.MRG.3IMERG.20230715-S000000-E002959.0000.V07B.HDF5"


def resize_to_shape(array_2d, target_shape):
    """
    Simple nearest-neighbor resize (no scipy dependency needed) so that
    every channel -- real or synthetic, whatever its native resolution --
    ends up at the same (H, W) size before stacking.
    """
    src_h, src_w = array_2d.shape
    tgt_h, tgt_w = target_shape
    row_idx = (np.linspace(0, src_h - 1, tgt_h)).round().astype(int)
    col_idx = (np.linspace(0, src_w - 1, tgt_w)).round().astype(int)
    return array_2d[row_idx][:, col_idx]


def build_merged_frame(shape=(64, 64), use_radar=True):
    """
    Merge one time step of all sources into a single multi-channel array.
    Automatically uses real satellite + NWP data if USE_REAL_DATA is True
    above, otherwise falls back to synthetic placeholders for everything.
    """
    if USE_REAL_DATA:
        satellite_channel = resize_to_shape(load_real_imerg_frame(REAL_SATELLITE_FILE), shape)
        nwp_dict = load_real_nwp_frame(REAL_NWP_FILE)
        nwp_channels = {k: resize_to_shape(v, shape) for k, v in nwp_dict.items()}
    else:
        satellite_channel = generate_synthetic_satellite_frame(shape)
        nwp_scalars = generate_synthetic_nwp_features()
        nwp_channels = {k: np.full(shape, v) for k, v in nwp_scalars.items()}

    # lightning and radar stay synthetic until IMD/lightning access arrives
    lightning_channel = generate_synthetic_lightning_frame(shape)

    channels = [satellite_channel, lightning_channel]

    if use_radar:
        radar_channel = generate_synthetic_radar_frame(shape)   # -> replace once IMD radar access arrives
        channels.append(radar_channel)

    for channel_array in nwp_channels.values():
        channels.append(channel_array)

    merged = np.stack(channels, axis=0)  # shape: (num_channels, H, W)
    return merged


def normalize_channels(merged_frame):
    """Min-max normalize each channel independently -- required before feeding a neural net."""
    normalized = np.zeros_like(merged_frame, dtype=np.float32)
    for i in range(merged_frame.shape[0]):
        channel = merged_frame[i]
        c_min, c_max = channel.min(), channel.max()
        normalized[i] = (channel - c_min) / (c_max - c_min + 1e-6)
    return normalized


if __name__ == "__main__":
    frame = build_merged_frame()
    print("Merged frame shape (channels, H, W):", frame.shape)
    print("Using real data:", USE_REAL_DATA)
    norm_frame = normalize_channels(frame)
    print("Normalized frame value range:", norm_frame.min(), "to", norm_frame.max())
