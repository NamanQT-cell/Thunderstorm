"""
LABEL GENERATION
==================
Converts lightning strike density into the binary "thunderstorm / no
thunderstorm" target your model will learn to predict.

WHAT YOU NEED TO DO:
Nothing for now (works with synthetic data). Once real lightning data is
in, this file automatically uses it -- no changes needed here since it
imports from ingest_lightning.py.
"""

import numpy as np
from config import LIGHTNING_DENSITY_THRESHOLD
from ingest_lightning import generate_synthetic_lightning_frame


def make_binary_label(lightning_density_grid, threshold=LIGHTNING_DENSITY_THRESHOLD):
    """1 = thunderstorm occurred in that grid cell during this time window, 0 = did not."""
    return (lightning_density_grid >= threshold).astype(np.uint8)


if __name__ == "__main__":
    density = generate_synthetic_lightning_frame()
    label = make_binary_label(density)
    print("Label grid shape:", label.shape, "| positive cells:", label.sum(), "/", label.size)
