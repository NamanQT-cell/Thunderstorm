"""
EVALUATION — standard meteorological nowcasting metrics
===========================================================
Plain accuracy is misleading for thunderstorm prediction because most
grid cells, most of the time, do NOT have a storm (severe class
imbalance). Meteorologists instead use:

  POD (Probability of Detection)  = hits / (hits + misses)
      "Of all the storms that actually happened, what fraction did we catch?"

  FAR (False Alarm Ratio)         = false_alarms / (hits + false_alarms)
      "Of all our storm predictions, what fraction were wrong?"

  CSI (Critical Success Index)    = hits / (hits + misses + false_alarms)
      "Overall skill score combining both of the above -- 1.0 is perfect,
       0.0 is worthless. This is the single number most commonly quoted
       in nowcasting papers/reports."

WHAT YOU NEED TO DO:
Nothing to edit -- just run this after training (model.py must have
already saved a model and you must have data/processed/X_real.npy +
y_real.npy from build_dataset.py).
    python evaluate.py
"""

import os
import numpy as np

try:
    import torch
    from model import ThunderstormCNN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import PROCESSED_DIR, MODEL_DIR


def compute_pod_far_csi(y_true, y_pred, threshold=0.5):
    """
    y_true: binary ground-truth grid(s), any shape
    y_pred: predicted probability grid(s), same shape, values in [0, 1]
    threshold: probability above which a cell counts as a "predicted storm"
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true_binary = y_true.astype(int)

    hits = int(np.sum((y_pred_binary == 1) & (y_true_binary == 1)))
    misses = int(np.sum((y_pred_binary == 0) & (y_true_binary == 1)))
    false_alarms = int(np.sum((y_pred_binary == 1) & (y_true_binary == 0)))
    correct_negatives = int(np.sum((y_pred_binary == 0) & (y_true_binary == 0)))

    pod = hits / (hits + misses) if (hits + misses) > 0 else float("nan")
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else float("nan")
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else float("nan")

    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "POD": pod,
        "FAR": far,
        "CSI": csi,
    }


def evaluate_saved_model(model_filename="model.pt", val_split=0.2, threshold=0.5):
    """Loads the saved model and the real dataset, evaluates on the held-out validation split."""
    if not TORCH_AVAILABLE:
        print("Install torch first: pip install torch --break-system-packages")
        return None

    x_path = os.path.join(PROCESSED_DIR, "X_real.npy")
    y_path = os.path.join(PROCESSED_DIR, "y_real.npy")
    model_path = os.path.join(MODEL_DIR, model_filename)

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print("[!] No real dataset found. Run build_dataset.py first.")
        return None
    if not os.path.exists(model_path):
        print(f"[!] No trained model found at {model_path}. Run model.py first.")
        return None

    X_all = np.load(x_path)
    y_all = np.load(y_path)

    n = X_all.shape[0]
    n_val = max(1, int(n * val_split))
    X_val, y_val = X_all[-n_val:], y_all[-n_val:]

    print(f"Evaluating on {X_val.shape[0]} held-out example(s) (same split used during training).\n")

    model = ThunderstormCNN(in_channels=X_val.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_pred = model(X_val_t).squeeze(1).numpy()  # (n_val, H, W)

    metrics = compute_pod_far_csi(y_val, y_pred, threshold=threshold)

    print(f"--- Evaluation @ threshold={threshold} ---")
    print(f"Hits:              {metrics['hits']}")
    print(f"Misses:            {metrics['misses']}")
    print(f"False alarms:      {metrics['false_alarms']}")
    print(f"Correct negatives: {metrics['correct_negatives']}")
    print()
    print(f"POD (Probability of Detection): {metrics['POD']:.3f}  (higher is better, max 1.0)")
    print(f"FAR (False Alarm Ratio):        {metrics['FAR']:.3f}  (lower is better, min 0.0)")
    print(f"CSI (Critical Success Index):   {metrics['CSI']:.3f}  (higher is better, max 1.0)")
    print()
    print("NOTE: with a small dataset (a few dozen dates) these numbers will be noisy --")
    print("treat them as a directional signal, not a final benchmark, until you have")
    print("substantially more real dates and (ideally) real lightning labels instead of")
    print("the CAPE-informed proxy currently in use.")

    return metrics


def evaluate_region_model(model_filename="model_region.pt", val_split=0.2, threshold=0.5):
    """
    Evaluates the REGION-LEVEL model (one yes/no prediction per date,
    not a full grid) -- see model.py's ThunderstormRegionCNN docstring
    for why this exists. This is the more realistic evaluation given a
    small (tens of dates) real dataset.
    """
    if not TORCH_AVAILABLE:
        print("Install torch first: pip install torch --break-system-packages")
        return None

    from model import ThunderstormRegionCNN

    x_path = os.path.join(PROCESSED_DIR, "X_real.npy")
    y_path = os.path.join(PROCESSED_DIR, "y_real.npy")
    model_path = os.path.join(MODEL_DIR, model_filename)

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print("[!] No real dataset found. Run build_dataset.py first.")
        return None
    if not os.path.exists(model_path):
        print(f"[!] No trained region model found at {model_path}. Run model.py first.")
        return None

    X_all = np.load(x_path)
    y_pixel_all = np.load(y_path)
    y_region_all = y_pixel_all.max(axis=(1, 2))  # same derivation used during training

    n = X_all.shape[0]
    n_val = max(1, int(n * val_split))
    X_val, y_val = X_all[-n_val:], y_region_all[-n_val:]

    print(f"Evaluating region-level model on {X_val.shape[0]} held-out date(s).\n")

    model = ThunderstormRegionCNN(in_channels=X_val.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_pred = model(X_val_t).squeeze(1).numpy()  # (n_val,)

    metrics = compute_pod_far_csi(y_val, y_pred, threshold=threshold)

    print(f"--- Region-level evaluation @ threshold={threshold} ---")
    print(f"Predicted probabilities: {[round(float(p), 3) for p in y_pred]}")
    print(f"True labels:             {[int(v) for v in y_val]}")
    print()
    print(f"Hits: {metrics['hits']}  Misses: {metrics['misses']}  "
          f"False alarms: {metrics['false_alarms']}  Correct negatives: {metrics['correct_negatives']}")
    print(f"POD: {metrics['POD']:.3f}   FAR: {metrics['FAR']:.3f}   CSI: {metrics['CSI']:.3f}")

    return metrics


if __name__ == "__main__":
    print("=== PIXEL-LEVEL model evaluation ===")
    evaluate_saved_model()
    print("\n\n=== REGION-LEVEL model evaluation (recommended at this data scale) ===")
    evaluate_region_model()