"""
MODEL: simple CNN for grid-based thunderstorm probability prediction.

This starts intentionally simple (a small CNN) so it trains fast on a
laptop/Colab and is easy to explain in a report/viva. Once this works
end-to-end, you can upgrade to a ConvLSTM/U-Net for real temporal
sequences (see the "Next steps" note at the bottom of this file).

WHAT YOU NEED TO DO:
1. pip install torch --break-system-packages   (or use Google Colab, which
   has it pre-installed and gives you free GPU -- recommended for training)
2. Once you have real merged frames (from preprocessing.py) and real
   labels (from labels.py) across MANY time steps, replace the synthetic
   dataset loop in train_on_synthetic_data() with your real saved arrays.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class ThunderstormCNN(nn.Module):
        """Small CNN: input (channels, H, W) -> output (1, H, W) probability map."""

        def __init__(self, in_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)


    class ThunderstormRegionCNN(nn.Module):
        """
        REGION-LEVEL model: input (channels, H, W) -> output a SINGLE
        probability (not a grid) for "does this region have a thunderstorm
        during this window at all?"

        Why this exists: with a small number of real dates (tens, not
        hundreds+), asking the model to predict the EXACT grid cell of a
        storm (ThunderstormCNN above) is unrealistically hard -- it needs
        far more examples to learn precise spatial location. Predicting one
        yes/no per date is a much more learnable target at this data scale,
        while still answering the practically useful nowcasting question:
        "should this region be on alert or not?"
        """

        def __init__(self, in_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(8),  # shrink any input H,W down to 8x8 before the dense layer
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),  # global average pool -> one value per channel
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            features = self.conv(x)
            return self.fc(features)  # shape: (N, 1) -- one probability per example


def train_on_synthetic_data(num_samples=200, epochs=10):
    """
    Trains the CNN on synthetic merged frames + labels, purely to prove the
    pipeline works end-to-end before real data arrives.
    """
    if not TORCH_AVAILABLE:
        print("Install torch first: pip install torch --break-system-packages")
        return None

    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import build_merged_frame, normalize_channels
    from ingest_lightning import generate_synthetic_lightning_frame
    from labels import make_binary_label

    X_list, y_list = [], []
    for _ in range(num_samples):
        frame = normalize_channels(build_merged_frame())
        density = generate_synthetic_lightning_frame()
        label = make_binary_label(density)
        X_list.append(frame)
        y_list.append(label)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)          # (N, C, H, W)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)

    return _train_model(X, y, epochs=epochs)


def train_on_real_data(epochs=30, val_split=0.2):
    """
    Trains the CNN on the real dataset built by build_dataset.py
    (data/processed/X_real.npy and y_real.npy).

    Splits off a validation set so you can see if the model is actually
    learning something generalizable, not just memorizing the training dates.
    """
    if not TORCH_AVAILABLE:
        print("Install torch first: pip install torch --break-system-packages")
        return None

    import os
    from config import PROCESSED_DIR

    x_path = os.path.join(PROCESSED_DIR, "X_real.npy")
    y_path = os.path.join(PROCESSED_DIR, "y_real.npy")

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print(f"[!] No real dataset found. Run build_dataset.py first to create:")
        print(f"    {x_path}")
        print(f"    {y_path}")
        return None

    X_all = np.load(x_path)  # (N, C, H, W)
    y_all = np.load(y_path)  # (N, H, W)

    n = X_all.shape[0]
    if n < 4:
        print(f"[!] Only {n} real example(s) available -- that's too few to split into "
              f"train/validation meaningfully. Download more dates with batch_download.py "
              f"and re-run build_dataset.py before training seriously. Training on all "
              f"{n} example(s) anyway (no validation) so you can still test the pipeline.")
        X_train, y_train = X_all, y_all
        X_val, y_val = None, None
    else:
        n_val = max(1, int(n * val_split))
        X_train, y_train = X_all[:-n_val], y_all[:-n_val]
        X_val, y_val = X_all[-n_val:], y_all[-n_val:]
        print(f"Training on {X_train.shape[0]} example(s), validating on {X_val.shape[0]}.")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    model = _train_model(X_train_t, y_train_t, epochs=epochs)

    if X_val is not None:
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
            val_pred = model(X_val_t)
            val_loss = weighted_bce_loss(val_pred, y_val_t, pos_weight=1.0)  # unweighted for a fair readout
            print(f"Validation loss (unweighted, for reference): {val_loss.item():.4f}")

    return model


def train_region_model_on_real_data(epochs=50, val_split=0.2):
    """
    Trains ThunderstormRegionCNN (single yes/no per date -- see class
    docstring above for why this exists) on the real dataset built by
    build_dataset.py.

    The region-level label for each date is: "did AT LEAST ONE grid cell
    have a simulated storm?" -- derived by taking the max over the
    pixel-level label grid for that date.
    """
    if not TORCH_AVAILABLE:
        print("Install torch first: pip install torch --break-system-packages")
        return None

    import os
    from config import PROCESSED_DIR

    x_path = os.path.join(PROCESSED_DIR, "X_real.npy")
    y_path = os.path.join(PROCESSED_DIR, "y_real.npy")

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print(f"[!] No real dataset found. Run build_dataset.py first.")
        return None

    X_all = np.load(x_path)                       # (N, C, H, W)
    y_pixel_all = np.load(y_path)                  # (N, H, W)
    y_region_all = y_pixel_all.max(axis=(1, 2))    # (N,) -- 1 if ANY cell had a storm that date

    n = X_all.shape[0]
    print(f"Region-level labels: {int(y_region_all.sum())} storm date(s) / {n} total date(s)")

    if n < 4:
        print(f"[!] Only {n} example(s) -- training on all of them, no validation split.")
        X_train, y_train = X_all, y_region_all
        X_val, y_val = None, None
    else:
        n_val = max(1, int(n * val_split))
        X_train, y_train = X_all[:-n_val], y_region_all[:-n_val]
        X_val, y_val = X_all[-n_val:], y_region_all[-n_val:]
        print(f"Training on {X_train.shape[0]} example(s), validating on {X_val.shape[0]}.")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    model = ThunderstormRegionCNN(in_channels=X_train_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_positive = float(y_train_t.sum())
    num_total = float(y_train_t.numel())
    pos_weight_value = min((num_total - num_positive) / max(num_positive, 1.0), 20.0)
    print(f"Class balance: {num_positive:.0f} positive date(s) / {num_total:.0f} total "
          f"-> using pos_weight={pos_weight_value:.1f}")

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = weighted_bce_loss(pred, y_train_t, pos_weight_value)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {loss.item():.4f}")

    if X_val is not None:
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
            val_pred = model(X_val_t)
            val_loss = weighted_bce_loss(val_pred, y_val_t, pos_weight=1.0)
            print(f"\nValidation loss (unweighted): {val_loss.item():.4f}")
            print(f"Validation predictions: {val_pred.squeeze().tolist()}")
            print(f"Validation true labels: {y_val_t.squeeze().tolist()}")

    return model


def weighted_bce_loss(pred, target, pos_weight):
    """
    Manual weighted binary cross-entropy that works directly on the model's
    sigmoid-activated probability outputs (0-1), rather than requiring raw
    logits like nn.BCEWithLogitsLoss does -- this keeps the model's output
    format unchanged (still direct probabilities) so dashboard.py and
    evaluate.py don't need any changes.

    pos_weight > 1 makes the loss penalize a missed positive (actual storm,
    predicted no-storm) more heavily than a false alarm.
    """
    eps = 1e-7  # avoid log(0)
    pred = torch.clamp(pred, eps, 1 - eps)
    loss = -(pos_weight * target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    return loss.mean()


def _train_model(X, y, epochs=30):
    """
    Shared training loop used by both synthetic and real training paths.

    Uses a WEIGHTED loss (pos_weight) instead of plain BCELoss. Without
    this, since thunderstorm cells are extremely rare (often <1% of all
    grid cells), the model can achieve deceptively low loss by simply
    predicting "no storm" everywhere -- which is exactly what was
    happening before this fix (confirmed via evaluate.py showing 0 hits).
    pos_weight tells the loss function "a missed storm costs N times more
    than a false alarm," forcing the model to actually try to predict
    positives instead of taking that shortcut.
    """
    model = ThunderstormCNN(in_channels=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # compute how rare positives are in this specific dataset, and weight
    # the loss accordingly (capped at 50x to avoid instability if storms
    # are EXTREMELY rare in a very small dataset)
    num_positive = float(y.sum())
    num_total = float(y.numel())
    num_negative = num_total - num_positive
    pos_weight_value = min(num_negative / max(num_positive, 1.0), 50.0)
    print(f"Class balance: {num_positive:.0f} positive cells / {num_total:.0f} total "
          f"-> using pos_weight={pos_weight_value:.1f}")

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = weighted_bce_loss(pred, y, pos_weight_value)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - loss: {loss.item():.4f}")

    return model


def save_model(model, filename="model.pt"):
    """Saves a trained model's weights to outputs/models/ for later use in the dashboard."""
    import os
    from config import MODEL_DIR
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")
    return path


# ---------------------------------------------------------------------
# NEXT STEPS once this basic version works:
# 1. Swap ThunderstormCNN for a ConvLSTM to use LOOKBACK_STEPS of history
#    instead of a single time step (needed for true "nowcasting" of motion).
# 2. Move from single time-step frames to sequences: shape becomes
#    (N, T, C, H, W) and you'd use nn.ConvLSTM2d equivalents (e.g. via the
#    `convlstm` pip package) or a 3D-CNN.
# 3. Add multiple lead-time output heads (30/60/90 min) instead of one.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from config import PROCESSED_DIR

    if os.path.exists(os.path.join(PROCESSED_DIR, "X_real.npy")):
        print("Real dataset found.\n")

        print("--- Training REGION-LEVEL model (recommended at this data scale) ---")
        region_model = train_region_model_on_real_data()
        if region_model is not None:
            save_model(region_model, filename="model_region.pt")

        print("\n\n--- Also training the original pixel-level model (dashboard.py still uses this one) ---")
        pixel_model = train_on_real_data()
        if pixel_model is not None:
            save_model(pixel_model, filename="model.pt")
    else:
        print("No real dataset found yet -- training on synthetic data instead.")
        print("(Run build_dataset.py first once you have real downloads to switch to real training.)")
        synthetic_model = train_on_synthetic_data()
        if synthetic_model is not None:
            save_model(synthetic_model, filename="model.pt")