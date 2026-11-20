
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. Load saved training history
# ============================================================

HISTORY_PATH = Path("results/training_history.csv")
RESULTS_DIR = Path("results")

history = pd.read_csv(HISTORY_PATH)

# One row per completed training epoch
epochs = history["epoch"]


# ============================================================
# 2. Plot training and validation loss
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    label="Training loss",
    linewidth=2
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation loss",
    linewidth=2
)

# Highlight the epoch with minimum validation loss
best_idx = history["val_loss"].idxmin()
best_epoch = history.loc[best_idx, "epoch"]
best_loss = history.loc[best_idx, "val_loss"]

plt.scatter(
    best_epoch,
    best_loss,
    s=80,
    zorder=3,
    label=f"Best epoch: {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Cloud CNN — Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "loss_curves.png", dpi=300)
plt.show()


# ============================================================
# 3. Plot training and validation accuracy
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_accuracy"] * 100,
    label="Training accuracy",
    linewidth=2
)

plt.plot(
    epochs,
    history["val_accuracy"] * 100,
    label="Validation accuracy",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Cloud CNN — Training and Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "accuracy_curves.png", dpi=300)
plt.show()

print("Training plots saved to:", RESULTS_DIR)
