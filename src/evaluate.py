
from pathlib import Path
import argparse

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data_loader import dataloader_test, dataset_test
from model import CloudCNN


# ============================================================
# 1. Command-line configuration
# ============================================================

parser = argparse.ArgumentParser(
    description="Evaluate a trained Cloud CNN on the test set"
)

parser.add_argument(
    "--experiment",
    type=str,
    default="stratified",
    help="Experiment folder name, e.g. stratified or higher_lr"
)

args = parser.parse_args()

DEVICE = torch.device("cpu")

# Keep model and results together for each experiment
EXPERIMENT_DIR = Path("experiments") / args.experiment
MODEL_PATH = EXPERIMENT_DIR / "best_cloud_cnn.pt"

CLASS_NAMES = dataset_test.classes
NUM_CLASSES = len(CLASS_NAMES)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


# ============================================================
# 2. Load the selected model
# ============================================================

# CNN input: (B, 3, 64, 64)
# CNN output: (B, 7) - one logit per cloud class
model = CloudCNN(num_classes=NUM_CLASSES).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print(f"Evaluating model: {MODEL_PATH}")


# ============================================================
# 3. Generate predictions
# ============================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in dataloader_test:

        # images: (B, 3, 64, 64)
        # labels: (B,)
        images = images.to(DEVICE)

        # Forward pass: (B, 3, 64, 64) -> (B, 7)
        logits = model(images)

        # Select highest-scoring class for each image
        predictions = logits.argmax(dim=1)  # (B,)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())


# ============================================================
# 4. Calculate classification metrics
# ============================================================

accuracy = accuracy_score(all_labels, all_predictions)

report_text = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

report_dict = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report_text)


# ============================================================
# 5. Confusion matrix
# ============================================================

# Shape: (7, 7)
# Rows = actual classes; columns = predicted classes
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 6. Save evaluation results
# ============================================================

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(report_dict).transpose().to_csv(
    EXPERIMENT_DIR / "classification_report.csv"
)

pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
).to_csv(EXPERIMENT_DIR / "confusion_matrix.csv")

print(f"\nEvaluation results saved to: {EXPERIMENT_DIR}")
