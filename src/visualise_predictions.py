
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data_loader import dataloader_test, dataset_test
from model import CloudCNN


# ============================================================
# 1. Configuration
# ============================================================

DEVICE = torch.device("cpu")
MODEL_PATH = Path("models/best_cloud_cnn.pt")
OUTPUT_PATH = Path("results/misclassified_images.png")

CLASS_NAMES = dataset_test.classes
NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# 2. Load the best trained CNN
# ============================================================

model = CloudCNN(num_classes=NUM_CLASSES).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


# ============================================================
# 3. Identify misclassified test images
# ============================================================

misclassified = []

with torch.no_grad():

    for images, labels in dataloader_test:

        # images: (B, 3, 64, 64)
        # labels: (B,)
        images = images.to(DEVICE)

        # Raw model outputs (logits): (B, 7)
        logits = model(images)

        # Convert logits into class probabilities
        probabilities = torch.softmax(logits, dim=1)
        # Shape: (B, 7)

        # Highest probability and predicted class per image
        confidences, predictions = probabilities.max(dim=1)
        # Both shapes: (B,)

        for i in range(images.size(0)):

            true_label = labels[i].item()
            predicted_label = predictions[i].item()

            if predicted_label != true_label:

                misclassified.append({
                    "image": images[i].cpu(),
                    "true": CLASS_NAMES[true_label],
                    "predicted": CLASS_NAMES[predicted_label],
                    "confidence": confidences[i].item()
                })


print(f"Total misclassified images: {len(misclassified)}")


# ============================================================
# 4. Visualise 12 misclassified images
# ============================================================

# ============================================================
# 4. Select misclassified images across different classes
# ============================================================

from collections import defaultdict
import random

# Group errors by their true cloud class
errors_by_class = defaultdict(list)

for example in misclassified:
    errors_by_class[example["true"]].append(example)

# Select up to two errors per class
# Use a fixed seed for reproducibility
rng = random.Random(42)

examples = []

for class_name in CLASS_NAMES:
    class_errors = errors_by_class[class_name]

    if class_errors:
        selected = rng.sample(
            class_errors,
            k=min(2, len(class_errors))
        )
        examples.extend(selected)

# Display at most 12 images
examples = examples[:12]

print("Number of selected examples:", len(examples))

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for ax, example in zip(axes.flat, examples):

    # PyTorch format: (C, H, W) = (3, 64, 64)
    image = example["image"]

    # Matplotlib format: (H, W, C) = (64, 64, 3)
    image = image.permute(1, 2, 0)

    ax.imshow(image)

    ax.set_title(
        f"True: {example['true']}\n"
        f"Pred: {example['predicted']}\n"
        f"Confidence: {example['confidence']:.1%}",
        fontsize=9
    )

    ax.axis("off")

# Hide any unused axes
for ax in axes.flat[len(examples):]:
    ax.axis("off")

plt.suptitle(
    "Cloud CNN — Misclassified Test Images",
    fontsize=16
)

plt.tight_layout()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Figure saved to: {OUTPUT_PATH}")
