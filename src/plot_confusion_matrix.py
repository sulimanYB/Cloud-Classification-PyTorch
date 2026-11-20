
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


# ============================================================
# 1. Load the saved confusion matrix
# ============================================================

RESULTS_DIR = Path("results")
CM_PATH = RESULTS_DIR / "confusion_matrix.csv"

# Rows = actual classes
# Columns = predicted classes
cm_df = pd.read_csv(CM_PATH, index_col=0)

class_names = cm_df.index.tolist()
cm = cm_df.to_numpy()

# Shape: (7, 7)
# 7 actual classes x 7 predicted classes
print("Confusion matrix shape:", cm.shape)


# ============================================================
# 2. Plot the confusion matrix
# ============================================================

fig, ax = plt.subplots(figsize=(11, 9))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=True
)

plt.title("Cloud CNN — Test Confusion Matrix", fontsize=15)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.xlabel("Predicted class")
plt.ylabel("Actual class")

plt.tight_layout()


# ============================================================
# 3. Save the figure for our GitHub README
# ============================================================

OUTPUT_PATH = RESULTS_DIR / "confusion_matrix.png"

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Confusion matrix saved to: {OUTPUT_PATH}")
