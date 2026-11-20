
from pathlib import Path

import torch
import torch.nn as nn

from data_loader import dataloader_train, dataloader_val
from model import CloudCNN

import pandas as pd


# ============================================================
# 1. Training configuration
# ============================================================

DEVICE = torch.device("cpu")
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5

# Reproducibility
torch.manual_seed(42)

# Directory for saving the best trained model
MODEL_PATH = Path("experiments/stratified/best_cloud_cnn.pt")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Initialise model, loss function and optimiser
# ============================================================

# CNN input:  (B, 3, 64, 64)
# CNN output: (B, 7) - raw logits for seven cloud classes
model = CloudCNN(num_classes=7).to(DEVICE)

# CrossEntropyLoss expects:
#   logits: (B, 7)
#   labels: (B,) containing integer class indices 0-6
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 3. Early stopping and training history
# ============================================================

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

# Store metrics for plotting learning curves later
history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
}


# ============================================================
# 4. Training and validation loop
# ============================================================

for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # Training phase
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in dataloader_train:

        # images: (B, 3, 64, 64)
        # labels: (B,)
        # B is usually 16, except possibly the final batch
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear gradients from the previous batch
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)  # Shape: (B, 7)

        # Cross-entropy returns a scalar loss
        loss = criterion(logits, labels)

        # Backpropagation: calculate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Accumulate loss, weighted by actual batch size
        train_loss += loss.item() * images.size(0)

        # Select the class with the highest logit
        predictions = logits.argmax(dim=1)  # Shape: (B,)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    # Average metrics over all training images
    train_loss /= train_total
    train_accuracy = train_correct / train_total

    # --------------------------------------------------------
    # Validation phase
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Disable gradient calculations during validation
    with torch.no_grad():

        for images, labels in dataloader_val:

            # images: (B, 3, 64, 64)
            # labels: (B,)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            logits = model(images)  # Shape: (B, 7)

            # Calculate validation loss
            loss = criterion(logits, labels)

            val_loss += loss.item() * images.size(0)

            # Predicted class indices
            predictions = logits.argmax(dim=1)  # Shape: (B,)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    # Average metrics over all validation images
    val_loss /= val_total
    val_accuracy = val_correct / val_total

    # --------------------------------------------------------
    # Record and display metrics
    # --------------------------------------------------------

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f}, "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Model checkpointing and early stopping
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        # Save model weights when validation loss improves
        torch.save(model.state_dict(), MODEL_PATH)

        print(
            f"  Best model saved "
            f"(Val Loss: {best_val_loss:.4f})"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"  No improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

    # Stop if validation loss has not improved
    # for PATIENCE consecutive epochs
    if epochs_without_improvement >= PATIENCE:

        print(f"\nEarly stopping at epoch {epoch + 1}")
        break


# ============================================================
# 5. Save training history
# ============================================================

# Save metrics as CSV for later plotting and analysis

history_df = pd.DataFrame(history)
history_df.index += 1
history_df.index.name = "epoch"

# Save training history for the selected stratified model
RESULTS_DIR = Path("experiments/stratified")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_PATH = RESULTS_DIR / "training_history.csv"
history_df.to_csv(HISTORY_PATH)


# ============================================================
# 6. Training summary
# ============================================================

print("\nTraining complete.")
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved to: {MODEL_PATH}")
print(f"Training history saved to: {HISTORY_PATH}")
