
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms


# ============================================================
# 1. Configuration
# ============================================================

DATA_DIR = Path("data/clouds")
TRAIN_DIR = DATA_DIR / "clouds_train"
TEST_DIR = DATA_DIR / "clouds_test"

IMAGE_SIZE = 64
BATCH_SIZE = 16
SEED = 42


# ============================================================
# 2. Image transformations
# ============================================================

# Training: resize and randomly augment images
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    # Single image shape: (C, H, W) = (3, 64, 64)
])

# Validation/test: deterministic preprocessing only
eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # Single image shape: (3, 64, 64)
])


# ============================================================
# 3. Load training data and create a stratified split
# ============================================================

# Two ImageFolder objects provide different transformations
# while retaining identical image paths and class mappings.
full_dataset = ImageFolder(
    root=TRAIN_DIR,
    transform=eval_transforms
)

augmented_dataset = ImageFolder(
    root=TRAIN_DIR,
    transform=train_transforms
)

# Original image indices and their integer class labels
indices = list(range(len(full_dataset)))
labels = full_dataset.targets

# Preserve class proportions in the train/validation split
train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,
    random_state=SEED
)

# Augmentation applies only to the training subset
dataset_train = Subset(augmented_dataset, train_indices)
dataset_val = Subset(full_dataset, val_indices)

# Keep the original test set separate and untouched
dataset_test = ImageFolder(
    root=TEST_DIR,
    transform=eval_transforms
)

# Verify identical class-to-index mappings
assert full_dataset.class_to_idx == dataset_test.class_to_idx


# ============================================================
# 4. Create DataLoaders
# ============================================================

dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_val = DataLoader(
    dataset_val,
    batch_size=BATCH_SIZE,
    shuffle=False
)

dataloader_test = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 5. Inspect dataset and tensor dimensions
# ============================================================

if __name__ == "__main__":

    print("Classes:", full_dataset.classes)
    print("Class mapping:", full_dataset.class_to_idx)

    print("Training images:", len(dataset_train))
    print("Validation images:", len(dataset_val))
    print("Test images:", len(dataset_test))

    # Inspect one batch
    images, labels = next(iter(dataloader_train))

    # images: (B, C, H, W) = (16, 3, 64, 64)
    # labels: (B,) = (16,)
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    print("\nClass distribution:")
    for class_name, class_idx in full_dataset.class_to_idx.items():
        train_count = sum(
            full_dataset.targets[i] == class_idx
            for i in train_indices
        )
        val_count = sum(
            full_dataset.targets[i] == class_idx
            for i in val_indices
        )

        print(
            f"{class_name:<25} "
            f"Train: {train_count:>3} | "
            f"Val: {val_count:>3}"
        )
