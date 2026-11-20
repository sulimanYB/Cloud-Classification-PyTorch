
from pathlib import Path

# Dataset directories relative to the project root
DATA_DIR = Path("data/clouds")
TRAIN_DIR = DATA_DIR / "clouds_train"
TEST_DIR = DATA_DIR / "clouds_test"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def inspect_dataset(directory):
    """Display image counts for each cloud category."""

    print(f"\nDataset: {directory.name}")
    print("-" * 45)

    total_images = 0

    for class_dir in sorted(directory.iterdir()):
        if not class_dir.is_dir():
            continue

        image_count = sum(
            1 for file in class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{class_dir.name:<30} {image_count:>5}")
        total_images += image_count

    print("-" * 45)
    print(f"Total images: {total_images}")


# Inspect training and test datasets
inspect_dataset(TRAIN_DIR)
inspect_dataset(TEST_DIR)
