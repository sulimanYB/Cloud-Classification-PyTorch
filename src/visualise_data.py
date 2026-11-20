
import matplotlib.pyplot as plt

from data_loader import dataloader_train, full_dataset


# Retrieve one batch of training images
images, labels = next(iter(dataloader_train))

# images shape: (16, 3, 64, 64)
# labels shape: (16,)

class_names = full_dataset.classes

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    # Select one image from the batch
    image = images[i]  # Shape: (3, 64, 64)

    # PyTorch uses (C, H, W).
    # Matplotlib expects (H, W, C).
    image = image.permute(1, 2, 0)
    # New shape: (64, 64, 3)

    label = labels[i].item()
    class_name = class_names[label]

    ax.imshow(image)
    ax.set_title(class_name, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()
