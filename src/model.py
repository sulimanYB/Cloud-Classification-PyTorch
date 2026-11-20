
import torch
import torch.nn as nn


class CloudCNN(nn.Module):
    """CNN for classifying 64x64 RGB images into 7 cloud classes."""

    def __init__(self, num_classes=7):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # Input: (B, 3, 64, 64)

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # (B, 32, 64, 64)
            # 32 filters; padding preserves height and width

            nn.ELU(),
            # (B, 32, 64, 64) - activation preserves shape

            nn.MaxPool2d(kernel_size=2),
            # (B, 32, 32, 32) - halves height and width

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # (B, 64, 32, 32)
            # 64 filters; padding preserves spatial dimensions

            nn.ELU(),
            # (B, 64, 32, 32)

            nn.MaxPool2d(kernel_size=2),
            # (B, 64, 16, 16)

            nn.Flatten(),
            # (B, 64 * 16 * 16) = (B, 16384)
        )

        self.classifier = nn.Linear(64 * 16 * 16, num_classes)
        # (B, 16384) -> (B, 7) raw class scores (logits)

    def forward(self, x):
        x = self.feature_extractor(x)  # (B, 16384)
        x = self.classifier(x)         # (B, 7)
        return x


if __name__ == "__main__":
    model = CloudCNN(num_classes=7)

    # Simulate a batch of 16 RGB images, each 64x64
    dummy_input = torch.randn(16, 3, 64, 64)

    # Forward pass
    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    # Count all trainable parameters (weights and biases)
    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable parameters:", total_params)
