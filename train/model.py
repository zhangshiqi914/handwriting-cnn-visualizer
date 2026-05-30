import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNForMNIST(nn.Module):
    """
    CNN model for MNIST handwritten digit recognition.

    Input:
        [B, 1, 28, 28]

    Feature maps:
        conv1: [B, 16, 28, 28]
        conv2: [B, 32, 14, 14]

    Output:
        logits: [B, 10]
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        feature_maps = {}

        x = self.conv1(x)
        x = F.relu(x)
        feature_maps["conv1"] = x.detach()
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        feature_maps["conv2"] = x.detach()
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)

        logits = self.fc2(x)

        if return_features:
            return logits, feature_maps

        return logits