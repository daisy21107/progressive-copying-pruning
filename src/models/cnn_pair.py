from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallEncoder(nn.Module):
    def __init__(self, width: int = 32, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(width, 2 * width, 3, padding=1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, padding=1)
        # Adaptive pooling to handle variable input sizes (28x28, 32x32, 36x36, etc.)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(7)  # Always outputs 7x7
        self.out_dim = 2 * width * 7 * 7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 14x14 from 28x28, or 16x16 from 32x32, etc.
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.adaptive_pool(x)  # Always 7x7 regardless of input size
        x = torch.flatten(x, 1)
        return x


class PairClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        width: int = 32,
        hidden: int = 128,
        shared_encoder: bool = True,
        in_channels: int = 1,
    ):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.enc1 = SmallEncoder(width, in_channels=in_channels)
        self.enc2 = self.enc1 if shared_encoder else SmallEncoder(width, in_channels=in_channels)
        feat_dim = self.enc1.out_dim * 2
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.enc1(x1)
        z2 = self.enc2(x2)
        z = torch.cat([z1, z2], dim=1)
        z = F.relu(self.fc1(z))
        logits = self.fc2(z)
        return logits
