"""ResNet18-v1 for 32x32 images, matching the flax model of the original Data Diet code."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual_branch = self.norm2(self.conv2(F.relu(self.norm1(self.conv1(inputs)))))
        shortcut = inputs if self.shortcut is None else self.shortcut(inputs)
        return F.relu(shortcut + residual_branch)


class ResNet18LowRes(nn.Module):
    """3x3 stride-1 stem and no max-pool (low-resolution variant); weights use flax's lecun_normal init."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        blocks, in_channels = [], 64
        for out_channels, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            blocks += [ResidualBlock(in_channels, out_channels, stride), ResidualBlock(out_channels, out_channels, 1)]
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(512, num_classes)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                std = math.sqrt(1.0 / module.weight[0].numel()) / 0.87962566103423978  # truncated-normal correction
                nn.init.trunc_normal_(module.weight, 0.0, std, -2 * std, 2 * std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.adaptive_avg_pool2d(self.blocks(self.stem(images)), 1).flatten(1)
        return self.classifier(features)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
