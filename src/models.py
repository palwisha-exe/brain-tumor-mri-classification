from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.config import CLASS_NAMES


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        channels = (3, 16, 32, 64, 128)
        blocks = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(128, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(inputs)))


def build_model(architecture: str, pretrained: bool = False) -> nn.Module:
    if architecture == "baseline_cnn":
        return BaselineCNN()
    if architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_features, len(CLASS_NAMES)))
        return model
    raise ValueError(f"Unknown architecture: {architecture}")


def configure_trainable_layers(model: nn.Module, architecture: str, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
    if architecture == "efficientnet_b0" and stage == "frozen":
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    elif architecture == "efficientnet_b0" and stage == "finetune":
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        for block in model.features[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def last_convolution(model: nn.Module) -> nn.Conv2d:
    convs = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not convs:
        raise ValueError("Model has no convolution layer for Grad-CAM")
    return convs[-1]
