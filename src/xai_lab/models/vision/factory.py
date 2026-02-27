from __future__ import annotations

import torch.nn as nn
from xai_lab.models.vision.resnet import build_resnet18


def build_model(name: str, num_classes: int, pretrained: bool = True, **kwargs) -> nn.Module:
    name = name.lower()

    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unknown model name: {name}")