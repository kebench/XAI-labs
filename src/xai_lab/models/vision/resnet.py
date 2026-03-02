from __future__ import annotations

import torch.nn as nn
from torchvision import models


_RESNET_WEIGHTS = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet34": models.ResNet34_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
    "resnet101": models.ResNet101_Weights.DEFAULT,
    "resnet152": models.ResNet152_Weights.DEFAULT,
}


def build_resnet(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a ResNet classifier by name and replace the final FC layer.

    Parameters
    ----------
    name:
        One of: resnet18, resnet34, resnet50, resnet101, resnet152
    num_classes:
        Number of output classes for your dataset.
    pretrained:
        If True, load ImageNet weights.

    Notes
    -----
    This function keeps all ResNet variants behind one interface, so scripts
    don't need hard-coded `if model == resnet18` branches.
    """
    name = name.lower().strip()
    if name not in _RESNET_WEIGHTS:
        raise ValueError(f"Unsupported ResNet '{name}'. Supported: {sorted(_RESNET_WEIGHTS.keys())}")

    model_fn = getattr(models, name)
    weights = _RESNET_WEIGHTS[name] if pretrained else None
    model = model_fn(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Convenience wrapper (kept for backwards-compatibility)."""
    return build_resnet("resnet18", num_classes=num_classes, pretrained=pretrained)