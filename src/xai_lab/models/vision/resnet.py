from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create a ResNet18 classifier.

    Pretrained ResNet18 was trained on ImageNet (1000 classes).
    We reuse the feature extractor and replace the final layer
    to output num_classes logits for our dataset.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
