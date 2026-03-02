from __future__ import annotations
from typing import Any, Dict
import torch.nn as nn
from xai_lab.models.vision.resnet import build_resnet, build_resnet18


def build_model(name: str, num_classes: int, pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Calls a function that builds a model from a name and configuration. Kept for backward compatibility.
    
    Args
    --------
    name: str
        Model name (e.g., "resnet18")
    num_classes: int
        Number of classes for the model
    pretrained: bool
        Whether to use pretrained weights
    **kwargs: Additional arguments to pass to the model builder
    
    Returns
    --------
    nn.Module
        The built model
    """
    name = name.lower()

    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unknown model name: {name}")

def build_model_from_config(model_cfg: Dict[str, Any]) -> nn.Module:
    """
    Build a model based on the experiment config.

    Expected fields in model_cfg
    ----------------------------
    name: str
    num_classes: int
    pretrained: bool (optional)

    Args
    --------
    model_cfg: Dict[str, Any]
        The model configuration dictionary.

    Returns
    -------
    nn.Module
        The built model.
    """
    name = str(model_cfg.get("name", "resnet18")).lower().strip()
    num_classes = int(model_cfg["num_classes"])
    pretrained = bool(model_cfg.get("pretrained", True))
    params = model_cfg.get("params", {}) or {}

    # Vision ResNets
    if name.startswith("resnet"):
        # params currently unused for ResNet; reserved for future extensions
        return build_resnet(name=name, num_classes=num_classes, pretrained=pretrained)

    raise ValueError(
        f"Unknown model name '{name}'. "
        f"Add a builder in src/xai_lab/models/vision/ and register it in factory.py."
    )