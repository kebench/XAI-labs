from __future__ import annotations

from typing import Any, Dict
from xai_lab.explainers.base import Explainer
from xai_lab.explainers.saliency import SaliencyExplainer
from xai_lab.explainers.gradcam import GradCAMExplainer


def build_explainer_from_config(cfg: Dict[str, Any]) -> Explainer:
    """
    Build an explainer from a config dict extracted from YAML.
    Modify this function to add new explainers.
    
    cfg example:
      {"name": "gradcam", "params": {"target_layer_path": "layer4"}}
    """
    name = cfg["name"].lower()
    params = cfg.get("params", {}) or {}

    if name == "saliency":
        return SaliencyExplainer()
    if name == "gradcam":
        return GradCAMExplainer(**params)

    raise ValueError(f"Unknown explainer: {name}")