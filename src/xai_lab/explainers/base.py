from __future__ import annotations

from abc import ABC, abstractmethod
import torch


class Explainer(ABC):
    """
    Minimal interface for explainers.

    Contract:
      - x: torch.Tensor [1, C, H, W] (single image, preprocessed)
      - target_class: int
      - returns: heatmap torch.Tensor [H, W] normalized to [0,1]
    """

    name: str

    @abstractmethod
    def explain(self, model: torch.nn.Module, x: torch.Tensor, target_class: int) -> torch.Tensor:
        raise NotImplementedError