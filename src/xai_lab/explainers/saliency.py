from __future__ import annotations

import torch
from xai_lab.explainers.base import Explainer


class SaliencyExplainer(Explainer):
    """
    Vanilla saliency = | d(score_class) / d(input_pixels) |.

    Returns a [H,W] heatmap in [0,1].
    """
    name = "saliency"

    def explain(self, model: torch.nn.Module, x: torch.Tensor, target_class: int) -> torch.Tensor:
        model.eval()

        # Need gradients on the input.
        x = x.clone().detach().requires_grad_(True)

        logits = model(x)
        score = logits[:, target_class].sum()

        model.zero_grad(set_to_none=True)
        score.backward()

        # x.grad: [1,C,H,W] -> magnitude -> collapse channels -> [H,W]
        g = x.grad.detach().abs()
        heat = g.max(dim=1).values[0]

        # normalize [0,1]
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        return heat