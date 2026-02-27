from __future__ import annotations

import torch
import torch.nn.functional as F
from xai_lab.explainers.base import Explainer


def get_module_by_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Resolve a module via dotted path string.
    Examples:
      "layer4" -> model.layer4
      "layer4.1.conv2" -> model.layer4[1].conv2
    """
    cur = model
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


class GradCAMExplainer(Explainer):
    """
    Grad-CAM: builds a coarse spatial heatmap using gradients wrt a conv layer.
    Returns [H,W] in [0,1].
    """
    name = "gradcam"

    def __init__(self, target_layer_path: str = "layer4"):
        self.target_layer_path = target_layer_path
        self._activations = None
        self._gradients = None
        self._hooks = []

    def _register_hooks(self, layer: torch.nn.Module):
        def fwd_hook(_m, _inp, out):
            self._activations = out.detach()

        def bwd_hook(_m, _gin, gout):
            self._gradients = gout[0].detach()

        self._hooks.append(layer.register_forward_hook(fwd_hook))
        self._hooks.append(layer.register_full_backward_hook(bwd_hook))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def explain(self, model: torch.nn.Module, x: torch.Tensor, target_class: int) -> torch.Tensor:
        model.eval()

        layer = get_module_by_path(model, self.target_layer_path)
        self._register_hooks(layer)

        logits = model(x)
        score = logits[:, target_class].sum()

        model.zero_grad(set_to_none=True)
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # [B,K,1,1]
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [B,1,h,w]
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        cam0 = cam[0, 0]
        cam0 = (cam0 - cam0.min()) / (cam0.max() - cam0.min() + 1e-8)

        self._remove_hooks()
        return cam0