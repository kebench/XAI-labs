from __future__ import annotations

import torch
import torch.nn.functional as F

"""
Core idea:
"Which spatial regions in the last conv layer are important for class c?"

Pros:
- often aligns better with human intuition (“eyes/mouth region”)
- easier to compare across images

Cons / watchouts:
- spatial resolution is coarse (depends on layer)
- can highlight the entire face (not very specific)
- if model is wrong, CAM explains the wrong decision confidently

Best use:
- sanity-check attention: does it light up face not background?
- compare correct vs incorrect predictions
"""

class GradCAM:
    """
    Grad-CAM implementation.

    Why Grad-CAM:
      Uses gradients in a chosen conv layer to produce a coarse spatial map
      showing "where" the network looked to make a class decision.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()  # [B,K,h,w]

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()  # [B,K,h,w]

    def __call__(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        self.model.eval()
        logits = self.model(x)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()

        # Channel weights: average gradient over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B,K,1,1]

        # Weighted sum of activations -> [B,1,h,w]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        cam0 = cam[0, 0]
        cam0 = (cam0 - cam0.min()) / (cam0.max() - cam0.min() + 1e-8)
        return cam0
