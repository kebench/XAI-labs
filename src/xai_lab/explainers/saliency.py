from __future__ import annotations

import torch

"""
Core idea:
"How sensitive is the model's class score to each input pixel?"
Pros:
- simplest XAI method
- teaches you backprop + sensitivity
- fast

Cons / common failure mode:
- can be very noisy (high-frequency speckles)
- gradients can highlight edges everywhere, not “meaningful concepts”
- sensitive to preprocessing and model confidence

Best use: sanity checks:
- does the model attend to the face region at all?
- or is it focusing on borders/background?
"""

def saliency_map(model: torch.nn.Module, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
    """
    Vanilla saliency (input gradients).

    Idea:
      Compute d(score_class)/d(input_pixels).
      Pixels with larger magnitude gradients are "more influential" for that class.

    Notes:
      - Saliency can look noisy (common). It's still great for learning gradient-based attribution.
      - x should be shape [1, 3, H, W].
    """
    model.eval()

    x = x.clone().detach().requires_grad_(True)
    logits = model(x)

    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    score = logits[:, target_class].sum()

    model.zero_grad(set_to_none=True)
    score.backward()

    # [1,3,H,W] -> magnitude -> collapse channels -> [H,W]
    g = x.grad.detach().abs()
    heat = g.max(dim=1).values[0]

    # Normalize to [0,1] for visualization
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat
