from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Keep this consistent with your training normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def denorm_to_uint8(x_chw: torch.Tensor) -> np.ndarray:
    """[3,H,W] normalized tensor -> uint8 RGB [H,W,3]"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img01 = (x_chw.detach().cpu() * std + mean).clamp(0, 1)
    return (img01.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def save_triplet(out_path: Path, img_uint8: np.ndarray, heat01: np.ndarray, title: str) -> None:
    """Save input / heatmap / overlay"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_uint8); plt.axis("off"); plt.title("Input (denorm)")

    plt.subplot(1, 3, 2)
    plt.imshow(heat01, cmap="inferno"); plt.axis("off"); plt.title("Attribution")

    plt.subplot(1, 3, 3)
    plt.imshow(img_uint8)
    plt.imshow(heat01, cmap="inferno", alpha=0.45)
    plt.axis("off")
    plt.title("Overlay")

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()