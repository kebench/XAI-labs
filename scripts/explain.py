#!/usr/bin/env python3
"""
scripts/explain.py (Saliency first)

What this script does
---------------------
1) Loads a trained checkpoint (best.pt) from a run folder
2) Loads the exact config used for that run (config_used.yaml)
3) Runs inference on the test split to get predictions + confidence
4) Selects a small set of samples to explain:
   - high-confidence (model very sure)
   - low-confidence  (model least sure; "harder" cases)
5) Generates saliency maps (input gradients) for the predicted class
6) Saves images to:
   artifacts/reports/<experiment>/<run_id>/xai/saliency/

Saliency refresher (input-focused)
----------------------------------
Saliency = | d(score_class) / d(input_pixels) |
It highlights which pixels the model is most sensitive to for that class.

Notes:
- Saliency can be noisy (that's normal).
- Later add SmoothGrad / Integrated Gradients to reduce noise.

Inputs:
-----------------
--ckpt: Path to checkpoint (e.g., artifacts/runs/exp001_resnet18_saliency_ckplus/20251015_123456/best.pt)
--run_dir: Alternative to --ckpt; without the checkpoint filename (e.g., artifacts/runs/exp001_resnet18_saliency_ckplus/20251015_123456)
--split: Which split to explain (train/val/test). Default: test
--num_each: How many high-conf and low-conf samples to explain. Default: 8

Sample usage:
-------------
python scripts/explain.py --run_dir artifacts/runs/exp001_resnet18_saliency_ckplus/20251015_123456 --split test --num_each 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------
# Bootstrap so `import xai_lab...` works without pip install -e .
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xai_lab.data.datasets.image_csv import CsvImageDataset, CsvImageDatasetConfig
from xai_lab.data.transforms.image import AugmentConfig, build_transforms, IMAGENET_MEAN, IMAGENET_STD
from xai_lab.models.vision.resnet import build_resnet18
from xai_lab.utils.paths import find_project_root, load_yaml
from xai_lab.explainers.saliency import saliency_map

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_ckpt_and_run_dir(ckpt: Optional[Path], run_dir: Optional[Path]) -> Tuple[Path, Path]:
    """
    - If run_dir is provided, use run_dir/best.pt
    - Else use --ckpt and infer run_dir as ckpt.parent
    """
    if run_dir is not None:
        run_dir = run_dir.resolve()
        ckpt_path = (run_dir / "best.pt").resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"best.pt not found: {ckpt_path}")
        return ckpt_path, run_dir

    if ckpt is not None:
        ckpt_path = ckpt.resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path, ckpt_path.parent

    raise ValueError("Provide either --run_dir or --ckpt.")


def reports_dir_from_run_dir(repo_root: Path, cfg: Dict[str, Any], run_dir: Path) -> Path:
    """
    artifacts/runs/<run_name>/<run_id>/  -> artifacts/reports/<run_name>/<run_id>/
    """
    run_name = cfg["run"]["name"]
    base_out = repo_root / cfg["run"].get("output_dir", "artifacts")
    run_id = run_dir.name
    out = (base_out / "reports" / run_name / run_id).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def denorm_to_uint8(x: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized tensor [3,H,W] back to a uint8 image [H,W,3] for saving/plotting.
    Assumes ImageNet mean/std.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img01 = (x.detach().cpu() * std + mean).clamp(0, 1)
    return (img01.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def save_saliency_triplet(
    out_path: Path,
    img_uint8: np.ndarray,
    heatmap01: np.ndarray,
    title: str,
) -> None:
    """
    Save a 3-panel image:
      (1) original
      (2) saliency heatmap
      (3) overlay
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_uint8)
    plt.axis("off")
    plt.title("Input (denorm)")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap01, cmap="inferno")  # chosen for visibility
    plt.axis("off")
    plt.title("Saliency (|dS/dx|)")

    plt.subplot(1, 3, 3)
    plt.imshow(img_uint8)
    plt.imshow(heatmap01, cmap="inferno", alpha=0.45)
    plt.axis("off")
    plt.title("Overlay")

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def infer_confidences(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the split to collect:
      - predicted label for each sample
      - true label for each sample
      - confidence = softmax probability of predicted class
    """
    model.eval()

    all_pred: List[int] = []
    all_true: List[int] = []
    all_conf: List[float] = []

    softmax = nn.Softmax(dim=1)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = softmax(logits)
        pred = probs.argmax(dim=1)

        # confidence for each sample = prob of its predicted class
        conf = probs.gather(1, pred.view(-1, 1)).view(-1)

        all_pred.extend(pred.cpu().tolist())
        all_true.extend(y.cpu().tolist())
        all_conf.extend(conf.cpu().tolist())

    return np.array(all_pred), np.array(all_true), np.array(all_conf)


def select_indices(conf: np.ndarray, k: int) -> Tuple[List[int], List[int]]:
    """
    Pick k highest-confidence and k lowest-confidence indices.
    """
    if len(conf) == 0:
        return [], []

    k = min(k, len(conf))
    sorted_idx = np.argsort(conf)  # ascending

    low = sorted_idx[:k].tolist()
    high = sorted_idx[-k:][::-1].tolist()  # descending for convenience
    return high, low


def main(
    ckpt: Optional[Path],
    run_dir: Optional[Path],
    num_images_each: int,
    split: str,
) -> None:
    repo_root = find_project_root(PROJECT_ROOT)
    device = get_device()

    ckpt_path, resolved_run_dir = resolve_ckpt_and_run_dir(ckpt, run_dir)
    print(f"[explain] device={device}")
    print(f"[explain] ckpt={ckpt_path}")
    print(f"[explain] run_dir={resolved_run_dir}")

    # ---- Load checkpoint + config ----
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config_path = Path(checkpoint["config_path"]).resolve()
    cfg = load_yaml(config_path)

    # ---- Prepare reports output folders ----
    reports_dir = reports_dir_from_run_dir(repo_root, cfg, resolved_run_dir)
    out_dir = reports_dir / "xai" / "saliency"
    (out_dir / "high_conf").mkdir(parents=True, exist_ok=True)
    (out_dir / "low_conf").mkdir(parents=True, exist_ok=True)

    # ---- Build dataset (split) + deterministic transforms ----
    data_cfg = cfg["data"]
    csv_key = f"{split}_csv"
    if csv_key not in data_cfg:
        raise ValueError(f"Config missing data.{csv_key}. Available keys: {list(data_cfg.keys())}")

    split_csv = (repo_root / data_cfg[csv_key]).resolve()
    path_col = data_cfg.get("path_col", "path")
    label_col = data_cfg.get("label_col", "label")

    label_name_col = data_cfg.get("label_name_col", "label_name")

    # Read the split CSV to get label names (same ordering as CsvImageDataset)
    split_df = pd.read_csv(split_csv)

    # Build an id -> name mapping (e.g., 3 -> "happy")
    # drop_duplicates() prevents repeated rows from overwriting.
    if label_name_col in split_df.columns:
        id_to_name = dict(
            split_df[[label_col, label_name_col]]
            .dropna()
            .drop_duplicates()
            .assign(**{label_col: lambda d: d[label_col].astype(int),
                      label_name_col: lambda d: d[label_name_col].astype(str).str.strip()})
            .values
        )
    else:
        id_to_name = {}

    input_size = int(cfg["model"].get("input_size", 224))

    # Aug config is read for consistency, but train=False means no random augmentation is applied here.
    aug_cfg = AugmentConfig(
        crop_scale_min=float(cfg.get("augment", {}).get("crop_scale_min", 0.85)),
        hflip_p=float(cfg.get("augment", {}).get("hflip_p", 0.5)),
        rotation_deg=int(cfg.get("augment", {}).get("rotation_deg", 10)),
        jitter_brightness=float(cfg.get("augment", {}).get("jitter_brightness", 0.15)),
        jitter_contrast=float(cfg.get("augment", {}).get("jitter_contrast", 0.15)),
    )
    tfms = build_transforms(input_size=input_size, train=False, aug=aug_cfg)

    ds = CsvImageDataset(
        CsvImageDatasetConfig(csv_path=split_csv, path_col=path_col, label_col=label_col, project_root=repo_root),
        transform=tfms,
    )

    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    # ---- Rebuild model + load weights ----
    model_name = checkpoint.get("model_name", cfg["model"].get("name", "resnet18"))
    num_classes = int(checkpoint.get("num_classes", cfg["model"]["num_classes"]))
    pretrained = bool(cfg["model"].get("pretrained", True))

    if model_name != "resnet18":
        raise ValueError(f"Only resnet18 wired here for now. Got: {model_name}")

    model = build_resnet18(num_classes=num_classes, pretrained=pretrained)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    # ---- Step 1: inference pass to rank by confidence ----
    pred, true, conf = infer_confidences(model, loader, device=device)
    high_idx, low_idx = select_indices(conf, k=num_images_each)

    print(f"[select] {split}: total={len(ds)}")
    print(f"[select] high_conf={len(high_idx)} low_conf={len(low_idx)}")

    # ---- Step 2: compute saliency for selected indices (requires gradients) ----
    # We do these one-by-one because saliency needs backward() and simpler bookkeeping.
    manifest_path = out_dir / "saliency_manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()  # start fresh each run

    def explain_one(index: int, bucket: str) -> None:
        x, y = ds[index]                 # x is normalized tensor [3,H,W]
        x1 = x.unsqueeze(0).to(device)   # [1,3,H,W]
        y_true = int(y)

        y_pred = int(pred[index])
        y_conf = float(conf[index])

        true_name = id_to_name.get(y_true, str(y_true))
        pred_name = id_to_name.get(y_pred, str(y_pred))

        # Saliency for the predicted class (default in saliency_map)
        heat = saliency_map(model, x1, target_class=y_pred)  # [H,W] in [0,1]
        heat_np = heat.detach().cpu().numpy()

        img_uint8 = denorm_to_uint8(x)

        # Make a readable filename
        stem = Path(ds.paths[index]).stem
        out_file = out_dir / bucket / f"{index:04d}_{stem}_t{y_true}_p{y_pred}_c{y_conf:.3f}.png"

        title = (
            f"{bucket} | idx={index} | "
            f"true={y_true}:{true_name} pred={y_pred}:{pred_name} conf={y_conf:.3f} | {stem}"
        )
        save_saliency_triplet(out_file, img_uint8=img_uint8, heatmap01=heat_np, title=title)

        # Write one-line record so you can trace images back to data
        rec = {
            "split": split,
            "bucket": bucket,
            "index": index,
            "path": ds.paths[index],
            "true": y_true,
            "pred": y_pred,
            "confidence": y_conf,
            "output_png": str(out_file),
        }
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    for idx in high_idx:
        explain_one(idx, "high_conf")
    for idx in low_idx:
        explain_one(idx, "low_conf")

    print(f"[done] saved saliency images under: {out_dir}")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate saliency explanations for a trained run.")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to artifacts/runs/<exp>/<run_id>/ (uses best.pt)")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (e.g., .../best.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to explain.")
    parser.add_argument("--num_each", type=int, default=8, help="How many high-conf and low-conf samples to explain.")
    args = parser.parse_args()

    if not args.ckpt and not args.run_dir:
        print("Error: Provide either --run_dir or --ckpt.")
        sys.exit(1)

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    run_dir_path = Path(args.run_dir) if args.run_dir else None

    main(
        ckpt=ckpt_path,
        run_dir=run_dir_path,
        num_images_each=args.num_each,
        split=args.split,
    )