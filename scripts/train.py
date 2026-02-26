#!/usr/bin/env python3
"""
scripts/train.py

What this script does (high-level)
----------------------------------
This is the "one command" entrypoint to train a model reproducibly.

It:
1) Loads an experiment YAML config
2) Builds train/val datasets and dataloaders (with optional class balancing)
3) Builds the model (e.g., ResNet18) and training components (loss/optimizer)
4) Runs the train + validation loop for N epochs
5) Logs metrics and saves checkpoints:
   - last.pt (latest)
   - best.pt (best validation macro-F1, by default)

Why a script (not a notebook)?
------------------------------
A script runs top-to-bottom with no hidden state, which makes it easier to:
- reproduce runs
- compare experiments
- review changes in git

Inputs:
-----------------
--config: Path to experiment YAML config (e.g., configs/experiments/exp001_resnet18_saliency_ckplus.yaml)

Sample usage:
-------------------
python scripts/train.py --config configs/experiments/exp001_resnet18_saliency_ckplus.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# ----------------------------
# Make `src/` importable even if you didn't pip install -e .
# This lets you run: python scripts/train.py --config ...
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Now we can import xai-lab package modules
from xai_lab.core.engine import evaluate, train_one_epoch
from xai_lab.data.datasets.image_csv import CsvImageDataset, CsvImageDatasetConfig
from xai_lab.data.transforms.image import AugmentConfig, build_transforms
from xai_lab.models.vision.resnet import build_resnet18
from xai_lab.utils.paths import find_project_root, load_yaml

# ----------------------------
# Reproducibility helpers
# ----------------------------
def set_seed(seed: int) -> None:
    """Set seeds for python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # These can make results more deterministic, but may reduce speed:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------
# Class balancing helpers
# ----------------------------
def compute_class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Count samples per class."""
    return np.bincount(labels, minlength=num_classes).astype(np.float32)


def build_weighted_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """
    WeightedRandomSampler makes batches more class-balanced by sampling minority
    classes more often (with replacement).

    Why it helps:
      With imbalanced data, plain shuffling can create batches dominated by the
      majority class, slowing minority learning.

    Important:
      When using a sampler, DataLoader must have shuffle=False.
    """
    counts = compute_class_counts(labels, num_classes=num_classes)
    class_weights = 1.0 / (counts + 1e-8)            # inverse frequency per class so that rare classes are sampled more often. We add 1e-8 to avoid division by zero
    sample_weights = class_weights[labels]           # per-sample weights
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def effective_num_class_weights(labels: np.ndarray, num_classes: int, beta: float = 0.999) -> torch.Tensor:
    """
    Compute class weights using "effective number of samples". This is to avoid the model from overreacting to tiny classes.
    Rare classes stop being rare after a while. Beta controls this.

    Intuition:
      Inverse-frequency weights can get too extreme for small classes.
      Effective-number smooths the weighting, often more stable on small datasets.

    Formula:
      eff_num[c] = 1 - beta^n_c # If class count is small, Beta Counts isn't too tiny, so eff_num is smaller. If class count is large, Beta Counts is tiny, so eff_num approaches 1.
      w[c] = (1 - beta) / eff_num[c]
      then normalize so average weight ≈ 1
    """
    counts = compute_class_counts(labels, num_classes=num_classes)
    eff_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / (eff_num + 1e-8)
    w = w / (w.mean() + 1e-8)
    return torch.tensor(w, dtype=torch.float32)


# ----------------------------
# Config loading
# ----------------------------
def now_stamp() -> str:
    """Timestamp for run folder naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_jsonl_line(path: Path, obj: Dict[str, Any]) -> None:
    """Append one JSON line to a .jsonl file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def save_confusion_matrix_csv(cm: torch.Tensor, out_path: Path) -> None:
    """
    Save confusion matrix to CSV. Rows=true, cols=pred.
    Useful for quickly seeing which classes are confused.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, cm.cpu().numpy().astype(int), fmt="%d", delimiter=",")


# ----------------------------
# Main training routine
# ----------------------------
def main(config_path: Path) -> None:
    repo_root = find_project_root(PROJECT_ROOT)  # robust even if run from other working dirs
    cfg = load_yaml(config_path)

    # ----- Run folder setup -----
    run_name = cfg["run"]["name"]
    base_out = repo_root / cfg["run"].get("output_dir", "artifacts")

    run_id = now_stamp()  # keep using timestamp as the run identifier

    # 1) Run folder: checkpoints + logs + config copy (machine/repro stuff)
    run_dir = base_out / "runs" / run_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2) Reports folder: confusion matrix, plots, XAI images (human-facing outputs)
    reports_dir = base_out / "reports" / run_name / run_id
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config used for this run (important for reproducibility)
    cfg_copy_path = run_dir / "config_used.yaml"
    with open(cfg_copy_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ----- Seeds + device -----
    seed = int(cfg["run"].get("seed", 42))
    set_seed(seed)

    device = get_device()
    print(f"[train] device={device} seed={seed}")
    print(f"[train] run_dir={run_dir}")

    # ----- Data: transforms -----
    input_size = int(cfg["model"].get("input_size", 224))
    aug_cfg = AugmentConfig(
        crop_scale_min=float(cfg["augment"].get("crop_scale_min", 0.85)),
        hflip_p=float(cfg["augment"].get("hflip_p", 0.5)),
        rotation_deg=int(cfg["augment"].get("rotation_deg", 10)),
        jitter_brightness=float(cfg["augment"].get("jitter_brightness", 0.15)),
        jitter_contrast=float(cfg["augment"].get("jitter_contrast", 0.15)),
    )

    train_tfms = build_transforms(input_size=input_size, train=True, aug=aug_cfg)
    val_tfms = build_transforms(input_size=input_size, train=False, aug=aug_cfg)

    # ----- Data: datasets -----
    data_cfg = cfg["data"]
    path_col = data_cfg.get("path_col", "path")
    label_col = data_cfg.get("label_col", "label")

    train_csv = repo_root / data_cfg["train_csv"]
    val_csv = repo_root / data_cfg["val_csv"]

    train_ds = CsvImageDataset(
        CsvImageDatasetConfig(csv_path=train_csv, path_col=path_col, label_col=label_col, project_root=repo_root),
        transform=train_tfms,
    )
    val_ds = CsvImageDataset(
        CsvImageDatasetConfig(csv_path=val_csv, path_col=path_col, label_col=label_col, project_root=repo_root),
        transform=val_tfms,
    )

    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    # Labels are stored in the dataset (we created it from the CSV)
    train_labels = np.array(train_ds.labels, dtype=int)

    # ----- Data: loaders + balancing -----
    batch_size = int(cfg["train"].get("batch_size", 64))
    num_workers = int(cfg["train"].get("num_workers", 4))

    num_classes = int(cfg["model"]["num_classes"])

    balance_cfg = cfg["train"].get("balance", {"method": "none"})
    balance_method = balance_cfg.get("method", "none")

    sampler = None
    train_shuffle = True  # default shuffling for training

    if balance_method == "sampler":
        sampler = build_weighted_sampler(train_labels, num_classes=num_classes)
        train_shuffle = False  # IMPORTANT: shuffle must be False when sampler is used
        print("[balance] Using WeightedRandomSampler (balanced sampling).")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ----- Model -----
    model_name = cfg["model"].get("name", "resnet18")
    pretrained = bool(cfg["model"].get("pretrained", True))

    if model_name == "resnet18":
        model = build_resnet18(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model.name={model_name}. Add a builder in src/xai_lab/models/ and extend train.py.")

    model = model.to(device)

    # ----- Loss (criterion) -----
    # Default: plain CrossEntropyLoss (multi-class single-label classification)
    criterion: nn.Module = nn.CrossEntropyLoss()

    if balance_method == "class_weights":
        method = balance_cfg.get("class_weight_method", "effective_num")
        beta = float(balance_cfg.get("beta", 0.999))

        if method == "effective_num":
            w = effective_num_class_weights(train_labels, num_classes=num_classes, beta=beta).to(device)
        elif method == "inverse_freq":
            counts = compute_class_counts(train_labels, num_classes=num_classes)
            w = torch.tensor(1.0 / (counts + 1e-8), dtype=torch.float32).to(device)
            w = w / (w.mean() + 1e-8)
        else:
            raise ValueError(f"Unknown class_weight_method={method}")

        criterion = nn.CrossEntropyLoss(weight=w)
        print(f"[balance] Using class-weighted loss ({method}).")

    # ----- Optimizer -----
    lr = float(cfg["train"].get("lr", 3e-4))
    weight_decay = float(cfg["train"].get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # (Optional) scheduler can be added later; keep v1 simple.

    # ----- Training loop -----
    epochs = int(cfg["train"].get("epochs", 15))
    metrics_path = run_dir / "metrics.jsonl"

    # Why Macro F1? 
    # Computes F1 for each class and takes the average (weighted by class count)
    best_macro_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_res = evaluate(model, val_loader, criterion, device, num_classes=num_classes)

        # Console log (human friendly)
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_res.loss:.4f} "
            f"val_acc={val_res.accuracy:.4f} "
            f"val_macro_f1={val_res.macro_f1:.4f}"
        )

        # JSONL log (machine friendly; one line per epoch)
        save_jsonl_line(metrics_path, {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_res.loss,
            "val_accuracy": val_res.accuracy,
            "val_macro_f1": val_res.macro_f1,
        })

        # Always save last checkpoint
        last_ckpt = {
            "epoch": epoch,
            "model_name": model_name,
            "num_classes": num_classes,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_macro_f1_so_far": best_macro_f1,
            "config_path": str(cfg_copy_path),
        }
        torch.save(last_ckpt, run_dir / "last.pt")

        # Save best checkpoint by macro-F1 (better than accuracy for imbalance)
        if val_res.macro_f1 > best_macro_f1:
            best_macro_f1 = val_res.macro_f1
            best_epoch = epoch

            best_ckpt = dict(last_ckpt)
            best_ckpt["best_macro_f1_so_far"] = best_macro_f1
            torch.save(best_ckpt, run_dir / "best.pt")

            # Save confusion matrix for the best epoch (helpful diagnostic artifact)
            save_confusion_matrix_csv(val_res.cm, reports_dir / "best_val_confusion_matrix.csv")

    # ----- End summary -----
    summary = {
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "reports_dir": str(reports_dir),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "device": str(device),
    }

    # 1) Main human-facing summary goes in reports/
    with open(reports_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 2) Optional: tiny pointer file in runs/ so you can find reports quickly
    pointer = {"reports_dir": str(reports_dir)}
    with open(run_dir / "reports_pointer.json", "w", encoding="utf-8") as f:
        json.dump(pointer, f, indent=2)

    print("[done] best_epoch=", best_epoch, "best_val_macro_f1=", round(best_macro_f1, 4))
    print("[done] run saved:", run_dir)
    print("[done] reports saved:", reports_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()

    if not Path(args.config).exists():
        raise SystemExit(f"Config not found: {args.config}")

    main(Path(args.config))