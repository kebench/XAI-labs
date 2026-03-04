#!/usr/bin/env python3
#!/usr/bin/env python3
"""
scripts/train.py (pipeline-based)

What this script does
---------------------
- Loads experiment config (YAML)
- Builds train/val datasets + dataloaders via the configured data pipeline
- Builds model via the configured model factory (model-agnostic)
- Applies training-only features:
    - class balancing (sampler OR class-weighted loss)
    - optimizer + training loop
- Writes outputs:
    runs/: checkpoints + metrics logs + config copy
    reports/: confusion matrix + summary.json (human-facing)

Why pipelines?
--------------
Scripts shouldn't know how to load datasets. That should live in src/xai_lab/data/pipelines/.
Then train/eval/explain all stay clean and dataset-agnostic.

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
import random
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# Bootstrap imports without requiring `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xai_lab.core.engine import evaluate, train_one_epoch
from xai_lab.data.pipelines.factory import build_split_loader
from xai_lab.models.vision.factory import build_model_from_config
from xai_lab.utils.paths import find_project_root
from xai_lab.utils.paths import load_yaml
from xai_lab.utils.device_check import get_device

# ----------------------------
# Small helpers
# ----------------------------
def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_jsonl_line(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def save_confusion_matrix_csv(cm: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, cm.cpu().numpy().astype(int), fmt="%d", delimiter=",")


# ----------------------------
# Balancing helpers
# ----------------------------
def compute_class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.bincount(labels, minlength=num_classes).astype(np.float32)


def build_weighted_sampler(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    counts = compute_class_counts(labels, num_classes=num_classes)
    class_weights = 1.0 / (counts + 1e-8)       # inverse frequency
    sample_weights = class_weights[labels]      # per-sample weights
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def effective_num_class_weights(labels: np.ndarray, num_classes: int, beta: float = 0.999) -> torch.Tensor:
    counts = compute_class_counts(labels, num_classes=num_classes)
    eff_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / (eff_num + 1e-8)
    w = w / (w.mean() + 1e-8)
    return torch.tensor(w, dtype=torch.float32)


# ----------------------------
# Main
# ----------------------------
def main(config_path: Path) -> None:
    repo_root = find_project_root(PROJECT_ROOT)
    cfg = load_yaml(config_path)

    # ----- Run folder layout -----
    run_name = cfg["run"]["name"]
    base_out = repo_root / cfg["run"].get("output_dir", "artifacts")
    run_id = now_stamp()

    run_dir = (base_out / "runs" / run_name / run_id).resolve()
    reports_dir = (base_out / "reports" / run_name / run_id).resolve()

    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save config used (important for reproducibility)
    cfg_copy_path = run_dir / "config_used.yaml"
    with open(cfg_copy_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ----- Seed + device -----
    seed = int(cfg["run"].get("seed", 42))
    set_seed(seed)

    device = get_device()
    print(f"[train] device={device} seed={seed}")
    print(f"[train] run_dir={run_dir}")
    print(f"[train] reports_dir={reports_dir}")

    # ----- Build train/val data via pipeline (dataset-agnostic) -----
    # The pipeline builder comes from cfg["data_pipeline"]["builder"].
    # It returns (dataset, dataloader, meta).
    train_ds, train_loader_base, train_meta = build_split_loader(
        exp_cfg=cfg,
        repo_root=repo_root,
        split="train",
        stage="train",
        device_type=device.type,
    )
    val_ds, val_loader, val_meta = build_split_loader(
        exp_cfg=cfg,
        repo_root=repo_root,
        split="val",
        stage="eval",
        device_type=device.type,
    )

    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    # ----- Model (model-agnostic) -----
    model_cfg = dict(cfg["model"])
    model = build_model_from_config(model_cfg).to(device)

    # ----- Loss (criterion) + balancing options -----
    num_classes = int(model_cfg["num_classes"])
    balance_cfg = cfg.get("train", {}).get("balance", {}) or {}
    balance_method = str(balance_cfg.get("method", "none")).lower()

    # Use labels from the pipeline meta - this is dataset-agnostic
    labels_obj = train_meta.get("labels", None)
    if labels_obj is None:
        raise KeyError("Pipeline meta is missing 'labels'. Add meta['labels'] in the pipeline builder.")
    train_labels = np.array(labels_obj, dtype=int)
    
    sampler = None
    criterion: nn.Module = nn.CrossEntropyLoss()

    # If sampler is enabled, we rebuild the train loader to use sampler instead of shuffle.
    if balance_method == "sampler":
        sampler = build_weighted_sampler(train_labels, num_classes=num_classes)
        print("[balance] Using WeightedRandomSampler (balanced sampling).")

        # Recreate DataLoader using the same base loader settings but with sampler.
        train_loader = DataLoader(
            train_ds,
            batch_size=train_loader_base.batch_size,
            sampler=sampler,
            shuffle=False,  # IMPORTANT: must be False when sampler is set
            num_workers=train_loader_base.num_workers,
            pin_memory=train_loader_base.pin_memory,
        )
    else:
        # Use the pipeline-provided loader as-is (usually shuffle=True for train stage)
        train_loader = train_loader_base

    if balance_method == "class_weights":
        method = str(balance_cfg.get("class_weight_method", "effective_num")).lower()
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
    train_cfg = cfg["train"]
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ----- Training loop -----
    epochs = int(train_cfg.get("epochs", 15))
    metrics_path = run_dir / "metrics.jsonl"

    best_macro_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_res = evaluate(model, val_loader, criterion, device, num_classes=num_classes)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_res.loss:.4f} "
            f"val_acc={val_res.accuracy:.4f} "
            f"val_macro_f1={val_res.macro_f1:.4f}"
        )

        save_jsonl_line(metrics_path, {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_res.loss,
            "val_accuracy": val_res.accuracy,
            "val_macro_f1": val_res.macro_f1,
        })

        # Always save "last"
        last_ckpt = {
            "epoch": epoch,
            "model_name": str(model_cfg.get("name", "")),
            "model_cfg": model_cfg,                 # helps eval/explain rebuild model without guessing
            "num_classes": num_classes,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_macro_f1_so_far": best_macro_f1,
            "config_path": str(cfg_copy_path),
        }
        torch.save(last_ckpt, run_dir / "last.pt")

        # Save "best" by val macro-F1
        if val_res.macro_f1 > best_macro_f1:
            best_macro_f1 = val_res.macro_f1
            best_epoch = epoch

            best_ckpt = dict(last_ckpt)
            best_ckpt["best_macro_f1_so_far"] = best_macro_f1
            torch.save(best_ckpt, run_dir / "best.pt")

            # Confusion matrix is a report artifact -> store in reports/
            save_confusion_matrix_csv(val_res.cm, reports_dir / "best_val_confusion_matrix.csv")

    # ----- Summary (human-facing) -----
    summary = {
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "reports_dir": str(reports_dir),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "device": str(device),
    }

    with open(reports_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Optional pointer file inside run_dir
    with open(run_dir / "reports_pointer.json", "w", encoding="utf-8") as f:
        json.dump({"reports_dir": str(reports_dir)}, f, indent=2)

    print("[done] best_epoch=", best_epoch, "best_val_macro_f1=", round(best_macro_f1, 4))
    print("[done] run saved:", run_dir)
    print("[done] reports saved:", reports_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    args = parser.parse_args()

    main(Path(args.config))