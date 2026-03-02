#!/usr/bin/env python3
"""
scripts/eval.py

Purpose
-------
Evaluate a trained checkpoint on the test split (test.csv) and write a clean report.

Inputs
------
Either:
  --ckpt   artifacts/runs/<experiment>/<run_id>/best.pt
or:
  --run_dir artifacts/runs/<experiment>/<run_id>   (uses best.pt automatically)

Usage
-----
python scripts/eval.py --run_dir artifacts/runs/exp001_ckplus_resnet18/20251015_123456

What it produces (in artifacts/reports/<experiment>/<run_id>/)
--------------------------------------------------------------
- test_summary.json
- test_confusion_matrix.csv

Why this is a script (not a notebook)
-------------------------------------
This creates a repeatable, reviewable evaluation path with no notebook state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Bootstrap imports without requiring `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xai_lab.core.engine import evaluate
from xai_lab.core.metrics import precision_recall_f1_from_cm
from xai_lab.data.datasets.image_csv import CsvImageDataset, CsvImageDatasetConfig
from xai_lab.utils.paths import find_project_root, load_yaml
from xai_lab.utils.imports import import_callable
from xai_lab.models.vision.factory import build_model_from_config
from xai_lab.utils.transform_factory import build_transform_pipeline

# Backward-compat fallback (if transforms section not present yet)
from xai_lab.data.transforms.image import AugmentConfig, build_transforms

def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_ckpt_and_run_dir(ckpt: Optional[Path], run_dir: Optional[Path]) -> Tuple[Path, Path]:
    if run_dir is not None:
        run_dir = run_dir.resolve()
        ckpt_path = (run_dir / "best.pt").resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"best.pt not found under run_dir: {ckpt_path}")
        return ckpt_path, run_dir

    if ckpt is not None:
        ckpt_path = ckpt.resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path, ckpt_path.parent

    raise ValueError("Provide either --ckpt or --run_dir.")


def reports_dir_from_run_dir(repo_root: Path, cfg: Dict[str, Any], run_dir: Path) -> Path:
    run_name = cfg["run"]["name"]
    base_out = repo_root / cfg["run"].get("output_dir", "artifacts")
    run_id = run_dir.name
    out = (base_out / "reports" / run_name / run_id).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out

def main(ckpt: Optional[Path], run_dir: Optional[Path], device_pref: Optional[str]) -> None:
    repo_root = find_project_root(PROJECT_ROOT)

    ckpt_path, resolved_run_dir = resolve_ckpt_and_run_dir(ckpt, run_dir)
    device = get_device(device_pref)

    print(f"[eval] device={device}")
    print(f"[eval] ckpt={ckpt_path}")
    print(f"[eval] run_dir={resolved_run_dir}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Load the exact config used in training
    config_path = Path(checkpoint["config_path"]).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config_used.yaml not found at: {config_path}")
    cfg = load_yaml(config_path)

    # Build test dataset
    data_cfg = cfg["data"]
    test_csv = (repo_root / data_cfg["test_csv"]).resolve()

    path_col = data_cfg.get("path_col", "path")
    label_col = data_cfg.get("label_col", "label")

    test_tfms = build_transform_pipeline(cfg, "eval")

    test_ds = CsvImageDataset(
        CsvImageDatasetConfig(
            csv_path=test_csv,
            path_col=path_col,
            label_col=label_col,
            project_root=repo_root,
        ),
        transform=test_tfms,
    )

    batch_size = int(cfg["train"].get("batch_size", 64))
    num_workers = int(cfg["train"].get("num_workers", 4))

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[data] test={len(test_ds)}")

    # Build model (model-agnostic)
    # Prefer model_cfg from checkpoint if present, else use cfg["model"].
    model_cfg = checkpoint.get("model_cfg", cfg["model"])
    model_cfg = dict(model_cfg)
    model_cfg["num_classes"] = int(checkpoint.get("num_classes", model_cfg["num_classes"]))

    model = build_model_from_config(model_cfg)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    model_name = str(model_cfg.get("name", "unknown"))
    num_classes = int(model_cfg["num_classes"])

    # Loss is for reporting; core metrics matter more (macro-F1)
    criterion: nn.Module = nn.CrossEntropyLoss()

    test_res = evaluate(model, test_loader, criterion, device, num_classes=num_classes)
    precision, recall, f1, macro_f1 = precision_recall_f1_from_cm(test_res.cm)

    print(f"[test] loss={test_res.loss:.4f} acc={test_res.accuracy:.4f} macro_f1={test_res.macro_f1:.4f}")

    # Write reports
    reports_dir = reports_dir_from_run_dir(repo_root, cfg, resolved_run_dir)

    cm_csv_path = reports_dir / "test_confusion_matrix.csv"
    np.savetxt(cm_csv_path, test_res.cm.cpu().numpy().astype(int), fmt="%d", delimiter=",")

    summary = {
        "run_dir": str(resolved_run_dir),
        "ckpt": str(ckpt_path),
        "test_csv": str(test_csv),
        "test_size": len(test_ds),
        "model_name": model_name,
        "num_classes": num_classes,
        "device": str(device),
        "metrics": {
            "loss": test_res.loss,
            "accuracy": test_res.accuracy,
            "macro_f1": test_res.macro_f1,
        },
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "macro_f1": macro_f1,
        },
        "files": {
            "test_confusion_matrix_csv": str(cm_csv_path),
        },
    }

    with open(reports_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[eval] reports_dir={reports_dir}")
    print("[eval] wrote test_summary.json + test_confusion_matrix.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on test.csv.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (e.g., .../best.pt)")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to run directory (uses best.pt)")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"], help="Force device.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    run_dir_path = Path(args.run_dir) if args.run_dir else None

    main(ckpt=ckpt_path, run_dir=run_dir_path, device_pref=args.device)