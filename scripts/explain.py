#!/usr/bin/env python3
"""
scripts/explain.py (Modular XAI explanations)

What this script does
---------------------
1) Loads a trained checkpoint (best.pt) from a run folder
2) Loads the exact config used for that run (config_used.yaml)
3) Loads a method-specific explainer config (e.g., configs/explainers/saliency.yaml)
4) Runs inference on the specified split to get predictions + confidence
5) Selects a small set of samples to explain:
   - high-confidence (model very sure)
   - low-confidence  (model least sure; "harder" cases)
6) Generates explanations using the configured method (saliency, Grad-CAM, etc.)
7) Saves visualization triplets (input, heatmap, overlay) to:
   artifacts/reports/<experiment>/<run_id>/xai/<method>/

Supported XAI methods
---------------------
- Saliency: Vanilla input gradients |d(score_class)/d(input_pixels)|
- Grad-CAM: Coarse spatial maps using conv layer gradients

Configuration
-------------
Method configs in configs/explainers/ define:
- name: "saliency" or "gradcam"
- target: "pred" (explain predicted class) or "true" (explain ground truth)
- selection: split and k_each (how many samples per confidence bucket)
- params: method-specific parameters (e.g., target_layer_path for Grad-CAM)

Inputs:
-------
--ckpt: Path to checkpoint (e.g., artifacts/runs/exp001_ckplus_resnet18/20260223_151702/best.pt)
--run_dir: Alternative to --ckpt; without the checkpoint filename
--method_config: Path to explainer config (required)

Sample usage:
-------------
python scripts/explain.py --run_dir artifacts/runs/exp001_ckplus_resnet18/20260223_151702 --method_config configs/explainers/saliency.yaml
python scripts/explain.py --run_dir artifacts/runs/exp001_ckplus_resnet18/20260303_153317 --method_config configs/explainers/gradcam.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xai_lab.data.pipelines.factory import build_split_loader
from xai_lab.utils.paths import find_project_root, load_yaml, reports_dir_from_run_dir, resolve_ckpt_and_run_dir
from xai_lab.data.datasets.image_csv import CsvImageDataset, CsvImageDatasetConfig
from xai_lab.data.transforms.image import AugmentConfig, build_transforms
from xai_lab.explainers.registry import build_explainer_from_config
from xai_lab.utils.vis import denorm_to_uint8, save_triplet
from xai_lab.utils.device_check import get_device

# optional but recommended if you added it:
from xai_lab.models.vision.factory import build_model

@torch.no_grad()
def infer_predictions_and_confidence(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    softmax = nn.Softmax(dim=1)

    all_pred: List[int] = []
    all_true: List[int] = []
    all_conf: List[float] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = softmax(logits)
        pred = probs.argmax(dim=1)
        conf = probs.gather(1, pred.view(-1, 1)).view(-1)

        all_pred.extend(pred.cpu().tolist())
        all_true.extend(y.cpu().tolist())
        all_conf.extend(conf.cpu().tolist())

    return np.array(all_pred), np.array(all_true), np.array(all_conf)


def select_high_low(conf: np.ndarray, k_each: int) -> Tuple[List[int], List[int]]:
    if len(conf) == 0:
        return [], []
    k_each = min(k_each, len(conf))
    idx_sorted = np.argsort(conf)
    low = idx_sorted[:k_each].tolist()
    high = idx_sorted[-k_each:][::-1].tolist()
    return high, low


def main(run_dir: Optional[Path], ckpt: Optional[Path], method_config: Path) -> None:
    repo_root = find_project_root(PROJECT_ROOT)
    device = get_device()
    ckpt_path, resolved_run_dir = resolve_ckpt_and_run_dir(ckpt, run_dir)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    print(f"[explain] device={device}")
    print(f"[explain] ckpt={ckpt_path}")
    print(f"[explain] run_dir={resolved_run_dir}")

    #Load experiment config
    exp_cfg_path = Path(checkpoint["config_path"]).resolve()
    exp_cfg = load_yaml(exp_cfg_path)

    # Load explainer config
    method_cfg = load_yaml(method_config)
    explainer = build_explainer_from_config(method_cfg)

    # Set directories for results of explainer
    reports_dir = reports_dir_from_run_dir(repo_root, exp_cfg, resolved_run_dir)
    subdir = method_cfg.get("output", {}).get("subdir", f"xai/{method_cfg['name']}")
    out_dir = (reports_dir / subdir).resolve()
    (out_dir / "high_conf").mkdir(parents=True, exist_ok=True)
    (out_dir / "low_conf").mkdir(parents=True, exist_ok=True)

    split = method_cfg.get("selection", {}).get("split", "test")
    k_each = int(method_cfg.get("selection", {}).get("k_each", 8))
    target_mode = method_cfg.get("target", "pred")  # pred|true

    data_cfg = exp_cfg["data"]
    split_csv = (repo_root / data_cfg[f"{split}_csv"]).resolve()
    label_col = data_cfg.get("label_col", "label")
    label_name_col = data_cfg.get("label_name_col", "label_name")

    split_df = pd.read_csv(split_csv)
    id_to_name = {}
    if label_name_col in split_df.columns:
        tmp = split_df[[label_col, label_name_col]].dropna().drop_duplicates()
        tmp[label_col] = tmp[label_col].astype(int)
        tmp[label_name_col] = tmp[label_name_col].astype(str).str.strip()
        id_to_name = dict(tmp.values.tolist())

    # Build the dataset, data loader, and labels
    ds, loader, meta = build_split_loader(exp_cfg, repo_root, split=split, stage="explain", device_type=device.type)

    model_name = checkpoint.get("model_name", exp_cfg["model"].get("name", "resnet18"))
    num_classes = int(checkpoint.get("num_classes", exp_cfg["model"]["num_classes"]))
    pretrained = bool(exp_cfg["model"].get("pretrained", True))

    # Create the model
    model = build_model(name=model_name, num_classes=num_classes, pretrained=pretrained)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    # Infer predictions and select highest and lowest confidence based on the predictions
    # TO DO: Might have to make this callable via yaml config instead--let's see how it goes
    pred, true, conf = infer_predictions_and_confidence(model, loader, device)
    high_idx, low_idx = select_high_low(conf, k_each=k_each)

    print(f"[select] {split}: total={len(ds)}")
    print(f"[select] high_conf={len(high_idx)} low_conf={len(low_idx)}")

    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    def explain_index(index: int, bucket: str):
        x, y_true = ds[index]             # [3,H,W]
        x1 = x.unsqueeze(0).to(device)    # [1,3,H,W]

        y_pred = int(pred[index])
        y_true = int(y_true)
        y_conf = float(conf[index])

        target_class = y_pred if target_mode == "pred" else y_true
        heat = explainer.explain(model, x1, target_class=target_class)  # [H,W]
        heat_np = heat.detach().cpu().numpy()

        img_uint8 = denorm_to_uint8(x)

        true_name = id_to_name.get(y_true, str(y_true))
        pred_name = id_to_name.get(y_pred, str(y_pred))
        target_name = id_to_name.get(target_class, str(target_class))

        stem = Path(ds.paths[index]).stem
        out_file = out_dir / bucket / f"{index:04d}_{stem}.png"

        title = (
            f"{method_cfg['name']} | {bucket} | idx={index} | "
            f"true={y_true}:{true_name} pred={y_pred}:{pred_name} "
            f"target={target_class}:{target_name} conf={y_conf:.3f}"
        )

        save_triplet(out_file, img_uint8=img_uint8, heat01=heat_np, title=title)

        rec = {
            "method": method_cfg["name"],
            "split": split,
            "bucket": bucket,
            "index": index,
            "path": ds.paths[index],
            "true": y_true,
            "pred": y_pred,
            "target_class": target_class,
            "confidence": y_conf,
            "output_png": str(out_file),
        }
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    for idx in high_idx:
        explain_index(idx, "high_conf")
    for idx in low_idx:
        explain_index(idx, "low_conf")

    print(f"[done] wrote explanations to: {out_dir}")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate XAI explanations for a trained run.")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--method_config", type=str, required=True)
    args = parser.parse_args()

    if not args.ckpt and not args.run_dir:
        print("Error: Provide either --run_dir or --ckpt.")
        sys.exit(1)

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    run_dir_path = Path(args.run_dir) if args.run_dir else None

    main(
        run_dir=Path(args.run_dir) if args.run_dir else None,
        ckpt=Path(args.ckpt) if args.ckpt else None,
        method_config=Path(args.method_config),
    )