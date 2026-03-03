from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from torch.utils.data import DataLoader

from xai_lab.data.datasets.image_csv import CsvImageDataset, CsvImageDatasetConfig

# Prefer using generic transform factory if adopted:
try:
    from xai_lab.utils.transform_factory import build_transform_pipeline
    _HAS_TRANSFORM_FACTORY = True
    print("[pipeline_image_csv] Using transform factory.")
except Exception:
    _HAS_TRANSFORM_FACTORY = False
    print("[pipeline_image_csv] Using fallback transforms.")

# Fallback (if transforms.* isn't implemented yet)
from xai_lab.data.transforms.image import AugmentConfig, build_transforms


def _infer_id_to_name(split_csv: Path, label_col: str, label_name_col: str) -> Dict[int, str]:
    """
    Build label_id -> label_name mapping from a split CSV.
    """
    df = pd.read_csv(split_csv)
    if label_name_col not in df.columns or label_col not in df.columns:
        return {}

    tmp = df[[label_col, label_name_col]].dropna().drop_duplicates()
    tmp[label_col] = tmp[label_col].astype(int)
    tmp[label_name_col] = tmp[label_name_col].astype(str).str.strip()
    return dict(tmp.values.tolist())


def build_loaders(
    *,
    exp_cfg: Dict[str, Any],
    repo_root: Path,
    split: str,
    stage: str,
    device_type: str,
    # pipeline params (optional overrides via YAML data_pipeline.params)
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> Tuple[CsvImageDataset, DataLoader, Dict[str, Any]]:
    """
    Image CSV pipeline: builds (dataset, dataloader, meta) for a given split/stage.

    stage:
      - "train": should normally use train transforms (augmentation)
      - "eval": deterministic transforms
      - "explain": deterministic transforms (usually same as eval)

    Returns:
      dataset: CsvImageDataset
      loader:  DataLoader
      meta:    dict (class names etc.)
    """
    data_cfg = exp_cfg["data"]

    csv_key = f"{split}_csv"
    split_csv = (repo_root / data_cfg[csv_key]).resolve()

    path_col = data_cfg.get("path_col", "path")
    label_col = data_cfg.get("label_col", "label")
    label_name_col = data_cfg.get("label_name_col", "label_name")

    # ---- transforms ----
    if _HAS_TRANSFORM_FACTORY:
        # Uses exp_cfg["transforms"][stage] (or eval fallback for explain)
        tfms = build_transform_pipeline(exp_cfg, stage)
    else:
        # Back-compat fallback: deterministic image transforms using input_size from model cfg
        input_size = int(exp_cfg.get("model", {}).get("input_size", 224))
        aug = AugmentConfig()  # no randomness when train=False
        tfms = build_transforms(input_size=input_size, train=(stage == "train"), aug=aug)

    ds = CsvImageDataset(
        CsvImageDatasetConfig(
            csv_path=split_csv,
            path_col=path_col,
            label_col=label_col,
            project_root=repo_root,
        ),
        transform=tfms,
    )

    # Batch size/num_workers: prefer pipeline params, else fall back to train config
    bs = int(batch_size or exp_cfg.get("train", {}).get("batch_size", 64))
    nw = int(num_workers if num_workers is not None else exp_cfg.get("train", {}).get("num_workers", 4))

    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=(stage == "train"),  # sampler logic belongs to train.py, not eval
        num_workers=nw,
        pin_memory=(device_type == "cuda"),
    )

    id_to_name = _infer_id_to_name(split_csv, label_col=label_col, label_name_col=label_name_col)
    class_names = [id_to_name.get(i, str(i)) for i in range(int(exp_cfg["model"]["num_classes"]))] if id_to_name else None

    meta = {
        "split": split,
        "stage": stage,
        "split_csv": str(split_csv),
        "id_to_name": id_to_name,
        "class_names": class_names,
    }
    return ds, loader, meta