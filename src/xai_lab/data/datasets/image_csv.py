from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from xai_lab.utils.paths import resolve_path


@dataclass(frozen=True)
class CsvImageDatasetConfig:
    """
    Dataset config for CSV-based image datasets.

    A split CSV (train/val/test) is the "index": it tells you which file belongs
    to which class label. This config tells the Dataset which columns to read.
    """
    csv_path: Path
    path_col: str = "path"
    label_col: str = "label"
    project_root: Optional[Path] = None


class CsvImageDataset(Dataset):
    """
    Turn rows of a CSV into samples PyTorch can train on.

    Each __getitem__ returns:
      x: image tensor (after transforms)
      y: integer label

    Why this exists (even if you already have train.csv/val.csv/test.csv):
      - CSVs are just lists. This class loads images from those lists.
      - Keeps your pipeline generic (swap datasets by swapping CSV paths).
    """

    def __init__(self, cfg: CsvImageDatasetConfig, transform: Optional[Callable] = None):
        self.cfg = cfg
        self.transform = transform

        df = pd.read_csv(cfg.csv_path)
        df[cfg.path_col] = df[cfg.path_col].astype(str).str.strip()

        self.paths = df[cfg.path_col].tolist()
        self.labels = df[cfg.label_col].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        # Resolve path consistently no matter where the script/notebook is run from.
        img_path = resolve_path(self.paths[idx], project_root=self.cfg.project_root)

        # Convert to RGB:
        # - If the file is grayscale ('L'), this replicates it to 3 channels.
        # - Pretrained ResNet expects 3-channel input, so this is the simplest safe default.
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label
