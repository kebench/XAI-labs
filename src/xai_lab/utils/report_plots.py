"""
xai_lab.utils.report_plots

Purpose
-------
Turn raw run outputs (metrics.jsonl, confusion_matrix.csv) into human-friendly
graphs saved as PNGs.

Why this exists
---------------
- metrics.jsonl is great for logging but hard to "see" trends in.
- graphs make it obvious if training is stable, overfitting, improving, etc.
- confusion matrix plots make class-level performance visible at a glance.

Design goals
------------
- Simple, readable, minimal dependencies (matplotlib/pandas/numpy)
- Safe to call from train.py *or* from a separate command
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xai_lab.utils.paths import load_yaml


def load_metrics_jsonl(metrics_path: Path) -> pd.DataFrame:
    """
    Load a metrics.jsonl file (one JSON object per line) into a DataFrame.

    Expected keys per line (based on our train.py):
      epoch, train_loss, val_loss, val_accuracy, val_macro_f1
    """
    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metrics found in {metrics_path}")
    df = pd.DataFrame(rows).sort_values("epoch")
    return df


def load_confusion_matrix_csv(cm_path: Path) -> np.ndarray:
    """
    Load confusion matrix saved as CSV (rows=true, cols=pred).
    """
    cm = np.loadtxt(cm_path, delimiter=",", dtype=int)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got shape={cm.shape}")
    return cm


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_line(
    x: Sequence[float],
    y: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """
    Helper to plot one simple line chart and save it.
    (One chart per file keeps things easy to read and compare.)
    """
    _ensure_dir(out_path.parent)
    plt.figure()
    plt.plot(x, y)  # default matplotlib styling; no custom colors needed
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[make_report_plots] Saved line chart to {out_path}")


def plot_loss_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Plot train loss and val loss on separate charts (clearer than overlaying).
    """
    epochs = df["epoch"].tolist()
    plot_line(
        epochs, df["train_loss"].tolist(),
        title="Training Loss vs Epoch",
        xlabel="epoch", ylabel="train_loss",
        out_path=out_dir / "train_loss.png",
    )
    plot_line(
        epochs, df["val_loss"].tolist(),
        title="Validation Loss vs Epoch",
        xlabel="epoch", ylabel="val_loss",
        out_path=out_dir / "val_loss.png",
    )


def plot_val_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Plot validation accuracy and validation macro-F1 as separate charts.
    """
    epochs = df["epoch"].tolist()

    if "val_accuracy" in df.columns:
        plot_line(
            epochs, df["val_accuracy"].tolist(),
            title="Validation Accuracy vs Epoch",
            xlabel="epoch", ylabel="val_accuracy",
            out_path=out_dir / "val_accuracy.png",
        )

    if "val_macro_f1" in df.columns:
        plot_line(
            epochs, df["val_macro_f1"].tolist(),
            title="Validation Macro-F1 vs Epoch",
            xlabel="epoch", ylabel="val_macro_f1",
            out_path=out_dir / "val_macro_f1.png",
        )


def plot_confusion_matrix(
    cm: np.ndarray,
    out_path: Path,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = False,
) -> None:
    """
    Plot confusion matrix as an image and save it.

    Parameters
    ----------
    cm:
      Square matrix (C x C). Rows=true class, cols=pred class.
    class_names:
      Optional list of C class names for axis ticks.
    normalize:
      If True, normalize each row to sum to 1 (shows per-class recall patterns).
      If False, show raw counts.

    How to interpret
    ----------------
    - Strong diagonal = good.
    - Off-diagonal hotspots show which classes are confused with which.
    - Normalized view helps when classes are imbalanced.
    """
    _ensure_dir(out_path.parent)

    mat = cm.astype(np.float32)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True) + 1e-8
        mat = mat / row_sums

    plt.figure(figsize=(7, 6))
    plt.imshow(mat, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.colorbar()

    C = cm.shape[0]
    ticks = np.arange(C)
    plt.xticks(ticks)
    plt.yticks(ticks)

    if class_names is not None and len(class_names) == C:
        plt.xticks(ticks, class_names, rotation=45, ha="right")
        plt.yticks(ticks, class_names)

    # Add text labels (counts or proportions)
    for i in range(C):
        for j in range(C):
            val = mat[i, j]
            txt = f"{val:.2f}" if normalize else str(int(cm[i, j]))
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[make_report_plots] Saved confusion matrix chart to {out_path}")

def infer_repo_root_from_anywhere(start: Path) -> Optional[Path]:
    """
    Walk upward from `start` until we find a folder that looks like the repo root.

    Heuristic: a repo root typically contains "src/" (and usually "scripts/").
    This keeps the function usable when run_dir is inside artifacts/.
    """
    current = start.resolve()
    for _ in range(10):
        if (current / "src").exists():
            return current
        current = current.parent
        if current == current.parent:
            break
    return None

def infer_class_names_from_run_dir(
    run_dir: Path,
    label_col: str = "label",
    label_name_col: str = "label_name",
) -> Optional[list[str]]:
    """
    Infer class names from the run's config_used.yaml + one of the split CSVs.

    How it works:
    - Reads run_dir/config_used.yaml to locate the dataset CSVs and num_classes.
    - Opens a split CSV (prefers test.csv, then val.csv, then train.csv).
    - Builds a mapping: label(int) -> label_name(str)
    - Returns a list of class names in label index order: [name_for_0, name_for_1, ...]

    Returns None if we can't infer names (e.g., CSV has no label_name column).
    """
    cfg_path = run_dir / "config_used.yaml"
    if not cfg_path.exists():
        return None

    cfg = load_yaml(cfg_path)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    num_classes = int(model_cfg.get("num_classes", 0))
    if num_classes <= 0:
        return None

    # Find repo root so we can resolve relative CSV paths from config.
    repo_root = infer_repo_root_from_anywhere(run_dir)
    if repo_root is None:
        return None

    # Choose a CSV to read label_name mapping from.
    # Prefer test -> val -> train (any split works if it contains label_name).
    candidate_keys = ["test_csv", "val_csv", "train_csv"]
    csv_path = None
    for k in candidate_keys:
        if k in data_cfg:
            p = (repo_root / data_cfg[k]).resolve()
            if p.exists():
                csv_path = p
                break

    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        return None

    if label_name_col not in df.columns:
        # If label_name isn't present, we can't name classes automatically.
        return None

    # Build mapping label -> name
    pairs = (
        df[[label_col, label_name_col]]
        .dropna()
        .drop_duplicates()
        .copy()
    )
    pairs[label_col] = pairs[label_col].astype(int)
    pairs[label_name_col] = pairs[label_name_col].astype(str)

    label_to_name = dict(zip(pairs[label_col].tolist(), pairs[label_name_col].tolist()))

    # Build ordered list (0..num_classes-1). Fallback to string index if missing.
    class_names = [label_to_name.get(i, str(i)) for i in range(num_classes)]
    return class_names

def generate_report_plots(
    run_dir: Path,
    reports_dir: Path,
    class_names: Optional[Sequence[str]] = None,
    cm_filenames: Sequence[str] = ("best_val_confusion_matrix.csv", "test_confusion_matrix.csv"),
) -> None:
    """
    One-call function: generate all plots for a run.

    Inputs (expected)
    -----------------
    - run_dir/metrics.jsonl
    - reports_dir/<confusion_matrix_csv> (one or more, e.g. val + test)

    Outputs (written to reports_dir)
    --------------------------------
    Curves (if metrics.jsonl exists):
      - train_loss.png
      - val_loss.png
      - val_accuracy.png
      - val_macro_f1.png

    Confusion matrix plots (for every CSV that exists):
      - <stem>.png
      - <stem>_normalized.png

    Notes
    -----
    Safe to call even if some inputs are missing: it prints warnings and continues.
    """
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        df = load_metrics_jsonl(metrics_path)
        plot_loss_curves(df, reports_dir)
        plot_val_metrics(df, reports_dir)
    else:
        print(f"[report_plots] metrics.jsonl not found at {metrics_path}")

    # Plot every confusion matrix CSV that exists (e.g. val + test)
    for cm_file in cm_filenames:
        cm_path = reports_dir / cm_file
        if not cm_path.exists():
            print(f"[report_plots] confusion matrix not found at {cm_path}")
            continue

        cm = load_confusion_matrix_csv(cm_path)

        stem = Path(cm_file).stem  # e.g. "test_confusion_matrix"

        plot_confusion_matrix(
            cm,
            out_path=reports_dir / f"{stem}.png",
            class_names=class_names,
            normalize=False,
        )
        plot_confusion_matrix(
            cm,
            out_path=reports_dir / f"{stem}_normalized.png",
            class_names=class_names,
            normalize=True,
        )