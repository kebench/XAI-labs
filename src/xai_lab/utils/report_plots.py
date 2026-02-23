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


def generate_report_plots(
    run_dir: Path,
    reports_dir: Path,
    class_names: Optional[Sequence[str]] = None,
    cm_filename: str = "best_val_confusion_matrix.csv",
) -> None:
    """
    One-call function: generate all plots for a run.

    Expected inputs:
      - run_dir/metrics.jsonl
      - reports_dir/<cm_filename> (default: best_val_confusion_matrix.csv)

    Outputs (written to reports_dir):
      - train_loss.png
      - val_loss.png
      - val_accuracy.png
      - val_macro_f1.png
      - confusion_matrix.png
      - confusion_matrix_normalized.png
    """
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        df = load_metrics_jsonl(metrics_path)
        plot_loss_curves(df, reports_dir)
        plot_val_metrics(df, reports_dir)
    else:
        # Not fatal — you might call this before a run finishes.
        print(f"[report_plots] metrics.jsonl not found at {metrics_path}")

    cm_path = reports_dir / cm_filename
    if cm_path.exists():
        cm = load_confusion_matrix_csv(cm_path)
        plot_confusion_matrix(cm, reports_dir / "confusion_matrix.png", class_names=class_names, normalize=False)
        plot_confusion_matrix(cm, reports_dir / "confusion_matrix_normalized.png", class_names=class_names, normalize=True)
    else:
        print(f"[report_plots] confusion matrix not found at {cm_path}")