#!/usr/bin/env python3
"""
scripts/make_report_plots.py

Generate plots (PNGs) from a completed training run.

It reads:
- run_dir/metrics.jsonl
- reports_dir/best_val_confusion_matrix.csv (if exists)
- reports_dir/test_confusion_matrix.csv (if exists)

And writes plots into reports_dir.

-------------
python scripts/make_report_plots.py \
  --run_dir     artifacts/runs/<experiment>/<run_id>/ \
  --reports_dir artifacts/reports/<experiment>/<run_id>/

Notes
-----
- run_dir is where reproducibility artifacts live (checkpoints + metrics.jsonl + config copy).
- reports_dir is where human-facing files live (plots, confusion matrix images, XAI outputs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap imports (so xai_lab is importable without pip install -e .)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xai_lab.utils.report_plots import (
    generate_report_plots,
    infer_class_names_from_run_dir,
)


def main(run_dir: Path, reports_dir: Path) -> None:
    """
     Entry point for report plotting.

    Parameters
    ----------
    run_dir:
        Folder containing raw run outputs such as metrics.jsonl (and checkpoints).
    reports_dir:
        Folder where plots will be written.

    Behavior
    --------
    - Infers class names from config_used.yaml + CSV (label_name column).
    - Reads run_dir/metrics.jsonl (if it exists) and plots curves.
    - Reads the confusion matrix files reports_dir (if it exists) and plots confusion matrices.
    """
    run_dir = run_dir.resolve()
    reports_dir = reports_dir.resolve()

    print(f"[make_report_plots] run_dir={run_dir}")
    print(f"[make_report_plots] reports_dir={reports_dir}")

    # Try to infer class names from config_used.yaml + CSV (label_name column).
    class_names = infer_class_names_from_run_dir(run_dir)
    if class_names is not None:
        print(f"[make_report_plots] inferred class names: {class_names}")
    else:
        print("[make_report_plots] no class names inferred (using numeric ticks).")

    generate_report_plots(run_dir=run_dir, reports_dir=reports_dir, class_names=class_names)

    print("[make_report_plots] done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate report plots for a training run.")
    parser.add_argument(
        "--run_dir",
        required=True,
        type=str,
        help="Path to artifacts/runs/<experiment>/<run_id>/",
    )
    parser.add_argument(
        "--reports_dir",
        required=True,
        type=str,
        help="Path to artifacts/reports/<experiment>/<run_id>/",
    )
    args = parser.parse_args()

    if not Path(args.run_dir).exists():
        raise SystemExit(f"Run dir not found: {args.run_dir}")
    if not Path(args.reports_dir).exists():
        raise SystemExit(f"Reports dir not found: {args.reports_dir}")

    main(Path(args.run_dir), Path(args.reports_dir))