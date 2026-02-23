#!/usr/bin/env python3
"""
scripts/make_report_plots.py

Purpose
-------
Generate human-friendly plots (PNGs) from a completed training run.

This script converts "raw run outputs" into visual reports:
- metrics.jsonl  -> loss curves + accuracy/F1 curves
- confusion matrix CSV -> confusion matrix heatmap (raw + normalized)

Why a separate script?
----------------------
Even though we can call plotting at the end of train.py, having a standalone command is useful:
- Re-generate plots after tweaking plotting code (without retraining).
- Create reports for old runs (e.g., after you pulled from another machine).
- Debug plot generation separately from training.

Typical usage
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

# --------------------------------------------------------------------
# Make `src/` importable even if you didn't do `pip install -e .`
# This matches what we do in train.py.
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import the one-call function that generates all plots.
from xai_lab.utils.report_plots import generate_report_plots


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
    - Reads run_dir/metrics.jsonl (if it exists) and plots curves.
    - Reads reports_dir/best_val_confusion_matrix.csv (if it exists) and plots it.
    """
    # Convert strings to absolute-ish paths (helps avoid confusion when running from different cwd)
    run_dir = run_dir.resolve()
    reports_dir = reports_dir.resolve()

    print(f"[make_report_plots] run_dir={run_dir}")
    print(f"[make_report_plots] reports_dir={reports_dir}")

    # Generate all plots. This function is safe: it prints warnings if inputs are missing.
    generate_report_plots(run_dir=run_dir, reports_dir=reports_dir)

    print("[make_report_plots] done")


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # CLI args: keep it explicit so you can generate plots for any run.
    # ----------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Generate report plots for a training run.")
    parser.add_argument(
        "--run_dir",
        required=True,
        type=str,
        help="Path to artifacts/runs/<experiment>/<run_id>/ (contains metrics.jsonl).",
    )
    parser.add_argument(
        "--reports_dir",
        required=True,
        type=str,
        help="Path to artifacts/reports/<experiment>/<run_id>/ (where plots will be saved).",
    )
    args = parser.parse_args()

    # Run
    main(Path(args.run_dir), Path(args.reports_dir))