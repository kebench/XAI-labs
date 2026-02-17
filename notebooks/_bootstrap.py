from __future__ import annotations

import sys
from pathlib import Path

def bootstrap(src_dirname: str = "src") -> None:
    """
    Adds <repo>/<src_dirname> to sys.path so notebooks can import src/xai_lab/...

    Assumes this file lives in <repo>/notebooks/_bootstrap.py
    """
    repo_root = Path(__file__).resolve().parents[1]  # notebooks/ -> repo root
    src_path = repo_root / src_dirname
    if src_path.exists():
        sys.path.insert(0, str(src_path))