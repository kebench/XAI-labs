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

    # Some error handling to notify the user
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory {src_path} does not exist. Please check your repository structure.")

    if not src_path.is_dir():
        raise NotADirectoryError(f"Source path {src_path} exists but is not a directory.")

    expected_package_dir = src_path / "xai_lab"
    if not expected_package_dir.exists() or not expected_package_dir.is_dir():
        raise ImportError(f"Expected package directory {expected_package_dir} does not exist or is not a directory. Please check your repository structure.")

    # If all are good, add the source directory to sys.path
    sys.path.insert(0, str(src_path))