from __future__ import annotations

from pathlib import Path
from typing import Iterable
import yaml


def find_project_root(start: Path | None = None, markers: tuple[str, ...] = ("pyproject.toml", ".git", "requirements.txt")) -> Path:
    """
    Walk upward from `start` (or current working directory) until we find a folder
    containing a marker file/dir like .git, pyproject.toml, or requirements.txt.
    """
    cur = (start or Path.cwd()).resolve()
    for p in (cur, *cur.parents):
        for m in markers:
            if (p / m).exists():
                return p
    # Fallback: just use current directory
    return cur


def resolve_path(path_str: str, project_root: Path | None = None) -> Path:
    """
    Resolve a dataset path string to an absolute Path.
    - If it's already absolute: return it.
    - Else: treat it as relative to project_root.
    """
    p = Path(str(path_str).strip())
    if p.is_absolute():
        return p
    root = project_root or find_project_root()
    return (root / p).resolve()


def resolve_paths(paths: Iterable[str], project_root: Path | None = None) -> list[Path]:
    root = project_root or find_project_root()
    return [resolve_path(p, root) for p in paths]

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
