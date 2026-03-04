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

def reports_dir_from_run_dir(repo_root: Path, cfg: Dict[str, Any], run_dir: Path) -> Path:
    run_name = cfg["run"]["name"]
    base_out = repo_root / cfg["run"].get("output_dir", "artifacts")
    run_id = run_dir.name
    out = (base_out / "reports" / run_name / run_id).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out

def resolve_ckpt_and_run_dir(ckpt: Optional[Path], run_dir: Optional[Path]) -> Tuple[Path, Path]:
    """
    Decide which checkpoint to load and which run directory it belongs to.

    - If run_dir is provided, use run_dir/best.pt by default.
    - If ckpt is provided, infer run_dir as ckpt.parent.
    """
    if run_dir is not None:
        run_dir = run_dir.resolve()
        ckpt_path = (run_dir / "best.pt").resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"best.pt not found under run_dir: {ckpt_path}")
        return ckpt_path, run_dir

    if ckpt is not None:
        ckpt_path = ckpt.resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path, ckpt_path.parent

    raise ValueError("Provide either --ckpt or --run_dir.")