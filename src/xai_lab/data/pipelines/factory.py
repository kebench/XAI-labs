from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from xai_lab.utils.imports import import_callable


def get_pipeline_builder(exp_cfg: Dict[str, Any]) -> Callable[..., Tuple[Any, Any, Dict[str, Any]]]:
    """
    Returns the pipeline builder callable.

    Expected YAML:
      data_pipeline:
        builder: "some.module:function"
        params: {...}

    The returned callable must have signature:
      build_loaders(exp_cfg, repo_root, split, stage, device_type, **params)
        -> (dataset, dataloader, meta)
    """
    dp = exp_cfg.get("data_pipeline", {}) or {}
    builder_path = dp.get("builder")
    if not builder_path:
        raise KeyError(
            "Missing data_pipeline.builder in config.\n"
            "Example:\n"
            "data_pipeline:\n"
            "  builder: xai_lab.data.pipelines.image_csv:build_loaders\n"
            "  params: {}\n"
        )
    return import_callable(builder_path)


def build_split_loader(
    exp_cfg: Dict[str, Any],
    repo_root: Path,
    split: str,
    stage: str,
    device_type: str,
):
    """
    Generic entrypoint used by scripts like eval.py / explain.py / train.py.

    split:  "train" | "val" | "test"
    stage:  "train" | "eval" | "explain"
    """
    dp = exp_cfg.get("data_pipeline", {}) or {}
    params = dp.get("params", {}) or {}

    builder = get_pipeline_builder(exp_cfg)
    return builder(
        exp_cfg=exp_cfg,
        repo_root=repo_root,
        split=split,
        stage=stage,
        device_type=device_type,
        **params,
    )