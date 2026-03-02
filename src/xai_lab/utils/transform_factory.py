from __future__ import annotations

from typing import Any, Dict

from xai_lab.utils.imports import import_callable


def build_transform_pipeline(exp_cfg: Dict[str, Any], stage: str):
    """
    Build a preprocessing/transform pipeline for a given stage.

    stage:
      - "train"   -> random augmentation (usually)
      - "eval"    -> deterministic preprocessing
      - "explain" -> deterministic preprocessing (usually same as eval)

    Expected YAML shape
    -------------------
    transforms:
      train:
        builder: some.module:function
        params: {...}
      eval:
        builder: some.module:function
        params: {...}
      explain: (optional)
        builder: some.module:function
        params: {...}

    Behavior
    --------
    - If stage == "explain" and transforms.explain is missing, we fall back to transforms.eval
      (because explain usually wants deterministic preprocessing).
    - If builder is missing, we raise a clear error (forces config clarity).
    """
    transforms_cfg = exp_cfg.get("transforms", {}) or {}

    stage_cfg = transforms_cfg.get(stage)

    print(f"[transform_factory] stage_cfg: {stage_cfg}")

    # Explain usually wants the same preprocessing as evaluation.
    if stage == "explain" and stage_cfg is None:
        stage_cfg = transforms_cfg.get("eval")

    if not stage_cfg or "builder" not in stage_cfg:
        raise KeyError(
            f"Missing transforms config for stage='{stage}'. "
            f"Expected exp_cfg['transforms']['{stage}']['builder'].\n"
            f"Tip: define transforms.eval and transforms.train in your experiment YAML."
        )

    builder_path = stage_cfg["builder"]
    params = stage_cfg.get("params", {}) or {}

    builder_fn = import_callable(builder_path)
    return builder_fn(**params)