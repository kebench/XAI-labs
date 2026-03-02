from __future__ import annotations

import importlib
from typing import Any, Callable


def import_callable(path: str) -> Callable[..., Any]:
    """
    Import a callable from a string like:
        "package.module:function_name"

    Why this exists:
      YAML can't directly store Python functions, so we store a string reference
      and import it at runtime.

    Raises:
      ValueError if format is wrong
      ImportError/AttributeError if module/function can't be found
    """
    if ":" not in path:
        raise ValueError(f"Expected 'module:function' format, got: {path}")

    module_path, fn_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Imported object is not callable: {path}")
    return fn