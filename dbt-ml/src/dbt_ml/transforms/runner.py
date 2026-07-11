from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from ..config.profile import LLMConfig, WarehouseConfig
from ..versioning import resolve_module_file


@dataclass(frozen=True)
class TransformContext:
    """Passed to transforms that declare a second arg.

    Lets transforms reach the resolved profile (e.g. LLM config) and the
    per-model `options` block (from `transform.options:` in YAML) without
    hard-coding values in the transform module.
    """

    project_dir: Path
    profile_name: str
    target_name: str
    warehouse: WarehouseConfig
    llm: LLMConfig | None
    options: dict[str, Any] = field(default_factory=dict)


class TransformFn(Protocol):
    def __call__(self, deps: dict[str, pl.DataFrame], *args: Any) -> pl.DataFrame: ...


def transform_call_arity(transform_fn: TransformFn) -> int:
    if inspect.iscoroutinefunction(transform_fn):
        raise TypeError("async transform functions are not supported")
    signature = inspect.signature(transform_fn)
    marker = object()
    for arity in (2, 1):
        try:
            signature.bind(*([marker] * arity))
        except TypeError:
            continue
        return arity
    raise TypeError(
        f"run{signature} must accept either (deps) or (deps, ctx) positional arguments"
    )


def load_transform(module_path: str, project_dir: Path) -> TransformFn:
    """Load a transform module's `run` callable.

    Resolution order:
        1. Project-local file (so users can override built-ins by writing
           their own `transforms/<name>.py`).
        2. Installed Python package (lets us ship built-ins like
           `dbt_ml.text.transforms.text_stats`).
    """
    file_path = resolve_module_file(module_path, project_dir)
    if file_path.exists():
        spec = importlib.util.spec_from_file_location(module_path, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load transform module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path] = module
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise FileNotFoundError(
                f"Transform '{module_path}' not found as a project file "
                f"({file_path}) or as an importable Python module: {e}"
            ) from e

    run_fn = getattr(module, "run", None)
    if run_fn is None or not callable(run_fn):
        raise AttributeError(
            f"Transform '{module_path}' must define a top-level "
            f"`run(deps: dict[str, polars.DataFrame], ctx=None) -> polars.DataFrame`"
        )
    return run_fn  # type: ignore[no-any-return]
