from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

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


class TransformOptionsValidator(Protocol):
    def __call__(self, options: Mapping[str, Any]) -> None: ...


class TransformDependencyDeclaration(Protocol):
    def __call__(self, options: Mapping[str, Any]) -> Iterable[str]: ...


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
    module = _load_transform_module(module_path, project_dir)
    return _transform_fn(module, module_path)


def validate_transform_contract(
    module_path: str,
    project_dir: Path,
    options: Mapping[str, Any],
    dependencies: Sequence[str] | None = None,
) -> None:
    """Validate a transform's callable and optional configuration hooks.

    A module may expose ``validate_options(options)`` to reject invalid
    configuration during compilation, before execution initializes optional
    SDKs, language models, credentials, or warehouse reads.

    A module may also expose ``declared_dependencies(options)`` returning the
    complete set of dependency model names its options require. Implementing it
    asserts that the options fully determine the transform's inputs, so the
    compiler enforces that ``depends_on`` matches exactly — catching a
    misspelled or stale dependency reference before any model is materialized.
    """
    module = _load_transform_module(module_path, project_dir)
    transform_call_arity(_transform_fn(module, module_path))
    _validate_options_hook(module, module_path, options)
    _validate_declared_dependencies(module, module_path, options, dependencies)


def _validate_options_hook(
    module: ModuleType,
    module_path: str,
    options: Mapping[str, Any],
) -> None:
    validator = getattr(module, "validate_options", None)
    if validator is None:
        return
    if not callable(validator):
        raise AttributeError(
            f"Transform '{module_path}' `validate_options` must be callable"
        )
    if inspect.iscoroutinefunction(validator) or inspect.iscoroutinefunction(
        type(validator).__call__
    ):
        raise TypeError("async transform option validators are not supported")
    cast(TransformOptionsValidator, validator)(dict(options))


def _validate_declared_dependencies(
    module: ModuleType,
    module_path: str,
    options: Mapping[str, Any],
    dependencies: Sequence[str] | None,
) -> None:
    declarer = getattr(module, "declared_dependencies", None)
    if declarer is None or dependencies is None:
        return
    if not callable(declarer):
        raise AttributeError(
            f"Transform '{module_path}' `declared_dependencies` must be callable"
        )
    if inspect.iscoroutinefunction(declarer) or inspect.iscoroutinefunction(
        type(declarer).__call__
    ):
        raise TypeError("async transform dependency declarations are not supported")

    declared = cast(TransformDependencyDeclaration, declarer)(dict(options))
    if isinstance(declared, str) or not isinstance(declared, Iterable):
        raise TypeError(
            f"Transform '{module_path}' `declared_dependencies` must return an "
            "iterable of model names"
        )
    declared_names = tuple(declared)
    if any(not isinstance(name, str) or not name.strip() for name in declared_names):
        raise TypeError(
            f"Transform '{module_path}' `declared_dependencies` must return "
            "non-empty model-name strings"
        )

    missing = sorted(set(declared_names) - set(dependencies))
    extra = sorted(set(dependencies) - set(declared_names))
    if missing or extra:
        details = []
        if missing:
            details.append(f"referenced by options but not in depends_on: {missing}")
        if extra:
            details.append(f"in depends_on but unused by options: {extra}")
        raise ValueError(
            f"transform options and `depends_on` disagree ({'; '.join(details)}). "
            f"Options require exactly {sorted(set(declared_names))}; "
            f"depends_on declares {sorted(set(dependencies))}"
        )


def _load_transform_module(module_path: str, project_dir: Path) -> ModuleType:
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
    return module


def _transform_fn(module: ModuleType, module_path: str) -> TransformFn:
    run_fn = getattr(module, "run", None)
    if run_fn is None or not callable(run_fn):
        raise AttributeError(
            f"Transform '{module_path}' must define a top-level "
            f"`run(deps: dict[str, polars.DataFrame], ctx=None) -> polars.DataFrame`"
        )
    return run_fn  # type: ignore[no-any-return]
