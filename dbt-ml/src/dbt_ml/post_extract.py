"""Project-local extraction-time field derivation.

A post-extract hook receives one backend result while the verified source
snapshot is still available, then returns only the fields that may cross the
warehouse boundary. Backend warnings and metrics stay owned by the runner and
are preserved automatically.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Protocol

from .backends import ExtractionResult
from .sources import DocumentRef
from .versioning import resolve_module_file


class PostExtractFn(Protocol):
    def __call__(
        self, fields: dict[str, Any], *args: Any
    ) -> Mapping[str, Any]: ...


class PostExtractOptionsValidator(Protocol):
    def __call__(self, options: Mapping[str, Any]) -> None: ...


class PostExtractError(Exception):
    """Safe public failure for a project-local post-extract hook."""


@dataclass(frozen=True)
class PostExtractContext:
    """Per-document context passed to a two-argument post-extract hook."""

    document_id: str
    source_name: str
    source_path: str
    source_uri: str | None
    source_metadata: Mapping[str, Any]
    local_path: Path
    options: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedPostExtract:
    """Validated hook loaded once per extraction model run."""

    module: str
    function: PostExtractFn
    arity: int
    options: Mapping[str, Any]

    def apply(
        self,
        result: ExtractionResult,
        *,
        document: DocumentRef,
        local_path: Path,
    ) -> ExtractionResult:
        context = PostExtractContext(
            document_id=document.document_id,
            source_name=document.source_name,
            source_path=document.relative_path,
            source_uri=document.source_uri,
            source_metadata=MappingProxyType(dict(document.source_metadata or {})),
            local_path=local_path,
            options=self.options,
        )
        try:
            output = (
                self.function(dict(result.fields), context)
                if self.arity == 2
                else self.function(dict(result.fields))
            )
            if not isinstance(output, Mapping):
                raise TypeError("run must return a mapping of field names to values")
            if any(not isinstance(name, str) for name in output):
                raise TypeError("run returned a field name that is not a string")
            fields = dict(output)
        except KeyboardInterrupt:
            raise
        except (Exception, SystemExit):
            # Hook inputs can contain the raw document payload. Do not retain or
            # surface an exception message that may interpolate those values.
            raise PostExtractError(
                f"Post-extract hook '{self.module}' failed"
            ) from None
        return ExtractionResult(
            fields=fields,
            warnings=list(result.warnings),
            metrics=dict(result.metrics),
        )


def load_post_extract(
    module_path: str,
    project_dir: Path,
    options: Mapping[str, Any],
) -> LoadedPostExtract:
    module = _load_project_module(module_path, project_dir)
    function = _post_extract_fn(module, module_path)
    arity = post_extract_call_arity(function)
    _validate_options_hook(module, module_path, options)
    return LoadedPostExtract(
        module=module_path,
        function=function,
        arity=arity,
        options=MappingProxyType(dict(options)),
    )


def validate_post_extract_contract(
    module_path: str,
    project_dir: Path,
    options: Mapping[str, Any],
) -> None:
    """Load and validate a hook before source discovery or warehouse access."""
    load_post_extract(module_path, project_dir, options)


def post_extract_call_arity(function: PostExtractFn) -> int:
    if inspect.iscoroutinefunction(function):
        raise TypeError("async post-extract functions are not supported")
    signature = inspect.signature(function)
    marker = object()
    for arity in (2, 1):
        try:
            signature.bind(*([marker] * arity))
        except TypeError:
            continue
        return arity
    raise TypeError(
        f"run{signature} must accept either (fields) or (fields, ctx) "
        "positional arguments"
    )


def _load_project_module(module_path: str, project_dir: Path) -> ModuleType:
    file_path = resolve_module_file(module_path, project_dir)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Post-extract hook '{module_path}' was not found at {file_path}"
        )
    spec = importlib.util.spec_from_file_location(module_path, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load post-extract hook: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = module
    spec.loader.exec_module(module)
    return module


def _post_extract_fn(module: ModuleType, module_path: str) -> PostExtractFn:
    function = getattr(module, "run", None)
    if function is None or not callable(function):
        raise AttributeError(
            f"Post-extract hook '{module_path}' must define a top-level "
            "`run(fields, ctx=None) -> Mapping[str, Any]`"
        )
    return function  # type: ignore[no-any-return]


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
            f"Post-extract hook '{module_path}' `validate_options` must be callable"
        )
    if inspect.iscoroutinefunction(validator):
        raise TypeError("async post-extract option validators are not supported")
    cast_validator: PostExtractOptionsValidator = validator
    cast_validator(dict(options))
