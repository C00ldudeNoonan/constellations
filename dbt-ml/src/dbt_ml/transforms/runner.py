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

from ..budget import BudgetLedger
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
    # The invocation-wide LLM budget ledger, threaded so a `uses_llm` transform
    # charges and enforces `llm.budget` like the native `llm:` kind (issue #240).
    run_budget: BudgetLedger | None = None


@dataclass(frozen=True)
class IncrementalContract:
    """How an incremental one-to-many transform maps input parents to output
    child rows (issue #218). Returned by a transform module's optional
    ``declared_incremental_contract(options)`` hook; a transform that does not
    implement the hook stays ``materialization: full``.

    The runner uses it to skip unchanged parents and replace a changed parent's
    children by deleting on the parent key and upserting on the child key:

    parent_key         output column carrying the parent identity — the delete
                       scope (e.g. ``document_id``).
    child_key          output column, the stable unique child-row identity —
                       the upsert scope (e.g. ``token_id``).
    parent_source      dependency model whose rows define the parents; it is the
                       only dependency filtered to changed/new parents. ``None``
                       means the transform has exactly one dependency and that is
                       the parent source (for transforms that take a single
                       upstream by position and cannot name it).
    parent_source_key  the string column in ``parent_source`` identifying the
                       parent.
    reference_deps     dependencies passed whole to the transform; a change to
                       any of them invalidates every parent (folded into each
                       parent's input fingerprint).

    ``parent_source`` and ``reference_deps`` together must account for every
    entry in ``depends_on`` so no input escapes invalidation.
    """

    parent_key: str
    child_key: str
    parent_source_key: str
    parent_source: str | None = None
    reference_deps: tuple[str, ...] = ()

    def resolve_parent_source(self, dependencies: Sequence[str]) -> str:
        """The effective parent-source model name for ``dependencies``. When
        ``parent_source`` is None the sole dependency is used; callers validate
        first, so this assumes a single dependency in that case."""
        if self.parent_source is not None:
            return self.parent_source
        return next(iter(dependencies))

    def validate_against(self, dependencies: Sequence[str]) -> None:
        for name, value in (
            ("parent_key", self.parent_key),
            ("child_key", self.child_key),
            ("parent_source_key", self.parent_source_key),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"IncrementalContract.{name} must be a non-empty string")
        if self.parent_key == self.child_key:
            raise ValueError("IncrementalContract parent_key and child_key must differ")
        refs = tuple(self.reference_deps)
        if any(not isinstance(ref, str) or not ref.strip() for ref in refs):
            raise ValueError(
                "IncrementalContract.reference_deps must be non-empty model-name strings"
            )
        ref_set = set(refs)
        if len(ref_set) != len(refs):
            raise ValueError("IncrementalContract.reference_deps contains duplicates")
        deps = set(dependencies)

        if self.parent_source is None:
            if ref_set:
                raise ValueError(
                    "IncrementalContract with no parent_source cannot declare "
                    "reference_deps; name the parent_source explicitly instead"
                )
            if len(deps) != 1:
                raise ValueError(
                    "IncrementalContract without a parent_source requires exactly one "
                    f"dependency; depends_on is {sorted(deps)}"
                )
            return

        if not self.parent_source.strip():
            raise ValueError("IncrementalContract.parent_source must be non-empty or None")
        if self.parent_source not in deps:
            raise ValueError(
                f"IncrementalContract.parent_source '{self.parent_source}' is not in "
                f"depends_on {sorted(deps)}"
            )
        if self.parent_source in ref_set:
            raise ValueError(
                "IncrementalContract.parent_source must not also be a reference_dep"
            )
        unknown = sorted(ref_set - deps)
        if unknown:
            raise ValueError(
                f"IncrementalContract.reference_deps not in depends_on: {unknown}"
            )
        unclassified = sorted(deps - ({self.parent_source} | ref_set))
        if unclassified:
            raise ValueError(
                "IncrementalContract must account for every dependency; unclassified: "
                f"{unclassified} (make one the parent_source or list it in reference_deps)"
            )


class TransformFn(Protocol):
    def __call__(self, deps: dict[str, pl.DataFrame], *args: Any) -> pl.DataFrame: ...


class TransformOptionsValidator(Protocol):
    def __call__(self, options: Mapping[str, Any]) -> None: ...


class TransformDependencyDeclaration(Protocol):
    def __call__(self, options: Mapping[str, Any]) -> Iterable[str]: ...


class TransformIncrementalContractDeclaration(Protocol):
    def __call__(self, options: Mapping[str, Any]) -> IncrementalContract | None: ...


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
    *,
    materialization: str = "full",
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

    A one-to-many transform may expose ``declared_incremental_contract(options)``
    returning an :class:`IncrementalContract`. It is required for
    ``materialization: incremental`` (issue #218) and validated against
    ``depends_on`` here so a grain mismatch is a compile error, not a wrong-key
    delete at build time.
    """
    module = _load_transform_module(module_path, project_dir)
    transform_call_arity(_transform_fn(module, module_path))
    _validate_options_hook(module, module_path, options)
    _validate_declared_dependencies(module, module_path, options, dependencies)
    _validate_incremental_contract(
        module, module_path, options, dependencies, materialization
    )


def load_incremental_contract(
    module_path: str,
    project_dir: Path,
    options: Mapping[str, Any],
) -> IncrementalContract | None:
    """Resolve a transform's ``declared_incremental_contract(options)`` hook, or
    None when the module does not implement it."""
    module = _load_transform_module(module_path, project_dir)
    return _incremental_contract_hook(module, module_path, options)


def transform_requires_llm(
    module_path: str,
    project_dir: Path,
    options: Mapping[str, Any],
) -> bool:
    """Whether a transform's optional ``requires_llm(options)`` hook reports that
    the configuration needs a resolved ``llm:`` provider. Such a model must set
    ``transform.uses_llm: true`` so the executor resolves inference and the code
    version folds the provider identity. Absent hook → ``False``."""
    module = _load_transform_module(module_path, project_dir)
    hook = getattr(module, "requires_llm", None)
    if hook is None:
        return False
    if not callable(hook):
        raise AttributeError(
            f"Transform '{module_path}' `requires_llm` must be callable"
        )
    return bool(hook(dict(options)))


def _incremental_contract_hook(
    module: ModuleType,
    module_path: str,
    options: Mapping[str, Any],
) -> IncrementalContract | None:
    declarer = getattr(module, "declared_incremental_contract", None)
    if declarer is None:
        return None
    if not callable(declarer):
        raise AttributeError(
            f"Transform '{module_path}' `declared_incremental_contract` must be callable"
        )
    if inspect.iscoroutinefunction(declarer) or inspect.iscoroutinefunction(
        type(declarer).__call__
    ):
        raise TypeError(
            "async transform incremental-contract declarations are not supported"
        )
    contract = cast(TransformIncrementalContractDeclaration, declarer)(dict(options))
    if contract is None:
        return None
    if not isinstance(contract, IncrementalContract):
        raise TypeError(
            f"Transform '{module_path}' `declared_incremental_contract` must return an "
            "IncrementalContract or None"
        )
    return contract


def _validate_incremental_contract(
    module: ModuleType,
    module_path: str,
    options: Mapping[str, Any],
    dependencies: Sequence[str] | None,
    materialization: str,
) -> None:
    contract = _incremental_contract_hook(module, module_path, options)
    if materialization != "incremental":
        return
    if contract is None:
        raise ValueError(
            "`materialization: incremental` requires the transform to declare "
            "`declared_incremental_contract(options)`; only one-to-many child-table "
            "transforms that opt in support incremental materialization"
        )
    if dependencies is None:
        return
    contract.validate_against(dependencies)


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
