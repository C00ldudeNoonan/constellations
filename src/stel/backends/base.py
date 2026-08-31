from __future__ import annotations

import ast
import functools
import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from .._distribution import distribution_version
from ..budget import BudgetGuard
from ..hashing import HASH_DIGEST_SIZE


@dataclass
class ExtractionResult:
    """Output of a single document extraction.

    `fields` holds the projected field values. `warnings` collects
    non-fatal issues surfaced by the backend. `metrics` carries numeric
    accounting the runner sums per model (issue #75) — today the llm backend's
    token/call/cache-hit counts; other backends leave it empty.
    """

    fields: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchExtractionOutput:
    items: list[ExtractionResult | Exception]
    metrics: dict[str, Any] = field(default_factory=dict)


class BaseBackend(ABC):
    """Contract every extraction backend implements."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def supported_formats(self) -> list[str]: ...

    @abstractmethod
    def extract(self, path: Path, options: dict[str, Any]) -> ExtractionResult: ...

    def extract_with_budget(
        self,
        path: Path,
        options: dict[str, Any],
        *,
        budget: BudgetGuard | None,
    ) -> ExtractionResult:
        """Extract one document with an optional provider-budget boundary.

        The default preserves the historical check-before-extract and
        charge-after-extract behavior for third-party backends. Provider-backed
        implementations override this so cache lookup can happen before atomic
        call admission.
        """
        if budget is not None:
            budget.ensure_headroom()
        result = self.extract(path, options)
        if budget is not None:
            budget.charge_metrics(result.metrics)
        return result

    def parse_options(self, options: dict[str, Any]) -> dict[str, Any]:
        """Validate this backend's options and return their canonical form."""
        from .options import validate_backend_options

        return validate_backend_options(self.name(), options)

    def extract_batch(
        self, paths: list[Path], options: dict[str, Any]
    ) -> list[ExtractionResult | Exception]:
        """Extract many documents in one call, returning one entry per input
        path (aligned): an ExtractionResult, or the Exception that document
        raised — per-document failures never abort the batch. Default is a
        sequential extract() loop; backends with a native batch path override."""
        out: list[ExtractionResult | Exception] = []
        for path in paths:
            try:
                out.append(self.extract(path, options))
            except Exception as e:
                out.append(e)
        return out

    def extract_batch_with_metrics(
        self,
        paths: list[Path],
        options: dict[str, Any],
        *,
        budget: BudgetGuard | None = None,
    ) -> BatchExtractionOutput:
        del budget
        return BatchExtractionOutput(self.extract_batch(paths, options))

    def version(self) -> str:
        """Parser identity recorded on every extracted row (issue #85), so a
        row can always be traced to the code that produced it. Backends built
        on a parsing library report that library's version."""
        return f"stel/{distribution_version()}"

    def implementation_identity(self) -> str:
        """Source identity for incremental invalidation.

        Deliberately excludes the stel release (issue #363): the source
        digests — the backend class, its module, `BaseBackend`, and every
        stel module the backend module transitively imports — already move
        whenever backend-reachable code changes, so a release that ships no
        such change leaves `extraction:` state intact instead of re-keying
        every corpus. Third-party parser upgrades invalidate separately via
        `parser_identity()`.
        """
        backend_type = type(self)
        backend_module = inspect.getmodule(backend_type)
        payload = {
            "backend_class": f"{backend_type.__module__}.{backend_type.__qualname__}",
            "base_source": _source_digest(BaseBackend),
            "backend_class_source": _source_digest(backend_type),
            "backend_module_source": _source_digest(backend_module),
            "dependency_sources": (
                dict(_dependency_source_digests(backend_module.__name__))
                if backend_module is not None
                else None
            ),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.blake2b(
            canonical.encode(), digest_size=HASH_DIGEST_SIZE
        ).hexdigest()
        return f"stel/backend/{digest}"

    def parser_identity(self) -> str | None:
        """Third-party parser identity for incremental invalidation.

        None means extraction is implemented wholly in stel source, which
        `implementation_identity()` already covers. Backends built on a
        parsing library return that library's identity (typically their
        `version()` string), so a library upgrade re-keys extraction state
        even though no stel source changed (issue #363)."""
        return None

    def validate(self) -> None:
        """Raise if the backend's runtime deps are missing. Default: no-op."""
        return None


def _source_digest(obj: Any) -> str | None:
    if obj is None:
        return None
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):
        return None
    return hashlib.blake2b(
        source.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()


_OWN_PACKAGE = __name__.partition(".")[0]


@functools.cache
def _dependency_source_digests(
    module_name: str, package: str = _OWN_PACKAGE
) -> tuple[tuple[str, str | None], ...]:
    """Source digests of every `package` module `module_name` transitively
    imports, sorted by module name; the origin module itself is excluded
    (its digest is a separate payload field). A dependency whose import or
    source fails to resolve digests as None rather than being dropped, so
    the failure still perturbs identity instead of hiding it.

    Cached per process: source cannot change under a running interpreter,
    and identity is recomputed on every run over every configured model.
    """
    digests: dict[str, str | None] = {}
    seen: set[str] = {module_name}
    queue: list[str] = [module_name]
    while queue:
        name = queue.pop()
        try:
            module = importlib.import_module(name)
        except Exception:
            digests[name] = None
            continue
        if name != module_name:
            digests[name] = _source_digest(module)
        for dep in _imported_package_modules(module, package):
            if dep not in seen:
                seen.add(dep)
                queue.append(dep)
    return tuple(sorted(digests.items()))


def _imported_package_modules(module: ModuleType, package: str) -> set[str]:
    """Module names within `package` that `module`'s source imports, at any
    nesting depth (module-level, function-local, TYPE_CHECKING blocks): a
    lazy or type-only import still names code that shapes behavior."""
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                try:
                    resolved = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""),
                        module.__package__ or "",
                    )
                except ImportError:
                    continue
            else:
                resolved = node.module or ""
            if not resolved:
                continue
            names.add(resolved)
            if resolved != package and not resolved.startswith(package + "."):
                continue
            # `from x import y` may bind the submodule x.y, not an attribute
            # of x; include it only when it actually resolves to a module.
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = f"{resolved}.{alias.name}"
                if candidate in sys.modules or _is_module(candidate):
                    names.add(candidate)
    prefix = package + "."
    return {name for name in names if name == package or name.startswith(prefix)}


def _is_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
