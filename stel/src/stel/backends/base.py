from __future__ import annotations

import hashlib
import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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
        """stel release and source identity for incremental invalidation."""
        backend_type = type(self)
        backend_module = inspect.getmodule(backend_type)
        payload = {
            "dbt_ml_version": distribution_version(),
            "backend_class": f"{backend_type.__module__}.{backend_type.__qualname__}",
            "base_source": _source_digest(BaseBackend),
            "backend_class_source": _source_digest(backend_type),
            "backend_module_source": _source_digest(backend_module),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.blake2b(
            canonical.encode(), digest_size=HASH_DIGEST_SIZE
        ).hexdigest()
        return f"stel/{payload['dbt_ml_version']}+backend/{digest}"

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
