from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


class BaseBackend(ABC):
    """Contract every extraction backend implements."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def supported_formats(self) -> list[str]: ...

    @abstractmethod
    def extract(self, path: Path, options: dict[str, Any]) -> ExtractionResult: ...

    def extract_batch(
        self, paths: list[Path], options: dict[str, Any]
    ) -> list[ExtractionResult | Exception]:
        """Extract many documents in one call, returning one entry per input
        path (aligned): an ExtractionResult, or the Exception that document
        raised — per-document failures never abort the batch. Default is a
        sequential extract() loop; backends with a native batch path (llm →
        Anthropic Message Batches, issue #75) override."""
        out: list[ExtractionResult | Exception] = []
        for path in paths:
            try:
                out.append(self.extract(path, options))
            except Exception as e:
                out.append(e)
        return out

    def version(self) -> str:
        """Parser identity recorded on every extracted row (issue #85), so a
        row can always be traced to the code that produced it. Backends built
        on a parsing library report that library's version."""
        from importlib.metadata import PackageNotFoundError, version

        try:
            return f"dbt-ml/{version('dbt-ml')}"
        except PackageNotFoundError:
            return "dbt-ml/unknown"

    def validate(self) -> None:
        """Raise if the backend's runtime deps are missing. Default: no-op."""
        return None
