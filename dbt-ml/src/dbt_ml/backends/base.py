from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExtractionResult:
    """Output of a single document extraction.

    `fields` holds the projected field values. `warnings` collects
    non-fatal issues surfaced by the backend.
    """

    fields: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class BaseBackend(ABC):
    """Contract every extraction backend implements."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def supported_formats(self) -> list[str]: ...

    @abstractmethod
    def extract(self, path: Path, options: dict[str, Any]) -> ExtractionResult: ...

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
