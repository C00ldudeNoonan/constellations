from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RunError(Exception):
    pass


@dataclass
class ModelRunResult:
    model_name: str
    materialization: str
    kind: str
    # None derives success/error from `errors`; explicit values represent
    # distinct budget-exceeded and cancelled outcomes.
    status: str | None = None
    backend: str | None = None
    provider: str | None = None
    provider_model: str | None = None
    provider_implementation: str | None = None
    documents_processed: int = 0
    documents_skipped: int = 0
    documents_deleted: int = 0
    rows_written: int = 0
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_failed: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    # Warnings are aggregated by safe message and never change the run status.
    warnings: dict[str, int] = field(default_factory=dict)
    artifact_path: str | None = None
    artifact_version: str | None = None
    training_input: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_metadata: dict[str, Any] | None = None
    serving_resource: dict[str, Any] | None = None
