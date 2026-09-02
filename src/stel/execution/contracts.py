from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RunError(Exception):
    """A model failed.

    `metrics` carries partial run metrics across the error boundary. A
    provider call that failed still spent the time -- `PhaseTimings.phase()`
    credits it deliberately -- and a *slow* failure is exactly the one an
    operator needs attributed. Without this the runner builds a fresh result
    with empty metrics and that timing never reaches `run_results.json`
    (issue #432, PR #460 review).
    """

    def __init__(self, *args: Any, metrics: dict[str, Any] | None = None) -> None:
        super().__init__(*args)
        self.metrics: dict[str, Any] = metrics or {}


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
    # Resolved prompt identity for `llm:` models (issue #303), carried so the
    # run log can group cost and throughput by prompt version.
    prompt_name: str | None = None
    prompt_version: str | None = None
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
