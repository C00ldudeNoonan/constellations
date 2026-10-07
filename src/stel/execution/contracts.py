from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..adapters import StateValue


class RunError(Exception):
    """A model failed.

    `metrics` carries partial run metrics across the error boundary. A
    provider call that failed still spent the time -- `PhaseTimings.phase()`
    credits it deliberately -- and a *slow* failure is exactly the one an
    operator needs attributed. Without this the runner builds a fresh result
    with empty metrics and that timing never reaches `run_results.json`
    (issue #432, PR #460 review).

    `progress` carries the row counters the same way, keyed by
    `ModelRunResult` field name (`documents_processed`, `rows_written`, ...).
    A search publish that wrote 1.8 million rows over six hours before its
    index build failed was logged as zero rows in zero seconds (issue #623);
    the serving ledger had the numbers, and the run log -- the place an
    operator looks -- did not.
    """

    def __init__(
        self,
        *args: Any,
        metrics: dict[str, Any] | None = None,
        progress: dict[str, int] | None = None,
    ) -> None:
        super().__init__(*args)
        self.metrics: dict[str, Any] = metrics or {}
        self.progress: dict[str, int] = progress or {}


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
    # The model's own wall-clock span, ISO 8601 UTC. None only on a result the
    # runner never started (a skipped model), where the run log falls back to
    # the invocation's span. Before this every run-log row carried the
    # invocation's timestamps, so a model that failed six hours in read as
    # having started with the first model of the build (issue #623).
    started_at: str | None = None
    completed_at: str | None = None
    errors: list[str] = field(default_factory=list)
    # Warnings are aggregated by safe message and never change the run status.
    warnings: dict[str, int] = field(default_factory=dict)
    artifact_path: str | None = None
    artifact_version: str | None = None
    training_input: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_metadata: dict[str, Any] | None = None
    serving_resource: dict[str, Any] | None = None


def state_for_skipping(
    state: Mapping[str, StateValue], *, reprocess_all: bool
) -> Mapping[str, StateValue]:
    """The state a stage may skip records on -- which `--reprocess-all` empties.

    Deliberately *not* the same mapping the stage reconciles removals against,
    which stays whole. Ignoring state is not the same as clearing it, and the
    difference is load-bearing three times over (issue #655):

    - **Removals still reconcile.** A stage works out what vanished upstream
      from the state it published last time: by a warehouse anti-join against
      `stel_state` where the id column's cast round-trips (issue #428), and
      by a Python set difference over this mapping otherwise. Clearing the
      state empties the left side of both, so a row deleted upstream survives
      in the target -- silently, because an empty removal set is also what a
      run with no removals produces. Mutation-checked: implementing this flag
      as `clear_state(scope)` leaves that row published while every assertion
      about reuse still passes, which is what makes it the plausible wrong
      answer rather than an obviously broken one.
    - **A transform stays incremental.** `_run_incremental_transform` reads an
      empty state baseline over an existing target as "rebuild with a full
      replace", because a child-keyed upsert onto rows no per-parent state
      owns would leave orphan children. Clearing state would trip that branch
      and turn an announced reprocess into a silent full rebuild -- a
      different operation, at a different cost.
    - **A failed run stays resumable.** Nothing on disk is destroyed, so an
      interrupted reprocess leaves the baseline that the next ordinary run
      needs. Clearing first means a crash costs the baseline too.

    The caller keeps both mappings in scope under separate names so which
    question is being asked -- "may I skip this record?" versus "what did the
    last run publish?" -- is visible at each site. `--full-refresh` answers
    both with "nothing" by not fetching state at all; this flag answers only
    the first.
    """
    return {} if reprocess_all else state
