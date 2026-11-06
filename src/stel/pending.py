"""`stel plan --pending`: upstream rows a model's state has never seen (issue #751).

`stel plan` compares code versions against rows already published, so it cannot
see rows that arrived upstream since the last run. An orchestrator deciding
whether to start a run needs exactly that, and until now had to hand-write an
anti-join per model. This is that question asked once, from the data and
`stel_state` and nothing else.

What it reports is **new** rows (an upstream key with no state row) and
**deleted** rows (a state row whose key is gone upstream), both exact. It does
not report **changed** rows: a changed row has a state row, telling it from an
unchanged one needs the row content fingerprint, and embed hashes the whole
upstream row in Python while chunk and transform digest rows the same way. None
of that is computable from key and layout columns, so `changed_rows` is null
and `changed_reason` says why. Reading the wide columns is the cost the issue
forbids at sensor frequency.

Read-only and cheap by construction: both statements project the key, the
partition column and the first cluster column, plus stel's state table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from .adapters import AdapterError, StateScope, WarehouseAdapter, create_adapter
from .adapters.base import PendingUpstreamRequest
from .compiler import validate_project_contract, validate_warehouse_capabilities
from .config.loader import load_project
from .config.model import ModelConfig
from .dag import parse_ref
from .execution.warehouse import warehouse_options
from .plan import PlanError
from .profile import resolve_profile
from .transforms.runner import load_incremental_contract
from .vocabulary_sources import resolve_warehouse_vocabularies

PENDING_FILENAME = "pending.json"
PENDING_SCHEMA_VERSION = 1

#: `stel plan --pending` exit codes. 1 is Click's, for an error.
EXIT_NONE_PENDING = 0
EXIT_PENDING = 3

PendingStatus = Literal["pending", "none", "unsupported", "upstream_missing", "dry_run"]
_STATUSES: tuple[PendingStatus, ...] = (
    "pending",
    "none",
    "unsupported",
    "upstream_missing",
    "dry_run",
)

CHANGED_REASON = (
    "changed rows need the row content fingerprint, which reads every column; "
    "this report reads the key, partition and cluster columns only"
)


@dataclass(frozen=True)
class PendingSource:
    """Where one model's state keys come from."""

    upstream: str
    key_column: str


@dataclass(frozen=True)
class ModelPending:
    name: str
    kind: str
    status: PendingStatus
    reason: str
    upstream: str | None = None
    key_column: str | None = None
    new_rows: int | None = None
    partition_column: str | None = None
    partition_min: str | None = None
    partition_max: str | None = None
    cluster_column: str | None = None
    cluster_distinct: int | None = None
    deleted_rows: int | None = None
    estimated_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "reason": self.reason,
            "upstream": self.upstream,
            "key_column": self.key_column,
            "new_rows": self.new_rows,
            "changed_rows": None,
            "changed_reason": CHANGED_REASON,
            "partition_column": self.partition_column,
            "partition_min": self.partition_min,
            "partition_max": self.partition_max,
            "oldest_pending": self.partition_min,
            "cluster_column": self.cluster_column,
            "cluster_distinct": self.cluster_distinct,
            "deleted_rows": self.deleted_rows,
            "estimated_bytes": self.estimated_bytes,
        }


@dataclass(frozen=True)
class PendingReport:
    models: tuple[ModelPending, ...]
    target: dict[str, Any]
    generated_at: str
    dry_run: bool

    @property
    def has_pending(self) -> bool:
        return any(model.status == "pending" for model in self.models)

    @property
    def exit_code(self) -> int:
        return EXIT_PENDING if self.has_pending else EXIT_NONE_PENDING

    def to_dict(self) -> dict[str, Any]:
        counts = dict.fromkeys(_STATUSES, 0)
        for model in self.models:
            counts[model.status] += 1
        return {
            "metadata": {
                "schema_version": PENDING_SCHEMA_VERSION,
                "invocation": "plan --pending",
                "generated_at": self.generated_at,
                "target": self.target,
                "dry_run": self.dry_run,
                "counts": {**counts, "total": len(self.models)},
                "has_pending": self.has_pending,
            },
            "models": [model.to_dict() for model in self.models],
        }


def pending_source(model: ModelConfig, *, project_dir: Path) -> PendingSource | str:
    """The upstream and key column `model` keys its state on, or the reason it
    has none this report can read."""
    if model.embed is not None:
        if not model.depends_on or len(model.depends_on) != 1:
            return "embed model does not declare exactly one upstream"
        return PendingSource(parse_ref(model.depends_on[0]), model.embed.id_field)
    if model.chunk is not None:
        if not model.depends_on:
            return "chunk model declares no upstream"
        return PendingSource(parse_ref(model.depends_on[0]), "document_id")
    if model.transform is not None:
        if model.materialization != "incremental":
            return "a full-refresh transform keeps no per-row state"
        if model.transform.module is None:
            return "a transform without a Python module keeps no per-row state"
        dependencies = [parse_ref(ref) for ref in model.depends_on or []]
        contract = load_incremental_contract(
            model.transform.module, project_dir, model.transform.options
        )
        if contract is None or not dependencies:
            return "transform declares no incremental contract"
        return PendingSource(
            contract.resolve_parent_source(dependencies), contract.parent_source_key
        )
    return f"a {model.kind_label()} model keeps no per-row upstream state"


def _render(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _pending_for_model(
    model: ModelConfig,
    source: PendingSource,
    *,
    adapter: WarehouseAdapter,
    dry_run: bool,
) -> ModelPending:
    kind = model.kind_label()
    if not adapter.relation_exists(source.upstream):
        return ModelPending(
            model.name,
            kind,
            "upstream_missing",
            f"upstream '{source.upstream}' does not exist yet",
            upstream=source.upstream,
            key_column=source.key_column,
        )
    available = set(adapter.read_table_schema(source.upstream).columns)
    if source.key_column not in available:
        raise PlanError(
            f"Model '{model.name}': upstream '{source.upstream}' has no "
            f"`{source.key_column}` column. Available: {sorted(available)}"
        )
    layout = adapter.layout_pruning_columns(warehouse_options(adapter, model))
    partition = layout.range_column if layout.range_column in available else None
    cluster = next(
        (column for column in layout.membership_columns if column in available),
        None,
    )
    summary = adapter.pending_upstream(
        PendingUpstreamRequest(
            scope=StateScope(model.name),
            upstream=source.upstream,
            key_column=source.key_column,
            partition_column=partition,
            cluster_column=cluster,
        ),
        dry_run=dry_run,
    )
    if dry_run:
        return ModelPending(
            model.name,
            kind,
            "dry_run",
            "dry run: nothing was read",
            upstream=source.upstream,
            key_column=source.key_column,
            partition_column=partition,
            cluster_column=cluster,
            estimated_bytes=summary.estimated_bytes,
        )
    pending = summary.new_rows > 0
    return ModelPending(
        model.name,
        kind,
        "pending" if pending else "none",
        (
            f"{summary.new_rows} upstream row(s) have no state row"
            if pending
            else "every upstream key has a state row"
        ),
        upstream=source.upstream,
        key_column=source.key_column,
        new_rows=summary.new_rows,
        partition_column=partition,
        partition_min=_render(summary.partition_min),
        partition_max=_render(summary.partition_max),
        cluster_column=cluster,
        cluster_distinct=summary.cluster_distinct,
        deleted_rows=summary.deleted_rows,
        estimated_bytes=summary.estimated_bytes,
    )


def pending_models(
    planned: list[ModelConfig],
    *,
    project_dir: Path,
    adapter: WarehouseAdapter,
    dry_run: bool,
) -> list[ModelPending]:
    results: list[ModelPending] = []
    for model in planned:
        source = pending_source(model, project_dir=project_dir)
        if isinstance(source, str):
            results.append(
                ModelPending(model.name, model.kind_label(), "unsupported", source)
            )
            continue
        try:
            results.append(
                _pending_for_model(model, source, adapter=adapter, dry_run=dry_run)
            )
        except AdapterError as error:
            raise PlanError(f"Model '{model.name}': {error}") from error
    return results


def pending_project(
    project_dir: Path,
    *,
    select: str | None,
    exclude: str | None,
    target: str | None,
    profiles_dir: Path | None,
    dry_run: bool,
) -> PendingReport:
    """Report pending upstream rows for every selected model.

    Same preflight as `plan` and `run`, so a bad configuration fails here with
    the message it would fail a run with."""
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    adapter = create_adapter(resolved.warehouse, project_dir=project_dir)
    selected = dag.select_models(select=select, exclude=exclude)
    models_by_name = {model.name: model for model in models}
    planned = [models_by_name[name] for name in selected]
    validate_warehouse_capabilities(planned, adapter)
    resolve_warehouse_vocabularies(project, sources, models, project_dir, adapter)
    with adapter:
        results = pending_models(
            planned, project_dir=project_dir, adapter=adapter, dry_run=dry_run
        )
    warehouse = resolved.warehouse
    return PendingReport(
        models=tuple(results),
        target={
            "profile": resolved.profile_name,
            "name": resolved.target_name,
            "adapter_type": warehouse.type,
            "schema": warehouse.schema_name,
            "catalog": warehouse.catalog_name(),
            "location": warehouse.storage_location(),
        },
        generated_at=datetime.now(UTC).isoformat(),
        dry_run=dry_run,
    )


def write_pending_artifact(project_dir: Path, report: PendingReport) -> Path:
    project = load_project(project_dir)[0]
    target_dir = (project_dir / project.target_path).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / PENDING_FILENAME
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return out


def format_pending_table(report: PendingReport) -> list[str]:
    lines = [f"{'model':<32} {'status':<17} {'new':>10} {'deleted':>9}  span"]
    for model in report.models:
        span = (
            f"{model.partition_min} .. {model.partition_max}"
            if model.partition_min is not None
            else ""
        )
        new = "-" if model.new_rows is None else str(model.new_rows)
        deleted = "-" if model.deleted_rows is None else str(model.deleted_rows)
        lines.append(
            f"{model.name:<32} {model.status:<17} {new:>10} {deleted:>9}  {span}"
        )
    lines.append("")
    lines.append(f"changed rows are not reported: {CHANGED_REASON}.")
    estimates = [
        m.estimated_bytes for m in report.models if m.estimated_bytes is not None
    ]
    if estimates:
        lines.append(f"estimated bytes billed: {sum(estimates):,}")
    return lines
