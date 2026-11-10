"""`stel plan --pending`: upstream rows a model's state has never seen (issue #751).

`stel plan` compares code versions against rows already published, so it cannot
see rows that arrived upstream since the last run. An orchestrator deciding
whether to start a run needs exactly that, and until now had to hand-write an
anti-join per model. This is that question asked once, from the data and
`stel_state` and nothing else.

What it reports is **new** rows (an upstream key with no state row) and
**deleted** rows (a state row whose key is gone upstream), both exact and both
read from the key, partition and cluster columns plus stel's state table.

**Changed** rows are opt-in (`include_changed`). A changed row has a state row,
and telling it from an unchanged one needs the row content fingerprint, which
embed and chunk compute in Python over the whole upstream row. So the default
leaves `changed_rows` null, with `changed_reason` saying why, and
`include_changed` streams every column of the upstream and compares each row's
fingerprint to the one state recorded. That is the cost a sensor must not pay
every tick, which is why it is a flag, and `--dry-run` prices it first. The
fingerprint functions are the runners' own, so the count cannot drift from what
a run would reprocess. A transform digests rows against reference tables too,
so it reports `changed_rows` null even then.

`changed_rows` counts rows whose *input* fingerprint moved. A row whose only
change is the model's `code_version` is what plain `stel plan` reports.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl

from .adapters import AdapterError, StateScope, WarehouseAdapter, create_adapter
from .adapters.base import PendingUpstreamRequest, ReadPredicate
from .adapters.pending import estimate_table_read_bytes, pending_upstream
from .compiler import validate_project_contract, validate_warehouse_capabilities
from .config.loader import load_project
from .config.model import ModelConfig
from .dag import parse_ref
from .execution import RunError
from .execution.chunk import chunk_input_hash
from .execution.embed import embed_input_fingerprint
from .execution.warehouse import warehouse_options
from .plan import PlanError
from .profile import resolve_profile
from .read_filters import parse_read_filters
from .transforms.runner import load_incremental_contract
from .vocabulary_sources import resolve_warehouse_vocabularies

_SCAN_BATCH_ROWS = 10_000
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
    "this report reads the key, partition and cluster columns only. Pass "
    "--include-changed to read them"
)
TRANSFORM_CHANGED_REASON = (
    "a transform fingerprints its parent rows together with its reference "
    "tables, which this report does not recompute"
)


@dataclass(frozen=True)
class PendingSource:
    """Where one model's state keys come from."""

    upstream: str
    key_column: str
    #: The fingerprint the runner records in state for one upstream row; None
    #: for a kind whose fingerprint is not a function of the row alone.
    fingerprint: Callable[[dict[str, Any]], str] | None = None


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
    changed_rows: int | None = None
    changed_reason: str = CHANGED_REASON

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "reason": self.reason,
            "upstream": self.upstream,
            "key_column": self.key_column,
            "new_rows": self.new_rows,
            "changed_rows": self.changed_rows,
            "changed_reason": None if self.changed_rows is not None else self.changed_reason,
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
    artifact_dir: Path

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
        return PendingSource(
            parse_ref(model.depends_on[0]),
            model.embed.id_field,
            embed_input_fingerprint,
        )
    if model.chunk is not None:
        if not model.depends_on:
            return "chunk model declares no upstream"
        text_field = model.chunk.text_field
        return PendingSource(
            parse_ref(model.depends_on[0]),
            "document_id",
            lambda record: chunk_input_hash(record, text_field=text_field),
        )
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
    include_changed: bool,
    predicates: tuple[ReadPredicate, ...],
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
    schema = adapter.read_table_schema(source.upstream)
    available = set(schema.columns)
    missing_filters = sorted({predicate.column for predicate in predicates} - available)
    if missing_filters:
        raise PlanError(
            f"Model '{model.name}': --read-filter column(s) {', '.join(missing_filters)} "
            f"are not in upstream '{source.upstream}'. Available: {sorted(available)}"
        )
    if source.key_column not in available:
        raise PlanError(
            f"Model '{model.name}': upstream '{source.upstream}' has no "
            f"`{source.key_column}` column. Available: {sorted(available)}"
        )
    key_type = schema.schema[source.key_column]
    if not (key_type == pl.String or key_type.is_integer()):
        # State keys are Python `str(value)`; only strings and integers have
        # the same text in every warehouse's CAST, so any other type would
        # report every row as both new and deleted.
        return ModelPending(
            model.name,
            kind,
            "unsupported",
            f"key column `{source.key_column}` is {key_type}; only string and "
            "integer keys can be compared with state from SQL",
            upstream=source.upstream,
            key_column=source.key_column,
        )
    layout = adapter.layout_pruning_columns(warehouse_options(adapter, model))
    partition = layout.range_column if layout.range_column in available else None
    cluster = next(
        (column for column in layout.membership_columns if column in available),
        None,
    )
    summary = pending_upstream(
        adapter,
        PendingUpstreamRequest(
            scope=StateScope(model.name),
            upstream=source.upstream,
            key_column=source.key_column,
            partition_column=partition,
            cluster_column=cluster,
        ),
        predicates=predicates,
        dry_run=dry_run,
    )
    identity: dict[str, Any] = {
        "upstream": source.upstream,
        "key_column": source.key_column,
        "partition_column": partition,
        "cluster_column": cluster,
    }
    changed_reason = CHANGED_REASON
    if source.fingerprint is None:
        changed_reason = TRANSFORM_CHANGED_REASON
    full_scan = include_changed and source.fingerprint is not None
    scan_bytes = (
        estimate_table_read_bytes(adapter, source.upstream, predicates)
        if full_scan and dry_run
        else None
    )
    if dry_run:
        estimated = summary.estimated_bytes
        if scan_bytes is not None:
            estimated = (estimated or 0) + scan_bytes
        return ModelPending(
            model.name,
            kind,
            "dry_run",
            "dry run: nothing was read",
            **identity,
            estimated_bytes=estimated,
            changed_reason=changed_reason,
        )
    changed_rows = (
        _count_changed(model, source, adapter=adapter, predicates=predicates)
        if full_scan
        else None
    )
    # Unfiltered runs reconcile removed rows, so deletions are work too.
    pending = summary.new_rows > 0 or summary.deleted_rows > 0 or bool(changed_rows)
    reason = _pending_reason(summary.new_rows, summary.deleted_rows, changed_rows)
    if not pending and not predicates:
        reason = "every upstream key has a state row and no state row is orphaned"
    return ModelPending(
        model.name,
        kind,
        "pending" if pending else "none",
        reason,
        **identity,
        new_rows=summary.new_rows,
        partition_min=_render(summary.partition_min),
        partition_max=_render(summary.partition_max),
        cluster_distinct=summary.cluster_distinct,
        deleted_rows=None if predicates else summary.deleted_rows,
        estimated_bytes=summary.estimated_bytes,
        changed_rows=changed_rows,
        changed_reason=changed_reason,
    )


def _pending_reason(new_rows: int, deleted_rows: int, changed_rows: int | None) -> str:
    reason = f"{new_rows} upstream row(s) have no state row"
    if deleted_rows:
        reason += f"; {deleted_rows} state row(s) have no upstream row"
    if changed_rows:
        reason += f"; {changed_rows} have a different input fingerprint"
    return reason


def _count_changed(
    model: ModelConfig,
    source: PendingSource,
    *,
    adapter: WarehouseAdapter,
    predicates: tuple[ReadPredicate, ...],
) -> int:
    """Rows whose state fingerprint differs from the one the row hashes to now.

    Streams matching upstream rows one batch at a time. The state is read
    whole, which the runners do too. Keys with no state row are `new`, not `changed`."""
    assert source.fingerprint is not None
    state = adapter.fetch_state(StateScope(model.name))
    changed = 0
    with adapter.table_snapshot(
        source.upstream, batch_size=_SCAN_BATCH_ROWS, predicate=predicates
    ) as snapshot:
        for batch in snapshot:
            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            for record in frame.iter_rows(named=True):
                prior = state.get(str(record[source.key_column]))
                if prior is not None and prior.input_fingerprint != source.fingerprint(record):
                    changed += 1
    return changed


def pending_models(
    planned: list[ModelConfig],
    *,
    project_dir: Path,
    adapter: WarehouseAdapter,
    dry_run: bool,
    include_changed: bool,
    predicates: tuple[ReadPredicate, ...] = (),
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
                _pending_for_model(
                    model,
                    source,
                    adapter=adapter,
                    dry_run=dry_run,
                    include_changed=include_changed,
                    predicates=predicates,
                )
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
    include_changed: bool,
    read_filter: tuple[tuple[str, str, str], ...] = (),
) -> PendingReport:
    """Report pending upstream rows for every selected model.

    Same preflight as `plan` and `run`, so a bad configuration fails here with
    the message it would fail a run with."""
    try:
        predicates = parse_read_filters(read_filter)
    except (AdapterError, RunError) as error:
        raise PlanError(str(error)) from error
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
            planned,
            project_dir=project_dir,
            adapter=adapter,
            dry_run=dry_run,
            include_changed=include_changed,
            predicates=predicates,
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
        artifact_dir=(project_dir / project.target_path).resolve(),
    )


def write_pending_artifact(project_dir: Path, report: PendingReport) -> Path:
    report.artifact_dir.mkdir(parents=True, exist_ok=True)
    out = report.artifact_dir / PENDING_FILENAME
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
    if any(model.changed_rows is None for model in report.models):
        lines.append(f"changed rows are not reported: {CHANGED_REASON}.")
    estimates = [
        m.estimated_bytes for m in report.models if m.estimated_bytes is not None
    ]
    if estimates:
        lines.append(f"estimated bytes billed: {sum(estimates):,}")
    return lines
