"""Native `llm:` model executor (issue #190).

Owns the llm-only lifecycle: upstream validation, runtime resolution,
incremental state, bounded concurrent generation under the run budget, output
row/schema shaping (including `output_cardinality: many` fan-out), and
materialization. runner.py keeps selection, DAG scheduling, the run budget
ledger, and result aggregation, and re-exports run_llm_model for compatibility.
"""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ..adapters import (
    AdapterError,
    StateRecord,
    StateScope,
    StateValue,
    WarehouseAdapter,
)
from ..budget import BudgetExceededError, BudgetGuard, BudgetLedger
from ..config.model import LLMTransformConfig, ModelConfig
from ..config.project import ProjectConfig
from ..dag import parse_ref
from ..hashing import canonical_fingerprint
from ..llm_map import (
    LLMMapError,
    LLMMapRuntime,
    execute_map_item,
    resolve_llm_runtime,
)
from ..profile import ResolvedProfile
from ..progress import get_reporter
from ..providers import get_inference_provider
from ..versioning import compute_model_code_version
from .checkpoint import FlushPublisher
from .contracts import ModelRunResult, RunError
from .cost import budget_cost_estimator
from .errors import artifact_error_text
from .extraction import EXTRACTION_FIELD_DTYPES  # shared declared-data_type contract
from .usage import add_provider_usage
from .values import scalarize
from .warehouse import warehouse_options

_LLM_METADATA_COLUMNS = (
    "llm_provider",
    "llm_model",
    "llm_provider_implementation",
    "llm_input_hash",
    "llm_config_hash",
    "prompt_name",
    "prompt_version",
    "generated_at",
)

# Read batch sizes for the two projected paths (issue #424). These are
# deliberately independent of `flush_every`: publication cadence is
# operator-configurable and has no upper bound, while these constants bound
# warehouse-read residency. Existing-target ids are narrow; input batches also
# carry the potentially large prompt content and therefore stay smaller.
_ID_BATCH_ROWS = 100_000
_INPUT_BATCH_ROWS = 10_000


@dataclass
class _LLMWork:
    record_id: str  # str(id value) — state key and dedup identity
    id_value: Any  # original upstream id value, carried to the output row
    input_fingerprint: str  # hash of the content that drove the generation
    content: str
    rows: list[dict[str, Any]] | None = None  # projected declared output fields
    generated_at: str | None = None


@dataclass(frozen=True)
class _LLMInputPlan:
    current_ids: set[str]
    work_count: int
    skipped: int


def _llm_content_fingerprint(value: Any) -> tuple[str, str]:
    content = "" if value is None else str(value)
    return content, canonical_fingerprint(
        {"content": content},
        domain="llm-input-content",
        version=1,
    )


def _stream_llm_input_plan(
    adapter: WarehouseAdapter,
    table: str,
    *,
    config: LLMTransformConfig,
    model_name: str,
    processed_state: dict[str, StateValue],
    code_version: str,
) -> _LLMInputPlan:
    """Validate and classify every input without retaining corpus text.

    This eager projected pass stays ahead of provider execution for two public
    contracts: invalid ids fail before any paid call, and `max_documents` can
    reject a run before its first publication. The cumulative id set is the
    known O(distinct keys) reconciliation trade tracked by issue #428; input
    text and row frames remain bounded by `_INPUT_BATCH_ROWS`.
    """
    current_ids: set[str] = set()
    total = 0
    null_count = 0
    empty_count = 0
    work_count = 0
    skipped = 0
    with adapter.table_snapshot(
        table,
        columns=[config.id_field, config.input_field],
        batch_size=_INPUT_BATCH_ROWS,
    ) as snapshot:
        for batch in snapshot:
            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            for record in frame.iter_rows(named=True):
                total += 1
                id_value = record[config.id_field]
                if id_value is None:
                    null_count += 1
                    continue
                record_id = str(id_value)
                if not record_id:
                    empty_count += 1
                    continue
                current_ids.add(record_id)
                _, input_fingerprint = _llm_content_fingerprint(
                    record[config.input_field]
                )
                if processed_state.get(record_id) == StateValue(
                    input_fingerprint, code_version
                ):
                    skipped += 1
                else:
                    work_count += 1
    if null_count:
        raise RunError(
            f"llm model '{model_name}': upstream `{config.id_field}` contains "
            f"{null_count} NULL value(s)"
        )
    if empty_count:
        raise RunError(
            f"llm model '{model_name}': upstream `{config.id_field}` contains "
            f"{empty_count} empty value(s)"
        )
    duplicate_count = total - len(current_ids)
    if duplicate_count:
        raise RunError(
            f"llm model '{model_name}': upstream `{config.id_field}` contains "
            f"{duplicate_count} duplicate value(s)"
        )
    return _LLMInputPlan(
        current_ids=current_ids,
        work_count=work_count,
        skipped=skipped,
    )


def _existing_llm_id_values(
    adapter: WarehouseAdapter,
    table: str,
    *,
    id_field: str,
) -> dict[str, Any]:
    # Probe without opening an unconsumed DuckDB Arrow reader; rows then stream
    # as the one narrow column deletion reconciliation actually needs (#424).
    if id_field not in adapter.read_table(table, limit=0).columns:
        return {}
    mapping: dict[str, Any] = {}
    with adapter.table_snapshot(
        table,
        columns=[id_field],
        batch_size=_ID_BATCH_ROWS,
    ) as snapshot:
        for batch in snapshot:
            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            for value in frame[id_field].to_list():
                if value is not None:
                    mapping.setdefault(str(value), value)
    return mapping


def _iter_llm_work_windows(
    adapter: WarehouseAdapter,
    table: str,
    *,
    config: LLMTransformConfig,
    processed_state: dict[str, StateValue],
    code_version: str,
) -> Iterator[list[_LLMWork]]:
    """Stream changed inputs into flush-sized work windows."""
    window: list[_LLMWork] = []
    with adapter.table_snapshot(
        table,
        columns=[config.id_field, config.input_field],
        batch_size=_INPUT_BATCH_ROWS,
    ) as snapshot:
        for batch in snapshot:
            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            for record in frame.iter_rows(named=True):
                id_value = record[config.id_field]
                record_id = str(id_value)
                content, input_fingerprint = _llm_content_fingerprint(
                    record[config.input_field]
                )
                if processed_state.get(record_id) == StateValue(
                    input_fingerprint, code_version
                ):
                    continue
                window.append(
                    _LLMWork(
                        record_id=record_id,
                        id_value=id_value,
                        input_fingerprint=input_fingerprint,
                        content=content,
                    )
                )
                if len(window) >= config.flush_every:
                    yield window
                    window = []
    if window:
        yield window


def _llm_output_schema(
    model: ModelConfig,
    config: LLMTransformConfig,
    source: pl.DataFrame,
) -> dict[str, Any]:
    schema: dict[str, Any] = {config.id_field: source.schema[config.id_field]}
    if config.output_cardinality == "many":
        schema[config.row_id_field] = pl.String
        schema[config.ordinal_field] = pl.Int64
    for field_config in model.fields:
        schema[field_config.name] = (
            EXTRACTION_FIELD_DTYPES[field_config.data_type]
            if field_config.data_type is not None
            else pl.String
        )
    for column in _LLM_METADATA_COLUMNS:
        schema[column] = pl.String
    return schema


def _llm_output_rows(
    item: _LLMWork,
    *,
    config: LLMTransformConfig,
    runtime: LLMMapRuntime,
    generated_at: str,
) -> list[dict[str, Any]]:
    assert item.rows is not None
    rows: list[dict[str, Any]] = []
    for ordinal, fields in enumerate(item.rows):
        row: dict[str, Any] = {name: scalarize(value) for name, value in fields.items()}
        row[config.id_field] = item.id_value
        if config.output_cardinality == "many":
            row[config.row_id_field] = f"{item.record_id}__{ordinal}"
            row[config.ordinal_field] = ordinal
        row.update(
            {
                "llm_provider": runtime.provider,
                "llm_model": runtime.model,
                "llm_provider_implementation": runtime.implementation,
                "llm_input_hash": item.input_fingerprint,
                "llm_config_hash": runtime.config_hash,
                # Which prompt produced this row, readable and groupable —
                # `llm_config_hash` only says that something changed (#303).
                "prompt_name": runtime.prompt_name,
                "prompt_version": runtime.prompt_version,
                "generated_at": generated_at,
            }
        )
        rows.append(row)
    return rows


def _llm_output_frame(
    rows: list[dict[str, Any]],
    *,
    schema: dict[str, Any],
    model: ModelConfig,
) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    frame = pl.DataFrame(rows)
    typed = {
        field.name: field.data_type
        for field in model.fields
        if field.data_type is not None
    }
    expressions: list[pl.Expr] = []
    for name, dtype in schema.items():
        if name not in frame.columns:
            expressions.append(pl.lit(None, dtype=dtype).alias(name))
            continue
        data_type = typed.get(name)
        if data_type == "date" and frame.schema[name] == pl.String:
            expressions.append(pl.col(name).str.to_date(strict=True))
        elif data_type == "timestamp" and frame.schema[name] == pl.String:
            expressions.append(
                pl.col(name).str.to_datetime(time_zone="UTC", strict=True)
            )
        else:
            expressions.append(pl.col(name).cast(dtype, strict=True))
    try:
        return frame.with_columns(expressions).select(list(schema))
    except Exception as e:
        raise RunError(
            f"llm model '{model.name}' produced a value that does not match its "
            f"declared field data_type: {e}"
        ) from e


def run_llm_model(
    *,
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    full_refresh: bool,
    run_budget: BudgetLedger | None = None,
) -> ModelRunResult:
    assert model.llm is not None
    config = model.llm
    if not model.depends_on or len(model.depends_on) != 1:
        raise RunError(
            f"llm model '{model.name}' must declare exactly one upstream in "
            "`depends_on:`"
        )
    upstream = parse_ref(model.depends_on[0])
    # Contract-only probe. Both input passes below project the two columns they
    # need and consume bounded Arrow batches instead of materializing the
    # upstream relation (#424).
    schema_probe = adapter.read_table(upstream, limit=0)
    missing = sorted(
        {config.id_field, config.input_field} - set(schema_probe.columns)
    )
    if missing:
        raise RunError(
            f"llm model '{model.name}': upstream '{upstream}' is missing required "
            f"column(s): {', '.join(missing)}. Available: "
            f"{sorted(schema_probe.columns)}"
        )
    generated = set(_LLM_METADATA_COLUMNS) | {field.name for field in model.fields}
    if config.output_cardinality == "many":
        generated |= {config.row_id_field, config.ordinal_field}
    collisions = sorted(
        column for column in schema_probe.columns if column in generated
    )
    if collisions:
        raise RunError(
            f"llm model '{model.name}': upstream '{upstream}' already contains "
            f"generated column(s): {', '.join(collisions)}"
        )

    try:
        runtime = resolve_llm_runtime(
            config,
            model.fields,
            resolved,
            project_dir=project_dir,
            model_name=model.name,
        )
    except LLMMapError as e:
        raise RunError(f"llm model '{model.name}': {e}") from e

    code_version = compute_model_code_version(
        model,
        project,
        project_dir,
        resolved=resolved,
    )
    warehouse_opts = warehouse_options(adapter, model)
    state_scope = StateScope(model.name)
    existing_tables = set(adapter.list_tables())
    is_incremental = model.materialization == "incremental" and not full_refresh
    rebuild_target = is_incremental and model.name not in existing_tables
    processed_state = (
        adapter.fetch_state(state_scope)
        if is_incremental and not rebuild_target
        else {}
    )
    input_plan = _stream_llm_input_plan(
        adapter,
        upstream,
        config=config,
        model_name=model.name,
        processed_state=processed_state,
        code_version=code_version,
    )
    existing_id_values = (
        _existing_llm_id_values(adapter, model.name, id_field=config.id_field)
        if is_incremental and not rebuild_target
        else {}
    )

    removed = sorted(set(processed_state) - input_plan.current_ids)
    removed_id_values = [
        existing_id_values[record_id]
        for record_id in removed
        if record_id in existing_id_values
    ]

    budget_guard: BudgetGuard | None = None
    if run_budget is not None:
        budget_guard = BudgetGuard(
            None,
            run_budget,
            cost_estimator=budget_cost_estimator(
                resolved,
                batch=False,
                provider=get_inference_provider(runtime.provider),
            ),
        )

    usage_totals: dict[str, int | float] = {}
    usage_lock = threading.Lock()
    provider_calls = 0

    def _one(item: _LLMWork) -> _LLMWork:
        rows, usage = execute_map_item(
            item.content,
            runtime,
            budget=budget_guard,
        )
        with usage_lock:
            nonlocal provider_calls
            provider_calls += 1
            add_provider_usage(usage_totals, usage)
        item.rows = rows
        return item

    run_status: str | None = None
    errors: list[str] = []
    rows_generated = 0
    key_col = (
        config.row_id_field
        if config.output_cardinality == "many"
        else config.id_field
    )
    use_full = model.materialization == "full" or full_refresh or rebuild_target
    output_schema = _llm_output_schema(model, config, schema_probe)
    publisher = FlushPublisher(
        adapter,
        model_name=model.name,
        state_scope=state_scope,
        use_full=use_full,
    )

    def _publish_window(window: list[_LLMWork]) -> None:
        """Publish one window's completions, then advance state for its inputs.

        Publish-then-state, matching extraction (#139) and embed (#401): an
        interrupted run never records an input it did not write, so the re-run
        re-calls the provider only for what was actually lost. That matters
        more here than anywhere else in the pipeline — an llm map model spends
        one provider call per input, so a failure at input 500,000 used to
        discard half a million paid completions.
        """
        nonlocal rows_generated
        now = datetime.now(UTC).isoformat()
        window_rows: list[dict[str, Any]] = []
        for item in window:
            window_rows.extend(
                _llm_output_rows(
                    item, config=config, runtime=runtime, generated_at=now
                )
            )
        rows_generated += len(window_rows)
        output = _llm_output_frame(window_rows, schema=output_schema, model=model)
        state_records = [
            StateRecord(item.record_id, item.input_fingerprint, code_version)
            for item in window
        ]
        fan_out = config.output_cardinality == "many"
        publisher.publish(
            write_full=lambda: adapter.materialize_full(
                model.name,
                output,
                options=warehouse_opts,
            ),
            write_incremental=lambda: (
                # Scoped to this window's parents, so it never disturbs
                # children published by an earlier window.
                adapter.replace_children(
                    model.name,
                    parent_key=config.id_field,
                    parent_ids=[item.id_value for item in window],
                    child_key=key_col,
                    new_rows=output,
                    state_scope=state_scope,
                    state_records=state_records,
                    on_schema_change=(
                        model.on_schema_change
                        if publisher.first_publication
                        else "append_new_columns"
                    ),
                    options=warehouse_opts,
                )
                if fan_out
                else adapter.materialize_incremental(
                    model.name,
                    output,
                    key_col=key_col,
                    on_schema_change=(
                        model.on_schema_change
                        if publisher.first_publication
                        else "append_new_columns"
                    ),
                    options=warehouse_opts,
                    update_when_changed=model.update_when_changed,
                )
                if window_rows
                else 0
            ),
            state_records=state_records,
            # replace_children applies the state records in the same
            # transaction as the rows, so the publisher must not repeat it.
            advances_state_itself=fan_out and not (use_full and publisher.first_publication),
        )

    try:
        if input_plan.work_count and budget_guard is not None:
            # Preserve the native llm contract: an over-cap corpus is rejected
            # before any provider call or partial publication. The projected
            # planning pass above makes that possible without retaining work.
            budget_guard.charge_documents(input_plan.work_count)
        # One bar across every window, counted in records: the windows are a
        # memory bound (issue #401), not something the operator tracks.
        with get_reporter().model_task(
            model.name, "llm", input_plan.work_count
        ) as task:
            windows = (
                _iter_llm_work_windows(
                    adapter,
                    upstream,
                    config=config,
                    processed_state=processed_state,
                    code_version=code_version,
                )
                if input_plan.work_count
                else ()
            )
            for window in windows:
                max_workers = max(1, min(config.max_concurrent, len(window)))
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as pool:
                    # Preserve input order; surface the first failure
                    # deterministically.
                    for completed in pool.map(_one, window):
                        del completed
                        task.advance(1)
                _publish_window(window)
                # Release this window's completions before the next one runs.
                for item in window:
                    item.rows = []
                window.clear()
    except BudgetExceededError as e:
        # Exhaustion fires before the next provider call. Windows already
        # published stay, with their state advanced; the partial window in
        # flight is discarded and re-called next run, matching how extraction
        # treats an unpublished buffer. Return budget_exceeded so run_project
        # records the status and skips descendants.
        run_status = "budget_exceeded"
        errors.append(f"BudgetExceededError: {e}")
    except RunError:
        raise
    except AdapterError as e:
        raise RunError(str(e)) from e
    except Exception as e:
        raise RunError(
            f"llm model '{model.name}' provider execution failed: "
            f"{artifact_error_text(e)}"
        ) from e

    if run_status is None:
        try:
            if not publisher.published_any and use_full:
                # Nothing to write, but a rebuild still owes the target its table.
                adapter.replace_state(state_scope, [])
                publisher.rows_written += adapter.materialize_full(
                    model.name,
                    _llm_output_frame([], schema=output_schema, model=model),
                    options=warehouse_opts,
                )
            if removed and not use_full:
                adapter.delete_rows_and_state(
                    model.name,
                    key_col=config.id_field,
                    keys=removed_id_values,
                    state_scope=state_scope,
                    state_record_keys=removed,
                )
        except AdapterError as e:
            raise RunError(str(e)) from e

    metrics: dict[str, Any] = {
        "provider_calls": provider_calls,
        # Always present so the run summary labels this an llm model even when
        # every input was skipped (no provider calls this run).
        "api_calls": usage_totals.get("api_calls", 0),
        "cache_hits": usage_totals.get("cache_hits", 0),
        "rows_generated": rows_generated,
        "inputs_processed": input_plan.work_count,
        **usage_totals,
    }
    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="llm",
        status=run_status,
        provider=runtime.provider,
        provider_model=runtime.model,
        provider_implementation=runtime.implementation,
        prompt_name=runtime.prompt_name,
        prompt_version=runtime.prompt_version,
        documents_processed=input_plan.work_count,
        documents_skipped=input_plan.skipped,
        documents_deleted=len(removed),
        rows_written=publisher.rows_written,
        errors=errors,
        metrics=metrics,
        artifact_metadata={"llm": runtime.identity()},
    )
