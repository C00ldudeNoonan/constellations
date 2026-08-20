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
from ..providers import get_inference_provider
from ..versioning import compute_model_code_version
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
    "generated_at",
)


@dataclass
class _LLMWork:
    record_id: str  # str(id value) — state key and dedup identity
    id_value: Any  # original upstream id value, carried to the output row
    input_fingerprint: str  # hash of the content that drove the generation
    content: str
    rows: list[dict[str, Any]] | None = None  # projected declared output fields
    generated_at: str | None = None


def _llm_record_ids(
    frame: pl.DataFrame,
    id_field: str,
    model_name: str,
) -> tuple[list[str], list[Any]]:
    values = frame[id_field].to_list()
    null_count = sum(value is None for value in values)
    if null_count:
        raise RunError(
            f"llm model '{model_name}': upstream `{id_field}` contains "
            f"{null_count} NULL value(s)"
        )
    record_ids = [str(value) for value in values]
    empty_count = sum(not value for value in record_ids)
    if empty_count:
        raise RunError(
            f"llm model '{model_name}': upstream `{id_field}` contains "
            f"{empty_count} empty value(s)"
        )
    duplicate_count = len(record_ids) - len(set(record_ids))
    if duplicate_count:
        raise RunError(
            f"llm model '{model_name}': upstream `{id_field}` contains "
            f"{duplicate_count} duplicate value(s)"
        )
    return record_ids, list(values)


def _existing_llm_id_values(
    adapter: WarehouseAdapter,
    table: str,
    *,
    id_field: str,
) -> dict[str, Any]:
    existing = adapter.read_table(table)
    if id_field not in existing.columns:
        return {}
    mapping: dict[str, Any] = {}
    for value in existing[id_field].to_list():
        if value is not None:
            mapping.setdefault(str(value), value)
    return mapping


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
    source = adapter.read_table(upstream)
    missing = sorted({config.id_field, config.input_field} - set(source.columns))
    if missing:
        raise RunError(
            f"llm model '{model.name}': upstream '{upstream}' is missing required "
            f"column(s): {', '.join(missing)}. Available: {sorted(source.columns)}"
        )
    generated = set(_LLM_METADATA_COLUMNS) | {field.name for field in model.fields}
    if config.output_cardinality == "many":
        generated |= {config.row_id_field, config.ordinal_field}
    collisions = sorted(column for column in source.columns if column in generated)
    if collisions:
        raise RunError(
            f"llm model '{model.name}': upstream '{upstream}' already contains "
            f"generated column(s): {', '.join(collisions)}"
        )

    try:
        runtime = resolve_llm_runtime(config, model.fields, resolved)
    except LLMMapError as e:
        raise RunError(f"llm model '{model.name}': {e}") from e

    record_ids, id_values = _llm_record_ids(source, config.id_field, model.name)
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
    existing_id_values = (
        _existing_llm_id_values(adapter, model.name, id_field=config.id_field)
        if is_incremental and not rebuild_target
        else {}
    )

    current_ids = set(record_ids)
    removed = sorted(set(processed_state) - current_ids)
    removed_id_values = [
        existing_id_values[record_id]
        for record_id in removed
        if record_id in existing_id_values
    ]

    work: list[_LLMWork] = []
    skipped = 0
    for record_id, id_value, record in zip(
        record_ids, id_values, source.iter_rows(named=True), strict=True
    ):
        content_value = record[config.input_field]
        content = "" if content_value is None else str(content_value)
        input_fingerprint = canonical_fingerprint(
            {"content": content},
            domain="llm-input-content",
            version=1,
        )
        if processed_state.get(record_id) == StateValue(input_fingerprint, code_version):
            skipped += 1
            continue
        work.append(
            _LLMWork(
                record_id=record_id,
                id_value=id_value,
                input_fingerprint=input_fingerprint,
                content=content,
            )
        )

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
        if budget_guard is not None:
            budget_guard.ensure_headroom()
        rows, usage = execute_map_item(item.content, runtime)
        if budget_guard is not None:
            budget_guard.charge_metrics(usage)
        with usage_lock:
            nonlocal provider_calls
            provider_calls += 1
            add_provider_usage(usage_totals, usage)
        item.rows = rows
        return item

    run_status: str | None = None
    errors: list[str] = []
    try:
        if work and budget_guard is not None:
            budget_guard.charge_documents(len(work))
        if work:
            max_workers = max(1, min(config.max_concurrent, len(work)))
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as pool:
                # Preserve input order; surface the first failure deterministically.
                for completed in pool.map(_one, work):
                    del completed
    except BudgetExceededError as e:
        # Exhaustion fires before the next provider call. This model writes once
        # at the end, so nothing is published and state is unchanged; return a
        # budget_exceeded result (with partial usage) so run_project records the
        # status and skips descendants instead of aborting the invocation.
        run_status = "budget_exceeded"
        errors.append(f"BudgetExceededError: {e}")
    except Exception as e:
        raise RunError(
            f"llm model '{model.name}' provider execution failed: "
            f"{artifact_error_text(e)}"
        ) from e

    output_rows: list[dict[str, Any]] = []
    rows_written = 0
    if run_status is None:
        now = datetime.now(UTC).isoformat()
        output_schema = _llm_output_schema(model, config, source)
        for item in work:
            output_rows.extend(
                _llm_output_rows(
                    item, config=config, runtime=runtime, generated_at=now
                )
            )
        output = _llm_output_frame(output_rows, schema=output_schema, model=model)
        state_records = [
            StateRecord(item.record_id, item.input_fingerprint, code_version)
            for item in work
        ]
        key_col = (
            config.row_id_field
            if config.output_cardinality == "many"
            else config.id_field
        )

        use_full = model.materialization == "full" or full_refresh or rebuild_target
        try:
            if use_full:
                rows_written = adapter.materialize_full(
                    model.name,
                    output,
                    options=warehouse_opts,
                )
                adapter.replace_state(state_scope, state_records)
            else:
                if config.output_cardinality == "many":
                    rows_written = adapter.replace_children(
                        model.name,
                        parent_key=config.id_field,
                        parent_ids=[item.id_value for item in work],
                        child_key=key_col,
                        new_rows=output,
                        state_scope=state_scope,
                        state_records=state_records,
                        on_schema_change=model.on_schema_change,
                        options=warehouse_opts,
                    )
                    if removed:
                        adapter.delete_rows_and_state(
                            model.name,
                            key_col=config.id_field,
                            keys=removed_id_values,
                            state_scope=state_scope,
                            state_record_keys=removed,
                        )
                else:
                    if output_rows:
                        rows_written = adapter.materialize_incremental(
                            model.name,
                            output,
                            key_col=key_col,
                            on_schema_change=model.on_schema_change,
                            options=warehouse_opts,
                            update_when_changed=model.update_when_changed,
                        )
                    if removed:
                        adapter.delete_rows_and_state(
                            model.name,
                            key_col=config.id_field,
                            keys=removed_id_values,
                            state_scope=state_scope,
                            state_record_keys=removed,
                        )
                    if state_records:
                        adapter.upsert_state(state_scope, state_records)
        except AdapterError as e:
            raise RunError(str(e)) from e

    metrics: dict[str, Any] = {
        "provider_calls": provider_calls,
        # Always present so the run summary labels this an llm model even when
        # every input was skipped (no provider calls this run).
        "api_calls": usage_totals.get("api_calls", 0),
        "cache_hits": usage_totals.get("cache_hits", 0),
        "rows_generated": len(output_rows),
        "inputs_processed": len(work),
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
        documents_processed=len(work),
        documents_skipped=skipped,
        documents_deleted=len(removed),
        rows_written=rows_written,
        errors=errors,
        metrics=metrics,
        artifact_metadata={"llm": runtime.identity()},
    )
