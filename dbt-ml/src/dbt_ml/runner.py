from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Any, cast

import polars as pl
import pyarrow as pa

from .adapters import (
    AdapterError,
    StateRecord,
    StateScope,
    StateValue,
    TableReadSnapshot,
    WarehouseAdapter,
    WarehouseCapability,
    create_adapter,
)
from .budget import BudgetExceededError, BudgetGuard, BudgetLedger
from .checks import TestResult, run_model_tests
from .classic_ml import run_classic_ml_model
from .compiler import (
    validate_project_contract,
    validate_retrieval_capabilities,
    validate_warehouse_capabilities,
)
from .config import load_project
from .config.model import (
    LLMTransformConfig,
    ModelConfig,
)
from .config.project import ProjectConfig
from .config.source import SourceConfig
from .dag import ProjectDAG, parse_ref
from .embedding import (
    effective_search_config,
    resolve_search_embedding_options,
)
from .execution import ModelRunResult as ModelRunResult
from .execution import RunError as RunError
from .execution import chunk as _chunk_execution
from .execution import cost as _cost_execution
from .execution import embed as _embed_execution
from .execution import errors as _errors_execution
from .execution import extraction as _extraction_execution
from .execution import scalarize as _scalarize
from .execution import transform as _transform_execution
from .execution import usage as _usage_execution
from .execution import warehouse_options as _warehouse_options
from .hashing import canonical_fingerprint
from .llm_map import LLMMapError, LLMMapRuntime, execute_map_item, resolve_llm_runtime
from .manifest import compute_modified_models
from .paths import resolve_within_project
from .profile import (
    ResolvedProfile,
    apply_source_path_overrides,
    resolve_profile,
)
from .providers import (
    get_inference_provider,
)
from .retrieval import (
    CollectionSpec,
    IndexedRow,
    PublishLease,
    RetrievalError,
    ServingCoordinationError,
    ServingCoordinator,
    collection_config_fingerprint,
    create_store,
)
from .sources import SourceError, get_document_source
from .state_reconciliation import BoundedReconciler, UpstreamRecord
from .versioning import compute_model_code_version

log = logging.getLogger(__name__)

_CHUNK_INPUT_EXCLUDED_FIELDS = _chunk_execution._CHUNK_INPUT_EXCLUDED_FIELDS
_chunk_document_ids = _chunk_execution.chunk_document_ids
_chunk_input_hash = _chunk_execution.chunk_input_hash
_chunk_row = _chunk_execution.chunk_row
_run_chunk_model = _chunk_execution.run_chunk_model

_run_sql_model = _transform_execution.run_sql_model
_run_transform_model = _transform_execution.run_transform_model
_validate_agent_context_output = _transform_execution._validate_agent_context_output
_artifact_error_text = _errors_execution.artifact_error_text

_estimate_cost = _cost_execution.estimate_cost
_budget_cost_estimator = _cost_execution.budget_cost_estimator

DiscoveredSource = _extraction_execution.DiscoveredSource
_run_extraction_model = _extraction_execution.run_extraction_model
# The native `llm:` output schema reuses the extraction declared-field dtype
# contract, so the llm executor (still in runner) imports it from here.
_EXTRACTION_FIELD_DTYPES = _extraction_execution.EXTRACTION_FIELD_DTYPES

_run_embed_model = _embed_execution.run_embed_model
# Shared with the llm executor (still in runner).
_add_provider_usage = _usage_execution.add_provider_usage


# Deterministic column order for the generation-metadata columns a native
# `llm:` model appends to every output row (issue #144). Same set as
# LLM_METADATA_FIELDS; the tuple pins column order in the materialized table.
_LLM_METADATA_COLUMNS = (
    "llm_provider",
    "llm_model",
    "llm_provider_implementation",
    "llm_input_hash",
    "llm_config_hash",
    "generated_at",
)


def _modified_set(
    models: list[ModelConfig],
    project_dir: Path,
    state: Path | None,
    *,
    project: ProjectConfig,
    resolved: ResolvedProfile,
) -> set[str] | None:
    """None when no state manifest was given (state:modified then errors in
    selection); otherwise the models whose code_version diverged from it."""
    if state is None:
        return None
    return compute_modified_models(
        models,
        project_dir,
        state,
        project=project,
        resolved=resolved,
    )


class _SerializedAdapter:
    """Serializes every adapter method call behind a lock so independent models
    can run on separate threads while sharing one warehouse connection. Property
    access (schema_ref, catalog, …) passes through untouched; only callables are
    guarded, which covers all the read/write paths the runner uses."""

    def __init__(self, adapter: WarehouseAdapter, lock: threading.Lock) -> None:
        self._adapter = adapter
        self._lock = lock

    @contextmanager
    def table_snapshot(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 10_000,
        predicate: Any = None,
        key_column: str | None = None,
    ) -> Iterator[TableReadSnapshot]:
        with self._lock:
            with self._adapter.table_snapshot(
                table,
                columns=columns,
                batch_size=batch_size,
                predicate=predicate,
                key_column=key_column,
            ) as snapshot:
                yield snapshot

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._adapter, name)
        if not callable(attr):
            return attr

        def guarded(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attr(*args, **kwargs)

        return guarded


@dataclass
class BuildResult:
    run_results: list[ModelRunResult] = field(default_factory=list)
    test_results: list[TestResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def run_project(
    project_dir: Path,
    *,
    full_refresh: bool = False,
    select: str | None = None,
    exclude: str | None = None,
    target: str | None = None,
    profiles_dir: Path | None = None,
    threads: int = 1,
    state: Path | None = None,
) -> list[ModelRunResult]:
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    sources = apply_source_path_overrides(sources, resolved)
    selected = dag.select_models(
        select=select,
        exclude=exclude,
        modified=_modified_set(
            models,
            project_dir,
            state,
            project=project,
            resolved=resolved,
        ),
    )
    validate_warehouse_capabilities(
        [model for model in models if model.name in set(selected)],
        resolved.warehouse.type,
    )
    validate_retrieval_capabilities(
        [model for model in models if model.name in set(selected)], project, resolved
    )
    if full_refresh and any(
        model.search is not None and model.name in set(selected) for model in models
    ):
        raise RunError(
            "--full-refresh is unavailable for search resources until atomic store "
            "activation and state replacement are implemented by #153"
        )

    required_sources = set(dag.required_sources(selected))
    source_docs = _discover_sources(
        [source for source in sources if source.name in required_sources],
        project_dir,
    )

    models_by_name = {m.name: m for m in models}

    run_budget = _run_budget_ledger(resolved)

    def _run(name: str, adapter: WarehouseAdapter) -> ModelRunResult:
        return _run_model(
            model=models_by_name[name],
            models_by_name=models_by_name,
            project=project,
            project_dir=project_dir,
            source_docs=source_docs,
            adapter=adapter,
            resolved=resolved,
            full_refresh=full_refresh,
            threads=threads,
            run_budget=run_budget,
        )

    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        if threads > 1 and len(selected) > 1:
            results_by_name = _run_in_batches(dag, selected, adapter, _run, threads)
        else:
            results_by_name = {name: _run(name, adapter) for name in selected}

    return [results_by_name[name] for name in selected]


def build_project(
    project_dir: Path,
    *,
    full_refresh: bool = False,
    select: str | None = None,
    exclude: str | None = None,
    target: str | None = None,
    profiles_dir: Path | None = None,
    threads: int = 1,
    store_failures: bool = False,
    state: Path | None = None,
) -> BuildResult:
    """Run + test each model in dependency order. A model whose run errors or
    whose tests hard-fail blocks all its descendants, which are reported as
    skipped (dbt `build` semantics)."""
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    sources = apply_source_path_overrides(sources, resolved)
    selected = dag.select_models(
        select=select,
        exclude=exclude,
        modified=_modified_set(
            models,
            project_dir,
            state,
            project=project,
            resolved=resolved,
        ),
    )
    validate_warehouse_capabilities(
        [model for model in models if model.name in set(selected)],
        resolved.warehouse.type,
    )
    validate_retrieval_capabilities(
        [model for model in models if model.name in set(selected)], project, resolved
    )
    if full_refresh and any(
        model.search is not None and model.name in set(selected) for model in models
    ):
        raise RunError(
            "--full-refresh is unavailable for search resources until atomic store "
            "activation and state replacement are implemented by #153"
        )

    required_sources = set(dag.required_sources(selected))
    source_docs = _discover_sources(
        [source for source in sources if source.name in required_sources],
        project_dir,
    )
    models_by_name = {m.name: m for m in models}

    run_budget = _run_budget_ledger(resolved)
    out = BuildResult()
    blocked: set[str] = set()

    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        for name in selected:
            if name in blocked:
                out.skipped.append(name)
                continue
            model = models_by_name[name]
            try:
                result = _run_model(
                    model=model,
                    models_by_name=models_by_name,
                    project=project,
                    project_dir=project_dir,
                    source_docs=source_docs,
                    adapter=adapter,
                    resolved=resolved,
                    full_refresh=full_refresh,
                    threads=threads,
                    run_budget=run_budget,
                )
            except RunError as e:
                out.run_results.append(
                    ModelRunResult(
                        model_name=name,
                        materialization=model.materialization,
                        kind="unknown",
                        errors=[_artifact_error_text(e)],
                    )
                )
                blocked |= dag.descendants(name)
                continue

            out.run_results.append(result)
            if result.errors:
                blocked |= dag.descendants(name)
                continue

            model_tests = (
                []
                if model.search is not None
                else run_model_tests(
                    model,
                    adapter,
                    project_dir=project_dir,
                    store_failures=store_failures,
                )
            )
            out.test_results.extend(model_tests)
            if any(t.is_hard_failure for t in model_tests):
                blocked |= dag.descendants(name)

    return out


def _run_in_batches(
    dag: ProjectDAG,
    selected: list[str],
    adapter: WarehouseAdapter,
    run_one: Any,
    threads: int,
) -> dict[str, ModelRunResult]:
    """Run topological generations: models within a batch are independent and
    run concurrently; all warehouse access is serialized behind a lock."""
    guarded = cast(WarehouseAdapter, _SerializedAdapter(adapter, threading.Lock()))
    results_by_name: dict[str, ModelRunResult] = {}
    for batch in dag.parallel_batches(selected):
        if len(batch) == 1:
            results_by_name[batch[0]] = run_one(batch[0], guarded)
            continue
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(threads, len(batch))
        ) as ex:
            futures = {ex.submit(run_one, name, guarded): name for name in batch}
            for future in concurrent.futures.as_completed(futures):
                results_by_name[futures[future]] = future.result()
    return results_by_name


def _discover_sources(
    sources: list[SourceConfig], project_dir: Path
) -> dict[str, DiscoveredSource]:
    out: dict[str, DiscoveredSource] = {}
    for source in sources:
        backend = get_document_source(source.path)
        try:
            refs = backend.discover(source, project_dir)
        except SourceError as e:
            raise RunError(str(e)) from e
        out[source.name] = DiscoveredSource(backend=backend, refs=refs)
    return out


def _run_budget_ledger(resolved: ResolvedProfile) -> BudgetLedger | None:
    """One shared run-scope ledger; every LLM extraction model charges it."""
    if resolved.llm is None or resolved.llm.budget is None:
        return None
    return BudgetLedger(resolved.llm.budget, scope="run")


def _run_model(
    *,
    model: ModelConfig,
    models_by_name: Mapping[str, ModelConfig],
    project: ProjectConfig,
    project_dir: Path,
    source_docs: dict[str, DiscoveredSource],
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    full_refresh: bool,
    threads: int = 1,
    run_budget: BudgetLedger | None = None,
) -> ModelRunResult:
    start = time.monotonic()
    if model.extraction is not None:
        result = _run_extraction_model(
            model=model,
            project=project,
            project_dir=project_dir,
            source_docs=source_docs,
            adapter=adapter,
            resolved=resolved,
            full_refresh=full_refresh,
            threads=threads,
            run_budget=run_budget,
        )
    elif model.ml is not None:
        result = _run_ml_model(
            model=model,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
        )
    elif model.transform is not None:
        if model.transform.type == "sql":
            result = _run_sql_model(
                model=model,
                project_dir=project_dir,
                adapter=adapter,
                resolved=resolved,
                full_refresh=full_refresh,
            )
        else:
            result = _run_transform_model(
                model=model,
                project_dir=project_dir,
                adapter=adapter,
                resolved=resolved,
            )
    elif model.chunk is not None:
        result = _run_chunk_model(
            model=model,
            project_dir=project_dir,
            adapter=adapter,
            full_refresh=full_refresh,
        )
    elif model.embed is not None:
        result = _run_embed_model(
            model=model,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
            resolved=resolved,
            full_refresh=full_refresh,
        )
    elif model.llm is not None:
        result = _run_llm_model(
            model=model,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
            resolved=resolved,
            full_refresh=full_refresh,
            run_budget=run_budget,
        )
    elif model.search is not None:
        result = _run_search_model(
            model=model,
            models_by_name=models_by_name,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
            resolved=resolved,
        )
    else:
        raise RunError(
            f"Model '{model.name}' has no extraction, transform, ml, chunk, embed, "
            "llm, or search block configured"
        )
    result.duration_seconds = round(time.monotonic() - start, 3)
    return result


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
            _EXTRACTION_FIELD_DTYPES[field_config.data_type]
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
        row: dict[str, Any] = {name: _scalarize(value) for name, value in fields.items()}
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


def _run_llm_model(
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
    warehouse_opts = _warehouse_options(adapter, model)
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
            cost_estimator=_budget_cost_estimator(
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
            _add_provider_usage(usage_totals, usage)
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
            f"{_artifact_error_text(e)}"
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
                if config.output_cardinality == "many" and work:
                    # Fan-out counts can change; clear each reprocessed parent's
                    # old rows before appending the fresh set (parent-scoped).
                    adapter.delete_rows(
                        model.name,
                        key_col=config.id_field,
                        keys=[item.id_value for item in work],
                    )
                if output_rows:
                    rows_written = adapter.materialize_incremental(
                        model.name,
                        output,
                        key_col=key_col,
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


def _run_search_model(
    *,
    model: ModelConfig,
    models_by_name: Mapping[str, ModelConfig],
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
) -> ModelRunResult:
    search = model.search
    assert search is not None
    if resolved.retrieval is None:
        raise RunError("Search publication requires a configured retrieval target")
    alias = search.store or resolved.retrieval.default
    store_config = resolved.retrieval.stores.get(alias)
    if store_config is None:
        raise RunError("Search publication selected an unknown retrieval target")
    upstream = parse_ref((model.depends_on or [""])[0])
    logical_collection = search.collection or model.name
    code_version = compute_model_code_version(
        model,
        project,
        project_dir,
        resolved=resolved,
    )
    store = create_store(
        store_config,
        project_name=project.name,
        target_name=resolved.target_name,
        alias=alias,
    )
    projected = search.projected_fields()
    inserted = 0
    updated = 0
    skipped = 0
    deleted = 0
    rows_written = 0
    rows_seen = 0
    spec: CollectionSpec | None = None
    state_scope: StateScope | None = None
    coordinator = ServingCoordinator(adapter)
    publish_lease: PublishLease | None = None
    active_generation: str | None = None

    try:
        # Publication-state residency is bounded (issue #153): reconciliation
        # never loads the full scope, so the adapter must provide ordered
        # paged state access before any store I/O begins.
        adapter.require_capability(
            WarehouseCapability.PAGED_STATE_RECONCILIATION,
            operation="bounded publication-state reconciliation",
        )
        with adapter.table_snapshot(
            upstream,
            columns=projected,
            batch_size=search.batch_size,
            key_column=search.id_field,
        ) as snapshot:
            physical = store.physical_collection(logical_collection)
            spec = _search_collection_spec(
                model=model,
                models_by_name=models_by_name,
                physical_collection=physical,
                upstream_schema=snapshot.schema,
                store_type=store_config.type,
                resolved=resolved,
            )
            state_target = store.state_descriptor(logical_collection)
            state_scope = StateScope.for_target_descriptor(
                model.name,
                stage="retrieval_publish",
                descriptor=state_target.descriptor(),
            )
            reconciler = BoundedReconciler(
                adapter,
                state_scope,
                code_version=code_version,
                page_size=search.batch_size,
            )

            publish_lease = coordinator.acquire_publish(
                state_scope,
                expected_code_version=code_version,
                config_fingerprint=spec.config_fingerprint,
            )
            with store.publisher_fence(physical), store:
                existing = store.inspect_collection(physical)
                force_publish = existing is None
                collection_exists = existing is not None
                if existing is not None:
                    if existing.config_fingerprint != spec.config_fingerprint:
                        raise RunError(
                            "Search index configuration changed; choose a new collection "
                            "or wait for atomic rebuild support"
                        )
                    if not existing.schema.equals(spec.arrow_schema, check_metadata=False):
                        raise RunError(
                            "Search index schema does not match the declared collection contract"
                        )

                for ordinal, batch in enumerate(snapshot):
                    indexed = _indexed_rows(
                        batch,
                        model,
                        spec.config_fingerprint,
                        max_id_bytes=store.capabilities().max_id_bytes,
                    )
                    rows_seen += len(indexed)
                    upstream_records = [
                        UpstreamRecord(row.record_id, row.input_fingerprint)
                        for row in indexed
                    ]
                    prior_state = (
                        reconciler.prior_state_for(upstream_records)
                        if upstream_records
                        else {}
                    )
                    outcome = reconciler.classify(
                        upstream_records,
                        prior=prior_state,
                        force_publish=force_publish,
                    )
                    skipped += len(outcome.unchanged)
                    pending_keys = {
                        record.record_key
                        for record in (*outcome.new, *outcome.changed)
                    }
                    pending: list[IndexedRow] = [
                        row for row in indexed if row.record_id in pending_keys
                    ]
                    pending_inserted = len(outcome.new)
                    pending_updated = len(outcome.changed)
                    if not pending:
                        continue
                    coordinator.verify_publish(publish_lease)
                    if not collection_exists:
                        store.create_collection(spec)
                        collection_exists = True
                    if search.access == "governed":
                        # Revoke-before-upsert: a changed governed record is
                        # removed first, so a failed publish leaves the old,
                        # possibly more permissive row absent, not queryable.
                        changed_ids = [
                            row.record_id
                            for row in pending
                            if row.record_id in prior_state
                        ]
                        if changed_ids:
                            revoke_digest = canonical_fingerprint(
                                {
                                    "snapshot": snapshot.fingerprint,
                                    "config": spec.config_fingerprint,
                                    "ordinal": ordinal,
                                    "revoked": changed_ids,
                                },
                                domain="dbt-ml-search-governed-revoke-batch",
                            )
                            receipt = store.delete(
                                physical,
                                changed_ids,
                                id_field=search.id_field,
                                mutation_digest=revoke_digest,
                            )
                            if not receipt.acknowledged or len(receipt.outcomes) != len(
                                changed_ids
                            ):
                                raise RunError(
                                    "Retrieval store did not return an exact durable "
                                    "governed-revoke receipt"
                                )
                    digest = canonical_fingerprint(
                        {
                            "snapshot": snapshot.fingerprint,
                            "config": spec.config_fingerprint,
                            "ordinal": ordinal,
                            "rows": [row.input_fingerprint for row in pending],
                        },
                        domain="dbt-ml-search-upsert-batch",
                    )
                    receipt = store.upsert(
                        physical,
                        pending,
                        id_field=search.id_field,
                        mutation_digest=digest,
                    )
                    if not receipt.acknowledged or len(receipt.outcomes) != len(pending):
                        raise RunError(
                            "Retrieval store did not return an exact durable upsert receipt"
                        )
                    # Advance state per batch, only for receipt-acknowledged
                    # rows: a later failure leaves those rows durably
                    # published and correctly recorded, while readiness stays
                    # gated by the serving ledger.
                    coordinator.verify_publish(publish_lease)
                    adapter.upsert_state(
                        state_scope,
                        [
                            StateRecord(
                                row.record_id, row.input_fingerprint, code_version
                            )
                            for row in pending
                        ],
                    )
                    inserted += pending_inserted
                    updated += pending_updated
                    rows_written += len(pending)

                # Stale discovery streams state pages whose keys no longer
                # exist upstream, in ascending key order — complete even for
                # an empty upstream, with residency bounded by one page.
                stale_pages = reconciler.iter_stale_pages(
                    upstream_table=upstream,
                    key_column=search.id_field,
                )
                try:
                    for ordinal, stale_page in enumerate(stale_pages):
                        record_ids = [record.record_key for record in stale_page]
                        coordinator.verify_publish(publish_lease)
                        digest = canonical_fingerprint(
                            {
                                "snapshot": snapshot.fingerprint,
                                "config": spec.config_fingerprint,
                                "ordinal": ordinal,
                                "state": [
                                    record.input_fingerprint
                                    for record in stale_page
                                ],
                            },
                            domain="dbt-ml-search-delete-batch",
                        )
                        receipt = store.delete(
                            physical,
                            record_ids,
                            id_field=search.id_field,
                            mutation_digest=digest,
                        )
                        if not receipt.acknowledged or len(receipt.outcomes) != len(
                            record_ids
                        ):
                            raise RunError(
                                "Retrieval store did not return an exact durable "
                                "delete receipt"
                            )
                        adapter.delete_state(state_scope, record_ids)
                        deleted += len(record_ids)
                finally:
                    stale_pages.close()

                if not collection_exists:
                    coordinator.verify_publish(publish_lease)
                    store.create_collection(spec)
                metadata = store.ensure_indexes(spec)
                if metadata.config_fingerprint != spec.config_fingerprint:
                    raise RunError(
                        "Retrieval collection failed post-publication configuration validation"
                    )
                if metadata.row_count != rows_seen:
                    raise RunError(
                        "Retrieval collection failed post-publication row-count validation"
                    )
                active_generation = metadata.physical_generation

        assert state_scope is not None
        coordinator.verify_publish(publish_lease)
        assert active_generation is not None
        assert spec is not None
        coordinator.mark_ready(
            publish_lease,
            active_generation=active_generation,
            config_fingerprint=spec.config_fingerprint,
            counts=(inserted, updated, skipped, deleted),
        )
    except (AdapterError, RetrievalError, RunError) as error:
        if publish_lease is not None:
            _mark_search_publication_failed(
                coordinator,
                publish_lease,
                error,
                counts=(inserted, updated, skipped, deleted),
            )
        if isinstance(error, RunError):
            raise
        raise RunError(str(error)) from None

    assert spec is not None
    safe_target = store.safe_descriptor()
    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="search",
        backend=store_config.type,
        documents_processed=inserted + updated,
        documents_skipped=skipped,
        documents_deleted=deleted,
        rows_written=rows_written,
        rows_inserted=inserted,
        rows_updated=updated,
        serving_resource={
            "type": "retrieval_index",
            "store_type": store_config.type,
            "safe_target_identity": safe_target.safe_target_identity,
            "logical_collection": logical_collection,
            "physical_collection": store.physical_collection(logical_collection),
            "config_fingerprint": spec.config_fingerprint,
            "status": "ready",
            "active_generation": active_generation,
            "fencing_token": publish_lease.fencing_token if publish_lease else None,
        },
    )


def _mark_search_publication_failed(
    coordinator: ServingCoordinator,
    lease: PublishLease,
    error: Exception,
    *,
    counts: tuple[int, int, int, int],
) -> None:
    """Record the failure under a safe code; a stale fence has nothing to record."""
    if isinstance(error, ServingCoordinationError):
        code = "coordination_error"
    elif isinstance(error, RetrievalError):
        code = "store_error"
    elif isinstance(error, AdapterError):
        code = "warehouse_error"
    else:
        code = "publication_failed"
    with suppress(ServingCoordinationError):
        coordinator.mark_failed(lease, safe_error_code=code, counts=counts)


def _search_collection_spec(
    *,
    model: ModelConfig,
    models_by_name: Mapping[str, ModelConfig],
    physical_collection: str,
    upstream_schema: pa.Schema,
    store_type: str,
    resolved: ResolvedProfile,
) -> CollectionSpec:
    search = model.search
    assert search is not None
    projected = search.projected_fields()
    missing = [field for field in projected if field not in upstream_schema.names]
    if missing:
        raise RunError(
            f"Search resource '{model.name}' upstream schema is missing declared fields: "
            f"{', '.join(missing)}"
        )
    fields: list[pa.Field] = []
    for name in projected:
        upstream = upstream_schema.field(name)
        if search.vector is not None and name == search.vector.field:
            if not _is_numeric_list_type(upstream.type):
                raise RunError("Search vector field must use a numeric list warehouse type")
            if (
                pa.types.is_fixed_size_list(upstream.type)
                and upstream.type.list_size != search.vector.dimensions
            ):
                raise RunError(
                    "Search vector warehouse type does not match configured dimensions"
                )
            fields.append(pa.field(name, pa.list_(pa.float32(), search.vector.dimensions)))
        else:
            fields.append(pa.field(name, upstream.type, nullable=upstream.nullable))
    schema = pa.schema(fields)
    _validate_search_schema(model, schema=schema)
    embedding_runtime = resolve_search_embedding_options(
        model,
        models_by_name,
        resolved,
    )
    config_fingerprint = collection_config_fingerprint(
        effective_search_config(
            model,
            models_by_name,
            profile_options=embedding_runtime.provider_options
            if embedding_runtime is not None
            else None,
        ),
        store_type=store_type,
    )
    return CollectionSpec(
        logical_name=search.collection or model.name,
        physical_name=physical_collection,
        id_field=search.id_field,
        text_fields=search.text_fields,
        full_text_fields=search.full_text.fields if search.full_text else (),
        attribute_fields=tuple(attribute.name for attribute in search.attributes),
        scalar_index_fields=tuple(
            attribute.name
            for attribute in search.attributes
            if attribute.filter_role != "none" or attribute.sortable
        ),
        display_fields=search.display_fields,
        vector_field=search.vector.field if search.vector else None,
        vector_dimensions=search.vector.dimensions if search.vector else None,
        distance_metric=search.vector.metric if search.vector else None,
        vector_search=search.vector.search if search.vector else None,
        config_fingerprint=config_fingerprint,
        arrow_schema=schema,
    )


def _indexed_rows(
    batch: pa.RecordBatch,
    model: ModelConfig,
    config_fingerprint: str,
    *,
    max_id_bytes: int | None,
) -> list[IndexedRow]:
    search = model.search
    assert search is not None
    rows: list[IndexedRow] = []
    for raw in batch.to_pylist():
        record_id = raw.get(search.id_field)
        if (
            not isinstance(record_id, str)
            or not record_id
            or "\x00" in record_id
            or (max_id_bytes is not None and len(record_id.encode()) > max_id_bytes)
        ):
            raise RunError(
                "Search input IDs must be non-empty, NUL-free strings within the "
                "retrieval store byte limit"
            )
        values = dict(raw)
        for optional_id_field in (search.document_id_field, search.chunk_id_field):
            if optional_id_field is None:
                continue
            optional_id = values.get(optional_id_field)
            if optional_id is not None and (
                not isinstance(optional_id, str)
                or not optional_id
                or "\x00" in optional_id
                or (max_id_bytes is not None and len(optional_id.encode()) > max_id_bytes)
            ):
                raise RunError("Search document and chunk IDs must be valid strings")
        for text_field in search.text_fields:
            if not isinstance(values.get(text_field), str):
                raise RunError("Search text fields must contain non-null strings")
        if not any(values[text_field] for text_field in search.text_fields):
            raise RunError("Search input must contain at least one non-empty text field")
        if search.vector is not None:
            vector = values.get(search.vector.field)
            if not isinstance(vector, Sequence) or isinstance(vector, str | bytes):
                raise RunError("Search vectors must be finite numeric sequences")
            if any(isinstance(item, bool) for item in vector):
                raise RunError("Search vectors must be finite numeric sequences")
            try:
                normalized = [float(item) for item in vector]
            except (TypeError, ValueError):
                raise RunError("Search vectors must be finite numeric sequences") from None
            if len(normalized) != search.vector.dimensions or any(
                not isfinite(item) for item in normalized
            ):
                raise RunError(
                    "Search vectors must match the configured dimensions and be finite"
                )
            values[search.vector.field] = normalized
        for attribute in search.attributes:
            _validate_search_attribute(values.get(attribute.name), attribute)
        for display_field in search.display_fields:
            if not _is_json_safe(values.get(display_field)):
                raise RunError("Search display fields must contain JSON-safe values")
        input_fingerprint = canonical_fingerprint(
            {"config": config_fingerprint, "row": values},
            domain="dbt-ml-search-indexed-row",
        )
        rows.append(IndexedRow(record_id, values, input_fingerprint))
    return rows


def _validate_search_attribute(value: Any, attribute: Any) -> None:
    if value is None:
        if attribute.nullable:
            return
        raise RunError("Search attributes declared non-nullable must not contain NULL")
    expected: dict[str, type[Any] | tuple[type[Any], ...]] = {
        "string": str,
        "integer": int,
        "float": (int, float),
        "boolean": bool,
        "date": (date, str),
        "timestamp": (datetime, str),
    }
    if attribute.data_type == "array[string]":
        if not isinstance(value, list | tuple) or any(
            not isinstance(item, str) for item in value
        ):
            raise RunError("Search array attributes must contain only strings")
        return
    required = expected[attribute.data_type]
    if attribute.data_type == "integer" and isinstance(value, bool):
        raise RunError("Search attribute type does not match its declaration")
    if attribute.data_type == "float" and isinstance(value, bool):
        raise RunError("Search attribute type does not match its declaration")
    if not isinstance(value, required):
        raise RunError("Search attribute type does not match its declaration")
    if attribute.data_type == "integer" and not -(2**63) <= value < 2**63:
        raise RunError("Search integer attributes must fit signed 64-bit values")
    if attribute.data_type == "float" and not isfinite(float(value)):
        raise RunError("Search float attributes must be finite")
    if attribute.data_type == "date" and isinstance(value, str):
        try:
            date.fromisoformat(value)
        except ValueError:
            raise RunError("Search date attributes must be ISO-8601 dates") from None
    if attribute.data_type == "timestamp":
        parsed = value
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                raise RunError(
                    "Search timestamp attributes must be ISO-8601 timestamps"
                ) from None
        if parsed.utcoffset() != timedelta(0):
            raise RunError("Search timestamp attributes must be UTC")


def _validate_search_schema(model: ModelConfig, *, schema: pa.Schema) -> None:
    search = model.search
    assert search is not None
    for field_name in (search.id_field, search.document_id_field, search.chunk_id_field):
        if field_name is not None and not _is_string_type(schema.field(field_name).type):
            raise RunError("Search ID fields must use a string warehouse type")
    for field_name in search.text_fields:
        if not _is_string_type(schema.field(field_name).type):
            raise RunError("Search text fields must use a string warehouse type")
    for attribute in search.attributes:
        if not _attribute_type_matches(
            attribute.data_type, schema.field(attribute.name).type
        ):
            raise RunError("Search attribute warehouse type does not match its declaration")


def _is_string_type(value: pa.DataType) -> bool:
    return cast(bool, pa.types.is_string(value) or pa.types.is_large_string(value))


def _is_numeric_list_type(value: pa.DataType) -> bool:
    if not (
        pa.types.is_list(value)
        or pa.types.is_large_list(value)
        or pa.types.is_fixed_size_list(value)
    ):
        return False
    return cast(
        bool,
        pa.types.is_integer(value.value_type) or pa.types.is_floating(value.value_type),
    )


def _attribute_type_matches(data_type: str, value: pa.DataType) -> bool:
    if data_type == "string":
        return _is_string_type(value)
    if data_type == "integer":
        return cast(bool, pa.types.is_integer(value))
    if data_type == "float":
        return cast(bool, pa.types.is_floating(value))
    if data_type == "boolean":
        return cast(bool, pa.types.is_boolean(value))
    if data_type == "date":
        return cast(bool, pa.types.is_date(value))
    if data_type == "timestamp":
        return cast(bool, pa.types.is_timestamp(value))
    if not (
        pa.types.is_list(value)
        or pa.types.is_large_list(value)
        or pa.types.is_fixed_size_list(value)
    ):
        return False
    return _is_string_type(value.value_type)


def _is_json_safe(value: Any) -> bool:
    if value is None or isinstance(value, str | bool | int):
        return True
    if isinstance(value, float):
        return isfinite(value)
    if isinstance(value, list | tuple):
        return all(_is_json_safe(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _is_json_safe(item) for key, item in value.items())
    return False


def _run_ml_model(
    *,
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
) -> ModelRunResult:
    assert model.ml is not None
    if model.materialization == "incremental":
        raise RunError(
            f"ML model '{model.name}' declares `materialization: incremental`, "
            "but ML models only support `full` today. Set `materialization: full` "
            "(or omit it) — see issue #53."
        )
    output = None
    try:
        output = run_classic_ml_model(
            model=model,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
        )
        rows_written = adapter.materialize_full(
            model.name, output.df, options=_warehouse_options(adapter, model)
        )
        for suffix, table_df in output.secondary_tables.items():
            adapter.materialize_full(
                f"{model.name}__{suffix}",
                table_df,
                options=_warehouse_options(adapter, model),
            )
        output.publish_artifact()
    except BaseException as e:
        if output is not None:
            output.discard_staged_artifact()
        if not isinstance(e, Exception):
            raise
        raise RunError(f"ML model '{model.name}' failed: {e}") from e

    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="ml",
        rows_written=rows_written,
        artifact_path=str(output.artifact_path),
        artifact_version=output.artifact_version,
        training_input=output.training_input,
        metrics=output.metrics,
        artifact_metadata=output.artifact_metadata,
    )


def clean_project(
    project_dir: Path,
) -> str:
    """Remove known dbt-ml artifacts without invoking warehouse cleanup."""
    project, _, _ = load_project(project_dir)
    project_root = project_dir.resolve()
    target_dir = resolve_within_project(
        project.target_path, project_dir, surface="`target-path`"
    )

    if target_dir == project_root:
        raise RunError(
            "Refusing to clean because `target-path` resolves to the project root."
        )
    relative_target = target_dir.relative_to(project_root)
    if relative_target.parts[0] in {".git", ".hg", ".svn"}:
        raise RunError(
            f"Refusing to clean reserved project metadata path {target_dir}."
        )

    for label, paths in (
        ("source-paths", project.source_paths),
        ("model-paths", project.model_paths),
        ("transform-paths", project.transform_paths),
    ):
        for configured_path in paths:
            protected = resolve_within_project(
                configured_path, project_dir, surface=f"`{label}`"
            )
            if target_dir.is_relative_to(protected) or protected.is_relative_to(
                target_dir
            ):
                raise RunError(
                    f"Refusing to clean {target_dir} because it overlaps "
                    f"configured `{label}` path {protected}."
                )

    lexical_target = Path(
        os.path.abspath(
            project.target_path
            if project.target_path.is_absolute()
            else project_root / project.target_path
        )
    )
    try:
        lexical_parts = lexical_target.relative_to(project_root).parts
    except ValueError as e:
        raise RunError(
            "Refusing to clean a target path that enters the project through "
            f"a symlink: {lexical_target}."
        ) from e
    current = project_root
    for part in lexical_parts:
        current /= part
        if current.is_symlink():
            raise RunError(
                f"Refusing to clean target path with symlink component {current}."
            )

    if not target_dir.exists():
        return str(target_dir)
    if not target_dir.is_dir():
        raise RunError(f"Configured target path is not a directory: {target_dir}")

    for filename in ("manifest.json", "run_results.json", "sources.yml"):
        artifact = target_dir / filename
        if artifact.is_symlink():
            raise RunError(f"Refusing to clean symlinked artifact {artifact}.")
        if artifact.exists():
            if not artifact.is_file():
                raise RunError(f"Expected generated artifact to be a file: {artifact}")
            artifact.unlink()

    for dirname in ("docs", "artifacts"):
        artifact_dir = target_dir / dirname
        if artifact_dir.is_symlink():
            raise RunError(f"Refusing to clean symlinked artifact {artifact_dir}.")
        if artifact_dir.exists():
            if not artifact_dir.is_dir():
                raise RunError(
                    f"Expected generated artifact to be a directory: {artifact_dir}"
                )
            shutil.rmtree(artifact_dir)

    try:
        target_dir.rmdir()
    except OSError:
        pass
    return str(target_dir)
