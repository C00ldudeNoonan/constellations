from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
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
from .agent_context import (
    AgentContextValidationError,
    empty_agent_context_frame,
    validate_agent_context_frame,
)
from .backends import (
    BackendOptionsError,
    BaseBackend,
    ExtractionResult,
    get_backend,
    validate_backend_options,
)
from .backends.llm_backend import BatchCancelledError
from .backends.options import LLMBackendOptions
from .budget import BudgetExceededError, BudgetGuard, BudgetLedger
from .checks import TestResult, run_model_tests
from .chunking import chunk_id, split_text
from .classic_ml import run_classic_ml_model
from .compiler import (
    validate_project_contract,
    validate_retrieval_capabilities,
    validate_warehouse_capabilities,
)
from .config import load_project
from .config.model import (
    EMBED_METADATA_FIELDS,
    INTERNAL_LINEAGE_FIELDS,
    LLMTransformConfig,
    ModelConfig,
)
from .config.profile import DEFAULT_LLM_PROVIDER, PricingConfig
from .config.project import ProjectConfig
from .config.source import SourceConfig
from .dag import ProjectDAG, parse_ref
from .embedding import EmbeddingIdentity, effective_search_config, embed_texts
from .hashing import canonical_fingerprint
from .llm_map import LLMMapError, LLMMapRuntime, execute_map_item, resolve_llm_runtime
from .paths import resolve_within_project
from .profile import (
    ResolvedProfile,
    apply_source_path_overrides,
    resolve_llm_options,
    resolve_profile,
)
from .providers import (
    InferenceProvider,
    ProviderBatchError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRequestError,
    ProviderResponseError,
    get_inference_provider,
    resolve_provider_model,
    sanitized_provider_error,
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
from .sources import DocumentRef, DocumentSource, SourceError, get_document_source
from .state_reconciliation import BoundedReconciler, UpstreamRecord
from .transforms import TransformContext, load_transform, transform_call_arity
from .versioning import compute_code_version, compute_model_code_version

log = logging.getLogger(__name__)

_EXTRACTION_LINEAGE_SCHEMA: dict[str, Any] = {
    "document_id": pl.String,
    "source_path": pl.String,
    "source_uri": pl.String,
    # Remote sources populate this JSON string; local rows retain it as NULL.
    "source_metadata": pl.String,
    "content_hash": pl.String,
    "code_version": pl.String,
    "backend_name": pl.String,
    "backend_version": pl.String,
    "extracted_at": pl.String,
}
_EXTRACTION_FIELD_DTYPES: dict[str, Any] = {
    "string": pl.String,
    "integer": pl.Int64,
    "float": pl.Float64,
    "boolean": pl.Boolean,
    "date": pl.Date,
    "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
    "json": pl.String,
}

_CHUNK_GENERATED_FIELDS = frozenset(
    {
        "chunk_id",
        "document_id",
        "chunk_index",
        "chunk_count",
        "text",
        "chunk_strategy",
        "code_version",
        "chunked_at",
    }
)
# These upstream values are replaced by the chunk model rather than carried.
# Everything else, including timestamps and ACL/filter metadata, participates
# in row invalidation because it is materialized on every generated chunk.
_CHUNK_INPUT_EXCLUDED_FIELDS = _CHUNK_GENERATED_FIELDS

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

class RunError(Exception):
    pass


def _validate_agent_context_output(
    frame: pl.DataFrame, model: ModelConfig
) -> pl.DataFrame:
    if model.agent_context is None:
        return frame
    if frame.is_empty() and not frame.columns:
        frame = empty_agent_context_frame(model.agent_context.grain)
    try:
        validate_agent_context_frame(frame, model.agent_context.grain)
    except AgentContextValidationError as error:
        raise RunError(f"Model '{model.name}' produced invalid {error}") from error
    return frame


def _artifact_error_text(error: Exception) -> str:
    provider_error = _provider_error_in_chain(error)
    if isinstance(provider_error, ProviderConfigurationError):
        return "ProviderConfigurationError: provider configuration is invalid"
    if isinstance(provider_error, ProviderRequestError):
        safe = sanitized_provider_error(
            provider_error.provider,
            provider_error.operation,
            provider_error,
        )
        return f"ProviderRequestError: {safe}"
    if isinstance(provider_error, ProviderResponseError):
        return "ProviderResponseError: provider response is invalid"
    if isinstance(provider_error, ProviderBatchError):
        return "ProviderBatchError: provider batch operation failed"
    if isinstance(provider_error, ProviderError):
        return "ProviderError: provider operation failed"
    return f"{type(error).__name__}: {error}"


def _provider_error_in_chain(error: BaseException) -> ProviderError | None:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, ProviderError):
            return current
        current = current.__cause__
    return None


class _FullExtractionFailed(Exception):
    pass


def _extraction_schema(model: ModelConfig) -> dict[str, Any]:
    schema = dict(_EXTRACTION_LINEAGE_SCHEMA)
    names = {name.casefold(): name for name in schema}
    for field_config in model.fields:
        folded = field_config.name.casefold()
        existing = names.get(folded)
        if existing is not None:
            if existing in _EXTRACTION_LINEAGE_SCHEMA:
                if field_config.data_type not in {None, "string", "json"}:
                    raise RunError(
                        f"Extraction model '{model.name}' declares lineage field "
                        f"'{field_config.name}' as {field_config.data_type}; lineage "
                        "fields use string storage"
                    )
                continue
            raise RunError(
                f"Extraction model '{model.name}' declares duplicate field "
                f"'{field_config.name}'"
            )
        names[folded] = field_config.name
        schema[field_config.name] = (
            _EXTRACTION_FIELD_DTYPES[field_config.data_type]
            if field_config.data_type is not None
            else pl.String
        )
    return schema


def _empty_extraction_frame(model: ModelConfig) -> pl.DataFrame:
    return pl.DataFrame(schema=_extraction_schema(model))


def _apply_extraction_contract(
    frame: pl.DataFrame, model: ModelConfig
) -> pl.DataFrame:
    schema = _extraction_schema(model)
    typed_names = {
        field_config.name: field_config.data_type
        for field_config in model.fields
        if field_config.data_type is not None
        and field_config.name.casefold() not in INTERNAL_LINEAGE_FIELDS
    }
    expressions: list[pl.Expr] = []
    for name, dtype in schema.items():
        if name in frame.columns:
            if name in typed_names:
                data_type = typed_names[name]
                if data_type == "date" and frame.schema[name] == pl.String:
                    expressions.append(pl.col(name).str.to_date(strict=True))
                elif data_type == "timestamp" and frame.schema[name] == pl.String:
                    expressions.append(
                        pl.col(name).str.to_datetime(time_zone="UTC", strict=True)
                    )
                else:
                    expressions.append(pl.col(name).cast(dtype, strict=True))
            continue
        expressions.append(pl.lit(None, dtype=dtype).alias(name))
    try:
        contracted = frame.with_columns(expressions) if expressions else frame
    except Exception as e:
        raise RunError(
            f"Extraction model '{model.name}' produced a value that does not "
            f"match its declared field data_type: {e}"
        ) from e
    if model.fields:
        return contracted.select(list(schema))
    return contracted


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
    # Local import: manifest.py imports ModelRunResult from this module.
    from .manifest import compute_modified_models

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
class DiscoveredSource:
    """A source's backend plus its discovered documents for this run."""

    backend: DocumentSource
    refs: list[DocumentRef]


@dataclass
class ModelRunResult:
    model_name: str
    materialization: str
    kind: str  # "extraction" | "transform"
    # None derives success/error from `errors`; set explicitly for the
    # distinct budget_exceeded / cancelled outcomes (issue #149).
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
    # Non-fatal backend warnings, aggregated: distinct message -> number of
    # documents that raised it. Never affects status or exit code.
    warnings: dict[str, int] = field(default_factory=dict)
    artifact_path: str | None = None
    artifact_version: str | None = None
    training_input: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_metadata: dict[str, Any] | None = None
    serving_resource: dict[str, Any] | None = None


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


def _warehouse_options(adapter: WarehouseAdapter, model: ModelConfig) -> Any:
    """Parse model-level warehouse_options through the active adapter,
    surfacing validation problems as a model config error."""
    try:
        return adapter.parse_warehouse_options(
            model.warehouse_options, model_name=model.name
        )
    except AdapterError as e:
        raise RunError(str(e)) from e


def _run_extraction_model(
    *,
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    source_docs: dict[str, DiscoveredSource],
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    full_refresh: bool,
    threads: int = 1,
    run_budget: BudgetLedger | None = None,
) -> ModelRunResult:
    assert model.extraction is not None
    backend_name = model.extraction.backend or project.extraction.default_backend
    backend = get_backend(backend_name)
    options = model.extraction.options
    if backend_name == "llm":
        if "cache_path" in options:
            # A model-level cache_path is project YAML: confine it. External
            # cache locations belong in the (trusted) profiles.yml llm block.
            options = {
                **options,
                "cache_path": str(
                    resolve_within_project(
                        options["cache_path"],
                        project_dir,
                        surface=f"Model '{model.name}' llm cache_path",
                        hint="Set llm.cache_path in profiles.yml for "
                        "locations outside the project.",
                    )
                ),
            }
        options = resolve_llm_options(options, resolved)
    try:
        options = validate_backend_options(backend_name, options)
    except BackendOptionsError as e:
        raise RunError(f"Extraction model '{model.name}' has {e}") from e

    inference_provider: InferenceProvider | None = None
    provider_name: str | None = None
    provider_model: str | None = None
    provider_implementation: str | None = None
    if backend_name == "llm":
        llm_options = LLMBackendOptions.model_validate(options)
        provider_name = llm_options.provider
        inference_provider = get_inference_provider(provider_name)
        if llm_options.model is None:
            raise RunError(
                f"Extraction model '{model.name}' has no effective LLM model; "
                "set one in the model options or profile"
            )
        provider_model = llm_options.model
        provider_implementation = inference_provider.implementation_identity()

    budget_guard: BudgetGuard | None = None
    if backend_name == "llm":
        model_ledger = (
            BudgetLedger(llm_options.budget, scope=f"model '{model.name}'")
            if llm_options.budget is not None
            else None
        )
        if model_ledger is not None or run_budget is not None:
            budget_guard = BudgetGuard(
                model_ledger,
                run_budget,
                cost_estimator=_budget_cost_estimator(
                    resolved,
                    batch=bool(options.get("batch")),
                    provider=inference_provider,
                ),
            )

    if not model.source:
        raise RunError(f"Extraction model '{model.name}' must declare a `source:`")
    source_name = parse_ref(model.source)
    discovered = source_docs.get(source_name)
    if discovered is None:
        raise RunError(
            f"Model '{model.name}' references unknown source '{source_name}'"
        )
    docs = discovered.refs
    source_backend = discovered.backend

    code_version = compute_model_code_version(
        model,
        project,
        project_dir,
        resolved=resolved,
    )
    warehouse_opts = _warehouse_options(adapter, model)
    state_scope = StateScope(model.name)

    is_incremental = model.materialization == "incremental" and not full_refresh
    processed_state = adapter.fetch_state(state_scope) if is_incremental else {}
    existing_tables = set(adapter.list_tables()) if is_incremental else set()
    empty_incremental_target = (
        is_incremental
        and not processed_state
        and model.name in existing_tables
        and adapter.row_count(model.name) == 0
    )

    docs_to_process: list[DocumentRef] = []
    for doc in docs:
        if is_incremental:
            prior = processed_state.get(doc.document_id)
            if prior == StateValue(doc.content_hash, code_version):
                continue
        docs_to_process.append(doc)

    deleted = 0
    if is_incremental:
        current_ids = {doc.document_id for doc in docs}
        removed = [doc_id for doc_id in processed_state if doc_id not in current_ids]
        if removed:
            adapter.delete_rows_and_state(
                model.name,
                key_col="document_id",
                keys=removed,
                state_scope=state_scope,
            )
            deleted = len(removed)

    skipped = len(docs) - len(docs_to_process)
    total_docs = len(docs_to_process)
    flush_every = model.extraction.flush_every
    use_full = model.materialization == "full" or full_refresh

    errors: list[str] = []
    warning_counts: Counter[str] = Counter()
    if not docs:
        warning_counts[
            f"Source '{source_name}' matched zero documents; verify its path and "
            "file_pattern."
        ] += 1
    usage_totals: dict[str, Any] = {}
    full_state_records: list[StateRecord] = []
    full_committed = False
    rows_written = 0
    docs_flushed = 0

    backend_version = backend.version()
    # One timestamp per model run: rows from the same run are batch-identifiable.
    extracted_at = datetime.now(UTC).isoformat()

    def _rows_for_chunk(
        extracted: list[tuple[DocumentRef, ExtractionResult | None, str | None]],
    ) -> tuple[list[dict[str, Any]], list[StateRecord]]:
        chunk_rows: list[dict[str, Any]] = []
        chunk_records: list[StateRecord] = []
        for doc, result, err in extracted:
            if err is not None or result is None:
                errors.append(f"{doc.relative_path}: {err}")
                continue
            warning_counts.update(set(result.warnings))
            for key, value in result.metrics.items():
                if isinstance(value, int | float):
                    usage_totals[key] = usage_totals.get(key, 0) + value
            chunk_rows.append(
                _row_for_extraction(
                    doc,
                    code_version,
                    result,
                    backend_name=backend_name,
                    backend_version=backend_version,
                    extracted_at=extracted_at,
                )
            )
            chunk_records.append(
                StateRecord(doc.document_id, doc.content_hash, code_version)
            )
        return chunk_rows, chunk_records

    run_status: str | None = None
    try:
        if budget_guard is not None and docs_to_process:
            budget_guard.charge_documents(len(docs_to_process))
        # Sources snapshot into a per-model scratch dir, lazily and only for
        # documents that actually need processing. Extraction streams through in
        # `flush_every`-sized chunks (issue #77): rows never accumulate beyond
        # one chunk, so corpus size is bounded by the flush size, not memory.
        with tempfile.TemporaryDirectory(prefix="dbt_ml_fetch_") as scratch:
            work_dir = Path(scratch)

            def _one(
                doc: DocumentRef,
            ) -> tuple[DocumentRef, ExtractionResult | None, str | None]:
                try:
                    if budget_guard is not None:
                        budget_guard.ensure_headroom()
                    local_path = source_backend.fetch(doc, work_dir)
                    if budget_guard is not None:
                        size = local_path.stat().st_size
                        budget_guard.check_file_bytes(size)
                        budget_guard.charge_bytes(size)
                    result = backend.extract(local_path, options)
                    if budget_guard is not None:
                        budget_guard.charge_metrics(result.metrics)
                    return doc, result, None
                except BudgetExceededError:
                    raise
                except Exception as e:
                    # Provider errors reach here already sanitized with their
                    # chains severed; redacted SDK diagnostics were logged at the
                    # provider boundary. Safe to log in full.
                    log.debug(
                        "extraction failed for %s", doc.relative_path, exc_info=True
                    )
                    return doc, None, _artifact_error_text(e)

            def _iter_extracted() -> (
                Iterator[list[tuple[DocumentRef, ExtractionResult | None, str | None]]]
            ):
                if options.get("batch") and docs_to_process:
                    # Deterministic windows bound fetch, text, and result memory
                    # (issue #149); each window is one or more resumable native
                    # batch submissions inside the backend.
                    on_partial = str(options.get("on_partial_batch", "fail"))
                    window_size = max(int(options.get("batch_size", 1000)), 1)
                    for start in range(0, total_docs, window_size):
                        window = docs_to_process[start : start + window_size]
                        extracted_window, batch_metrics = _extract_batched(
                            window,
                            source_backend,
                            backend,
                            options,
                            work_dir,
                            model.name,
                            budget=budget_guard,
                        )
                        for key, value in batch_metrics.items():
                            if isinstance(value, int | float):
                                usage_totals[key] = usage_totals.get(key, 0) + value
                        if on_partial == "fail":
                            failed = [
                                (doc, err)
                                for doc, _res, err in extracted_window
                                if err is not None
                            ]
                            if failed:
                                for doc, err in failed:
                                    errors.append(f"{doc.relative_path}: {err}")
                                raise RunError(
                                    f"Batch for model '{model.name}' returned "
                                    f"{len(failed)} failed document(s); "
                                    "on_partial_batch=fail publishes nothing "
                                    "further from this run. Set on_partial_batch: "
                                    "publish_successful to record per-document "
                                    "failures and keep successes instead."
                                )
                        for i in range(0, len(extracted_window), flush_every):
                            yield extracted_window[i : i + flush_every]
                    return
                for i in range(0, total_docs, flush_every):
                    chunk = docs_to_process[i : i + flush_every]
                    if threads > 1 and len(chunk) > 1:
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=threads
                        ) as ex:
                            yield list(ex.map(_one, chunk))
                    else:
                        yield [_one(d) for d in chunk]

            if use_full:
                # Chunks stream into a staging table that atomically replaces the
                # target at the end; state upserts once, after the swap.
                def _frames() -> Iterator[pl.DataFrame]:
                    nonlocal docs_flushed
                    yielded = False
                    for extracted in _iter_extracted():
                        chunk_rows, chunk_records = _rows_for_chunk(extracted)
                        full_state_records.extend(chunk_records)
                        docs_flushed += len(extracted)
                        if chunk_rows:
                            log.info(
                                "staged %d rows (%d/%d docs) for %s",
                                len(chunk_rows),
                                docs_flushed,
                                total_docs,
                                model.name,
                            )
                            yielded = True
                            yield _apply_extraction_contract(
                                pl.DataFrame(chunk_rows), model
                            )
                    if errors:
                        raise _FullExtractionFailed
                    if not yielded:
                        yield _empty_extraction_frame(model)

                try:
                    rows_written = adapter.materialize_full_chunks(
                        model.name, _frames(), options=warehouse_opts
                    )
                    full_committed = True
                except _FullExtractionFailed:
                    rows_written = 0
                except AdapterError as e:
                    raise RunError(str(e)) from e
            else:
                # Incremental: each flush upserts rows and its state immediately —
                # a killed run keeps completed chunks, and the re-run skips them.
                first_flush = True
                for extracted in _iter_extracted():
                    chunk_rows, chunk_records = _rows_for_chunk(extracted)
                    docs_flushed += len(extracted)
                    if not chunk_rows:
                        continue
                    try:
                        rows_written += adapter.materialize_incremental(
                            model.name,
                            _apply_extraction_contract(pl.DataFrame(chunk_rows), model),
                            key_col="document_id",
                            # The model's policy governs run-over-run drift on the
                            # first flush; later flushes union within-run drift,
                            # matching what one whole-run DataFrame did.
                            on_schema_change=(
                                "append_new_columns"
                                if first_flush and empty_incremental_target
                                else model.on_schema_change
                                if first_flush
                                else "append_new_columns"
                            ),
                            options=warehouse_opts,
                        )
                    except AdapterError as e:
                        # RunError so `build` fails this model and blocks
                        # descendants instead of aborting the whole invocation.
                        raise RunError(str(e)) from e
                    first_flush = False
                    adapter.upsert_state(state_scope, chunk_records)
                    log.info(
                        "flushed %d rows (%d/%d docs) for %s",
                        len(chunk_rows),
                        docs_flushed,
                        total_docs,
                        model.name,
                    )

                if not docs and model.name not in existing_tables:
                    try:
                        adapter.materialize_full(
                            model.name,
                            _empty_extraction_frame(model),
                            options=warehouse_opts,
                        )
                    except AdapterError as e:
                        raise RunError(str(e)) from e

    except BudgetExceededError as e:
        # Exhaustion fires before the next provider call. Chunks already
        # committed stay (state advanced only for published IDs, #139);
        # everything else is unpublished.
        run_status = "budget_exceeded"
        errors.append(f"BudgetExceededError: {e}")
    except BatchCancelledError as e:
        run_status = "cancelled"
        errors.append(f"BatchCancelledError: {e}")

    if usage_totals and options.get("batch"):
        usage_totals["batch"] = True
    if usage_totals and resolved.llm is not None and resolved.llm.pricing is not None:
        cost = _estimate_cost(usage_totals, resolved.llm.pricing)
        if options.get("batch") and inference_provider is not None:
            cost = round(cost * inference_provider.batch_cost_multiplier, 6)
        usage_totals["estimated_cost_usd"] = cost

    if use_full and full_committed:
        adapter.replace_state(state_scope, full_state_records)

    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="extraction",
        status=run_status,
        backend=backend_name,
        provider=provider_name,
        provider_model=provider_model,
        provider_implementation=provider_implementation,
        documents_processed=len(docs_to_process),
        documents_skipped=skipped,
        documents_deleted=deleted,
        rows_written=rows_written,
        errors=errors,
        warnings=dict(warning_counts),
        metrics=usage_totals,
    )


def _estimate_cost(totals: dict[str, Any], pricing: PricingConfig) -> float:
    """Token totals × user-supplied USD-per-Mtok rates. Cache rates are
    optional; tokens with no configured rate contribute nothing."""
    rates = [
        ("input_tokens", pricing.input_usd_per_mtok),
        ("output_tokens", pricing.output_usd_per_mtok),
        ("cache_read_input_tokens", pricing.cache_read_usd_per_mtok),
        ("cache_creation_input_tokens", pricing.cache_write_usd_per_mtok),
    ]
    cost = sum(
        float(totals.get(key, 0)) * rate for key, rate in rates if rate is not None
    )
    return round(cost / 1_000_000, 6)


def _budget_cost_estimator(
    resolved: ResolvedProfile,
    *,
    batch: bool,
    provider: InferenceProvider | None,
) -> Callable[[Mapping[str, Any]], float] | None:
    """Per-response USD estimate for spend budgets, honoring the provider's
    native-batch discount. Provider-reported cost wins when present."""
    if resolved.llm is None or resolved.llm.pricing is None:
        return None
    pricing = resolved.llm.pricing
    multiplier = (
        provider.batch_cost_multiplier if batch and provider is not None else 1.0
    )

    def _estimate(metrics: Mapping[str, Any]) -> float:
        return round(_estimate_cost(dict(metrics), pricing) * multiplier, 6)

    return _estimate


def _extract_batched(
    docs: list[DocumentRef],
    source_backend: DocumentSource,
    backend: BaseBackend,
    options: dict[str, Any],
    work_dir: Path,
    model_name: str,
    *,
    budget: BudgetGuard | None = None,
) -> tuple[
    list[tuple[DocumentRef, ExtractionResult | None, str | None]],
    dict[str, Any],
]:
    """Batch-mode extraction: fetch everything up front, hand the backend one
    extract_batch() call, and map its aligned results back per document. Fetch
    and per-item failures stay per-document; only batch submission itself
    fails the model."""
    entries: list[tuple[DocumentRef, Path | None, str | None]] = []
    for doc in docs:
        try:
            entries.append((doc, source_backend.fetch(doc, work_dir), None))
        except Exception as e:
            log.debug("fetch failed for %s", doc.relative_path, exc_info=True)
            entries.append((doc, None, f"{type(e).__name__}: {e}"))

    fetched = [(doc, path) for doc, path, err in entries if path is not None]
    try:
        batch_output = (
            backend.extract_batch_with_metrics(
                [p for _, p in fetched], options, budget=budget
            )
            if fetched
            else None
        )
    except (BudgetExceededError, BatchCancelledError):
        raise
    except Exception as e:
        raise RunError(
            f"Batch extraction failed for model '{model_name}': {e}"
        ) from e
    batch_out = batch_output.items if batch_output is not None else []
    by_doc_id = {
        doc.document_id: res
        for (doc, _), res in zip(fetched, batch_out, strict=True)
    }

    out: list[tuple[DocumentRef, ExtractionResult | None, str | None]] = []
    for doc, _path, err in entries:
        if err is not None:
            out.append((doc, None, err))
            continue
        res = by_doc_id[doc.document_id]
        if isinstance(res, Exception):
            out.append((doc, None, _artifact_error_text(res)))
        else:
            out.append((doc, res, None))
    return out, batch_output.metrics if batch_output is not None else {}


def _row_for_extraction(
    doc: DocumentRef,
    code_version: str,
    result: ExtractionResult,
    *,
    backend_name: str,
    backend_version: str,
    extracted_at: str,
) -> dict[str, Any]:
    conflicts = sorted(
        key
        for key in result.fields
        if key.casefold() in INTERNAL_LINEAGE_FIELDS
    )
    if conflicts:
        raise RunError(
            "Extracted fields collide with reserved dbt-ml lineage columns: "
            f"{', '.join(conflicts)}"
        )
    # The common output contract (issue #85): identity, lineage back to the
    # exact source object, and the parser that produced the row.
    row: dict[str, Any] = {
        "document_id": doc.document_id,
        "source_path": doc.relative_path,
        "source_uri": doc.source_uri,
        "content_hash": doc.content_hash,
        "code_version": code_version,
        "backend_name": backend_name,
        "backend_version": backend_version,
        "extracted_at": extracted_at,
    }
    if doc.source_metadata is not None:
        row["source_metadata"] = json.dumps(doc.source_metadata, default=str)
    for key, value in result.fields.items():
        row[key] = _scalarize(value)
    return row


def _scalarize(value: Any) -> Any:
    """Serialize nested types as JSON strings so DuckDB gets a flat schema."""
    if isinstance(value, dict | list):
        return json.dumps(value, default=str)
    return value


def _run_transform_model(
    *,
    model: ModelConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
) -> ModelRunResult:
    assert model.transform is not None
    if model.materialization == "incremental":
        raise RunError(
            f"Transform model '{model.name}' declares `materialization: incremental`, "
            "but transforms only support `full` today. Set `materialization: full` "
            "(or omit it) — see issue #53."
        )
    if model.transform.type != "python":
        raise RunError(
            f"Model '{model.name}': only `type: python` transforms are supported in v1"
        )
    if not model.transform.module:
        raise RunError(f"Model '{model.name}': transform requires a `module:`")
    if not model.depends_on:
        raise RunError(
            f"Transform model '{model.name}' must declare `depends_on:` for v1"
        )

    provider_name: str | None = None
    provider_model: str | None = None
    provider_implementation: str | None = None
    if model.transform.uses_llm:
        provider_name = (
            resolved.llm.provider
            if resolved.llm is not None
            else DEFAULT_LLM_PROVIDER
        )
        selected_model = resolved.llm.model if resolved.llm is not None else None
        try:
            provider = get_inference_provider(provider_name)
            provider_model = resolve_provider_model(provider, selected_model)
            provider_implementation = provider.implementation_identity()
        except Exception as e:
            raise RunError(
                f"Transform model '{model.name}' could not initialize inference: "
                f"{_artifact_error_text(e)}"
            ) from e

    transform_fn = load_transform(model.transform.module, project_dir)
    deps: dict[str, pl.DataFrame] = {}
    for dep_ref in model.depends_on:
        dep_name = parse_ref(dep_ref)
        deps[dep_name] = adapter.read_table(dep_name)

    try:
        if transform_call_arity(transform_fn) == 2:
            ctx = TransformContext(
                project_dir=project_dir,
                profile_name=resolved.profile_name,
                target_name=resolved.target_name,
                warehouse=resolved.warehouse,
                llm=resolved.llm,
                options=dict(model.transform.options),
            )
            output = transform_fn(deps, ctx)
        else:
            output = transform_fn(deps)
    except RunError:
        raise
    except Exception as e:
        log.debug("transform failed for %s", model.name, exc_info=True)
        raise RunError(
            f"Transform model '{model.name}' failed: {_artifact_error_text(e)}"
        ) from e

    if not isinstance(output, pl.DataFrame):
        raise RunError(
            f"Transform '{model.transform.module}' must return a polars.DataFrame"
        )

    adapter.materialize_full(
        model.name,
        _validate_agent_context_output(output, model),
        options=_warehouse_options(adapter, model),
    )

    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="transform",
        provider=provider_name,
        provider_model=provider_model,
        provider_implementation=provider_implementation,
        rows_written=output.height,
    )


def _run_chunk_model(
    *,
    model: ModelConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    full_refresh: bool,
) -> ModelRunResult:
    assert model.chunk is not None
    chunk_cfg = model.chunk
    if not model.depends_on or len(model.depends_on) != 1:
        raise RunError(
            f"Chunk model '{model.name}' must declare exactly one upstream in "
            "`depends_on:` (the extraction model to chunk)"
        )
    upstream = parse_ref(model.depends_on[0])
    df = adapter.read_table(upstream)
    if chunk_cfg.text_field not in df.columns:
        raise RunError(
            f"Chunk model '{model.name}': upstream '{upstream}' has no column "
            f"'{chunk_cfg.text_field}'. Available: {sorted(df.columns)}"
        )
    if "document_id" not in df.columns:
        raise RunError(
            f"Chunk model '{model.name}': upstream '{upstream}' has no "
            "`document_id`; chunk models read extraction outputs."
        )
    document_ids = _chunk_document_ids(df, model.name)

    code_version = compute_code_version(
        extraction=None,
        transform=None,
        chunk=chunk_cfg,
        depends_on=[upstream],
        project_dir=project_dir,
    )
    warehouse_opts = _warehouse_options(adapter, model)
    state_scope = StateScope(model.name)
    is_incremental = model.materialization == "incremental" and not full_refresh
    processed_state = adapter.fetch_state(state_scope) if is_incremental else {}

    # Carry every upstream column except the split text field (replaced by the
    # per-chunk text), so lineage (document_id, source_uri, content_hash, …)
    # flows onto every chunk row for free.
    carry_cols = [
        c
        for c in df.columns
        if c != chunk_cfg.text_field and c not in _CHUNK_GENERATED_FIELDS
    ]
    chunked_at = datetime.now(UTC).isoformat()

    rows: list[dict[str, Any]] = []
    state_records: list[StateRecord] = []
    processed = 0
    skipped = 0
    current_ids: set[str] = set()
    changed_ids: list[str] = []

    for document_id, record in zip(
        document_ids, df.iter_rows(named=True), strict=True
    ):
        current_ids.add(document_id)
        raw_text = record[chunk_cfg.text_field]
        text = "" if raw_text is None else str(raw_text)
        doc_hash = _chunk_input_hash(record, text_field=chunk_cfg.text_field)
        if is_incremental:
            prior = processed_state.get(document_id)
            if prior == StateValue(doc_hash, code_version):
                skipped += 1
                continue
            if prior is not None:
                changed_ids.append(document_id)
        processed += 1
        pieces = split_text(text, chunk_cfg)
        carried = {c: record[c] for c in carry_cols}
        for piece in pieces:
            rows.append(
                _chunk_row(
                    carried=carried,
                    document_id=document_id,
                    piece_index=piece.index,
                    chunk_count=len(pieces),
                    text=piece.text,
                    strategy=chunk_cfg.strategy,
                    code_version=code_version,
                    chunked_at=chunked_at,
                )
            )
        state_records.append(StateRecord(document_id, doc_hash, code_version))

    deleted = 0
    if is_incremental:
        removed = [d for d in processed_state if d not in current_ids]
        # Re-chunked docs: clear their old chunks so shrinking a document
        # doesn't leave orphan chunk rows (materialize_incremental keys on
        # chunk_id, which differs for the new chunks).
        stale = removed + changed_ids
        if stale:
            adapter.delete_rows_and_state(
                model.name,
                key_col="document_id",
                keys=stale,
                state_scope=state_scope,
            )
            deleted = len(removed)

    rows_written = 0
    if rows or full_refresh or model.materialization == "full":
        chunk_df = pl.DataFrame(rows) if rows else pl.DataFrame()
        if model.materialization == "full" or full_refresh:
            rows_written = adapter.materialize_full(
                model.name, chunk_df, options=warehouse_opts
            )
        else:
            try:
                rows_written = adapter.materialize_incremental(
                    model.name,
                    chunk_df,
                    key_col="chunk_id",
                    on_schema_change=model.on_schema_change,
                    options=warehouse_opts,
                )
            except AdapterError as e:
                raise RunError(str(e)) from e

    if model.materialization == "full" or full_refresh:
        adapter.replace_state(state_scope, state_records)
    else:
        adapter.upsert_state(state_scope, state_records)

    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="chunk",
        documents_processed=processed,
        documents_skipped=skipped,
        documents_deleted=deleted,
        rows_written=rows_written,
    )


def _chunk_document_ids(df: pl.DataFrame, model_name: str) -> list[str]:
    raw_ids = df["document_id"].to_list()
    null_count = sum(value is None for value in raw_ids)
    if null_count:
        raise RunError(
            f"Chunk model '{model_name}': upstream `document_id` contains "
            f"{null_count} NULL value(s)"
        )
    document_ids = [str(value) for value in raw_ids]
    empty_count = sum(not value for value in document_ids)
    if empty_count:
        raise RunError(
            f"Chunk model '{model_name}': upstream `document_id` contains "
            f"{empty_count} empty value(s)"
        )
    duplicate_count = len(document_ids) - len(set(document_ids))
    if duplicate_count:
        raise RunError(
            f"Chunk model '{model_name}': upstream `document_id` contains "
            f"{duplicate_count} duplicate value(s)"
        )
    return document_ids


def _chunk_input_hash(record: dict[str, Any], *, text_field: str) -> str:
    raw_text = record[text_field]
    effective_input = {
        "document_id": str(record["document_id"]),
        "text": "" if raw_text is None else str(raw_text),
        "carried": {
            key: value
            for key, value in record.items()
            if key != text_field and key not in _CHUNK_INPUT_EXCLUDED_FIELDS
        },
    }
    return canonical_fingerprint(effective_input, domain="chunk-input", version=2)


def _chunk_row(
    *,
    carried: dict[str, Any],
    document_id: str,
    piece_index: int,
    chunk_count: int,
    text: str,
    strategy: str,
    code_version: str,
    chunked_at: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {c: _scalarize(v) for c, v in carried.items()}
    row.update(
        {
            "chunk_id": chunk_id(document_id, piece_index, text),
            "document_id": document_id,
            "chunk_index": piece_index,
            "chunk_count": chunk_count,
            "text": text,
            "chunk_strategy": strategy,
            "code_version": code_version,
            "chunked_at": chunked_at,
        }
    )
    return row


@dataclass
class _EmbedWork:
    record_id: str
    record: dict[str, Any]
    input_fingerprint: str
    text_hash: str
    text: str
    vector: tuple[float, ...] | None = None
    embedded_at: str | None = None


def _run_embed_model(
    *,
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    full_refresh: bool,
) -> ModelRunResult:
    assert model.embed is not None
    config = model.embed
    if not model.depends_on or len(model.depends_on) != 1:
        raise RunError(
            f"Embed model '{model.name}' must declare exactly one upstream in "
            "`depends_on:`"
        )
    upstream = parse_ref(model.depends_on[0])
    source = adapter.read_table(upstream)
    missing = sorted({config.id_field, config.text_field} - set(source.columns))
    if missing:
        raise RunError(
            f"Embed model '{model.name}': upstream '{upstream}' is missing "
            f"required column(s): {', '.join(missing)}. Available: "
            f"{sorted(source.columns)}"
        )
    generated = set(EMBED_METADATA_FIELDS) | {config.vector_field}
    generated_names = {name.casefold() for name in generated}
    collisions = sorted(
        column for column in source.columns if column.casefold() in generated_names
    )
    if collisions:
        raise RunError(
            f"Embed model '{model.name}': upstream '{upstream}' already contains "
            f"generated embedding column(s): {', '.join(collisions)}"
        )

    record_ids = _embed_record_ids(source, config.id_field, model.name)
    identity = EmbeddingIdentity.from_config(config)
    code_version = compute_model_code_version(
        model,
        project,
        project_dir,
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
    existing_rows = (
        _existing_embedding_rows(
            adapter,
            model.name,
            id_field=config.id_field,
        )
        if is_incremental and not rebuild_target
        else {}
    )

    current_ids = set(record_ids)
    removed = sorted(set(processed_state) - current_ids)
    removed_target_keys = [
        existing_rows[record_id][config.id_field]
        for record_id in removed
        if record_id in existing_rows
    ]
    work: list[_EmbedWork] = []
    skipped = 0
    cache_hits = 0
    for record_id, record in zip(
        record_ids,
        source.iter_rows(named=True),
        strict=True,
    ):
        text_value = record[config.text_field]
        text = "" if text_value is None else str(text_value)
        input_fingerprint = canonical_fingerprint(
            record,
            domain="embedding-input-row",
            version=1,
        )
        if processed_state.get(record_id) == StateValue(
            input_fingerprint,
            code_version,
        ):
            skipped += 1
            continue
        text_hash = canonical_fingerprint(
            {"text": text},
            domain="embedding-input-text",
            version=1,
        )
        item = _EmbedWork(
            record_id=record_id,
            record=record,
            input_fingerprint=input_fingerprint,
            text_hash=text_hash,
            text=text,
        )
        existing = existing_rows.get(record_id)
        if (
            existing is not None
            and existing.get("embedding_input_hash") == text_hash
            and existing.get("embedding_config_hash") == identity.config_hash
        ):
            vector = _coerce_embedding_vector(
                existing.get(config.vector_field),
                dimensions=config.dimensions,
            )
            if vector is not None:
                item.vector = vector
                embedded_at = existing.get("embedded_at")
                item.embedded_at = (
                    str(embedded_at) if embedded_at is not None else None
                )
                cache_hits += 1
        work.append(item)

    pending = [item for item in work if item.vector is None]
    usage_totals: dict[str, int | float] = {}
    provider_batches = 0
    try:
        for offset in range(0, len(pending), config.batch_size):
            batch = pending[offset : offset + config.batch_size]
            embedded = embed_texts(
                [item.text for item in batch],
                identity,
                input_ids=[item.record_id for item in batch],
                max_retries=config.max_retries,
            )
            provider_batches += 1
            _add_provider_usage(usage_totals, embedded.usage.to_metrics())
            for item, vector in zip(batch, embedded.vectors, strict=True):
                item.vector = vector
    except Exception as e:
        raise RunError(
            f"Embed model '{model.name}' provider execution failed: "
            f"{_artifact_error_text(e)}"
        ) from e

    now = datetime.now(UTC).isoformat()
    rows = [
        _embedding_row(
            item,
            identity=identity,
            vector_field=config.vector_field,
            embedded_at=item.embedded_at or now,
        )
        for item in work
    ]
    state_records = [
        StateRecord(item.record_id, item.input_fingerprint, code_version)
        for item in work
    ]
    output = (
        pl.DataFrame(rows)
        if rows
        else _empty_embedding_frame(
            source,
            vector_field=config.vector_field,
        )
    )

    rows_written = 0
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
            if rows:
                rows_written = adapter.materialize_incremental(
                    model.name,
                    output,
                    key_col=config.id_field,
                    on_schema_change=model.on_schema_change,
                    options=warehouse_opts,
                )
            if removed:
                adapter.delete_rows_and_state(
                    model.name,
                    key_col=config.id_field,
                    keys=removed_target_keys,
                    state_scope=state_scope,
                    state_record_keys=removed,
                )
            if state_records:
                adapter.upsert_state(state_scope, state_records)
    except AdapterError as e:
        raise RunError(str(e)) from e

    metrics: dict[str, Any] = {
        "provider_calls": provider_batches,
        "batches": provider_batches,
        "cache_hits": cache_hits,
        "cache_misses": len(pending),
        "rows_embedded": len(pending),
        "metadata_updates": cache_hits,
        **usage_totals,
    }
    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="embed",
        provider=identity.provider,
        provider_model=identity.model,
        provider_implementation=identity.implementation,
        documents_processed=len(work),
        documents_skipped=skipped,
        documents_deleted=len(removed),
        rows_written=rows_written,
        metrics=metrics,
        artifact_metadata={"embedding": identity.to_dict()},
    )


def _embed_record_ids(
    frame: pl.DataFrame,
    id_field: str,
    model_name: str,
) -> list[str]:
    values = frame[id_field].to_list()
    null_count = sum(value is None for value in values)
    if null_count:
        raise RunError(
            f"Embed model '{model_name}': upstream `{id_field}` contains "
            f"{null_count} NULL value(s)"
        )
    record_ids = [str(value) for value in values]
    empty_count = sum(not value for value in record_ids)
    if empty_count:
        raise RunError(
            f"Embed model '{model_name}': upstream `{id_field}` contains "
            f"{empty_count} empty value(s)"
        )
    duplicate_count = len(record_ids) - len(set(record_ids))
    if duplicate_count:
        raise RunError(
            f"Embed model '{model_name}': upstream `{id_field}` contains "
            f"{duplicate_count} duplicate value(s)"
        )
    return record_ids


def _existing_embedding_rows(
    adapter: WarehouseAdapter,
    table: str,
    *,
    id_field: str,
) -> dict[str, dict[str, Any]]:
    existing = adapter.read_table(table)
    if id_field not in existing.columns:
        return {}
    return {
        str(row[id_field]): row
        for row in existing.iter_rows(named=True)
        if row[id_field] is not None
    }


def _coerce_embedding_vector(
    value: Any,
    *,
    dimensions: int,
) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return None
    if len(value) != dimensions:
        return None
    if any(
        isinstance(item, bool)
        or not isinstance(item, int | float)
        or not isfinite(item)
        for item in value
    ):
        return None
    return tuple(float(item) for item in value)


def _embedding_row(
    item: _EmbedWork,
    *,
    identity: EmbeddingIdentity,
    vector_field: str,
    embedded_at: str,
) -> dict[str, Any]:
    assert item.vector is not None
    row = {key: _scalarize(value) for key, value in item.record.items()}
    row[vector_field] = list(item.vector)
    row.update(
        {
            "embedding_provider": identity.provider,
            "embedding_model": identity.model,
            "embedding_dimensions": identity.dimensions,
            "embedding_provider_implementation": identity.implementation,
            "embedding_input_hash": item.text_hash,
            "embedding_config_hash": identity.config_hash,
            "embedded_at": embedded_at,
        }
    )
    return row


def _empty_embedding_frame(
    source: pl.DataFrame,
    *,
    vector_field: str,
) -> pl.DataFrame:
    schema: dict[str, Any] = dict(source.schema)
    schema.update(
        {
            vector_field: pl.List(pl.Float64),
            "embedding_provider": pl.String,
            "embedding_model": pl.String,
            "embedding_dimensions": pl.Int64,
            "embedding_provider_implementation": pl.String,
            "embedding_input_hash": pl.String,
            "embedding_config_hash": pl.String,
            "embedded_at": pl.String,
        }
    )
    return pl.DataFrame(schema=schema)


def _add_provider_usage(
    totals: dict[str, int | float],
    usage: dict[str, int | float],
) -> None:
    for key, value in usage.items():
        totals[key] = totals.get(key, 0) + value


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
    config_fingerprint = collection_config_fingerprint(
        effective_search_config(model, models_by_name), store_type=store_type
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
