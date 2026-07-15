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
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from .adapters import (
    AdapterError,
    StateRecord,
    StateScope,
    StateValue,
    WarehouseAdapter,
    create_adapter,
)
from .backends import (
    BackendOptionsError,
    BaseBackend,
    ExtractionResult,
    get_backend,
    validate_backend_options,
)
from .backends.options import LLMBackendOptions
from .checks import TestResult, run_model_tests
from .chunking import chunk_id, split_text
from .classic_ml import run_classic_ml_model
from .compiler import validate_project_contract, validate_warehouse_capabilities
from .config import load_project
from .config.model import INTERNAL_LINEAGE_FIELDS, ModelConfig
from .config.profile import DEFAULT_LLM_PROVIDER, PricingConfig
from .config.project import ProjectConfig
from .config.source import SourceConfig
from .dag import ProjectDAG, parse_ref
from .hashing import canonical_fingerprint
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
from .sources import DocumentRef, DocumentSource, SourceError, get_document_source
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


class RunError(Exception):
    pass


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
    backend: str | None = None
    provider: str | None = None
    provider_model: str | None = None
    provider_implementation: str | None = None
    documents_processed: int = 0
    documents_skipped: int = 0
    documents_deleted: int = 0
    rows_written: int = 0
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

    required_sources = set(dag.required_sources(selected))
    source_docs = _discover_sources(
        [source for source in sources if source.name in required_sources],
        project_dir,
    )

    models_by_name = {m.name: m for m in models}

    def _run(name: str, adapter: WarehouseAdapter) -> ModelRunResult:
        return _run_model(
            model=models_by_name[name],
            project=project,
            project_dir=project_dir,
            source_docs=source_docs,
            adapter=adapter,
            resolved=resolved,
            full_refresh=full_refresh,
            threads=threads,
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

    required_sources = set(dag.required_sources(selected))
    source_docs = _discover_sources(
        [source for source in sources if source.name in required_sources],
        project_dir,
    )
    models_by_name = {m.name: m for m in models}

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
                    project=project,
                    project_dir=project_dir,
                    source_docs=source_docs,
                    adapter=adapter,
                    resolved=resolved,
                    full_refresh=full_refresh,
                    threads=threads,
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

            model_tests = run_model_tests(
                model, adapter, project_dir=project_dir, store_failures=store_failures
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


def _run_model(
    *,
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    source_docs: dict[str, DiscoveredSource],
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    full_refresh: bool,
    threads: int = 1,
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
    else:
        raise RunError(
            f"Model '{model.name}' has no extraction, transform, ml, or chunk "
            "block configured"
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
                local_path = source_backend.fetch(doc, work_dir)
                return doc, backend.extract(local_path, options), None
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
                # One Batches API submission for the whole model (#75); only
                # the materialization of its results is chunked.
                all_extracted, batch_metrics = _extract_batched(
                    docs_to_process,
                    source_backend,
                    backend,
                    options,
                    work_dir,
                    model.name,
                )
                for key, value in batch_metrics.items():
                    if isinstance(value, int | float):
                        usage_totals[key] = usage_totals.get(key, 0) + value
                for i in range(0, len(all_extracted), flush_every):
                    yield all_extracted[i : i + flush_every]
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


def _extract_batched(
    docs: list[DocumentRef],
    source_backend: DocumentSource,
    backend: BaseBackend,
    options: dict[str, Any],
    work_dir: Path,
    model_name: str,
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
            backend.extract_batch_with_metrics([p for _, p in fetched], options)
            if fetched
            else None
        )
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
        model.name, output, options=_warehouse_options(adapter, model)
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
