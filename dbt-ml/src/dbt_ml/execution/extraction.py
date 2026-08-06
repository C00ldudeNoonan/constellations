"""Extraction model executor (issue #190).

Owns the extraction-only lifecycle: backend/provider resolution, incremental
state and deletion, bounded streaming/batch flush, the common output row and
lineage-schema contract, and safe error conversion. runner.py keeps selection,
DAG scheduling, source discovery, threading, the run budget ledger, and result
aggregation, and re-exports the public names below for compatibility.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import tempfile
from collections import Counter
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
from ..backends import (
    BackendOptionsError,
    BaseBackend,
    ExtractionResult,
    get_backend,
    validate_backend_options,
)
from ..backends.llm_backend import BatchCancelledError
from ..backends.options import LLMBackendOptions
from ..budget import BudgetExceededError, BudgetGuard, BudgetLedger
from ..config.model import INTERNAL_LINEAGE_FIELDS, ModelConfig
from ..config.project import ProjectConfig
from ..dag import parse_ref
from ..paths import resolve_within_project
from ..profile import ResolvedProfile, resolve_llm_options
from ..providers import InferenceProvider, get_inference_provider
from ..sources import DocumentRef, DocumentSource
from ..versioning import compute_model_code_version
from .contracts import ModelRunResult, RunError
from .cost import budget_cost_estimator, estimate_cost
from .errors import artifact_error_text
from .values import scalarize
from .warehouse import warehouse_options

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

EXTRACTION_FIELD_DTYPES: dict[str, Any] = {
    "string": pl.String,
    "integer": pl.Int64,
    "float": pl.Float64,
    "boolean": pl.Boolean,
    "date": pl.Date,
    "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
    "json": pl.String,
}


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
            EXTRACTION_FIELD_DTYPES[field_config.data_type]
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


@dataclass
class DiscoveredSource:
    """A source's backend plus its discovered documents for this run."""

    backend: DocumentSource
    refs: list[DocumentRef]


def run_extraction_model(
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
    subset_run: bool = False,
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
                cost_estimator=budget_cost_estimator(
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
    warehouse_opts = warehouse_options(adapter, model)
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
    # A subset run (`--source-filter`) only discovered a slice of the source, so
    # "absent from this run" does NOT mean "removed upstream" — every other
    # partition's document would look removed. Skip the delete pass entirely:
    # a filtered run is additive/upsert-only, and removed-document reconciliation
    # is the job of a periodic full (unfiltered) run (#266).
    if is_incremental and not subset_run:
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
                    return doc, None, artifact_error_text(e)

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
        cost = estimate_cost(usage_totals, resolved.llm.pricing)
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
            out.append((doc, None, artifact_error_text(res)))
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
        row[key] = scalarize(value)
    return row

