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
import os
import shutil
import subprocess
import tempfile
import time
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
from ..post_extract import LoadedPostExtract, load_post_extract
from ..profile import ResolvedProfile, resolve_llm_options
from ..progress import get_reporter
from ..providers import InferenceProvider, get_inference_provider
from ..sources import DocumentRef, DocumentSource
from ..versioning import compute_model_code_version
from .contracts import ModelRunResult, RunError
from .cost import budget_cost_estimator, estimate_cost
from .errors import artifact_error_text
from .values import scalarize
from .warehouse import warehouse_options

log = logging.getLogger(__name__)

# Fetch-staging directories (#273): a killed/crashed process never runs
# TemporaryDirectory.__exit__, leaking everything it fetched. Two mitigations:
# per-document cleanup below bounds a *live* run's peak disk use to in-flight
# documents, and the startup sweep self-heals directories a *dead* run left
# behind, which nothing else will ever clean up.
#
# The producer and the sweep filter must both use this constant. The sweep
# only ever sees directories a *dead* process left behind, so a prefix that
# drifts on one side disables the self-healing without failing anything.
_FETCH_DIR_PREFIX = "dbt_ml_fetch_"

# A directory's mtime is refreshed by every file *written* inside it, but a
# single long native-batch submission (#149) can sit idle for hours without
# creating any new entries while it waits on an external API — age alone
# cannot tell that apart from a directory a dead process abandoned (#273
# review). The owner-PID marker below resolves the ambiguity; this threshold
# only decides which directories are even worth checking.
_STALE_FETCH_DIR_MAX_AGE_SECONDS = 6 * 60 * 60
_OWNER_MARKER_NAME = ".dbt_ml_owner_pid"
_swept_stale_fetch_dirs = False


def _write_owner_marker(work_dir: Path) -> None:
    """Record this process's PID so a later sweep — possibly run by a
    different stel process — can tell a directory still owned by a live run
    apart from one a dead run left behind (#273)."""
    try:
        (work_dir / _OWNER_MARKER_NAME).write_text(str(os.getpid()))
    except OSError:
        log.debug(
            "failed to write fetch-staging owner marker in %s", work_dir, exc_info=True
        )


def _pid_is_alive(pid: int) -> bool:
    """Best-effort, non-destructive liveness check. An inconclusive result
    reports "alive" so a sweep skips the directory rather than risking
    deletion of a live run's staging area. Never uses `os.kill(pid, 0)` on
    Windows: CPython implements non-special `os.kill` signals there via
    `TerminateProcess`, so that "probe" would actually kill the process it is
    only trying to inspect."""
    if os.name == "nt":
        try:
            probe = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except OSError:
            return True
        return str(pid) in probe.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _fetch_dir_is_live(entry: Path) -> bool:
    """True if `entry` carries an owner marker naming a still-running process.
    A missing/unreadable marker means either a pre-fix stel version leaked
    this directory (no marker was ever written) or the marker write itself
    failed — both fall back to the age check the caller already made."""
    marker = entry / _OWNER_MARKER_NAME
    try:
        pid = int(marker.read_text().strip())
    except (OSError, ValueError):
        return False
    return _pid_is_alive(pid)


def _sweep_stale_fetch_dirs(root: Path, *, max_age_seconds: float) -> None:
    """Best-effort removal of `_FETCH_DIR_PREFIX`-prefixed directories under
    `root` that are both older than `max_age_seconds` and not owned by a
    still-running process. Never raises: a sweep failure must not fail the run
    it happens to run alongside."""
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    now = time.time()
    for entry in entries:
        if not entry.name.startswith(_FETCH_DIR_PREFIX):
            continue
        try:
            if not entry.is_dir() or now - entry.stat().st_mtime < max_age_seconds:
                continue
            if _fetch_dir_is_live(entry):
                continue
            shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            log.debug("failed to sweep stale fetch dir %s", entry, exc_info=True)


def _sweep_stale_fetch_dirs_once() -> None:
    """Run the stale-directory sweep at most once per process. Each stel
    invocation (standalone CLI or embedded `materialize()`) is a fresh process,
    so once-per-process is once-per-run without needing a shared choke point
    across runner.py and dbt_embed."""
    global _swept_stale_fetch_dirs
    if _swept_stale_fetch_dirs:
        return
    _swept_stale_fetch_dirs = True
    _sweep_stale_fetch_dirs(
        Path(tempfile.gettempdir()), max_age_seconds=_STALE_FETCH_DIR_MAX_AGE_SECONDS
    )


def _cleanup_fetched(path: Path, work_dir: Path) -> None:
    """Best-effort removal of one document's fetch-staging bytes right after
    extraction, so peak staging disk usage is bounded by in-flight documents
    rather than the whole run's corpus (#273). `fetch()`'s contract guarantees
    the returned path lives under `work_dir`, so this never touches a source
    document itself — only the snapshot stel made of it."""
    try:
        top = work_dir / path.relative_to(work_dir).parts[0]
        if top.is_dir():
            shutil.rmtree(top, ignore_errors=True)
        else:
            top.unlink(missing_ok=True)
    except (OSError, ValueError):
        log.debug("failed to clean up fetch-staging path %s", path, exc_info=True)


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
    # An `enum` field materializes as the string it constrains (issue #304);
    # `enum` is stel's declaration, not a warehouse column type. Shared with
    # `_llm_output_schema` in execution/llm.py, so a missing entry here is a
    # KeyError before any row or typed empty relation is written.
    "enum": pl.String,
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
    _sweep_stale_fetch_dirs_once()
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

    try:
        post_extract = (
            load_post_extract(
                model.extraction.post_extract.module,
                project_dir,
                model.extraction.post_extract.options,
            )
            if model.extraction.post_extract is not None
            else None
        )
    except (Exception, SystemExit):
        # Compilation already validated the hook. If its file changes between
        # compile and execution, fail the model without surfacing module-local
        # exception text that could contain project-sensitive configuration.
        module = model.extraction.post_extract
        name = module.module if module is not None else "unknown"
        raise RunError(f"Post-extract hook '{name}' could not be loaded") from None

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
    progress_task = get_reporter().model_task(
        model.name, "extraction", total_docs
    )
    progress_task.__enter__()
    try:
        if budget_guard is not None and docs_to_process:
            budget_guard.charge_documents(len(docs_to_process))
        # Sources snapshot into a per-model scratch dir, lazily and only for
        # documents that actually need processing. Extraction streams through in
        # `flush_every`-sized chunks (issue #77): rows never accumulate beyond
        # one chunk, so corpus size is bounded by the flush size, not memory.
        with tempfile.TemporaryDirectory(prefix=_FETCH_DIR_PREFIX) as scratch:
            work_dir = Path(scratch)
            _write_owner_marker(work_dir)

            def _one(
                doc: DocumentRef,
            ) -> tuple[DocumentRef, ExtractionResult | None, str | None]:
                local_path: Path | None = None
                try:
                    if budget_guard is not None:
                        budget_guard.ensure_headroom()
                    local_path = source_backend.fetch(doc, work_dir)
                    if budget_guard is not None:
                        size = local_path.stat().st_size
                        budget_guard.check_file_bytes(size)
                        budget_guard.charge_bytes(size)
                    result = backend.extract(local_path, options)
                    if post_extract is not None:
                        result = post_extract.apply(
                            result, document=doc, local_path=local_path
                        )
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
                finally:
                    # Delete this document's staged bytes now rather than at the
                    # end of the whole run (#273): peak fetch-staging disk usage
                    # is bounded by in-flight documents, not the corpus size.
                    if local_path is not None:
                        _cleanup_fetched(local_path, work_dir)

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
                            post_extract=post_extract,
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
                        progress_task.advance(len(extracted))
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
                # Incremental: each flush upserts rows and advances its state.
                # publish_every>1 coalesces that many flushes into one upsert
                # (issue #293) so a run of many small flushes shares one warehouse
                # MERGE. State advances only after a publication succeeds, so a
                # crash or budget exhaustion with a partial buffer leaves those
                # flushes unpublished and retryable; already-published batches
                # stay. publish_every==1 buffers exactly one frame per publish, so
                # the path is byte-for-byte the prior per-flush behavior.
                publish_every = model.extraction.publish_every
                buffered_frames: list[pl.DataFrame] = []
                buffered_records: list[StateRecord] = []
                buffered_flushes = 0
                buffered_schema: dict[str, Any] | None = None
                first_publication = True

                def _publish() -> None:
                    nonlocal rows_written, first_publication, buffered_flushes
                    nonlocal buffered_schema
                    if not buffered_frames:
                        return
                    combined = (
                        buffered_frames[0]
                        if len(buffered_frames) == 1
                        # Every buffered frame shares one schema (see below), so a
                        # plain vertical concat is exact — no column union or dtype
                        # coercion is applied that a single flush would not have had.
                        else pl.concat(buffered_frames)
                    )
                    try:
                        rows_written += adapter.materialize_incremental(
                            model.name,
                            combined,
                            key_col="document_id",
                            # The model's policy governs run-over-run drift on the
                            # first publication; later publications union within-run
                            # drift, matching what the prior per-flush path did (it
                            # applied the policy only to the first flush and
                            # append_new_columns to every later one).
                            on_schema_change=(
                                "append_new_columns"
                                if first_publication and empty_incremental_target
                                else model.on_schema_change
                                if first_publication
                                else "append_new_columns"
                            ),
                            options=warehouse_opts,
                            update_when_changed=model.update_when_changed,
                        )
                    except AdapterError as e:
                        # RunError so `build` fails this model and blocks
                        # descendants instead of aborting the whole invocation.
                        raise RunError(str(e)) from e
                    first_publication = False
                    # State only after the merge lands (publish-then-state), so an
                    # interrupted run never records unpublished documents.
                    adapter.upsert_state(state_scope, buffered_records)
                    log.info(
                        "published %d rows across %d flush(es) (%d/%d docs) for %s",
                        combined.height,
                        buffered_flushes,
                        docs_flushed,
                        total_docs,
                        model.name,
                    )
                    buffered_frames.clear()
                    buffered_records.clear()
                    buffered_flushes = 0
                    buffered_schema = None

                for extracted in _iter_extracted():
                    chunk_rows, chunk_records = _rows_for_chunk(extracted)
                    docs_flushed += len(extracted)
                    progress_task.advance(len(extracted))
                    if not chunk_rows:
                        continue
                    frame = _apply_extraction_contract(pl.DataFrame(chunk_rows), model)
                    frame_schema = dict(frame.schema)
                    # Only coalesce flushes that share one schema. A schema-on-read
                    # model can drift within a run; publishing at the boundary keeps
                    # each publication uniform so `on_schema_change` applies exactly
                    # as it did per flush — a later flush's new column is never
                    # folded into (and dropped/failed by) the first publication's
                    # policy, which would otherwise lose data under `ignore` (#293).
                    if buffered_frames and frame_schema != buffered_schema:
                        _publish()
                    if not buffered_frames:
                        buffered_schema = frame_schema
                    buffered_frames.append(frame)
                    buffered_records.extend(chunk_records)
                    buffered_flushes += 1
                    if buffered_flushes >= publish_every:
                        _publish()
                # Publish the trailing partial buffer on clean completion only; an
                # exception leaves it unpublished (and thus retryable) by design.
                _publish()

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
    finally:
        progress_task.__exit__(None, None, None)

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
    post_extract: LoadedPostExtract | None = None,
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
    out: list[tuple[DocumentRef, ExtractionResult | None, str | None]] = []
    try:
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
        by_doc_id: dict[
            str, tuple[ExtractionResult | None, str | None]
        ] = {}
        for index, ((doc, path), result) in enumerate(
            zip(fetched, batch_out, strict=True)
        ):
            if isinstance(result, Exception):
                by_doc_id[doc.document_id] = (
                    None,
                    artifact_error_text(result),
                )
                # Drop any provider exception that could retain raw response
                # state while the rest of a large batch is derived.
                batch_out[index] = RuntimeError("batch item extraction failed")
                continue
            if post_extract is not None:
                try:
                    result = post_extract.apply(
                        result, document=doc, local_path=path
                    )
                except Exception as e:
                    by_doc_id[doc.document_id] = (None, artifact_error_text(e))
                    batch_out[index] = RuntimeError("post-extract hook failed")
                    continue
                # Release the raw backend result as soon as this document has
                # crossed the hook boundary; do not retain a window of raw
                # payloads alongside the derived output.
                batch_out[index] = result
            by_doc_id[doc.document_id] = (result, None)
        for doc, _path, err in entries:
            if err is not None:
                out.append((doc, None, err))
                continue
            result, extraction_error = by_doc_id[doc.document_id]
            out.append((doc, result, extraction_error))
    finally:
        # Keep snapshots through post-extract so the hook can inspect the
        # verified bytes, then remove them before the window is published.
        for _doc, path in fetched:
            _cleanup_fetched(path, work_dir)
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
            "Extracted fields collide with reserved stel lineage columns: "
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

