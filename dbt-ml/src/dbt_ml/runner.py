from __future__ import annotations

import concurrent.futures
import json
import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from .adapters import AdapterError, WarehouseAdapter, create_adapter
from .backends import ExtractionResult, get_backend
from .checks import TestResult, run_model_tests
from .classic_ml import run_classic_ml_model
from .config import load_project
from .config.model import ModelConfig
from .config.project import ProjectConfig
from .config.source import SourceConfig
from .dag import ProjectDAG, parse_ref
from .profile import ResolvedProfile, resolve_llm_options, resolve_profile
from .sources import DocumentRef, DocumentSource, SourceError, get_document_source
from .transforms import load_transform
from .versioning import compute_code_version

log = logging.getLogger(__name__)


class RunError(Exception):
    pass


def _modified_set(
    models: list[ModelConfig], project_dir: Path, state: Path | None
) -> set[str] | None:
    """None when no state manifest was given (state:modified then errors in
    selection); otherwise the models whose code_version diverged from it."""
    if state is None:
        return None
    # Local import: manifest.py imports ModelRunResult from this module.
    from .manifest import compute_modified_models

    return compute_modified_models(models, project_dir, state)


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
    documents_processed: int = 0
    documents_skipped: int = 0
    documents_deleted: int = 0
    rows_written: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
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
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    dag = ProjectDAG(sources, models)
    selected = dag.select_models(
        select=select, exclude=exclude, modified=_modified_set(models, project_dir, state)
    )

    source_docs = _discover_sources(sources, project_dir)

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
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    dag = ProjectDAG(sources, models)
    selected = dag.select_models(
        select=select, exclude=exclude, modified=_modified_set(models, project_dir, state)
    )

    source_docs = _discover_sources(sources, project_dir)
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
                        errors=[str(e)],
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
    else:
        raise RunError(
            f"Model '{model.name}' has no extraction, transform, or ml block configured"
        )
    result.duration_seconds = round(time.monotonic() - start, 3)
    return result


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
        options = resolve_llm_options(options, resolved)

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

    code_version = compute_code_version(
        extraction=model.extraction,
        transform=None,
        project_dir=project_dir,
    )

    is_incremental = model.materialization == "incremental" and not full_refresh
    processed_state = adapter.fetch_state(model.name) if is_incremental else {}

    docs_to_process: list[DocumentRef] = []
    for doc in docs:
        if is_incremental:
            prior = processed_state.get(doc.document_id)
            if prior == (doc.content_hash, code_version):
                continue
        docs_to_process.append(doc)

    deleted = 0
    if is_incremental:
        current_ids = {doc.document_id for doc in docs}
        removed = [doc_id for doc_id in processed_state if doc_id not in current_ids]
        if removed:
            adapter.delete_rows(model.name, key_col="document_id", keys=removed)
            adapter.delete_state(model.name, removed)
            deleted = len(removed)

    skipped = len(docs) - len(docs_to_process)
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    state_records: list[tuple[str, str, str]] = []

    # Remote sources download into a per-model scratch dir, lazily and only
    # for documents that actually need processing; local sources pass their
    # real path through untouched.
    with tempfile.TemporaryDirectory(prefix="dbt_ml_fetch_") as scratch:
        work_dir = Path(scratch)

        def _one(
            doc: DocumentRef,
        ) -> tuple[DocumentRef, ExtractionResult | None, str | None]:
            try:
                local_path = source_backend.fetch(doc, work_dir)
                return doc, backend.extract(local_path, options), None
            except Exception as e:
                log.debug("extraction failed for %s", doc.relative_path, exc_info=True)
                return doc, None, f"{type(e).__name__}: {e}"

        if threads > 1 and len(docs_to_process) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
                extracted = list(ex.map(_one, docs_to_process))
        else:
            extracted = [_one(d) for d in docs_to_process]

    backend_version = backend.version()
    # One timestamp per model run: rows from the same run are batch-identifiable.
    extracted_at = datetime.now(UTC).isoformat()
    for doc, result, err in extracted:
        if err is not None or result is None:
            errors.append(f"{doc.relative_path}: {err}")
            continue
        rows.append(
            _row_for_extraction(
                doc,
                code_version,
                result,
                backend_name=backend_name,
                backend_version=backend_version,
                extracted_at=extracted_at,
            )
        )
        state_records.append((doc.document_id, doc.content_hash, code_version))

    rows_written = 0
    if rows or full_refresh or model.materialization == "full":
        df = pl.DataFrame(rows) if rows else pl.DataFrame()
        if model.materialization == "full" or full_refresh:
            rows_written = adapter.materialize_full(model.name, df)
        else:
            try:
                rows_written = adapter.materialize_incremental(
                    model.name,
                    df,
                    key_col="document_id",
                    on_schema_change=model.on_schema_change,
                )
            except AdapterError as e:
                # RunError so `build` fails this model and blocks descendants
                # instead of aborting the whole invocation.
                raise RunError(str(e)) from e

    if full_refresh:
        adapter.clear_model_state(model.name)
    adapter.upsert_state(model.name, state_records)

    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="extraction",
        backend=backend_name,
        documents_processed=len(docs_to_process),
        documents_skipped=skipped,
        documents_deleted=deleted,
        rows_written=rows_written,
        errors=errors,
    )


def _row_for_extraction(
    doc: DocumentRef,
    code_version: str,
    result: ExtractionResult,
    *,
    backend_name: str,
    backend_version: str,
    extracted_at: str,
) -> dict[str, Any]:
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

    import inspect

    from .transforms import TransformContext

    transform_fn = load_transform(model.transform.module, project_dir)
    deps: dict[str, pl.DataFrame] = {}
    for dep_ref in model.depends_on:
        dep_name = parse_ref(dep_ref)
        deps[dep_name] = adapter.query_df(
            f"SELECT * FROM {adapter.table_ref(dep_name)}"
        )

    sig = inspect.signature(transform_fn)
    if len(sig.parameters) >= 2:
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

    if not isinstance(output, pl.DataFrame):
        raise RunError(
            f"Transform '{model.transform.module}' must return a polars.DataFrame"
        )

    adapter.materialize_full(model.name, output)

    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="transform",
        rows_written=output.height,
    )


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
    try:
        output = run_classic_ml_model(
            model=model,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
        )
    except Exception as e:
        raise RunError(f"ML model '{model.name}' failed: {e}") from e

    rows_written = adapter.materialize_full(model.name, output.df)
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
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
) -> str:
    """Delegate to the adapter's clean(). Returns a description of what was removed."""
    project, _, _ = load_project(project_dir)
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    adapter = create_adapter(resolved.warehouse, project_dir=project_dir)
    return adapter.clean()
