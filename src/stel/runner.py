from __future__ import annotations

import concurrent.futures
import fnmatch
import logging
import os
import shutil
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .adapters import (
    TableReadSnapshot,
    WarehouseAdapter,
    create_adapter,
)
from .budget import BudgetLedger
from .checks import TestResult, run_model_tests, validate_test_requirements
from .compiler import (
    validate_project_contract,
    validate_retrieval_capabilities,
    validate_warehouse_capabilities,
)
from .config import load_project
from .config.model import (
    ModelConfig,
)
from .config.profile import WarehouseConfig
from .config.project import ProjectConfig
from .config.source import SourceConfig
from .dag import ProjectDAG, is_dbt_ref
from .execution import ModelRunResult as ModelRunResult
from .execution import RunError as RunError
from .execution import chunk as _chunk_execution
from .execution import cost as _cost_execution
from .execution import embed as _embed_execution
from .execution import errors as _errors_execution
from .execution import eval as _eval_execution
from .execution import extraction as _extraction_execution
from .execution import llm as _llm_execution
from .execution import ml as _ml_execution
from .execution import search as _search_execution
from .execution import transform as _transform_execution
from .execution import usage as _usage_execution
from .manifest import compute_modified_models
from .paths import resolve_within_project
from .profile import (
    ResolvedProfile,
    apply_source_path_overrides,
    resolve_profile,
)
from .progress import get_reporter
from .sources import SourceError, get_document_source

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
# Compatibility re-export: the declared-field dtype contract now consumed by
# execution/llm.py directly.
_EXTRACTION_FIELD_DTYPES = _extraction_execution.EXTRACTION_FIELD_DTYPES

_run_embed_model = _embed_execution.run_embed_model
_add_provider_usage = _usage_execution.add_provider_usage

_run_llm_model = _llm_execution.run_llm_model
_run_eval_model = _eval_execution.run_eval_model

_run_search_model = _search_execution.run_search_model
_run_ml_model = _ml_execution.run_ml_model


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
    source_filter: Sequence[str] = (),
) -> list[ModelRunResult]:
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)
    # Validate --source-filter before resolving the profile: it is a deterministic
    # option error that a missing credential / bad profile must not mask, and a
    # rejected filter should not resolve credentials first (#266 review). The
    # select/exclude closure is a superset of the final (state-narrowed)
    # selection, so validating it is safe.
    subset_run = (
        _prepare_subset_run(
            source_filter,
            full_refresh=full_refresh,
            selected=dag.select_models(select=select, exclude=exclude),
            models=models,
        )
        if source_filter
        else False
    )
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    adapter = create_adapter(resolved.warehouse, project_dir=project_dir)
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
        adapter,
    )
    if source_filter:
        subset_run = _prepare_subset_run(
            source_filter,
            full_refresh=full_refresh,
            selected=selected,
            models=models,
            adapter=adapter,
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

    dbt_ref_models = sorted(
        model.name
        for model in models
        if model.name in set(selected)
        and model.source is not None
        and is_dbt_ref(model.source)
    )
    if dbt_ref_models:
        # A dbt_ref('...') source reads a dbt-built table, which only dbt can
        # resolve. These models run in embedded mode (dbt-duckdb Python models via
        # `stel.dbt_embed.materialize`), not the standalone runner (#177).
        raise RunError(
            "Models "
            + ", ".join(f"'{name}'" for name in dbt_ref_models)
            + " use a `dbt_ref(...)` source and can only run in embedded mode "
            "(dbt-duckdb) via `stel codegen` + `dbt build`, not standalone "
            "`stel run`/`build`."
        )

    required_sources = set(dag.required_sources(selected))
    source_docs = _discover_sources(
        [source for source in sources if source.name in required_sources],
        project_dir,
        source_filter=source_filter,
        warehouse=resolved.warehouse,
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
            subset_run=subset_run,
        )

    with adapter:
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
    source_filter: Sequence[str] = (),
) -> BuildResult:
    """Run + test each model in dependency order. A model whose run errors or
    whose tests hard-fail blocks all its descendants, which are reported as
    skipped (dbt `build` semantics)."""
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)
    # Validate --source-filter before resolving the profile: it is a deterministic
    # option error that a missing credential / bad profile must not mask, and a
    # rejected filter should not resolve credentials first (#266 review). The
    # select/exclude closure is a superset of the final (state-narrowed)
    # selection, so validating it is safe.
    subset_run = (
        _prepare_subset_run(
            source_filter,
            full_refresh=full_refresh,
            selected=dag.select_models(select=select, exclude=exclude),
            models=models,
        )
        if source_filter
        else False
    )
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    adapter = create_adapter(resolved.warehouse, project_dir=project_dir)
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
        adapter,
    )
    if source_filter:
        subset_run = _prepare_subset_run(
            source_filter,
            full_refresh=full_refresh,
            selected=selected,
            models=models,
            adapter=adapter,
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

    dbt_ref_models = sorted(
        model.name
        for model in models
        if model.name in set(selected)
        and model.source is not None
        and is_dbt_ref(model.source)
    )
    if dbt_ref_models:
        # A dbt_ref('...') source reads a dbt-built table, which only dbt can
        # resolve. These models run in embedded mode (dbt-duckdb Python models via
        # `stel.dbt_embed.materialize`), not the standalone runner (#177).
        raise RunError(
            "Models "
            + ", ".join(f"'{name}'" for name in dbt_ref_models)
            + " use a `dbt_ref(...)` source and can only run in embedded mode "
            "(dbt-duckdb) via `stel codegen` + `dbt build`, not standalone "
            "`stel run`/`build`."
        )

    required_sources = set(dag.required_sources(selected))
    source_docs = _discover_sources(
        [source for source in sources if source.name in required_sources],
        project_dir,
        source_filter=source_filter,
        warehouse=resolved.warehouse,
    )
    models_by_name = {m.name: m for m in models}

    validate_test_requirements(
        [model for model in models if model.name in set(selected)], resolved
    )
    run_budget = _run_budget_ledger(resolved)
    out = BuildResult()
    blocked: set[str] = set()

    with adapter:
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
                    subset_run=subset_run,
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
                    resolved=resolved,
                    run_budget=run_budget,
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
    sources: list[SourceConfig],
    project_dir: Path,
    *,
    source_filter: Sequence[str] = (),
    warehouse: WarehouseConfig | None = None,
) -> dict[str, DiscoveredSource]:
    out: dict[str, DiscoveredSource] = {}
    reporter = get_reporter()
    for source in sources:
        backend = get_document_source(
            source.path, warehouse=warehouse, project_dir=project_dir
        )
        try:
            refs = backend.discover(source, project_dir)
        except SourceError as e:
            raise RunError(str(e)) from e
        if source_filter:
            # Subset a run to documents whose source-relative path matches any
            # filter glob (`*` spans `/`, so `AAPL/*` selects a whole prefix).
            refs = [
                ref
                for ref in refs
                if any(fnmatch.fnmatch(ref.relative_path, pat) for pat in source_filter)
            ]
        # The final selected count on both verbose channels: the bar reporter
        # (TTY) and the INFO log (non-TTY). The per-source discover() log lines
        # report the pre-filter — and for GCS pre-file-pattern — count, so
        # without this a captured run could show hundreds discovered while
        # processing zero after --source-filter.
        log.info("Source '%s': %d document(s) selected", source.name, len(refs))
        reporter.source_discovered(source.name, len(refs))
        out[source.name] = DiscoveredSource(backend=backend, refs=refs)
    return out


def _prepare_subset_run(
    source_filter: Sequence[str],
    *,
    full_refresh: bool,
    selected: Sequence[str],
    models: list[ModelConfig],
    adapter: WarehouseAdapter | None = None,
) -> bool:
    """Validate `--source-filter` against the run and return whether it is an
    additive subset run. A filtered run upserts a slice and never deletes, so it
    is incompatible with a full refresh or a non-incremental extraction model."""
    if not source_filter:
        return False
    if full_refresh:
        raise RunError(
            "--source-filter cannot be combined with --full-refresh: a filtered "
            "run is additive (upsert-only) and never deletes. Run a full refresh "
            "without a filter to rebuild the whole model."
        )
    selected_set = set(selected)
    unsafe: list[str] = []
    for model in models:
        if model.name not in selected_set or model.extraction is None:
            continue
        if model.materialization != "incremental":
            unsafe.append(f"{model.name} (materialization: {model.materialization})")
        else:
            strategy = model.warehouse_options.get("incremental_strategy")
            if adapter is not None:
                parsed = adapter.parse_warehouse_options(
                    model.warehouse_options,
                    model_name=model.name,
                )
                strategy = getattr(parsed, "incremental_strategy", strategy)
            if strategy != "insert_overwrite":
                continue
            # insert_overwrite replaces every partition a batch touches, so a
            # partial (filtered) batch would delete sibling documents sharing
            # those partitions — not additive. Only merge (upsert) is safe.
            unsafe.append(f"{model.name} (incremental_strategy: insert_overwrite)")
    if unsafe:
        raise RunError(
            "--source-filter requires additive extraction models — incremental "
            "materialization with the default merge strategy — because a filtered "
            "run upserts a subset and must never replace whole tables or "
            "partitions. Unsafe selected extraction models: "
            + ", ".join(sorted(unsafe))
        )
    return True


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
    subset_run: bool = False,
) -> ModelRunResult:
    kind = _model_kind_label(model)
    log.info("starting %s (%s)", model.name, kind)
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
            subset_run=subset_run,
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
                project=project,
                project_dir=project_dir,
                adapter=adapter,
                resolved=resolved,
                full_refresh=full_refresh,
                run_budget=run_budget,
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
    elif model.eval is not None:
        result = _run_eval_model(
            model=model,
            models_by_name=models_by_name,
            project_dir=project_dir,
            adapter=adapter,
            full_refresh=full_refresh,
        )
    else:
        raise RunError(
            f"Model '{model.name}' has no extraction, transform, ml, chunk, embed, "
            "llm, search, or eval block configured"
        )
    result.duration_seconds = round(time.monotonic() - start, 3)
    log.info(
        "finished %s: %d row(s) in %.3fs%s",
        model.name,
        result.rows_written,
        result.duration_seconds,
        f" [{result.status}]" if result.status else "",
    )
    get_reporter().model_finished(
        model.name, result.rows_written, result.duration_seconds, result.status
    )
    return result


def _model_kind_label(model: ModelConfig) -> str:
    if model.extraction is not None:
        return "extraction"
    if model.ml is not None:
        return "ml"
    if model.transform is not None:
        return "transform"
    if model.chunk is not None:
        return "chunk"
    if model.embed is not None:
        return "embed"
    if model.llm is not None:
        return "llm"
    if model.search is not None:
        return "search"
    if model.eval is not None:
        return "eval"
    return "unknown"


def clean_project(
    project_dir: Path,
) -> str:
    """Remove known stel artifacts without invoking warehouse cleanup."""
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
