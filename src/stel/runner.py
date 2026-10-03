from __future__ import annotations

import concurrent.futures
import fnmatch
import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from .adapters import (
    ReadPredicate,
    ReadPredicateOperator,
    StateScope,
    SyncWatermark,
    TableContentFingerprint,
    WarehouseAdapter,
    create_adapter,
)
from .adapters.serialized import SerializedAdapter
from .append_log import RUN_LOG_SCHEMA, run_log_rows, write_rows
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
from .dag import NodeKind, ProjectDAG, is_dbt_ref
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
from .logging_setup import REPORTER_ECHO_EXTRA
from .manifest import compute_modified_models
from .paths import resolve_within_project
from .plan import plan_models
from .profile import (
    ResolvedProfile,
    apply_source_path_overrides,
    resolve_profile,
)
from .progress import get_reporter
from .reprocess_guard import format_refusals, guard_reprocess
from .sources import SourceError, get_document_source
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
    read_filter: Sequence[tuple[str, str, str]] = (),
    accept_reprocess: bool = False,
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
    # Deliberately NOT resolved here. `--read-filter` with a `state:` selector
    # needs the modified set, which needs the manifest, which is loaded below
    # -- so validating this early raises "selector requires --state" for a
    # caller who supplied --state (issue #417 review). Validation happens once
    # the real selection exists.
    read_predicates: tuple[ReadPredicate, ...] = ()
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    log.info(
        "resolved profile '%s' target '%s' -> %s warehouse",
        project.profile,
        resolved.target_name,
        resolved.warehouse.type,
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
    if read_filter:
        read_predicates = _prepare_read_filter(
            read_filter,
            full_refresh=full_refresh,
            selected=selected,
            models=models,
        )
        # Either filter flag makes the whole invocation additive: absence from
        # a deliberately narrowed run is not removal (issue #417).
        subset_run = True
    validate_retrieval_capabilities(
        [model for model in models if model.name in set(selected)], project, resolved
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

    reporter = get_reporter()
    reporter.run_started(
        len(selected),
        target=resolved.target_name,
        warehouse=resolved.warehouse.type,
        project_total=len(models),
    )
    log.info(
        "selected %d of %d model(s)",
        len(selected),
        len(models),
        extra=REPORTER_ECHO_EXTRA,
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
            dag=dag,
            threads=threads,
            run_budget=run_budget,
            subset_run=subset_run,
            read_predicates=read_predicates,
        )

    started_at = datetime.now(UTC).isoformat()
    with adapter:
        log.info("connected to %s warehouse", resolved.warehouse.type)
        _enforce_reprocess_guard(
            selected,
            models_by_name=models_by_name,
            dag=dag,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
            resolved=resolved,
            full_refresh=full_refresh,
            accept_reprocess=accept_reprocess,
        )
        if threads > 1 and len(selected) > 1:
            results_by_name = _run_in_batches(dag, selected, adapter, _run, threads)
        else:
            results_by_name = {name: _run(name, adapter) for name in selected}

        results = [results_by_name[name] for name in selected]
        # Written inside the adapter context, after the models it describes:
        # best-effort by contract, so a log failure never turns a successful
        # run into a failed one (issue #306).
        write_rows(
            adapter,
            resolved.run_log,
            run_log_rows(
                results,
                invocation_id=uuid.uuid4().hex,
                started_at=started_at,
                completed_at=datetime.now(UTC).isoformat(),
                profile_target=resolved.target_name,
                # `run` has no notion of tests: null columns, not zeros.
                test_results=None,
            ),
            schema=RUN_LOG_SCHEMA,
            what="the run log",
        )

    errored = sum(1 for r in results if r.errors)
    reporter.run_finished(ok=len(results) - errored, errored=errored, skipped=0)
    return results


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
    read_filter: Sequence[tuple[str, str, str]] = (),
    accept_reprocess: bool = False,
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
    # Deliberately NOT resolved here. `--read-filter` with a `state:` selector
    # needs the modified set, which needs the manifest, which is loaded below
    # -- so validating this early raises "selector requires --state" for a
    # caller who supplied --state (issue #417 review). Validation happens once
    # the real selection exists.
    read_predicates: tuple[ReadPredicate, ...] = ()
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    log.info(
        "resolved profile '%s' target '%s' -> %s warehouse",
        project.profile,
        resolved.target_name,
        resolved.warehouse.type,
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
    if read_filter:
        read_predicates = _prepare_read_filter(
            read_filter,
            full_refresh=full_refresh,
            selected=selected,
            models=models,
        )
        # Either filter flag makes the whole invocation additive: absence from
        # a deliberately narrowed run is not removal (issue #417).
        subset_run = True
    validate_retrieval_capabilities(
        [model for model in models if model.name in set(selected)], project, resolved
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

    reporter = get_reporter()
    reporter.run_started(
        len(selected),
        target=resolved.target_name,
        warehouse=resolved.warehouse.type,
        project_total=len(models),
    )
    log.info(
        "selected %d of %d model(s)",
        len(selected),
        len(models),
        extra=REPORTER_ECHO_EXTRA,
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
    skipped_results: list[ModelRunResult] = []

    started_at = datetime.now(UTC).isoformat()
    with adapter:
        log.info("connected to %s warehouse", resolved.warehouse.type)
        _enforce_reprocess_guard(
            selected,
            models_by_name=models_by_name,
            dag=dag,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
            resolved=resolved,
            full_refresh=full_refresh,
            accept_reprocess=accept_reprocess,
        )
        for name in selected:
            model = models_by_name[name]
            if name in blocked:
                out.skipped.append(name)
                reporter.model_skipped(name, "upstream failed")
                # Only the log gets a row: `out.run_results` also feeds
                # `run_results.json` and the error count, which a skip is
                # not part of.
                skipped_results.append(
                    ModelRunResult(
                        model_name=name,
                        materialization=model.materialization,
                        kind=_model_kind_label(model),
                        status="skipped",
                    )
                )
                continue
            # Taken here rather than inside `_run_model`, which cannot report
            # them once it has raised: the failure row below carries the
            # model's own span, not the invocation's (issue #623).
            model_started_at = datetime.now(UTC).isoformat()
            model_start = time.monotonic()
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
                    dag=dag,
                    threads=threads,
                    run_budget=run_budget,
                    subset_run=subset_run,
                    read_predicates=read_predicates,
                )
            except RunError as e:
                failed = _failed_model_result(
                    model,
                    e,
                    started_at=model_started_at,
                    duration_seconds=round(time.monotonic() - model_start, 3),
                )
                out.run_results.append(failed)
                # _run_model raised before reaching its own model_finished, so
                # the ledger would skip this model entirely without this.
                reporter.model_finished(
                    name,
                    failed.kind,
                    failed.rows_written,
                    failed.duration_seconds,
                    None,
                    failed=True,
                )
                blocked |= dag.descendants(name)
                continue

            out.run_results.append(result)
            if result.errors:
                blocked |= dag.descendants(name)
                continue

            log.info("testing %s", name)
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
            if model_tests:
                log.info(
                    "tested %s: %d passed, %d failed",
                    name,
                    sum(1 for t in model_tests if t.status == "pass"),
                    sum(1 for t in model_tests if t.status == "fail"),
                )
            if any(t.is_hard_failure for t in model_tests):
                blocked |= dag.descendants(name)

        # Same contract as `run_project`: written inside the adapter context,
        # after the models it describes, best-effort (issue #575). A build's
        # per-model outcome includes tests, which a plain run has no notion
        # of, so its test counts ride along on the same row.
        write_rows(
            adapter,
            resolved.run_log,
            run_log_rows(
                [*out.run_results, *skipped_results],
                invocation_id=uuid.uuid4().hex,
                started_at=started_at,
                completed_at=datetime.now(UTC).isoformat(),
                profile_target=resolved.target_name,
                test_results=out.test_results,
            ),
            schema=RUN_LOG_SCHEMA,
            what="the run log",
        )

    errored = sum(1 for r in out.run_results if r.errors)
    hard_failed = {t.model_name for t in out.test_results if t.is_hard_failure}
    # A model whose run succeeded but whose tests hard-failed is not "ok" — the
    # footer would otherwise disagree with the exit code.
    errored += len(hard_failed - {r.model_name for r in out.run_results if r.errors})
    reporter.run_finished(
        ok=len(out.run_results) - errored,
        errored=errored,
        skipped=len(out.skipped),
    )
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
    guarded = cast(WarehouseAdapter, SerializedAdapter(adapter, threading.Lock()))
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
            # Passed as a listing hint only — the filter below is what
            # actually decides the run's documents (issue #348).
            refs = backend.discover(source, project_dir, source_filter=source_filter)
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
        log.info(
            "Source '%s': %d document(s) selected",
            source.name,
            len(refs),
            extra=REPORTER_ECHO_EXTRA,
        )
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


_READ_FILTER_OPERATORS = {
    "eq": ReadPredicateOperator.EQUAL,
    "ne": ReadPredicateOperator.NOT_EQUAL,
    "lt": ReadPredicateOperator.LESS_THAN,
    "le": ReadPredicateOperator.LESS_THAN_OR_EQUAL,
    "gt": ReadPredicateOperator.GREATER_THAN,
    "ge": ReadPredicateOperator.GREATER_THAN_OR_EQUAL,
    "in": ReadPredicateOperator.IN,
}


def _prepare_read_filter(
    read_filter: Sequence[tuple[str, str, str]],
    *,
    full_refresh: bool,
    selected: Sequence[str],
    models: list[ModelConfig],
) -> tuple[ReadPredicate, ...]:
    """Validate `--read-filter` and build the typed predicates (issue #417).

    A read filter narrows what transform parent reads and embed source reads
    see, so it carries the same additive contract as --source-filter -- and
    two constraints follow directly:

    - **Not with --full-refresh.** A full refresh rebuilds from what it reads;
      rebuilding from a slice silently truncates the model to the slice.
    - **Only onto incremental transform/embed models.** A `materialization:
      full` model replaces its whole table from its (now narrowed) read --
      the same truncation, without the flag to warn about it.
    """
    if not read_filter:
        return ()
    if full_refresh:
        raise RunError(
            "--read-filter cannot be combined with --full-refresh: a filtered "
            "read rebuilds from a slice, which would silently truncate the "
            "model to that slice."
        )
    selected_set = set(selected)
    unsafe: list[str] = []
    for model in models:
        if model.name not in selected_set:
            continue
        narrowed = model.embed is not None or (
            model.transform is not None and model.transform.type == "python"
        )
        if narrowed and model.materialization != "incremental":
            unsafe.append(f"{model.name} (materialization: {model.materialization})")
    if unsafe:
        raise RunError(
            "--read-filter narrows what a model reads, so every selected "
            "transform/embed model must be incremental -- a full "
            "materialization would replace the whole table with the slice. "
            "Unsafe: " + ", ".join(sorted(unsafe))
        )
    predicates: list[ReadPredicate] = []
    for column, operator_name, raw_value in read_filter:
        operator = _READ_FILTER_OPERATORS.get(operator_name)
        if operator is None:
            raise RunError(
                f"--read-filter operator '{operator_name}' is not one of "
                f"{sorted(_READ_FILTER_OPERATORS)}"
            )
        value: Any
        if operator is ReadPredicateOperator.IN:
            try:
                decoded = json.loads(raw_value)
            except json.JSONDecodeError:
                raise RunError(
                    "--read-filter 'in' takes a JSON array of strings"
                ) from None
            if (
                not isinstance(decoded, list)
                or not decoded
                or not all(isinstance(item, str) for item in decoded)
            ):
                raise RunError(
                    "--read-filter 'in' takes a non-empty JSON array of strings"
                )
            value = tuple(decoded)
        else:
            value = raw_value
        predicates.append(ReadPredicate(column, operator, value))
    return tuple(predicates)


def _run_budget_ledger(resolved: ResolvedProfile) -> BudgetLedger | None:
    """One shared run-scope ledger; every LLM extraction model charges it."""
    if resolved.llm is None or resolved.llm.budget is None:
        return None
    return BudgetLedger(resolved.llm.budget, scope="run")


def _enforce_reprocess_guard(
    selected: list[str],
    *,
    models_by_name: Mapping[str, ModelConfig],
    dag: ProjectDAG,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    full_refresh: bool,
    accept_reprocess: bool,
) -> None:
    """Stop before the first model runs if a paid model would reprocess
    published rows it was not told to (issue #530).

    `--full-refresh` and `--accept-reprocess` are the operator saying so.
    Otherwise the whole selection is planned -- every model, because the
    cascade an embed model pays for starts at a chunk model above it -- and
    any `on_code_change: fail` model over its limit refuses the run. The plan
    is one aggregate query per selected model; a selection with nothing under
    the guard skips it entirely."""
    if full_refresh or accept_reprocess:
        return
    planned = [models_by_name[name] for name in selected]
    guarded = [
        model
        for model in planned
        if (model.embed is not None and model.embed.on_code_change == "fail")
        or (model.llm is not None and model.llm.on_code_change == "fail")
    ]
    if not guarded:
        return
    plans = plan_models(
        planned,
        dag=dag,
        project=project,
        project_dir=project_dir,
        adapter=adapter,
        resolved=resolved,
    )
    refusals = guard_reprocess(plans)
    if refusals:
        raise RunError(format_refusals(refusals))
    log.info(
        "reprocess guard: %d guarded model(s) within their reprocess_limit",
        len(guarded),
    )


def _single_data_parent(model_name: str, dag: ProjectDAG) -> str | None:
    """The one upstream stel model whose content this model's classification
    scan actually reads, or None (issue #611).

    `dag.predecessors` also carries ordering-only edges -- `depends_on` noise,
    relationship-test targets, a retrieval test's golden set -- that are not
    data the scan reads. Rather than resolve the real one precisely per kind,
    this requires there to be *exactly one* non-source predecessor at all:
    over-restrictive for a model with, say, both a real data parent and an
    unrelated relationship test pointing at another model, but that only
    costs a missed optimization for that model, never an incorrect skip. Zero
    predecessors (a root model reading a raw source) must always return None
    too -- that scan, for new source documents, is exactly the one this
    mechanism can never justify skipping.
    """
    model_predecessors = [
        name
        for name in dag.predecessors.get(model_name, set())
        if dag.nodes[name].kind != NodeKind.SOURCE
    ]
    return model_predecessors[0] if len(model_predecessors) == 1 else None


def _can_skip_unchanged_scan(model: ModelConfig, *, full_refresh: bool, subset_run: bool) -> bool:
    """Whether this model kind and invocation are even eligible for the
    unchanged-parent skip (issue #611), before paying for the watermark and
    code_version checks that decide it for real.

    `search:` is excluded: `run_search_model` also sweeps stale retrieval
    generations inline, a side effect this skip must not suppress. `ml:` and
    `eval:` are excluded because they do not fit the same incremental,
    state-scoped-by-model-name contract the skip's state check assumes.
    `--full-refresh` and a source-filtered/read-filtered subset run each
    narrow or force what a normal run would do, so neither is safe to
    second-guess with a scan that was written for the unfiltered case."""
    return (
        not full_refresh
        and not subset_run
        and model.materialization == "incremental"
        and model.search is None
        and model.ml is None
        and model.eval is None
    )


def _watermark_safely[T](model_name: str, operation: str, action: Callable[[], T]) -> T | None:
    """Run one unchanged-scan-skip read or write (issue #611), never letting
    it fail the run it was only ever meant to speed up.

    This is the runner, one of the three places this codebase's exception
    policy names as a legitimate boundary. The skip is purely an
    optimization over an always-correct scan: a warehouse error here --
    including the brand-new `stel_sync_watermark` table existing but the
    caller lacking permission to create or read it, on a deployment that can
    otherwise mutate `stel_state` fine -- must fall back to the real scan,
    not abort an otherwise-successful model. The exception class, not its
    text, is logged: the same reasoning the store layer already applies to
    native warehouse errors.
    """
    try:
        return action()
    except Exception as error:
        log.warning(
            "%s: could not %s for the unchanged-scan skip [%s]; doing the full scan instead",
            model_name,
            operation,
            type(error).__name__,
        )
        return None


def _published_row_count_if_code_unchanged(
    model: ModelConfig,
    *,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
) -> int | None:
    """This model's published row count, when every one of those rows still
    carries the current code_version -- one aggregate query, the same one
    `stel plan` runs (issue #611). None (never skip) on a model with no
    published state or with any stale row: a first run, or a real code
    change, always has to do the real work regardless of what its parent did.

    The count doubles as what a full scan would have reported as
    `documents_skipped`: with the watermark also matching (checked
    separately, by the caller), every one of this model's previously
    published rows is still exactly what the parent holds.
    """
    counts = adapter.state_code_version_counts(StateScope(model.name))
    state_rows = sum(counts.values())
    if state_rows == 0:
        return None
    code_version = compute_model_code_version(
        model, project, project_dir, resolved=resolved
    )
    stale_rows = sum(count for version, count in counts.items() if version != code_version)
    return state_rows if stale_rows == 0 else None


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
    dag: ProjectDAG,
    threads: int = 1,
    run_budget: BudgetLedger | None = None,
    subset_run: bool = False,
    read_predicates: Sequence[ReadPredicate] = (),
) -> ModelRunResult:
    kind = _model_kind_label(model)
    log.info("starting %s (%s)", model.name, kind)
    started_at = datetime.now(UTC).isoformat()
    start = time.monotonic()
    # The unchanged-scan skip (issue #611), two tiers. `state` is cheap (one
    # aggregate query over stel's own narrow bookkeeping table) and catches
    # every ordinary case, including a real code/cascade change and a real
    # deletion -- but it is blind to a write that bypassed stel entirely (a
    # direct UPDATE/DELETE/ALTER TABLE against a model's own output table,
    # which this repo's own test suite does routinely to simulate scenarios
    # cheaply, so it is not a hypothetical). `content` is the authoritative
    # confirmation: a real-cost aggregate hash over the parent's actual
    # current rows.
    #
    # `content` is read exactly once whenever it is needed at all, strictly
    # before this model's own run starts, and reused as-is for the post-run
    # watermark write rather than ever re-read afterward (Codex review,
    # #612): a parent mutated *during* this model's own dispatch -- by
    # anything other than stel, since nothing stel-owned touches it while
    # this model depends on it -- must never be folded into a watermark
    # claiming this model is caught up with content it never actually
    # consumed. It is read in two situations, both cheap relative to what
    # happens next: to confirm a skip the cheap signal alone cannot (the
    # no-op case this mechanism exists for), and -- whenever real work is
    # about to happen for any reason (a first run, a code change, a cheap
    # mismatch) -- once more, so that work ends with a fresh watermark
    # established. Without the second case a model would never acquire its
    # first watermark: the next run would find nothing to compare against
    # either, and the skip could never engage for it at all.
    parent_name = (
        _single_data_parent(model.name, dag)
        if _can_skip_unchanged_scan(model, full_refresh=full_refresh, subset_run=subset_run)
        else None
    )
    parent_scope = StateScope(parent_name) if parent_name is not None else None
    parent_state = (
        _watermark_safely(
            model.name,
            "read the parent's generation",
            lambda: adapter.state_generation(parent_scope),
        )
        if parent_scope is not None
        else None
    )
    parent_content: TableContentFingerprint | None = None
    skipped_row_count = None
    if parent_state is not None:
        skipped_row_count = _published_row_count_if_code_unchanged(
            model, project=project, project_dir=project_dir, adapter=adapter, resolved=resolved
        )
        if skipped_row_count is not None:
            child_scope = StateScope(model.name)
            synced = _watermark_safely(
                model.name,
                "read its sync watermark",
                lambda: adapter.read_sync_watermark(child_scope, cast(StateScope, parent_scope)),
            )
            if synced is None or synced.state != parent_state:
                skipped_row_count = None
            else:
                # The cheap signal alone looks unchanged: pay for the
                # authoritative one now, to decide the skip for real.
                parent_content = _watermark_safely(
                    model.name,
                    "fingerprint the parent's content",
                    lambda: adapter.table_content_fingerprint(cast(str, parent_name)),
                )
                if parent_content is None or synced.content != parent_content:
                    skipped_row_count = None
        if skipped_row_count is None and parent_content is None:
            # Real work is happening regardless -- the model's own code
            # changed, no watermark was ever recorded, or the cheap signal
            # alone already proved a change. Reading content now, once, is
            # marginal next to the real scan about to run, and it is the only
            # way a watermark ever gets established for a model that has
            # never confirmed one before (otherwise it never would: the next
            # run would find no watermark to compare against either, and the
            # skip could never engage for this model at all).
            parent_content = _watermark_safely(
                model.name,
                "fingerprint the parent's content",
                lambda: adapter.table_content_fingerprint(cast(str, parent_name)),
            )
    if skipped_row_count is not None:
        # The parent published nothing since this model last synced to it (by
        # either signal), and this model's own code hasn't moved since its
        # last publish -- so there is nothing for a full parent scan to find.
        result = ModelRunResult(
            model_name=model.name,
            materialization=model.materialization,
            kind=kind,
            status="unchanged",
            documents_skipped=skipped_row_count,
        )
    elif model.extraction is not None:
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
                subset_run=subset_run,
                read_predicates=read_predicates,
            )
    elif model.chunk is not None:
        result = _run_chunk_model(
            model=model,
            project_dir=project_dir,
            adapter=adapter,
            full_refresh=full_refresh,
            subset_run=subset_run,
        )
    elif model.embed is not None:
        result = _run_embed_model(
            model=model,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
            resolved=resolved,
            full_refresh=full_refresh,
            run_budget=run_budget,
            subset_run=subset_run,
            read_predicates=read_predicates,
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
            full_refresh=full_refresh,
            subset_run=subset_run,
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
    if parent_scope is not None and parent_content is not None and not result.errors:
        # This model just did real work and came back clean: it is now caught
        # up with the parent as of `parent_content`, read strictly before
        # this run started and never re-read since (see the comment above).
        # A model whose parent could not be read at all this round (every
        # attempt above failed, or there is no eligible parent) writes no
        # watermark, never one derived from content read after the fact.
        # Best-effort (issue #611): losing this write costs one missed skip
        # next run, never a wrong one, so it is never worth failing over.
        assert parent_state is not None  # parent_content is only ever set inside that branch
        watermark = SyncWatermark(state=parent_state, content=parent_content)
        _watermark_safely(
            model.name,
            "record its sync watermark",
            lambda: adapter.write_sync_watermark(
                StateScope(model.name), parent_scope, watermark
            ),
        )
    result.duration_seconds = round(time.monotonic() - start, 3)
    result.started_at = started_at
    result.completed_at = datetime.now(UTC).isoformat()
    log.info(
        "finished %s: %d row(s) in %.3fs%s",
        model.name,
        result.rows_written,
        result.duration_seconds,
        f" [{result.status}]" if result.status else "",
        extra=REPORTER_ECHO_EXTRA,
    )
    get_reporter().model_finished(
        model.name,
        kind,
        result.rows_written,
        result.duration_seconds,
        result.status,
        failed=bool(result.errors),
    )
    return result


def _failed_model_result(
    model: ModelConfig,
    error: RunError,
    *,
    started_at: str,
    duration_seconds: float,
) -> ModelRunResult:
    """The run-results row for a model whose stage raised (issue #623).

    The configured kind, the model's own span, and whatever the stage managed
    to attribute before it failed. A slow failure is the one worth diagnosing,
    and a row that says "unknown, zero rows, zero seconds" for a six-hour
    search publish is the run log being wrong exactly where an operator reads
    it. The counters come from `RunError.progress` and are filtered to the
    fields the result has, so a stage cannot smuggle an unknown key into
    `run_results.json`.
    """
    result = ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind=_model_kind_label(model),
        errors=[_artifact_error_text(error)],
        duration_seconds=duration_seconds,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        metrics=error.metrics,
    )
    for name, value in error.progress.items():
        if name in _PROGRESS_FIELDS and isinstance(value, int) and not isinstance(value, bool):
            setattr(result, name, value)
    return result


_PROGRESS_FIELDS = frozenset(
    {
        "documents_processed",
        "documents_skipped",
        "documents_deleted",
        "rows_written",
        "rows_inserted",
        "rows_updated",
        "rows_failed",
    }
)


def _model_kind_label(model: ModelConfig) -> str:
    """The run-result label for a model's kind.

    Delegates rather than re-deriving. This label, `stel ls`'s kind column and
    the `kind:` selector all have to agree about what a model is, and the only
    way to guarantee that is for there to be one answer (issue #494). Kept as a
    named function because the runner calls it in two places and tests pin it.
    """
    return model.kind_label()


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

    for filename in ("manifest.json", "run_results.json", "plan.json", "sources.yml"):
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
