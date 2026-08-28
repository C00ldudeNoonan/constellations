from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa

from ..adapters import (
    AdapterError,
    ReadPredicate,
    ReadPredicateOperator,
    StateRecord,
    StateScope,
    StateValue,
    WarehouseAdapter,
    WarehouseCapability,
)
from ..agent_context import (
    AgentContextValidationError,
    empty_agent_context_frame,
    validate_agent_context_frame,
)
from ..budget import BudgetLedger
from ..config.model import ModelConfig
from ..config.profile import DEFAULT_LLM_PROVIDER
from ..config.project import ProjectConfig
from ..dag import is_dbt_ref, parse_dbt_ref, parse_ref
from ..hashing import canonical_fingerprint, canonical_json
from ..paths import resolve_within_project
from ..profile import ResolvedProfile
from ..providers import get_inference_provider, resolve_provider_model
from ..sql_models import compile_sql, discover_refs, read_sql_source
from ..transforms import (
    IncrementalContract,
    ReferenceDep,
    TransformContext,
    load_incremental_contract,
    load_transform,
    transform_call_arity,
)
from ..versioning import compute_model_code_version
from .contracts import ModelRunResult, RunError
from .errors import artifact_error_text
from .warehouse import warehouse_options

log = logging.getLogger(__name__)


def run_sql_model(
    *,
    model: ModelConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    full_refresh: bool = False,
) -> ModelRunResult:
    """Compile a SQL transform's refs to warehouse relations and materialize it
    with an adapter-owned CTAS/merge — upstream rows never enter the stel
    process.

    For `materialization: incremental` (#142), the SQL is compiled twice
    differently depending on whether the target already exists and
    `--full-refresh` isn't active: `is_incremental()` renders False (and
    `materialize_sql_full` runs a plain CTAS) on the first run or under
    `--full-refresh`; otherwise it renders True and the compiled delta query is
    merged into the existing target via `materialize_sql_incremental`."""
    assert model.transform is not None and model.transform.path is not None
    if WarehouseCapability.SQL_MODEL_MATERIALIZATION not in adapter.capabilities():
        raise RunError(
            f"Adapter '{adapter.adapter_type()}' does not support SQL models "
            "(SQL_MODEL_MATERIALIZATION)."
        )

    resolved_path = resolve_within_project(
        model.transform.path,
        project_dir,
        surface=f"model '{model.name}' transform.path",
    )
    sql_text = read_sql_source(resolved_path, model_name=model.name)
    refs = discover_refs(sql_text, model_name=model.name)
    # Each ref resolves to the upstream model's adapter-quoted target relation.
    relations = {name: adapter.table_ref(name) for name in refs}

    is_incremental_model = model.materialization == "incremental"
    target_exists = is_incremental_model and adapter.relation_exists(model.name)
    # Compile-time is_incremental() reflects what will actually run: False on
    # the first run or --full-refresh, even though the model is *configured*
    # incremental, so the .sql file's own branch stays truthful.
    run_incremental = is_incremental_model and target_exists and not full_refresh
    select_sql = compile_sql(
        sql_text,
        model_name=model.name,
        relations=relations,
        target_name=resolved.target_name,
        target_type=resolved.warehouse.type,
        this=adapter.table_ref(model.name) if is_incremental_model else None,
        is_incremental=run_incremental if is_incremental_model else None,
    )

    try:
        if run_incremental:
            assert model.unique_key is not None  # enforced at compile time
            if (
                WarehouseCapability.SQL_INCREMENTAL_MATERIALIZATION
                not in adapter.capabilities()
            ):
                raise RunError(
                    f"Adapter '{adapter.adapter_type()}' does not support "
                    "incremental SQL models (SQL_INCREMENTAL_MATERIALIZATION)."
                )
            result = adapter.materialize_sql_incremental(
                model.name,
                select_sql,
                unique_key=model.unique_key,
                on_schema_change=model.on_schema_change,
                options=warehouse_options(adapter, model),
            )
        else:
            result = adapter.materialize_sql_full(
                model.name, select_sql, options=warehouse_options(adapter, model)
            )
    except AdapterError as e:
        raise RunError(f"SQL model '{model.name}' failed: {e}") from e

    metrics: dict[str, Any] = {
        "compiled_sql": select_sql,
        "relation": result.relation,
        "refs": list(refs),
    }
    if is_incremental_model:
        metrics["is_incremental"] = run_incremental
    if result.job_metadata:
        metrics["job"] = result.job_metadata
    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="sql",
        rows_written=result.rows_written,
        rows_inserted=result.rows_inserted or 0,
        rows_updated=result.rows_updated or 0,
        metrics=metrics,
    )


def run_transform_model(
    *,
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    full_refresh: bool = False,
    run_budget: BudgetLedger | None = None,
    subset_run: bool = False,
    read_predicates: Sequence[ReadPredicate] = (),
) -> ModelRunResult:
    assert model.transform is not None
    if model.transform.type != "python":
        raise RunError(
            f"Model '{model.name}': only `type: python` transforms are supported in v1"
        )
    if not model.transform.module:
        raise RunError(f"Model '{model.name}': transform requires a `module:`")
    dbt_ref_source = (
        model.source if model.source and is_dbt_ref(model.source) else None
    )
    if not model.depends_on and dbt_ref_source is None:
        raise RunError(
            f"Transform model '{model.name}' must declare `depends_on:` for v1"
        )

    provider_name, provider_model, provider_implementation = _resolve_transform_provider(
        model, resolved
    )

    transform_fn = load_transform(model.transform.module, project_dir)
    dep_names: list[str] = []
    if dbt_ref_source is not None:
        # The single input is a dbt-built table (#177). In embedded mode the dbt
        # Python shim reads `dbt.ref('name')` and injects it as an upstream, which
        # the CaptureAdapter serves through the same `read_table` path.
        dep_names.append(parse_dbt_ref(dbt_ref_source))
    dep_names.extend(parse_ref(dep_ref) for dep_ref in model.depends_on or [])

    contract: IncrementalContract | None = None
    if model.materialization == "incremental":
        contract = load_incremental_contract(
            model.transform.module, project_dir, model.transform.options
        )
        # The compiler requires the contract for incremental materialization.
        assert contract is not None, "incremental transform is missing its contract"
        _require_incremental_capabilities(model, adapter)

    # The parent source is deliberately *not* read here when the model is
    # incremental: it is classified from a streamed scan and then re-read
    # scoped to the parents that actually changed, so peak memory follows the
    # change set rather than the corpus (issue #385).
    parent_source = (
        contract.resolve_parent_source(dep_names) if contract is not None else None
    )
    deps: dict[str, pl.DataFrame] = {
        name: adapter.read_table(name) for name in dep_names if name != parent_source
    }

    ctx = TransformContext(
        project_dir=project_dir,
        profile_name=resolved.profile_name,
        target_name=resolved.target_name,
        warehouse=resolved.warehouse,
        llm=resolved.llm,
        options=dict(model.transform.options),
        run_budget=run_budget,
    )

    result = ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="transform",
        provider=provider_name,
        provider_model=provider_model,
        provider_implementation=provider_implementation,
    )

    if contract is not None:
        assert parent_source is not None
        return _run_incremental_transform(
            model=model,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
            resolved=resolved,
            transform_fn=transform_fn,
            ctx=ctx,
            deps=deps,
            contract=contract,
            parent_source=parent_source,
            full_refresh=full_refresh,
            result=result,
            subset_run=subset_run,
            read_predicates=read_predicates,
        )

    output = _invoke_transform(transform_fn, deps, ctx, model)
    adapter.materialize_full(
        model.name,
        _validate_agent_context_output(output, model),
        options=warehouse_options(adapter, model),
    )
    result.rows_written = output.height
    return result


def _resolve_transform_provider(
    model: ModelConfig, resolved: ResolvedProfile
) -> tuple[str | None, str | None, str | None]:
    assert model.transform is not None
    if not model.transform.uses_llm:
        return None, None, None
    provider_name = (
        resolved.llm.provider if resolved.llm is not None else DEFAULT_LLM_PROVIDER
    )
    selected_model = resolved.llm.model if resolved.llm is not None else None
    try:
        provider = get_inference_provider(provider_name)
        provider_model = resolve_provider_model(provider, selected_model)
        provider_implementation = provider.implementation_identity()
    except Exception as e:
        raise RunError(
            f"Transform model '{model.name}' could not initialize inference: "
            f"{artifact_error_text(e)}"
        ) from e
    return provider_name, provider_model, provider_implementation


def _invoke_transform(
    transform_fn: Any,
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
    model: ModelConfig,
) -> pl.DataFrame:
    assert model.transform is not None
    try:
        if transform_call_arity(transform_fn) == 2:
            output = transform_fn(deps, ctx)
        else:
            output = transform_fn(deps)
    except RunError:
        raise
    except Exception as e:
        log.debug("transform failed for %s", model.name, exc_info=True)
        raise RunError(
            f"Transform model '{model.name}' failed: {artifact_error_text(e)}"
        ) from e
    if not isinstance(output, pl.DataFrame):
        raise RunError(
            f"Transform '{model.transform.module}' must return a polars.DataFrame"
        )
    return output


def _require_incremental_capabilities(
    model: ModelConfig, adapter: WarehouseAdapter
) -> None:
    """Fail preflight when the active adapter cannot atomically replace a
    changed parent's children and advance state in one transaction (issue #229)."""
    capabilities = adapter.capabilities()
    missing = [
        capability.value
        for capability in (
            WarehouseCapability.ATOMIC_PARENT_CHILD_REPLACE,
            WarehouseCapability.ATOMIC_STATE_SCOPE_REPLACE,
            # Classification streams the parent table rather than loading it
            # (issue #385); without this the only alternative is the
            # whole-corpus read the change exists to remove.
            WarehouseCapability.STREAMING_TABULAR_READS,
            # The scoped re-read pushes an IN predicate. Without this an
            # adapter would pass preflight, complete the whole classification
            # scan, and only then fail (Codex review).
            WarehouseCapability.TABULAR_PREDICATE_PUSHDOWN,
        )
        if capability not in capabilities
    ]
    if missing:
        raise RunError(
            f"Adapter '{adapter.adapter_type()}' cannot run incremental transform "
            f"'{model.name}': missing capabilities {sorted(missing)}."
        )


def _run_incremental_transform(
    *,
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    transform_fn: Any,
    ctx: TransformContext,
    deps: dict[str, pl.DataFrame],
    contract: IncrementalContract,
    parent_source: str,
    full_refresh: bool,
    result: ModelRunResult,
    subset_run: bool = False,
    read_predicates: Sequence[ReadPredicate] = (),
) -> ModelRunResult:
    """Generalize the chunk-model incremental pattern to a declared one-to-many
    transform: skip unchanged parents, invoke the transform only on changed/new
    parents, replace each changed parent's children by deleting on the parent
    key and upserting on the child key, and advance state only after a
    successful publication (issue #218)."""
    assert model.transform is not None
    code_version = compute_model_code_version(
        model, project, project_dir, resolved=resolved
    )
    state_scope = StateScope(model.name)
    options = warehouse_options(adapter, model)
    _reject_unsupported_incremental_strategy(options, model)
    is_incremental = not full_refresh
    prior_state = adapter.fetch_state(state_scope) if is_incremental else {}

    # No state baseline but a pre-existing target — e.g. a table switched from
    # `materialization: full`, or an interrupted first run. Its rows are not
    # owned by any per-parent state, so a child-keyed upsert would leave orphan
    # children a parent no longer emits. Rebuild with a full replace to
    # establish the baseline; subsequent runs are incremental (Codex review).
    if is_incremental and not prior_state and adapter.relation_exists(model.name):
        is_incremental = False

    reference_specs = contract.reference_specs()
    table_reference_fingerprints = {
        spec.name: _frame_fingerprint(deps[spec.name])
        for spec in reference_specs
        if spec.join_key is None
    }
    keyed_reference_fingerprints = {
        spec.name: _keyed_reference_fingerprints(deps[spec.name], spec, model.name)
        for spec in reference_specs
        if spec.join_key is not None
    }
    # Classification streams the parent table and keeps only a digest per row,
    # so the corpus is never resident (issue #385). Rows come back afterwards
    # for the changed parents alone.
    digests_by_parent = _stream_parent_digests(
        adapter,
        parent_source,
        contract.parent_source_key,
        model_name=model.name,
        predicates=read_predicates,
    )
    current_keys = set(digests_by_parent)

    processed_parents: list[str] = []
    state_records: list[StateRecord] = []
    changed: list[str] = []
    skipped = 0
    for parent_key, row_digests in digests_by_parent.items():
        reference_fingerprints = dict(table_reference_fingerprints)
        for name, by_parent in keyed_reference_fingerprints.items():
            reference_fingerprints[name] = by_parent.get(
                parent_key, _EMPTY_REFERENCE_ROWS_FINGERPRINT
            )
        fingerprint = _parent_fingerprint(
            parent_key, row_digests, reference_fingerprints
        )
        if is_incremental:
            prior = prior_state.get(parent_key)
            if prior == StateValue(fingerprint, code_version):
                skipped += 1
                continue
            if prior is not None:
                changed.append(parent_key)
        processed_parents.append(parent_key)
        state_records.append(StateRecord(parent_key, fingerprint, code_version))

    # Under a subset run the classification pass saw a deliberate slice of
    # the parents, so every other partition's parent would look removed;
    # reconciliation belongs to the next unfiltered run (issue #417).
    removed = (
        [key for key in prior_state if key not in current_keys]
        if is_incremental and not subset_run
        else []
    )

    # No changed or new parents: at most some removed parents need deleting.
    # Skip invoking the transform so an unchanged corpus performs no provider
    # work — the primary cost goal of incremental materialization.
    if is_incremental and not processed_parents:
        if removed:
            adapter.delete_rows_and_state(
                model.name,
                key_col=contract.parent_key,
                keys=removed,
                state_scope=state_scope,
            )
        result.documents_skipped = skipped
        result.documents_deleted = len(removed)
        result.metrics = _incremental_metrics(contract, is_incremental=True)
        return result

    if not is_incremental:
        full_deps = dict(deps)
        full_deps[parent_source] = _read_parent_rows(
            adapter, parent_source, contract.parent_source_key, keys=None
        )
        output = _validate_agent_context_output(
            _invoke_transform(transform_fn, full_deps, ctx, model), model
        )
        _validate_incremental_output(
            output, contract, model, set(processed_parents), is_incremental=False
        )
        # Full refresh: atomic full replace, then reset the per-parent state
        # baseline so the next incremental run classifies correctly. Never
        # batched — `materialize_full` replaces the table in one operation, and
        # committing it in pieces would expose a partially built table.
        result.rows_written = adapter.materialize_full(
            model.name, output, options=options
        )
        adapter.replace_state(state_scope, state_records)
        result.documents_processed = len(processed_parents)
        result.metrics = _incremental_metrics(contract, is_incremental=False)
        return result

    if removed:
        adapter.delete_rows_and_state(
            model.name,
            key_col=contract.parent_key,
            keys=removed,
            state_scope=state_scope,
        )

    # Invoke and publish in batches of changed parents, committing each the way
    # extraction commits each flush (issue #379). Each parent's children are
    # independent by the incremental contract's own definition, so a
    # partially-published run is coherent: the parents whose state advanced are
    # done, and a relaunch reclassifies only the rest. Before this, a failure
    # at the last parent — or at the publish — re-paid the whole corpus.
    changed_set = set(changed)
    records_by_parent = {record.record_key: record for record in state_records}
    rows_written = 0
    for batch in _batched(processed_parents, model.transform_commit_every()):
        call_deps = dict(deps)
        # Rows arrive per batch, scoped to that batch's parents (issue #385
        # composed with #379's batching): peak follows the commit batch, which
        # is tighter than either change gives on its own.
        batch_rows = _read_parent_rows(
            adapter,
            parent_source,
            contract.parent_source_key,
            keys=batch,
            predicates=read_predicates,
        )
        _verify_classified_rows(
            batch_rows,
            contract.parent_source_key,
            batch,
            digests_by_parent,
            model_name=model.name,
        )
        call_deps[parent_source] = batch_rows
        output = _validate_agent_context_output(
            _invoke_transform(transform_fn, call_deps, ctx, model), model
        )
        _validate_incremental_output(
            output, contract, model, set(batch), is_incremental=True
        )
        try:
            rows_written += adapter.replace_children(
                model.name,
                parent_key=contract.parent_key,
                parent_ids=[key for key in batch if key in changed_set],
                child_key=contract.child_key,
                new_rows=output,
                state_scope=state_scope,
                state_records=[records_by_parent[key] for key in batch],
                # Every batch, not just the first (Codex review). A transform's
                # output schema can be data-dependent, so a later batch may
                # introduce a column the first never emitted. Forcing `ignore`
                # after the first batch would drop that column silently while
                # still advancing the parents' state, making the loss
                # unrecoverable on later runs. Reconciling every batch costs
                # nothing when the schema is stable: `plan_schema_change`
                # early-returns once the column sets match.
                on_schema_change=model.on_schema_change,
                options=options,
            )
        except AdapterError as error:
            raise RunError(str(error)) from error

    result.rows_written = rows_written
    result.documents_processed = len(processed_parents)
    result.documents_skipped = skipped
    result.documents_deleted = len(removed)
    result.metrics = _incremental_metrics(contract, is_incremental=True)
    return result


def _verify_classified_rows(
    frame: pl.DataFrame,
    key_col: str,
    batch: list[str],
    classified: dict[str, list[str]],
    *,
    model_name: str,
) -> None:
    """Refuse to publish rows that are not the rows that were classified.

    Classification and the scoped read are separate snapshots (issue #385), so
    an upstream relation replaced between them would have a parent publish one
    generation's rows under another generation's fingerprint — a success that
    stores state describing content the table no longer holds (Codex review).
    The per-row digests are already in hand, so re-deriving them for the rows
    actually read costs one hash per changed row and turns that into a loud
    failure.

    This covers every parent whose rows are published. A parent classified as
    *unchanged* is never re-read, and a `--full-refresh` reads the whole table
    in one pass, so both remain exposed to a concurrent replacement for the
    rest of the run — self-correcting, because the next run reclassifies from
    current content, but not detected here.
    """
    observed: dict[str, list[str]] = {}
    for row in frame.iter_rows(named=True):
        observed.setdefault(str(row[key_col]), []).append(_row_digest(row))
    for parent_key in batch:
        if sorted(observed.get(parent_key, [])) != sorted(classified[parent_key]):
            raise RunError(
                f"Incremental transform '{model_name}': parent '{parent_key}' "
                "changed between classification and its scoped read, so its "
                "children would publish under a fingerprint describing "
                "different rows. Re-run once the upstream relation is stable."
            )


def _reject_unsupported_incremental_strategy(
    options: Any, model: ModelConfig
) -> None:
    """An incremental transform publishes only changed and new parents. A
    partition-replacing strategy (BigQuery `insert_overwrite`) would drop every
    row in a touched partition — including unchanged parents sharing it — while
    their state stays current, so they would be skipped forever. Reject it; the
    default key-merge strategy is required (Codex review)."""
    if getattr(options, "incremental_strategy", None) == "insert_overwrite":
        raise RunError(
            f"Incremental transform '{model.name}' cannot use "
            "warehouse_options.incremental_strategy: insert_overwrite — it replaces "
            "whole partitions, but incremental transforms publish only changed and "
            "new parents, which would drop unchanged parents sharing a partition. "
            "Use the default merge strategy."
        )


def _batched(keys: list[str], size: int) -> Iterator[list[str]]:
    """Split parents into commit batches, preserving order.

    A run with fewer changed parents than `size` yields exactly one batch, so
    the common case keeps its single warehouse MERGE and behaves as it did
    before issue #379.
    """
    for start in range(0, len(keys), size):
        yield keys[start : start + size]


def _validate_group_key(
    frame: pl.DataFrame, key_col: str, *, model_name: str, surface: str
) -> None:
    """Reject a grouping key that cannot identify rows, before any grouping.

    Vectorized rather than per-row: the null/empty check is the reason the old
    path had to visit every row as a Python object, and a column-wise test
    answers it over Arrow without materializing anything (issue #383).
    """
    if key_col not in frame.columns:
        raise RunError(
            f"Incremental transform '{model_name}': {surface} is missing the "
            f"parent key column '{key_col}'. Available: {sorted(frame.columns)}"
        )
    if frame.schema[key_col] != pl.String:
        raise RunError(
            f"Incremental transform '{model_name}': parent key column '{key_col}' "
            f"must be string-typed, got {frame.schema[key_col]}"
        )
    unusable = frame.select(
        (pl.col(key_col).is_null() | (pl.col(key_col).str.strip_chars() == ""))
        .any()
        .alias("bad")
    ).item()
    if unusable:
        raise RunError(
            f"Incremental transform '{model_name}': parent key column "
            f"'{key_col}' contains null or empty values"
        )


# Scratch column for the row indices `_row_groups` gathers by. Suffixed to
# stay clear of user columns; a collision is refused rather than shadowed.
_ROW_INDEX_COLUMN = "__stel_row_index__"


def _row_groups(
    frame: pl.DataFrame, key_col: str, *, model_name: str
) -> Iterator[tuple[str, pl.DataFrame]]:
    """Yield (key, rows) one group at a time, in first-appearance key order.

    Lazily, which is the whole point of issue #383 and the part
    `partition_by` cannot do: it returns a *list* of every partition, so the
    entire corpus is materialized as sub-frames before a generator over it can
    yield anything — the peak this was meant to remove, plus per-DataFrame
    overhead that is worst for the high-cardinality tables that motivated the
    change (Codex review).

    Instead one pass records each group's row indices — a few bytes per row,
    not a copy of it — and each group is gathered only when it is asked for.
    Peak memory is then the source frame plus the largest single group.
    """
    if _ROW_INDEX_COLUMN in frame.columns:
        raise RunError(
            f"Incremental transform '{model_name}': input column "
            f"'{_ROW_INDEX_COLUMN}' is reserved by stel; rename it"
        )
    grouped = (
        frame.with_row_index(_ROW_INDEX_COLUMN)
        .group_by(key_col, maintain_order=True)
        .agg(pl.col(_ROW_INDEX_COLUMN))
    )
    for key, indices in zip(
        grouped[key_col], grouped[_ROW_INDEX_COLUMN], strict=True
    ):
        yield str(key), frame[indices]


def _frame_fingerprint(frame: pl.DataFrame) -> str:
    # Order-insensitive: reordering a reference table (e.g. an alias list) must
    # not invalidate every parent.
    return canonical_fingerprint(
        {"rows": sorted(frame.iter_rows(named=True), key=canonical_json)},
        domain="dbt-ml.transform-incremental-reference",
    )


# What a keyed reference dep contributes to a parent with no joined rows: the
# fingerprint of an empty row group, so gaining a first reference row (or
# losing the last one) still moves that parent's fingerprint.
_EMPTY_REFERENCE_ROWS_FINGERPRINT = canonical_fingerprint(
    {"rows": []}, domain="dbt-ml.transform-incremental-reference"
)


def _keyed_reference_fingerprints(
    frame: pl.DataFrame, spec: ReferenceDep, model_name: str
) -> dict[str, str]:
    """Per-parent fingerprints over a keyed reference dep's joined rows
    (issue #364), keyed by parent identity. Mirrors `_parent_groups`'
    strictness: the declared join-key column must exist, be string-typed, and
    hold no null or empty values — a reference row that belongs to no parent
    would otherwise escape invalidation silently."""
    key_col = spec.join_key
    assert key_col is not None
    if key_col not in frame.columns:
        raise RunError(
            f"Incremental transform '{model_name}': reference dep '{spec.name}' is "
            f"missing its declared join_key column '{key_col}'. "
            f"Available: {sorted(frame.columns)}"
        )
    if frame.schema[key_col] != pl.String:
        raise RunError(
            f"Incremental transform '{model_name}': reference dep '{spec.name}' "
            f"join_key column '{key_col}' must be string-typed, "
            f"got {frame.schema[key_col]}"
        )
    unusable = frame.select(
        (pl.col(key_col).is_null() | (pl.col(key_col).str.strip_chars() == ""))
        .any()
        .alias("bad")
    ).item()
    if unusable:
        raise RunError(
            f"Incremental transform '{model_name}': reference dep '{spec.name}' "
            f"join_key column '{key_col}' contains null or empty values"
        )
    # One group's rows at a time, for the same reason as `_parent_groups`
    # (issue #383): only the digest is kept, so holding every group's rows as
    # Python objects bought nothing and cost a whole-table copy.
    return {
        key: canonical_fingerprint(
            {"rows": sorted(group.iter_rows(named=True), key=canonical_json)},
            domain="dbt-ml.transform-incremental-reference",
        )
        for key, group in _row_groups(frame, key_col, model_name=model_name)
    }


# Rows per streamed batch. Large enough that per-batch overhead is noise,
# small enough that one batch is never the memory story.
_CLASSIFY_BATCH_ROWS = 10_000

# Parent keys per scoped re-read. BigQuery rejects very large IN lists, and a
# huge one is a planning cost even where it is accepted, so the changed set is
# fetched in chunks and concatenated.
_READ_KEY_CHUNK = 1_000


def _stream_parent_digests(
    adapter: WarehouseAdapter,
    table: str,
    key_col: str,
    *,
    model_name: str,
    predicates: Sequence[ReadPredicate] = (),
) -> dict[str, list[str]]:
    """Per-parent row digests, in first-appearance parent order.

    The classification pass of issue #385. Rows are read in batches and
    reduced to one digest each, so what stays resident is ~32 bytes per row
    rather than the row itself — for a corpus like #383's SEC chunks, a few
    hundred MB in place of several GB. Nothing here needs the rows again:
    only the digests feed the fingerprint.

    Key validation happens per batch rather than over a materialized column,
    for the same reason.
    """
    digests: dict[str, list[str]] = {}
    with adapter.table_snapshot(
        table, batch_size=_CLASSIFY_BATCH_ROWS, predicate=list(predicates)
    ) as snapshot:
        for predicate in predicates:
            if predicate.column not in snapshot.schema.names:
                raise RunError(
                    f"Incremental transform '{model_name}': --read-filter "
                    f"column '{predicate.column}' is not in parent source "
                    f"'{table}'. Available: {sorted(snapshot.schema.names)}"
                )
        if key_col not in snapshot.schema.names:
            raise RunError(
                f"Incremental transform '{model_name}': parent_source is missing "
                f"the parent key column '{key_col}'. "
                f"Available: {sorted(snapshot.schema.names)}"
            )
        key_type = snapshot.schema.field(key_col).type
        if not pa.types.is_string(key_type) and not pa.types.is_large_string(key_type):
            # Coercing instead would be worse than a type error: the scoped
            # re-read pushes string parameters at a typed column, which an
            # adapter may answer with no rows — and an empty transform result
            # makes `replace_children` delete that parent's children and
            # advance its state (Codex review).
            raise RunError(
                f"Incremental transform '{model_name}': parent key column "
                f"'{key_col}' must be string-typed, got {key_type}"
            )
        for batch in snapshot:
            for row in batch.to_pylist():
                raw = row[key_col]
                if raw is None or not str(raw).strip():
                    raise RunError(
                        f"Incremental transform '{model_name}': parent key column "
                        f"'{key_col}' contains null or empty values"
                    )
                digests.setdefault(str(raw), []).append(_row_digest(row))
    return digests


def _read_parent_rows(
    adapter: WarehouseAdapter,
    table: str,
    key_col: str,
    *,
    keys: Sequence[str] | None,
    predicates: Sequence[ReadPredicate] = (),
) -> pl.DataFrame:
    """The parent rows the transform will actually see.

    `keys is None` means every row (a full refresh, which needs them all).
    Otherwise the read is pushed down to just those parents, so an incremental
    run moves data proportional to what changed (issue #385).
    """
    if keys is None:
        return adapter.read_table(table)
    if not keys:
        return adapter.read_table(table).clear()
    frames = [
        _read_keyed_chunk(
            adapter,
            table,
            key_col,
            tuple(keys[start : start + _READ_KEY_CHUNK]),
            predicates=predicates,
        )
        for start in range(0, len(keys), _READ_KEY_CHUNK)
    ]
    return frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical")


def _read_keyed_chunk(
    adapter: WarehouseAdapter,
    table: str,
    key_col: str,
    keys: tuple[str, ...],
    *,
    predicates: Sequence[ReadPredicate] = (),
) -> pl.DataFrame:
    # The read filter rides along with the key predicate, and must: the
    # classification pass hashed the *filtered* rows, so a reread by parent key
    # alone returns rows classification never saw. `_verify_classified_rows`
    # then correctly reports the parent as changed and aborts the run -- which
    # is what a filtered transform over a multi-row parent did before this
    # (issue #417 review).
    predicate = [
        ReadPredicate(column=key_col, operator=ReadPredicateOperator.IN, value=keys),
        *predicates,
    ]
    batches: list[pl.DataFrame] = []
    with adapter.table_snapshot(
        table, batch_size=_CLASSIFY_BATCH_ROWS, predicate=predicate
    ) as snapshot:
        empty = pl.from_arrow(snapshot.schema.empty_table())
        assert isinstance(empty, pl.DataFrame)
        for batch in snapshot:
            frame = pl.from_arrow(batch)
            assert isinstance(frame, pl.DataFrame)
            batches.append(frame)
    return pl.concat(batches, how="vertical") if batches else empty


def _row_digest(row: Mapping[str, Any]) -> str:
    """One row's whole contribution to its parent's fingerprint.

    Every column still participates — the coverage question (what invalidates
    a parent) is deliberately unchanged by issue #385, which is a memory
    question.
    """
    return canonical_fingerprint(row, domain="dbt-ml.transform-incremental-row")


def _parent_fingerprint(
    parent_key: str,
    row_digests: Sequence[str],
    reference_fingerprints: dict[str, str],
) -> str:
    """A parent's input fingerprint, over its rows' digests rather than its
    rows (issue #385).

    Sorting digests keeps the two properties the previous encoding had — the
    result is order-insensitive, and repeated rows still count, because
    duplicates survive a sort where they would cancel in an XOR or sum
    accumulator. What changes is that a parent's contribution can be
    accumulated from a streamed scan at 32 bytes per row instead of holding
    every row, which is the whole point: peak stops scaling with the corpus.

    `version=2` marks the encoding change. It re-keys every parent once — the
    same one-time cost #363 paid, and the reason `_INPUT_FINGERPRINT_VERSION`
    is pinned in `test_frozen_names.py` rather than left implicit.
    """
    return canonical_fingerprint(
        {
            "parent": parent_key,
            "rows": sorted(row_digests),
            "references": reference_fingerprints,
        },
        domain="dbt-ml.transform-incremental-input",
        version=2,
    )


def _validate_incremental_output(
    output: pl.DataFrame,
    contract: IncrementalContract,
    model: ModelConfig,
    processed_parents: set[str],
    *,
    is_incremental: bool,
) -> None:
    for column in (contract.parent_key, contract.child_key):
        if column not in output.columns:
            raise RunError(
                f"Incremental transform '{model.name}' output is missing declared "
                f"column '{column}'"
            )
    if output.height == 0:
        return
    if output[contract.parent_key].null_count():
        raise RunError(
            f"Incremental transform '{model.name}' produced null "
            f"{contract.parent_key} parent keys"
        )
    child = output[contract.child_key]
    if child.null_count():
        raise RunError(
            f"Incremental transform '{model.name}' produced null "
            f"{contract.child_key} child keys"
        )
    if child.n_unique() != output.height:
        raise RunError(
            f"Incremental transform '{model.name}' produced duplicate "
            f"{contract.child_key} child keys"
        )
    if is_incremental:
        emitted = set(output[contract.parent_key].cast(pl.String).to_list())
        unexpected = sorted(emitted - processed_parents)
        if unexpected:
            raise RunError(
                f"Incremental transform '{model.name}' emitted rows for parents "
                f"outside the changed/new set: {unexpected[:5]}"
            )


def _incremental_metrics(
    contract: IncrementalContract, *, is_incremental: bool
) -> dict[str, Any]:
    return {
        "incremental": is_incremental,
        "parent_key": contract.parent_key,
        "child_key": contract.child_key,
    }


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
