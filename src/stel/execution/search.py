"""Search publication executor (issue #190).

Owns the search-only lifecycle: retrieval-store selection, collection spec and
arrow-schema contract, publication lease/fence and serving coordination,
bounded reconciliation with per-batch state advancement, durable-receipt
verification, and the serving_resource descriptor. runner.py keeps selection,
DAG scheduling, and result aggregation, and re-exports run_search_model.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import suppress
from datetime import date, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pyarrow as pa

from ..adapters import (
    AdapterError,
    StateRecord,
    StateScope,
    StateScopeFence,
    WarehouseAdapter,
    WarehouseCapability,
)
from ..config.model import ModelConfig, SearchConfig
from ..config.project import ProjectConfig
from ..dag import parse_ref
from ..embedding import effective_search_config, resolve_search_embedding_options
from ..hashing import canonical_fingerprint
from ..memory import container_memory_limit_bytes
from ..profile import ResolvedProfile
from ..retrieval import (
    ChangeKind,
    CollectionMetadata,
    CollectionSpec,
    IndexedRow,
    PublishLease,
    RetrievalError,
    RetrievalFeature,
    ServingCoordinationError,
    ServingCoordinator,
    classify_descriptor_changes,
    collection_config_fingerprint,
    collection_descriptor,
    create_store,
    descriptor_json,
    legacy_collection_config_fingerprint,
    rebuild_required,
)
from ..retrieval.base import GENERATION_MARKER
from ..retrieval.publication_memory import log_publication_memory
from ..retrieval.retention import retire_superseded_generations
from ..retrieval.servability import (
    exact_advisory_row_threshold,
    exact_search_advisory,
    index_build_advisory,
    index_build_advisory_row_threshold,
)
from ..state_reconciliation import BoundedReconciler, UpstreamRecord
from ..timing import PhaseTimings
from ..versioning import compute_model_code_version
from .contracts import ModelRunResult, RunError

log = logging.getLogger(__name__)


def run_search_model(
    *,
    model: ModelConfig,
    models_by_name: Mapping[str, ModelConfig],
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    full_refresh: bool = False,
    subset_run: bool = False,
) -> ModelRunResult:
    """Publish the model, keeping its timings even if it fails.

    A publish that dies hours in is exactly the one whose attribution matters,
    and the runner builds a fresh result on `RunError` (PR #460 review).
    """
    timings = PhaseTimings()
    try:
        return _run_search_model(
            model=model,
            models_by_name=models_by_name,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
            resolved=resolved,
            full_refresh=full_refresh,
            subset_run=subset_run,
            timings=timings,
        )
    except RunError as error:
        if not error.metrics:
            error.metrics = timings.as_metrics()
        raise


def _run_search_model(
    *,
    model: ModelConfig,
    models_by_name: Mapping[str, ModelConfig],
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    full_refresh: bool = False,
    subset_run: bool = False,
    timings: PhaseTimings,
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
    state_swapped = False
    superseded_collection: str | None = None

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
            # The serving scope is keyed on the logical collection, so it
            # resolves without knowing which generation is live (issue #355).
            state_scope = StateScope.for_target_descriptor(
                model.name,
                stage="retrieval_publish",
                descriptor=store.state_descriptor(logical_collection).descriptor(),
            )
            # A null activation pointer means the unsuffixed default, which is
            # what every index published before generations existed uses.
            serving_entry = coordinator.status(state_scope)
            if serving_entry.publication_id is not None:
                raise RunError(
                    "Another publisher owns this serving scope; retry after it completes, "
                    "or recover only after terminating the old owners"
                )
            active_collection = serving_entry.active_collection
            # The generation serving queries before this publish, and the
            # configuration it was published under. A rebuild that fails hands
            # both back so the index keeps answering from it (issue #449); an
            # in-place publish must not, having written into what it names.
            # The fingerprint travels with the generation; a private claim
            # retains this pair until activation replaces both together.
            previous_generation = serving_entry.active_generation
            previous_fingerprint = serving_entry.config_fingerprint
            default_collection = store.physical_collection(logical_collection)
            rebuild = _rebuild_requested(
                model, full_refresh=full_refresh
            ) or _config_change_forces_rebuild(
                store,
                model=model,
                models_by_name=models_by_name,
                collection=active_collection or default_collection,
                upstream_schema=snapshot.schema,
                store_type=store_config.type,
                resolved=resolved,
            )
            if rebuild:
                _require_private_generation_build(store, model, subset_run=subset_run)
                generation_token = uuid4().hex[:12]
                physical = store.physical_collection(
                    logical_collection, generation=generation_token
                )
                log.info(
                    "%s: building a private search generation; activation follows "
                    "row and index validation",
                    model.name,
                )
            else:
                physical = active_collection or default_collection
            spec = _search_collection_spec(
                model=model,
                models_by_name=models_by_name,
                physical_collection=physical,
                upstream_schema=snapshot.schema,
                store_type=store_config.type,
                resolved=resolved,
            )
            # A rebuild accumulates publication state in its own scope, keyed
            # on the generation it is building. Writing into the serving scope
            # would corrupt the state of the generation still serving reads,
            # and the whole point is that the old one keeps serving until
            # activation. Activation then moves the scope atomically.
            publish_scope = (
                _generation_state_scope(model.name, physical)
                if rebuild
                else state_scope
            )
            reconciler = BoundedReconciler(
                adapter,
                publish_scope,
                code_version=code_version,
                page_size=search.batch_size,
            )

            publish_lease = coordinator.acquire_publish(
                state_scope,
                expected_code_version=code_version,
                config_fingerprint=spec.config_fingerprint,
                expected_fencing_token=serving_entry.fencing_token,
                # A rebuild writes to a private generation, so the live one
                # stays servable if this publisher dies; an in-place publish
                # mutates what is live, and the claim clears the pointer so a
                # crash recovers to `failed` rather than serving a half-
                # rewritten collection (issue #449).
                preserves_active_generation=rebuild,
            )
            with store.publisher_fence(physical), store:
                if rebuild:
                    # Reclaim generations left by publishers that died before
                    # activating. Under the publish claim, and before this
                    # build's collection exists, so the sweep cannot list it
                    # as a candidate. Sparing the live generation is what
                    # `active_collection` does here.
                    retire_superseded_generations(
                        store,
                        logical_collection=logical_collection,
                        active_collection=active_collection or default_collection,
                        coordinator=coordinator,
                        lease=publish_lease,
                    )
                warned_exact = False
                exact_scan_warn_after = (
                    exact_advisory_row_threshold(dimensions=spec.vector_dimensions)
                    if spec.vector_search == "exact"
                    and spec.vector_dimensions is not None
                    else None
                )
                # The other scale-dependent cost, and the one that killed the
                # #473 retest's predecessor at its last step: the ANN build's
                # peak against the container ceiling (issue #476). Read once;
                # None outside a container, where there is nothing to compare.
                build_limit = container_memory_limit_bytes()
                warned_build = False
                build_warn_after = (
                    index_build_advisory_row_threshold(
                        dimensions=spec.vector_dimensions,
                        vector_index=spec.vector_index,
                        limit_bytes=build_limit,
                    )
                    if spec.vector_search == "approximate"
                    and spec.vector_dimensions is not None
                    and spec.vector_index is not None
                    and build_limit is not None
                    else None
                )
                existing = store.inspect_collection(physical)
                force_publish = existing is None
                collection_exists = existing is not None
                # A private build inspects its own, still-empty target, so
                # `existing` is None on exactly the runs the advisories matter
                # most for: a configuration change over a large live
                # collection. #474 moved those off the in-place path and, with
                # them, the pre-run advisories' view of the row count -- the
                # exact-scan warning lost its minute-zero shot without anyone
                # noticing. The live collection is what the rows will be
                # republished from, so its count is the one to warn against
                # before any of them are read (issue #476).
                live = (
                    store.inspect_collection(active_collection or default_collection)
                    if rebuild and existing is None
                    else None
                )
                known_rows = (
                    existing.row_count
                    if existing is not None
                    else live.row_count
                    if live is not None
                    else None
                )
                if known_rows is not None:
                    # Before the run spends its time, not after: on every run
                    # but the first, a collection already knows how many rows
                    # an exact query has to scan (issue #461) and an index
                    # build has to hold (issue #476).
                    warned_exact = _warn_on_exact_vector_scan(model, spec, known_rows)
                    warned_build = _warn_on_index_build_memory(
                        model, spec, known_rows, limit_bytes=build_limit
                    )
                if existing is not None:
                    _verify_collection_config(
                        store, existing, spec, policy=search.on_index_change
                    )
                    _validate_collection_schema(existing.schema, spec)

                batches = iter(snapshot)
                ordinal = -1
                while True:
                    # Only the pull is credited to `read`: the row shaping
                    # below is stel's own CPU. Publishing this corpus was
                    # dominated by per-page round trips (#452), so which of
                    # these three terms is large is the whole question.
                    with timings.phase("read"):
                        batch = next(batches, None)
                    if batch is None:
                        # Only the adapter can attribute inside a read, so its
                        # transfer/decode/copy split is folded in once the
                        # stream is drained (issue #454).
                        timings.merge(snapshot.timings)
                        break
                    ordinal += 1
                    indexed = _indexed_rows(
                        batch,
                        model,
                        spec.config_fingerprint,
                        max_id_bytes=store.capabilities().max_id_bytes,
                    )
                    rows_seen += len(indexed)
                    if (
                        not warned_exact
                        and exact_scan_warn_after is not None
                        and rows_seen >= exact_scan_warn_after
                    ):
                        # A first publish has no prior row count, so without
                        # this the only advisory arrives after every batch has
                        # been read, upserted and indexed — and not at all if
                        # the index build fails (Codex review, #461).
                        warned_exact = _warn_on_exact_vector_scan(
                            model, spec, rows_seen, in_progress=True
                        )
                    if (
                        not warned_build
                        and build_warn_after is not None
                        and rows_seen >= build_warn_after
                    ):
                        warned_build = _warn_on_index_build_memory(
                            model, spec, rows_seen, limit_bytes=build_limit, in_progress=True
                        )
                    # A line rather than a bar: the snapshot is a one-shot
                    # bounded stream with no row count, so there is no total to
                    # render a determinate bar against.
                    log.info(
                        "%s: indexing batch %d (%d row(s) so far)",
                        model.name,
                        ordinal + 1,
                        rows_seen,
                    )
                    log_publication_memory(log, model.name, phase="batch", batch=ordinal + 1)
                    upstream_records = [
                        UpstreamRecord(row.record_id, row.input_fingerprint)
                        for row in indexed
                    ]
                    with timings.phase("state"):
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
                            with timings.phase("store_write"):
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
                    with timings.phase("store_write"):
                        write = store.append if rebuild else store.upsert
                        receipt = write(
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
                    with timings.phase("state"):
                        coordinator.verify_publish(publish_lease)
                        adapter.upsert_state(
                            publish_scope,
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
                # A subset invocation upstream means an id absent from this
                # run's view is not stale; reconciliation belongs to the next
                # unfiltered run, exactly as rebuild already skips it
                # (issue #417).
                stale_pages = (
                    reconciler.iter_stale_pages(
                        upstream_table=upstream,
                        key_column=search.id_field,
                    )
                    if not rebuild and not subset_run
                    else _no_stale_pages()
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
                        with timings.phase("store_write"):
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
                        with timings.phase("state"):
                            adapter.delete_state(publish_scope, record_ids)
                        deleted += len(record_ids)
                finally:
                    stale_pages.close()

                if not collection_exists:
                    coordinator.verify_publish(publish_lease)
                    store.create_collection(spec)
                log_publication_memory(log, model.name, phase="index_build", batch=ordinal + 1)
                metadata = store.ensure_indexes(spec)
                if not warned_exact:
                    _warn_on_exact_vector_scan(model, spec, metadata.row_count)
                if metadata.config_fingerprint != spec.config_fingerprint:
                    raise RunError(
                        "Retrieval collection failed post-publication configuration validation"
                    )
                # A subset run retains records its narrowed view did not
                # cover, so the collection legitimately holds MORE rows than
                # this run saw; fewer still means the store lost rows
                # (issue #417).
                if (
                    metadata.row_count < rows_seen
                    if subset_run
                    else metadata.row_count != rows_seen
                ):
                    raise RunError(
                        "Retrieval collection failed post-publication row-count validation"
                    )
                active_generation = metadata.physical_generation

        assert state_scope is not None
        coordinator.verify_publish(publish_lease)
        assert active_generation is not None
        assert spec is not None
        if rebuild:
            # The state snapshot is replaced first, fenced on this claim, and
            # only then does the ledger pointer move. The order is forced:
            # `mark_ready` releases the publication claim, so a fenced
            # replacement after it would be refused.
            #
            # That leaves one window. If the swap lands and activation does
            # not, the serving scope describes the new generation while the
            # pointer still names the old collection, and a later incremental
            # publish would skip rows the old collection never received.
            # `state_swapped` lets the failure path clear that state, so the
            # next run reconciles against nothing and republishes in full.
            with timings.phase("state"):
                _activate_generation(
                    adapter,
                    serving_scope=state_scope,
                    publish_scope=publish_scope,
                    lease=publish_lease,
                    page_size=search.batch_size,
                )
            state_swapped = True
        coordinator.mark_ready(
            publish_lease,
            active_generation=active_generation,
            config_fingerprint=spec.config_fingerprint,
            counts=(inserted, updated, skipped, deleted),
            active_collection=physical if rebuild else active_collection,
        )
        # Activation succeeded: the swapped state now describes the live
        # generation. Clearing it from here on would leave the ledger ready
        # with empty state and re-embed the whole index on the next run.
        state_swapped = False
        superseded_collection = (
            active_collection
            if rebuild and active_collection and GENERATION_MARKER in active_collection
            else None
        )
        if (
            rebuild and superseded_collection is not None
            and coordinator.status(state_scope).query_leases == 0
        ):
            # The collection this activation replaced. Dropped by name rather
            # than by sweeping: `mark_ready` has released the publish claim,
            # and a listing sweep without it can delete a generation another
            # publisher is building. Reader pins must also have drained; a
            # late admission rechecks the active pointer before store I/O.
            with store:
                store.drop_collection(superseded_collection)
    except (AdapterError, RetrievalError, RunError, MemoryError) as error:
        if state_swapped and state_scope is not None:
            # See the activation note above: this state no longer describes
            # the collection the pointer names.
            with suppress(AdapterError):
                adapter.clear_state(state_scope)
        if publish_lease is not None:
            # A claim exists, so the pre-publish read of the serving entry ran
            # and both of these are bound.
            retain_previous = bool(
                rebuild and previous_generation and previous_fingerprint
            )
            _mark_search_publication_failed(
                coordinator,
                publish_lease,
                error,
                counts=(inserted, updated, skipped, deleted),
                # A rebuild builds where nothing is reading, so a failure
                # leaves the previous generation intact and still correct --
                # both pointers survive, and the scope stays servable from
                # that generation (issues #355, #449). An in-place publish may
                # have corrupted what it wrote into, so neither pointer may be
                # trusted and the scope goes unavailable.
                # Retained only when the previous generation's own
                # configuration is known: a generation advertised under the
                # configuration this run was building for would be handed to
                # readers as answering for a configuration it never had.
                active_collection=active_collection if retain_previous else None,
                active_generation=previous_generation if retain_previous else None,
                config_fingerprint=previous_fingerprint if retain_previous else None,
            )
        if isinstance(error, RunError):
            raise
        if isinstance(error, MemoryError):
            raise RunError(
                "Search publication exhausted process memory; inspect the last "
                "publication memory sample and reduce batch_size or increase memory"
            ) from None
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
        # Where the wall clock went, so a slow publish can be attributed
        # rather than guessed at (#432, #454).
        metrics=timings.as_metrics(),
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


def _no_stale_pages() -> Generator[Sequence[Any], None, None]:
    """A rebuild has no stale rows: nothing was ever published into it.

    A generator rather than an empty iterator so it carries the `.close()` the
    caller's `finally` block calls on the real page stream.
    """
    return
    yield  # pragma: no cover - unreachable, makes this a generator


def _activate_generation(
    adapter: WarehouseAdapter,
    *,
    serving_scope: StateScope,
    publish_scope: StateScope,
    lease: PublishLease,
    page_size: int,
) -> None:
    """Move a generation's publication state into the serving scope.

    Fenced on this publication's claim, so a publisher that lost authority
    mid-build cannot overwrite the state of whatever replaced it. Streams the
    generation's state in pages rather than materializing it: a full index can
    be millions of rows, and bounded residency is the rule the whole
    publication path is written to.
    """
    fence = StateScopeFence(
        publication_id=lease.publication_id, fencing_token=lease.fencing_token
    )
    with adapter.state_page_reader(publish_scope, page_size=page_size) as reader:
        adapter.replace_state_scope(
            serving_scope, _state_batches(reader), fence=fence
        )
    # The generation scope has served its purpose; leaving it would accumulate
    # one dead scope per rebuild.
    adapter.clear_state(publish_scope)


def _state_batches(reader: Any) -> Iterator[Sequence[StateRecord]]:
    cursor = None
    while True:
        page = reader.fetch_page(cursor)
        if not page.records:
            return
        yield [
            StateRecord(record.record_key, record.input_fingerprint, record.code_version)
            for record in page.records
        ]
        if page.next_cursor is None:
            return
        cursor = page.next_cursor


def _rebuild_requested(model: ModelConfig, *, full_refresh: bool) -> bool:
    """Whether the operator asked for a full replacement.

    `materialization: full` states it in the project; `--full-refresh` states
    it for one run. Neither is inferred — an unannounced full re-embed is the
    behavior issue #344 rejected.
    """
    return full_refresh or model.materialization == "full"


def _require_private_generation_build(
    store: Any, model: ModelConfig, *, subset_run: bool
) -> None:
    if subset_run:
        raise RunError(
            "Search index replacement requires an unfiltered run; "
            "a subset cannot replace the complete serving generation"
        )
    if (
        RetrievalFeature.PRIVATE_GENERATION_BUILD
        not in store.capabilities().features
    ):
        raise RunError(
            f"Search index '{model.name}' needs a full replacement, but "
            f"retrieval store '{store.store_type()}' cannot build a private "
            "generation, so the running index cannot be replaced atomically"
        )


def _generation_state_scope(model_name: str, physical_collection: str) -> StateScope:
    """The publication scope a rebuild accumulates state in.

    Keyed on the physical generation, which is what makes it an independent
    publication: it is invisible to readers and to the serving ledger until
    activation moves it into the serving scope.
    """
    return StateScope.for_target_descriptor(
        model_name,
        stage="retrieval_generation_publish",
        descriptor={"physical_collection": physical_collection},
    )


def _config_change_forces_rebuild(
    store: Any,
    *,
    model: ModelConfig,
    models_by_name: Mapping[str, ModelConfig],
    collection: str,
    upstream_schema: pa.Schema,
    store_type: str,
    resolved: ResolvedProfile,
) -> bool:
    """Plan configuration changes before claiming or mutating a live target."""
    search = model.search
    assert search is not None
    # Validate the warehouse projection before opening the store. Even an
    # inspect context can create local storage or resolve cloud credentials.
    spec = _search_collection_spec(
        model=model,
        models_by_name=models_by_name,
        physical_collection=collection,
        upstream_schema=upstream_schema,
        store_type=store_type,
        resolved=resolved,
    )
    with store:
        existing = store.inspect_collection(collection)
    if existing is None:
        return False
    if existing.descriptor is None:
        if existing.config_fingerprint != spec.legacy_config_fingerprint:
            raise RunError(
                "Search index configuration changed without a stored descriptor; "
                "publish a private replacement with --full-refresh"
            )
        return False
    if existing.descriptor == spec.descriptor:
        return False
    changes = classify_descriptor_changes(
        json.loads(existing.descriptor), json.loads(spec.descriptor)
    )
    blocking = rebuild_required(changes)
    if search.on_index_change == "fail":
        _verify_collection_config(store, existing, spec)
    if blocking and search.on_index_change == "online":
        detail = "; ".join(change.describe() for change in blocking)
        raise RunError(
            f"Search index configuration changed and requires a rebuild: {detail}. "
            "Use on_index_change: rebuild for a private replacement."
        )
    # Both policies build away from readers. Online still accepts only
    # compatible changes; rebuild is an explicit opt-in for any change.
    return bool(changes)


def _warn_on_exact_vector_scan(
    model: ModelConfig,
    spec: CollectionSpec,
    row_count: int,
    *,
    in_progress: bool = False,
) -> bool:
    """Report an `exact` collection that has grown past what a scan can serve.

    Returns whether anything was said, so the same run does not say it twice.
    A warning rather than a refusal: `exact` stays a legitimate choice, and the
    threshold is an estimate anchored to one measurement, not a contract. What
    was missing was any signal at all — the collection in issue #461 published,
    validated, and reported ready while being structurally unqueryable.
    """
    search = model.search
    assert search is not None
    if spec.vector_search != "exact" or spec.vector_dimensions is None:
        return False
    advisory = exact_search_advisory(
        collection=spec.logical_name,
        rows=row_count,
        dimensions=spec.vector_dimensions,
        access=search.access,
        in_progress=in_progress,
    )
    if advisory is None:
        return False
    log.warning("Search resource '%s': %s", model.name, advisory)
    return True


def _warn_on_index_build_memory(
    model: ModelConfig,
    spec: CollectionSpec,
    row_count: int,
    *,
    limit_bytes: int | None,
    in_progress: bool = False,
) -> bool:
    """Report an ANN build that will not fit the container it runs in.

    Returns whether anything was said, so the same run does not say it twice.
    A warning, not a refusal, for the same reasons as the exact-scan advisory:
    the ratios are measurements, not a contract, and the operator may know
    better. The #473 predecessor to this appended for hours and died at the
    build with nothing having mentioned the cost (issue #476).
    """
    if (
        spec.vector_search != "approximate"
        or spec.vector_dimensions is None
        or spec.vector_index is None
    ):
        return False
    advisory = index_build_advisory(
        collection=spec.logical_name,
        rows=row_count,
        dimensions=spec.vector_dimensions,
        vector_index=spec.vector_index,
        limit_bytes=limit_bytes,
        in_progress=in_progress,
    )
    if advisory is None:
        return False
    log.warning("Search resource '%s': %s", model.name, advisory)
    return True


def _mark_search_publication_failed(
    coordinator: ServingCoordinator,
    lease: PublishLease,
    error: Exception,
    *,
    counts: tuple[int, int, int, int],
    active_collection: str | None = None,
    active_generation: str | None = None,
    config_fingerprint: str | None = None,
) -> None:
    """Record the failure under a safe code; a stale fence has nothing to record."""
    if isinstance(error, MemoryError):
        code = "memory_exhausted"
    elif isinstance(error, ServingCoordinationError):
        code = "coordination_error"
    elif isinstance(error, RetrievalError):
        code = "store_error"
    elif isinstance(error, AdapterError):
        code = "warehouse_error"
    else:
        code = "publication_failed"
    with suppress(ServingCoordinationError):
        coordinator.mark_failed(
            lease,
            safe_error_code=code,
            counts=counts,
            active_collection=active_collection,
            active_generation=active_generation,
            config_fingerprint=config_fingerprint,
        )


def _scalar_index_fields(search: SearchConfig) -> tuple[str, ...]:
    """Scalar columns worth a BTree, the merge key first (issue #475).

    `id_field` is the join key `upsert` hands to `merge_insert` on every
    incremental publish, and the column its acknowledging `count_rows` filters
    on. Unindexed, LanceDB evaluates that predicate with a full column scan and
    the ack scans the same column again — two O(table) passes per page, on a
    *daily* incremental rather than only on a reindex. It is indexed here
    rather than at the store because the store is handed a spec, and which
    columns deserve an index is a property of the search model.

    Ordered with the merge key first and de-duplicated: an id field that is
    also a filterable attribute must not ask for the same index twice.
    """
    fields = [search.id_field]
    fields.extend(
        attribute.name
        for attribute in search.attributes
        if attribute.filter_role != "none" or attribute.sortable
    )
    return tuple(dict.fromkeys(fields))


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
    search_payload = effective_search_config(
        model,
        models_by_name,
        profile_options=embedding_runtime.provider_options
        if embedding_runtime is not None
        else None,
    )
    config_fingerprint = collection_config_fingerprint(
        search_payload, store_type=store_type
    )
    descriptor = descriptor_json(
        collection_descriptor(search_payload, store_type=store_type)
    )
    legacy_fingerprint = legacy_collection_config_fingerprint(
        search_payload, store_type=store_type
    )
    return CollectionSpec(
        logical_name=search.collection or model.name,
        physical_name=physical_collection,
        id_field=search.id_field,
        text_fields=search.text_fields,
        full_text_fields=search.full_text.fields if search.full_text else (),
        attribute_fields=tuple(attribute.name for attribute in search.attributes),
        scalar_index_fields=_scalar_index_fields(search),
        display_fields=search.display_fields,
        vector_field=search.vector.field if search.vector else None,
        vector_dimensions=search.vector.dimensions if search.vector else None,
        distance_metric=search.vector.metric if search.vector else None,
        vector_search=search.vector.search if search.vector else None,
        vector_index=search.vector.index if search.vector else None,
        config_fingerprint=config_fingerprint,
        descriptor=descriptor,
        legacy_config_fingerprint=legacy_fingerprint,
        arrow_schema=schema,
    )


def _schema_matches_by_name(existing: pa.Schema, declared: pa.Schema) -> bool:
    """Compare two schemas by column name and type, ignoring order.

    This is the contract stel actually relies on: every read, write, and
    predicate addresses columns by name, so ordering carries no meaning, and a
    store column's nullability is not a guarantee stel makes — rows are
    validated on the way in. Types are still compared exactly.

    An ordered comparison was over-strict, and once collections could be
    widened it was wrong. `add_columns` can only append, while the declared
    schema orders columns by projection, so a newly added attribute lands ahead
    of the display fields in one and behind them in the other; added columns
    are also nullable until the republish fills them. Comparing strictly
    rejected a collection that had just been widened correctly, on every run
    after the widening (Codex review, #344).
    """
    return {field.name: field.type for field in existing} == {
        field.name: field.type for field in declared
    }


def _validate_collection_schema(existing: pa.Schema, spec: CollectionSpec) -> None:
    """Refuse a collection whose physical shape is not the declared one.

    One comparison for every path. A collection that was widened in place is
    reached again on the next run, when nothing has changed and no evolution
    happens, so a stricter check here than the one applied during the widening
    would reject the collection it had just accepted.
    """
    if not _schema_matches_by_name(existing, spec.arrow_schema):
        raise RunError(
            "Search index schema does not match the declared collection contract"
        )


def _verify_collection_config(
    store: Any,
    existing: CollectionMetadata,
    spec: CollectionSpec,
    *,
    policy: str = "fail",
) -> bool:
    """Refuse a publish whose configuration no longer describes the collection.

    Only legacy, semantically unchanged stamps can be rewritten here. Changed
    configurations must have been routed to a private target by the planner;
    never evolve a live collection as a fallback (issue #473).
    """
    if existing.descriptor is None:
        # Published before the descriptor existed: it carries only the legacy
        # digest. Recomputing that digest is the one way to prove the
        # configuration is unchanged, in which case the stamp is merely stale
        # in format and is rewritten in place — no rebuild, no re-embed.
        if existing.config_fingerprint == spec.legacy_config_fingerprint:
            store.restamp_collection(spec)
            return False
        raise RunError(
            "Search index configuration changed. This collection predates "
            "configuration classification, so the changed fields cannot be "
            "named; publish under a new collection name, validate it, and cut "
            "consumers over."
        )

    if existing.descriptor == spec.descriptor:
        return False

    changes = classify_descriptor_changes(
        json.loads(existing.descriptor), json.loads(spec.descriptor)
    )
    blocking = [c for c in changes if c.kind is ChangeKind.REBUILD_REQUIRED]
    detail = "; ".join(change.describe() for change in changes) or "store contract"
    if blocking:
        raise RunError(
            f"Search index configuration changed and requires a rebuild: {detail}. "
            "Rows already written were indexed under the previous definition, so "
            "publish under a new collection name, validate it, and cut consumers "
            "over."
        )
    if policy == "online":
        raise RunError(
            "Online configuration changes require a private generation; "
            "refusing to mutate the live collection"
        )

    raise RunError(
        f"Search index configuration changed: {detail}. The existing collection "
        "can serve the change — set `on_index_change: online` to build a private "
        "replacement, or publish under a new collection name."
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
