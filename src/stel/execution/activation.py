"""Activate a physically complete search generation without re-paging the corpus.

The publish path's own re-entry (ADR-0005) adopts an orphaned generation and
finishes it on the next run, which is right whenever that run can finish. It
could not for the corpus in issue #615: every leg re-read all 146 pages from
the warehouse and died at BigQuery's six-hour read session (#614), while a
generation holding every row sat in the store, an index refresh away from
serving. Nothing could turn it into the served one.

This module is that operator command. It activates a generation the operator
names, from the publication state stel already recorded for its rows, and it
re-stamps that state at the current `code_version` -- the operator's assertion,
made with `--rows-verified`, is that the rows are what the current code would
produce. A hash-only `code_version` change (#607) is exactly that case.

What it refuses to do is activate something it cannot check: the collection
must exist, carry this configuration's fingerprint, and hold no more rows than
the upstream relation; the state assembled for it must describe no more rows
than the collection holds; and a sample of those state keys must be present in
the collection. A generation that fails any of these is left as it was, and the
serving scope keeps whatever it was serving. The checks are the same ones a
publish applies before it activates, minus the one that needs the corpus read
-- that the rows *are* the upstream's -- which is the one the flag stands in
for, and which the next incremental run reconciles anyway.

A shortfall in either count is tolerated, with one piece of bookkeeping. Rows
the upstream has that the collection lacks are simply published by the next
run. Rows the collection holds that its state does not describe are a hole
the next run cannot see on its own: stale discovery enumerates *state* keys
absent upstream, so a row with no state whose upstream key is later deleted
would be served for good. Activation therefore walks the collection's ids and
records every such row under a marker fingerprint that no upstream row can
match; the next run then re-upserts the row if its key still exists upstream,
and deletes it as stale if not. The walk runs on every activation, because it
is also the exhaustive form of the membership check: a state key the
collection does not hold and a collection row the state does not describe
cancel in a row count, and only a walk that visits every id tells them apart.

Shares the publish path's spec, state-copy and swap helpers from `.search` on
purpose: an activation that built its own would be a second definition of
what a generation's state and fingerprint mean.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..adapters import (
    AdapterError,
    StateRecord,
    StateScope,
    StateScopeAbsenceProbe,
    WarehouseAdapter,
)
from ..config.model import ModelConfig
from ..config.project import ProjectConfig
from ..dag import parse_ref
from ..profile import ResolvedProfile
from ..retrieval import (
    CollectionMetadata,
    CollectionSpec,
    PublishLease,
    RetrievalError,
    RetrievalStore,
    ServingCoordinator,
    StoreRole,
    create_store,
)
from ..versioning import compute_model_code_version
from .contracts import RunError
from .heartbeat import Heartbeat
from .search import (
    _activate_generation,
    _generation_state_scope,
    _mark_search_publication_failed,
    _resolve_row_fingerprint,
    _search_collection_spec,
    _state_batches,
)

log = logging.getLogger(__name__)

# How many state keys are checked against the collection before its state is
# trusted. Drawn across the whole scope, not from its head: the failure this
# guards against is state describing a *different* collection, which a sample
# from anywhere detects and a sample from one end might not.
ACTIVATION_SAMPLE_SIZE = 1000

# The input fingerprint recorded for a collection row the assembled state did
# not describe. Real fingerprints are hex digests, so this can never equal
# one: the next incremental run classifies the row as changed and re-upserts
# it, idempotently, if the key still exists upstream, and finds it among the
# stale state keys and deletes it if not. Either way the marker is gone after
# one run; nothing reads it back.
UNVERIFIED_INPUT_FINGERPRINT = "unverified-by-activation"


@dataclass(frozen=True)
class GenerationActivation:
    """What `activate_search_generation` did, for the command to report."""

    model_name: str
    physical_collection: str
    active_generation: str
    rows: int
    # Rows whose state came from the generation's own publication scope,
    # and rows filled in from the serving scope's state for the collection it
    # had been serving. The split is what tells an operator how far the
    # interrupted publish had got.
    state_rows_from_generation: int
    state_rows_from_serving: int
    # What the next incremental run reconciles. `rows_behind_upstream` is the
    # upstream row count minus the collection's: a *net* figure, since a
    # deletion and an insertion upstream cancel in it, so it bounds the
    # staleness from below rather than counting the rows the run will write.
    # `rows_without_state` is the number of collection rows the assembled
    # state did not describe, each now recorded under
    # `UNVERIFIED_INPUT_FINGERPRINT` so the next run re-checks or deletes it.
    # Both are staleness, not damage (see `activation_refusal`).
    rows_behind_upstream: int
    rows_without_state: int
    code_version: str
    fencing_token: int


def activation_refusal(
    existing: CollectionMetadata | None,
    spec: CollectionSpec,
    *,
    upstream_rows: int,
    collection: str,
) -> str | None:
    """Why this collection may not be activated for this configuration, or None.

    Pure, so the refusals can be pinned without a store. Each names the two
    values that disagree: the operator's next step is different for each.

    Fewer rows than the upstream is not a refusal. The rows missing are ones
    the upstream gained after the generation's last complete write, and an
    index that lacks this week's filings is what every index is between
    incremental runs; the next one publishes them. Refusing would send the
    operator to a resume that re-reads the corpus to add a few hundred rows --
    the cost this command exists to avoid (#614) -- and a corpus that is
    always growing would otherwise refuse every activation attempted more than
    a few hours after the generation was written. More rows than the upstream
    is still refused: those rows are not in the relation at all, so either the
    upstream shrank (a resume deletes them in minutes) or this collection is
    not this relation's, and the row count cannot tell which.
    """
    if existing is None:
        return f"Retrieval collection '{collection}' does not exist in this store"
    if existing.config_fingerprint != spec.config_fingerprint:
        return (
            f"Retrieval collection '{collection}' was built for a different "
            "configuration than this model declares; activating it would answer "
            "queries with an index that was never built for them. Republish instead."
        )
    if existing.row_count > upstream_rows:
        return (
            f"Retrieval collection '{collection}' holds {existing.row_count} row(s) "
            f"but the upstream relation has only {upstream_rows}; a generation "
            "holding rows the upstream does not cannot be told apart from another "
            "relation's collection by its count. Resume the publish instead."
        )
    return None


def activate_search_generation(
    *,
    model: ModelConfig,
    models_by_name: Mapping[str, ModelConfig],
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
    resolved: ResolvedProfile,
    physical_collection: str,
    rows_verified: bool,
) -> GenerationActivation:
    """Make `physical_collection` the served generation of this search model.

    `rows_verified` is the operator's assertion that the collection's rows are
    what the current code would publish, so re-stamping their state at the
    current `code_version` is correct. Required rather than defaulted: without
    it this refuses, because nothing here can prove that claim.
    """
    search = model.search
    assert search is not None
    if not rows_verified:
        raise RunError(
            "Activating a generation re-stamps its rows as current under this "
            "code version, which nothing here can verify; confirm with "
            "--rows-verified that the collection's rows are the current corpus"
        )
    if resolved.retrieval is None:
        raise RunError("Search publication requires a configured retrieval target")
    alias = search.store or resolved.retrieval.default
    store_config = resolved.retrieval.stores.get(alias)
    if store_config is None:
        raise RunError("Search publication selected an unknown retrieval target")
    upstream = parse_ref((model.depends_on or [""])[0])
    logical_collection = search.collection or model.name
    code_version = compute_model_code_version(model, project, project_dir, resolved=resolved)
    store = create_store(
        store_config,
        project_name=project.name,
        target_name=resolved.target_name,
        alias=alias,
        # The index build is the one heavy step, and it wants the publish
        # ceiling rather than a serving cache.
        role=StoreRole.PUBLISH,
    )
    coordinator = ServingCoordinator(adapter, ensure_schema=True)

    # The schema is all that is read from the upstream. No `key_column`, so
    # on BigQuery this opens a storage session straight on the table instead
    # of running the uniqueness query job the publish needs; nothing pulls a
    # batch, and the session is dropped with the block. Re-reading the corpus
    # is the cost this command exists to avoid (#614).
    with adapter.table_snapshot(
        upstream, columns=search.projected_fields(), batch_size=search.batch_size
    ) as snapshot:
        upstream_schema = snapshot.schema
    upstream_rows = adapter.row_count(upstream)

    state_scope = StateScope.for_target_descriptor(
        model.name,
        stage="retrieval_publish",
        descriptor=store.state_descriptor(logical_collection).descriptor(),
    )
    entry = coordinator.status(state_scope)
    if entry.publication_id is not None:
        raise RunError(
            "Another publisher owns this serving scope; retry after it completes, "
            "or recover only after terminating the old owners"
        )
    spec = _search_collection_spec(
        model=model,
        models_by_name=models_by_name,
        physical_collection=physical_collection,
        upstream_schema=upstream_schema,
        store_type=store_config.type,
        resolved=resolved,
    )
    with store:
        existing = store.inspect_collection(physical_collection)
    refusal = activation_refusal(
        existing, spec, upstream_rows=upstream_rows, collection=physical_collection
    )
    if refusal is not None:
        raise RunError(refusal)
    assert existing is not None
    spec = _resolve_row_fingerprint(spec, existing)

    # The collection the serving scope's state describes, if the ledger still
    # knows. A lost pointer (the #615 row) leaves it unknown, and that state is
    # then offered to the generation and checked against it below.
    default_collection = store.physical_collection(logical_collection)
    served = (
        (entry.active_collection or default_collection)
        if entry.active_generation is not None
        else None
    )
    serving_state_applies = served is None or served == physical_collection
    publish_scope = _generation_state_scope(model.name, physical_collection)
    lease = coordinator.acquire_publish(
        state_scope,
        expected_code_version=code_version,
        config_fingerprint=spec.config_fingerprint,
        expected_fencing_token=entry.fencing_token,
        # Nothing here writes a row. The index build on a collection readers
        # resolve to is still an in-place mutation, so readers are excluded
        # for it; activating any other collection leaves them on theirs.
        preserves_active_generation=True,
        excludes_readers=served == physical_collection,
    )
    try:
        with store.publisher_fence(physical_collection), store:
            from_serving = 0
            if serving_state_applies:
                from_serving = _fill_state_from_serving(
                    adapter,
                    coordinator,
                    lease,
                    serving_scope=state_scope,
                    publish_scope=publish_scope,
                    page_size=search.batch_size,
                    code_version=code_version,
                )
            state_rows, sample = _restamp_and_sample(
                adapter,
                coordinator,
                lease,
                scope=publish_scope,
                page_size=search.batch_size,
                code_version=code_version,
                model_name=model.name,
                # The collection's row count, not the scope's: it is already
                # known, and it is the number the state is about to be checked
                # against anyway. The scope's own count would cost an extra
                # aggregate over the slice this phase exists to stop
                # re-scanning.
                total=existing.row_count,
            )
            # State naming more rows than the collection holds vouches for
            # rows that are not there, and the reconciler would skip them for
            # good: refused outright, since no sample is needed to know it.
            # Fewer is the other direction -- a page whose slices committed
            # before its state could advance (the 2026-09-20 failure died in
            # one) -- and those rows are recorded below under a marker the
            # next run cannot mistake for current.
            if state_rows > existing.row_count:
                raise RunError(
                    f"Publication state describes {state_rows} row(s) of "
                    f"'{physical_collection}' but the collection holds only "
                    f"{existing.row_count}; state that vouches for rows the "
                    "collection does not hold cannot be activated. Resume the "
                    "publish instead."
                )
            rows_without_state = existing.row_count - state_rows
            # The sample is the cheap refusal for gross mismatch (state that
            # describes another collection); the id walk below is the
            # exhaustive one.
            present = store.count_present(
                physical_collection, sample, id_field=search.id_field
            )
            if present != len(sample):
                raise RunError(
                    f"Publication state names {len(sample) - present} of "
                    f"{len(sample)} sampled row(s) that '{physical_collection}' "
                    "does not hold; that state describes another collection. "
                    "Resume the publish instead."
                )
            # Always, not only when the counts differ: a ghost state key and
            # a row without state cancel in the arithmetic above, and an
            # activation interrupted mid-walk can leave the counts equal with
            # rows still unmarked. The walk is the exhaustive form of the
            # sample check -- every collection id ends up either known or
            # marked, so a count that disagrees with `rows_without_state` is
            # exactly the number of state keys the collection does not hold.
            marked = _mark_rows_without_state(
                store,
                adapter,
                coordinator,
                lease,
                collection=physical_collection,
                id_field=search.id_field,
                scope=publish_scope,
                page_size=search.batch_size,
                code_version=code_version,
            )
            if marked != rows_without_state:
                raise RunError(
                    f"Publication state names {marked - rows_without_state} "
                    f"row(s) that '{physical_collection}' does not hold; that "
                    "state describes another collection. Resume the publish "
                    "instead."
                )
            coordinator.verify_publish(lease)
            metadata = store.ensure_indexes(spec)
            if metadata.config_fingerprint != spec.config_fingerprint:
                raise RunError(
                    "Retrieval collection failed post-publication configuration validation"
                )
            if metadata.row_count != existing.row_count:
                raise RunError(
                    "Retrieval collection failed post-publication row-count validation"
                )
            active_generation = metadata.physical_generation
        _activate_generation(
            adapter,
            serving_scope=state_scope,
            publish_scope=publish_scope,
            lease=lease,
            page_size=search.batch_size,
        )
        coordinator.mark_ready(
            lease,
            active_generation=active_generation,
            config_fingerprint=spec.config_fingerprint,
            counts=(0, 0, existing.row_count, 0),
            active_collection=physical_collection,
        )
    except (AdapterError, RetrievalError, RunError) as error:
        # No row was written anywhere, so whatever was serving still can: the
        # previous pointers go back whenever they were there to begin with.
        retain = bool(entry.active_generation and entry.config_fingerprint)
        _mark_search_publication_failed(
            coordinator,
            lease,
            error,
            counts=(0, 0, 0, 0),
            active_collection=entry.active_collection if retain else None,
            active_generation=entry.active_generation if retain else None,
            config_fingerprint=entry.config_fingerprint if retain else None,
        )
        if isinstance(error, RunError):
            raise
        raise RunError(str(error)) from None
    rows_behind_upstream = upstream_rows - existing.row_count
    log.info(
        "%s: activated %s (%d rows; state from the generation for %d, from the "
        "serving scope for %d; upstream count exceeds the collection's by %d, "
        "%d held row(s) marked unverified) at code version %s",
        model.name,
        physical_collection,
        existing.row_count,
        state_rows - from_serving,
        from_serving,
        rows_behind_upstream,
        rows_without_state,
        code_version,
    )
    return GenerationActivation(
        model_name=model.name,
        physical_collection=physical_collection,
        active_generation=active_generation,
        rows=existing.row_count,
        state_rows_from_generation=state_rows - from_serving,
        state_rows_from_serving=from_serving,
        rows_behind_upstream=rows_behind_upstream,
        rows_without_state=rows_without_state,
        code_version=code_version,
        fencing_token=lease.fencing_token,
    )


def _fill_state_from_serving(
    adapter: WarehouseAdapter,
    coordinator: ServingCoordinator,
    lease: PublishLease,
    *,
    serving_scope: StateScope,
    publish_scope: StateScope,
    page_size: int,
    code_version: str,
) -> int:
    """Add the serving scope's records to the generation's, where it has none.

    The generation's own receipts win: a row the resumed build rewrote carries
    the fingerprint of what it wrote, and the serving scope's older record for
    the same key would make the next run republish a row that is already
    current. Only keys the generation never recorded are taken from the serving
    scope, re-stamped to this code version. Returns how many were taken.
    """
    taken = 0
    # The warehouse evaluates the absence, so every record the walk yields is
    # already one the generation lacks (issue #635). Before this, the walk
    # yielded the whole serving scope and a per-batch `record_key IN UNNEST`
    # asked the generation scope which of them it had -- a lookup that
    # re-scanned that entire slice every batch, ~514 MB of the ~3.4 GB a batch
    # cost. `record_key IN UNNEST(@array)` does not prune on the clustering
    # #431 added, so the cost was the slice and not the keys.
    #
    # Resolving it locally instead -- holding one scope's keys and comparing
    # in Python -- is what #428 moved out of Python and what
    # `docs/architecture/bounded-memory.md` prices at 370-740 MB for 3.6M
    # rows. The anti-join is the bounded form of the same question.
    with adapter.state_page_reader(
        serving_scope,
        page_size=page_size,
        absent_from=StateScopeAbsenceProbe(publish_scope),
    ) as reader:
        for batch in _state_batches(reader):
            coordinator.verify_publish(lease)
            adapter.upsert_state(
                publish_scope,
                [
                    StateRecord(record.record_key, record.input_fingerprint, code_version)
                    for record in batch
                ],
            )
            taken += len(batch)
    return taken


def _mark_rows_without_state(
    store: RetrievalStore,
    adapter: WarehouseAdapter,
    coordinator: ServingCoordinator,
    lease: PublishLease,
    *,
    collection: str,
    id_field: str,
    scope: StateScope,
    page_size: int,
    code_version: str,
) -> int:
    """Record every collection row `scope` does not describe under the marker
    fingerprint; return how many were recorded.

    Walks the collection's ids, not the state: the rows being looked for are
    exactly the ones the state cannot name. Idempotent -- a second pass finds
    its own markers already recorded and writes nothing -- so an activation
    interrupted here is re-entered by running it again, and the caller's
    count check still holds on the retry because a marker counts as known.
    """
    marked = 0
    for page in store.iter_record_ids(collection, id_field=id_field, page_size=page_size):
        coordinator.verify_publish(lease)
        known = adapter.fetch_state_subset(scope, page)
        markers = [
            StateRecord(record_key, UNVERIFIED_INPUT_FINGERPRINT, code_version)
            for record_key in page
            if record_key not in known
        ]
        if markers:
            adapter.upsert_state(scope, markers)
        marked += len(markers)
    return marked


def _restamp_and_sample(
    adapter: WarehouseAdapter,
    coordinator: ServingCoordinator,
    lease: PublishLease,
    *,
    scope: StateScope,
    page_size: int,
    code_version: str,
    model_name: str,
    total: int,
) -> tuple[int, list[str]]:
    """Re-stamp every record in `scope` at `code_version`; return the count and
    a reservoir sample of its keys for the membership check.

    One pass does both. Re-stamping is idempotent -- a record already at this
    version is rewritten unchanged -- so an activation interrupted here is
    re-entered by running it again. The reservoir is seeded from the code
    version, so two runs over the same state check the same keys.
    """
    rng = random.Random(code_version)
    sample: list[str] = []
    seen = 0
    # This phase ran for hours in prod with nothing to watch: `serving status`
    # showed the previous publish's counts and `status: publishing`, so the
    # only way to tell the command was alive was INFORMATION_SCHEMA (issue
    # #635). The note goes on the ledger, which is what another terminal can
    # read; the log line is for whoever launched it.
    heartbeat = Heartbeat()

    def _log_heartbeat(count: int, elapsed: float) -> None:
        log.info(
            "%s: re-stamping publication state: %d of %d records (%.1fs elapsed)",
            model_name,
            count,
            total,
            elapsed,
        )

    with (
        heartbeat.watch(_log_heartbeat),
        adapter.state_page_reader(scope, page_size=page_size) as reader,
    ):
        for batch in _state_batches(reader):
            coordinator.verify_publish(lease)
            adapter.upsert_state(
                scope,
                [
                    StateRecord(record.record_key, record.input_fingerprint, code_version)
                    for record in batch
                ],
            )
            for record in batch:
                seen += 1
                if len(sample) < ACTIVATION_SAMPLE_SIZE:
                    sample.append(record.record_key)
                else:
                    slot = rng.randrange(seen)
                    if slot < ACTIVATION_SAMPLE_SIZE:
                        sample[slot] = record.record_key
            # Per batch, not per heartbeat: the ledger note is what another
            # process reads, and a batch is already ~28s in prod, so the
            # write is noise against the MERGE it follows.
            coordinator.record_progress(lease, f"re-stamped {seen} of {total} records")
            heartbeat.update(seen)
    return seen, sample
