"""Keep a search index's store mirror current, and restore from it (issue #666).

A store's `mirror:` is a second copy of every generation the ledger serves,
kept so a fresh host can be brought back without re-embedding the corpus. It
is never read by a query; builds and serving read the primary.

**A sync is a reader.** It holds a query lease on the active generation for
the length of the copy, and the lease is what makes the copy consistent: an
in-place publisher cannot claim the scope while any lease is held, and the
retirement of a superseded generation waits for leases to drain. A private
rebuild may still run alongside, exactly as it may alongside any query, and
the sync records nothing if that rebuild activates first.

**The mirror holds only what the ledger vouched for.** The collection is
copied only when its physical generation is the one the ledger names as
active. An in-place publish that failed on an interruption-safe store leaves
the scope `degraded`, serving a pointer whose collection has since moved on;
that collection is not mirrored, because a restore of it would put rows under
publication state that does not describe them.

**A restore checks what it wrote.** It copies the active generation onto a
primary that lacks it, never overwriting anything there, and then requires the
collection it produced to be that same generation before reporting success.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..adapters import AdapterError, StateScope
from ..retrieval import (
    RetrievalConfigError,
    RetrievalError,
    RetrievalStore,
    ServingBusyError,
    ServingCoordinator,
    ServingLedgerEntry,
)
from ..retrieval.base import MirrorCopy, mirror_fingerprint
from ..retrieval.retention import generation_prefix, superseded_among
from ..timing import PhaseTimings
from .contracts import RunError

log = logging.getLogger(__name__)

_NO_COPY = MirrorCopy(files_copied=0, bytes_copied=0, files_deleted=0)


@dataclass(frozen=True)
class MirrorSync:
    """What one sync of a search index's mirror did."""

    collection: str
    generation: str
    copy: MirrorCopy
    # Whether the ledger now records the mirror as holding `generation`.
    # False only when a later activation replaced it during the copy.
    recorded: bool
    # Superseded generations removed from the mirror after the record landed.
    retired: tuple[str, ...]


@dataclass(frozen=True)
class MirrorRestore:
    """What one restore of a search index from its mirror did."""

    collection: str
    generation: str
    # All zero when the primary already held the generation.
    copy: MirrorCopy


def sync_search_mirror(
    *,
    store: RetrievalStore,
    coordinator: ServingCoordinator,
    scope: StateScope,
    logical_collection: str,
) -> MirrorSync:
    """Bring the mirror up to the generation the ledger serves.

    Raises `ServingBusyError` when an in-place publisher holds the scope, and
    `ServingNotReadyError` when nothing is served; every other refusal is a
    `RetrievalError` naming what disagreed.
    """
    location = _mirror_location(store)
    lease = coordinator.acquire_query(scope)
    try:
        collection = lease.pinned_collection or store.physical_collection(logical_collection)
        with store:
            existing = store.inspect_collection(collection)
            if existing is None or existing.physical_generation != lease.pinned_generation:
                raise RetrievalError(
                    f"Collection '{collection}' is not the generation the serving "
                    "ledger activated, so it is not mirrored: a mirror holds only "
                    "what the ledger vouched for. The next successful publish "
                    "syncs it (code=mirror_generation_mismatch)"
                )
            copy = store.sync_to_mirror(collection)
        # The lease must have survived the copy. `serving recover` clears
        # every lease, and after it an in-place publisher may have written
        # into what was being copied.
        coordinator.validate_query(lease)
        recorded = coordinator.record_mirror(
            lease, mirror_target=mirror_fingerprint(location)
        )
        retired: tuple[str, ...] = ()
        if recorded:
            # Only once the record says the mirror holds this generation, so a
            # mirror is never left holding no servable generation at all.
            with store:
                stale = superseded_among(
                    store.mirror_collections(),
                    prefix=generation_prefix(store, logical_collection),
                    active_collection=collection,
                )
                for name in stale:
                    store.drop_mirror_collection(name)
            retired = tuple(stale)
        return MirrorSync(
            collection=collection,
            generation=lease.pinned_generation,
            copy=copy,
            recorded=recorded,
            retired=retired,
        )
    finally:
        coordinator.release_query(lease)


def sync_mirror_after_publish(
    *,
    store: RetrievalStore,
    coordinator: ServingCoordinator,
    scope: StateScope,
    logical_collection: str,
    model_name: str,
    timings: PhaseTimings,
    progress: dict[str, int],
) -> MirrorSync | None:
    """The sync a successful publish or activation ends with, as a model outcome.

    The generation is already published and serving when this runs, so a
    failure here does not touch the ledger. It does fail the model: a mirror
    that silently stops tracking is the failure a mirror exists to prevent,
    and an orchestrator's retry heals it, because the rerun reconciles nothing
    and syncs only the files the mirror lacks. A publisher that claimed the
    scope in between is not a failure; its own completion syncs.
    """
    try:
        with timings.phase("mirror_sync"):
            outcome = sync_search_mirror(
                store=store,
                coordinator=coordinator,
                scope=scope,
                logical_collection=logical_collection,
            )
    except ServingBusyError:
        log.info(
            "%s: another publisher holds the index; the mirror is synced when it "
            "completes",
            model_name,
        )
        return None
    except (AdapterError, RetrievalError) as error:
        raise RunError(
            f"{model_name} is published and serving, but syncing its store mirror "
            f"failed: {error}. Run `stel serving sync {model_name}` to retry; "
            "it copies only what the mirror lacks",
            metrics=timings.as_metrics(),
            # What the publish did, which stands: the run log should not
            # report a completed publish as zero rows (issue #623).
            progress=progress,
        ) from None
    log.info(
        "%s: mirror holds %s (%d file(s), %d byte(s) copied; %d removed; %d "
        "superseded generation(s) retired)",
        model_name,
        outcome.collection,
        outcome.copy.files_copied,
        outcome.copy.bytes_copied,
        outcome.copy.files_deleted,
        len(outcome.retired),
    )
    return outcome


def restore_search_mirror(
    *,
    store: RetrievalStore,
    coordinator: ServingCoordinator,
    scope: StateScope,
    logical_collection: str,
) -> MirrorRestore:
    """Copy the served generation from the mirror onto this host's primary."""
    location = _mirror_location(store)
    lease = coordinator.acquire_query(scope)
    try:
        # Read under the lease, which is what holds the active pointer still
        # from here to the end of the copy.
        refusal = mirror_restore_refusal(
            coordinator.status(scope), mirror_target=mirror_fingerprint(location)
        )
        if refusal is not None:
            raise RetrievalError(refusal)
        collection = lease.pinned_collection or store.physical_collection(logical_collection)
        with store.publisher_fence(collection), store:
            existing = store.inspect_collection(collection)
            if existing is not None and existing.physical_generation == lease.pinned_generation:
                return MirrorRestore(collection, lease.pinned_generation, _NO_COPY)
            copy = store.restore_from_mirror(collection)
            restored = store.inspect_collection(collection)
        if restored is None or restored.physical_generation != lease.pinned_generation:
            raise RetrievalError(
                f"Collection '{collection}' after the restore is not the generation "
                "the serving ledger activated; the primary held files that diverge "
                "from the mirror. Nothing was overwritten; move the primary's copy "
                "aside and restore again (code=mirror_restore_mismatch)"
            )
        return MirrorRestore(collection, lease.pinned_generation, copy)
    finally:
        coordinator.release_query(lease)


def mirror_restore_refusal(entry: ServingLedgerEntry, *, mirror_target: str) -> str | None:
    """Why the mirror cannot restore what this ledger entry serves, or None.

    Pure, so each refusal is pinned without a store. A restore serves the
    copy under the publication state the warehouse already holds, so it is
    correct only when the mirror holds exactly the generation that state
    describes -- the one the ledger names as active.
    """
    if entry.active_generation is None:
        return (
            "This search index serves no generation, so there is nothing to "
            "restore; publish it instead"
        )
    if entry.mirror_generation is None or entry.mirror_target != mirror_target:
        return (
            "The serving ledger has no record of this mirror holding the index. "
            "If the mirror was changed or never synced, run `stel serving sync` "
            "from a host that holds the primary, or republish"
        )
    if entry.mirror_generation != entry.active_generation:
        return (
            "The mirror holds an older generation than the one the ledger serves; "
            "restoring it would serve rows the publication state does not "
            "describe. Run `stel serving sync` from a host that holds the "
            "current generation, or republish"
        )
    return None


def _mirror_location(store: RetrievalStore) -> str:
    location = store.config.mirror_location()
    if location is None:
        raise RetrievalConfigError(
            f"Retrieval store '{store.alias}' has no mirror configured; add "
            "`mirror:` to the store in the profile"
        )
    return location
