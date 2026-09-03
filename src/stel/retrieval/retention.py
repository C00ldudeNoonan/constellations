"""Retirement of superseded retrieval generations (issue #355).

A private generation build leaves collections behind: the one it replaced once
activation moves the pointer, and the half-built one if the publisher dies
before activating. Neither is reachable through the serving ledger, so nothing
would ever reclaim them.

**Why there is no grace period.** Private builds allow concurrent readers.
An admitted query can still pin a superseded generation, so retirement defers
while any query lease exists. With zero pins, new admissions can reach only
the active generation (excluded from the sweep). Admission re-verifies the
active pointer before store I/O, so a late insertion for a retired generation
is refused. A publisher claim excludes concurrent builders.

Holding the lease is therefore the contract, verified rather than trusted:
`retire_superseded_generations` takes the coordinator and the caller's
`PublishLease` and aborts if the fence has moved. Running "just after
recovery" is *not* a safe state on its own — recovery only guarantees zero
leases at the instant it clears them, and another process may acquire the
publish lease and start building a new generation while the sweep is listing
and dropping. A post-recovery sweeper must acquire the lease like any other
publisher.

**Why prefix matching is safe.** A generation collection is named exactly
`<base>__g<token>` (1-16 lowercase alphanumerics). That shape is reserved:
`reject_generation_shaped_collection_name` refuses to resolve any *base*
collection name ending the same way, so no sibling logical collection can be
mistaken for a generation of this one. The sweep still matches the complete
shape — marker plus exact token — never a bare prefix. The active generation
is identified by the ledger; anything else matching the shape is unreachable
and safe to drop.
"""
from __future__ import annotations

import re

from .base import (
    GENERATION_MARKER,
    RetrievalCapabilityError,
    RetrievalFeature,
    RetrievalStore,
)
from .coordination import PublishLease, ServingCoordinator

_GENERATION_TOKEN_RE = re.compile(r"^[a-z0-9]{1,16}$")


def generation_prefix(store: RetrievalStore, logical_collection: str) -> str:
    """The prefix every generation collection for `logical_collection` shares."""
    return f"{store.physical_collection(logical_collection)}{GENERATION_MARKER}"


def superseded_generations(
    store: RetrievalStore,
    *,
    logical_collection: str,
    active_collection: str | None,
) -> list[str]:
    """Generation collections for `logical_collection` that nothing can reach.

    Excludes the active generation, and never includes the unsuffixed base
    collection: dropping that would destroy an in-place published index. Only
    complete generation names — the shared prefix followed by exactly one
    valid token — are candidates, so a collection that merely shares the
    prefix without the generation shape is never classified as one.
    """
    prefix = generation_prefix(store, logical_collection)
    return sorted(
        name
        for name in store.list_collections()
        if name.startswith(prefix)
        and _GENERATION_TOKEN_RE.fullmatch(name[len(prefix) :])
        and name != active_collection
    )


def retire_superseded_generations(
    store: RetrievalStore,
    *,
    logical_collection: str,
    active_collection: str | None,
    coordinator: ServingCoordinator,
    lease: PublishLease,
) -> list[str]:
    """Drop every unreachable generation, returning the names retired.

    The caller must hold the publish lease for this scope — see the module
    docstring for why that is what makes the sweep safe — and the lease is
    verified against the ledger before listing and before every drop, so a
    reassigned fence aborts the sweep before it can delete a newer
    publisher's build. Idempotent: a second run finds nothing.

    A collection stel does not own is refused by `drop_collection` rather than
    skipped. A foreign table sitting on this prefix means the namespace is not
    ours to sweep, and silently stepping around it would hide that.
    """
    capabilities = store.capabilities()
    if RetrievalFeature.PRIVATE_GENERATION_BUILD not in capabilities.features:
        raise RetrievalCapabilityError(
            f"Retrieval store '{store.store_type()}' does not build private "
            "generations, so it has none to retire"
        )
    coordinator.verify_publish(lease)
    # Pins can outlive activation. Never retire a collection while a reader
    # may still be using it. New admissions can pin only the active collection,
    # which the sweep excludes, so a zero count cannot gain an old-generation
    # reader after this check (admission re-verifies the active pointer).
    if coordinator.status(lease.scope).query_leases:
        return []
    retired = []
    for name in superseded_generations(
        store,
        logical_collection=logical_collection,
        active_collection=active_collection,
    ):
        coordinator.verify_publish(lease)
        if store.drop_collection(name):
            retired.append(name)
    return retired
