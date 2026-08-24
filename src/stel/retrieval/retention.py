"""Retirement of superseded retrieval generations (issue #355).

A private generation build leaves collections behind: the one it replaced once
activation moves the pointer, and the half-built one if the publisher dies
before activating. Neither is reachable through the serving ledger, so nothing
would ever reclaim them.

**Why there is no grace period.** The issue asks how long a superseded
generation should linger and what reclaims it if a publisher dies mid-flight.
Both answers fall out of the coordination protocol rather than needing a timer
or a generations table:

- `acquire_publish` refuses while any query lease exists, and `acquire_query`
  refuses while a publisher holds the claim. So while the publish lease is
  held there are *zero* query leases, by construction — a sweep running under
  that claim cannot race a reader, whichever generation the reader pinned.
- `serving recover` clears every lease before returning, giving the same
  guarantee from the other direction.

A sweep is therefore only ever correct in those two states, and in both of
them "is anything reading this?" is already answered. The caller is
responsible for being in one of them; that is the contract of this module.

The active generation is identified by the ledger, so anything else matching
the generation-suffixed pattern is unreachable and safe to drop.
"""
from __future__ import annotations

from .base import (
    RetrievalCapabilityError,
    RetrievalFeature,
    RetrievalStore,
)

# Collections built for a generation are named `<base>__g<token>` by
# `physical_collection(..., generation=...)`. The marker is what keeps the
# unsuffixed `<base>` — the collection an in-place incremental publish writes,
# and where every pre-#355 index still lives — out of every candidate set.
GENERATION_MARKER = "__g"


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
    collection: dropping that would destroy an in-place published index.
    """
    prefix = generation_prefix(store, logical_collection)
    return sorted(
        name
        for name in store.list_collections()
        if name.startswith(prefix) and name != active_collection
    )


def retire_superseded_generations(
    store: RetrievalStore,
    *,
    logical_collection: str,
    active_collection: str | None,
) -> list[str]:
    """Drop every unreachable generation, returning the names retired.

    The caller must hold the publish lease for this scope, or have just
    recovered it — see the module docstring for why that is what makes the
    sweep safe. Idempotent: a second run finds nothing.

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
    retired = []
    for name in superseded_generations(
        store,
        logical_collection=logical_collection,
        active_collection=active_collection,
    ):
        if store.drop_collection(name):
            retired.append(name)
    return retired
