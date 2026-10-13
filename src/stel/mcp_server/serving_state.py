"""How current a served index is, read from the serving ledger (issue #682).

`list_context_models` used to answer this from the compiled artifacts: the
model-level `freshness` was the literal string `"unknown"`, and
`last_successful_materialization` came from the latest `run_results.json`. Both
were wrong against a live index, and wrong in the direction that matters. The
timestamp went back to null after any run whose selection left the index out,
and after a container start that rehydrated the store from object storage
without running anything at all -- so an index that had stopped publishing
looked exactly like one that had just finished.

The ledger is the only record of when the rows now being served went live. It
is warehouse state, so it survives both of those, and `activated_epoch` moves
only on activation -- not on a failed publish that retains its predecessor.
That is what lets an index's reported age keep growing while its publishes
fail, which is the signal #617 showed an agent has no other way to get.

The scope is derived from the compiled descriptor rather than from a store:
building one inside the MCP process would mean a vector-store import and its
credentials, to learn a fingerprint compilation already wrote down.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ..adapters.base import (
    RETRIEVAL_PUBLISH_STAGE,
    SERVING_LEDGER_TABLE,
    ReadPredicate,
    ReadPredicateOperator,
)
from ..agent_context import FreshnessStatus
from .catalog import ContextResource
from .repository import ContextRepository, ContextRepositoryError

log = logging.getLogger(__name__)

# One row per scope, and a listing covers at most one page of models. The
# ceiling is a guard against a ledger corrupted by duplicate creation races,
# not a real expectation: `stel serving recover` repairs that case, and over
# it the read degrades to "unknown" rather than failing the listing.
_MAX_LEDGER_ROWS = 512


def read_activations(
    repository: ContextRepository,
    resources: Sequence[ContextResource],
) -> dict[str, datetime]:
    """When each resource's active generation was activated.

    Keyed by resource name, and absent for any index with no activation on
    record. One read for the whole page rather than one per model: the ledger
    is small and a listing is on a request path.

    A ledger that cannot be read is reported as an unknown age rather than
    raised, because a listing's job is to list the models -- failing it
    outright would turn a missing diagnostic into no catalog at all.
    """
    if not resources:
        return {}
    names = tuple({resource.name for resource in resources})
    try:
        rows = repository.read_owned_rows(
            SERVING_LEDGER_TABLE,
            predicates=[
                ReadPredicate(
                    "model_name",
                    ReadPredicateOperator.IN,
                    names,
                ),
                ReadPredicate(
                    "stage",
                    ReadPredicateOperator.EQUAL,
                    RETRIEVAL_PUBLISH_STAGE,
                ),
            ],
            max_rows=_MAX_LEDGER_ROWS,
            # `activated_epoch` is newer than the table. A ledger written by
            # an earlier release does not have it yet -- the ALTER that adds
            # it runs on the next publish, not here -- and `read_owned_rows`
            # reports a relation missing a projected column as absent, which
            # yields "no activation on record". That is both true and useful;
            # failing the listing would be neither.
            columns=[
                "model_name",
                "target_identity",
                "active_generation",
                "activated_epoch",
            ],
        )
    except ContextRepositoryError:
        # The class name is not even included: this error is already stel's
        # own sanitized wrapper, and warehouse text may carry SQL or
        # identifiers that must not reach logs or artifacts.
        log.warning(
            "Could not read the serving ledger; %d model(s) will report an "
            "unknown age and are otherwise unaffected",
            len(resources),
        )
        return {}
    by_scope = {
        (str(row["model_name"]), str(row["target_identity"])): row
        for row in rows
        if _scoped(row)
    }
    activations: dict[str, datetime] = {}
    for resource in resources:
        row = by_scope.get((resource.name, resource.scope_fingerprint))
        if row is None:
            continue
        # A date is about a generation, so a scope with no active generation
        # has no age to report -- whatever timestamp the row still carries.
        # Asserted here rather than left to the writers: several paths clear
        # the pointer (a fail-closed in-place failure, a non-preserving
        # publish claim, recovery with nothing servable), and a reader that
        # trusted the stamp alone would report a materialization for an index
        # that serves nothing.
        if row.get("active_generation") is None:
            continue
        activated = _epoch(row.get("activated_epoch"))
        if activated is not None:
            activations[resource.name] = activated
    return activations


def freshness(
    activated_at: datetime | None,
    *,
    stale_after_seconds: int | None,
    now: datetime,
) -> FreshnessStatus:
    """Judge one index's activation time against its declared policy.

    `unknown` means exactly what #682 asked it to mean: there is no activation
    on record. An index that has published but declares no policy reports
    `fresh` -- it materialized, and nothing says that is too long ago; the
    activation time travels alongside so a caller can judge for itself.

    Past the policy it is `pipeline_stale`, the same status the row-level
    contract uses when a refresh is overdue, rather than `source_stale`: what
    is late is stel's own publication, not the upstream document.
    """
    if activated_at is None:
        return FreshnessStatus.UNKNOWN
    if stale_after_seconds is None:
        return FreshnessStatus.FRESH
    if (now - activated_at).total_seconds() >= stale_after_seconds:
        return FreshnessStatus.PIPELINE_STALE
    return FreshnessStatus.FRESH


def _scoped(row: Mapping[str, Any]) -> bool:
    return isinstance(row.get("model_name"), str) and isinstance(
        row.get("target_identity"), str
    )


def _epoch(value: Any) -> datetime | None:
    """Epoch seconds as an aware UTC instant, or None when unusable.

    None is the answer for a scope last activated by a release that predates
    the column, and for a value no arithmetic can be done on -- both are
    "no activation on record" rather than an error worth refusing a listing
    over.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return datetime.fromtimestamp(value, tz=UTC)
