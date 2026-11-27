"""The pure rules of `get_entity_neighbors` (issue #796, ALE-424).

Kept apart from `service.py`, which owns authorization and reads, so the two
rules that decide what a walk may say are exercised without a warehouse:

- when an edge holds at an instant, which must agree with `relations_as_of`
  (a test pins the two against each other), and
- the order edges are returned in, so two calls over the same data agree.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from .contracts import EdgeDirection


def edge_holds_at(row: Mapping[str, Any], as_of: datetime | None) -> bool:
    """Whether an edge's validity interval admits `as_of`.

    Half-open, `valid_from <= as_of < valid_to`, with a null end open: at the
    instant a newer fact begins, only the newer fact holds. An edge with no
    `valid_from` does not claim to hold at any instant, so an `as_of` query leaves
    it out; with no `as_of` it is kept and its unknown interval is reported.
    """
    if as_of is None:
        return True
    start = row.get("valid_from")
    if start is None:
        return False
    end = row.get("valid_to")
    at = as_of.astimezone(UTC)
    return start <= at and (end is None or at < end)


def edge_direction(row: Mapping[str, Any], source_id: str) -> EdgeDirection:
    """The edge's orientation as seen from `source_id`, the entity the walk
    reached it from."""
    if not row["directed"]:
        return EdgeDirection.UNDIRECTED
    if row["subject_entity_id"] == source_id:
        return EdgeDirection.OUTGOING
    return EdgeDirection.INCOMING


def edge_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Deterministic order within a hop. The warehouse returns rows in no
    guaranteed order, and which edges survive a `limit` must not depend on it."""
    return (
        str(row["relation_type"]),
        str(row["subject_entity_id"]),
        str(row["object_entity_id"]),
        str(row["context_entity_edge_id"]),
    )
