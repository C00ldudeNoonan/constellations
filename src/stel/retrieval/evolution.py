"""Classify a search-configuration change before mutation (issue #344).

`docs/architecture/semantic-retrieval.md` specifies which configuration
changes invalidate a published collection and which do not. This module is
that table in code.

Two things follow from it that the previous single-fingerprint comparison
could not express:

- **Fields that change execution cadence are not configuration changes.**
  `batch_size`, `index_options`, and `on_index_change` itself never alter what
  a published row contains, so they must not invalidate an index. Hashing the
  whole `SearchConfig` made tuning `batch_size` demand a full re-embed, and
  put `on_index_change` inside the fingerprint that decides how to react to a
  fingerprint change — so adopting a non-default policy tripped the very gate
  it was adopted to escape.
- **A change has to be nameable, not just detectable.** Comparing two digests
  can say *something* moved; it cannot say which field, and so cannot tell an
  additive attribute apart from a changed vector dimension. Classification
  needs the stored descriptor, which is why one is persisted alongside the
  collection rather than only its hash.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# Excluded from the descriptor entirely: per the design doc's change table
# these are "no semantic invalidation" — they change execution cadence,
# never the content of a published row.
NON_SEMANTIC_FIELDS = frozenset({"batch_size", "index_options", "on_index_change"})

# Routing identity rather than configuration: these select *which* physical
# collection is published, so a change means an independent publication
# against a different collection, not an evolution of this one.
ROUTING_FIELDS = frozenset({"store", "collection"})

# Projection-only fields. Adding one changes what a query returns, never how a
# row was indexed, so a published collection can serve the wider projection
# without being rebuilt.
_ADDITIVE_FIELDS = ("display_fields", "return_text_fields")

# The `vector` sub-fields that describe how the collection is *indexed*
# rather than what a published row contains. Switching between `exact` and
# `approximate` builds or drops an ANN structure over vectors that are already
# there (issue #461), and `index` chooses which structure (issue #476); the
# field, dimensions, metric, and embedding identity all change what a row
# means. `index` is absent from a descriptor when it is the default, so a
# stored descriptor without it and a current one with the default compare
# equal, and only a deliberate choice registers as a change.
_VECTOR_INDEX_ONLY_FIELDS = frozenset({"search", "index"})


class ChangeKind(StrEnum):
    """Whether a classified change can be served by the existing collection."""

    COMPATIBLE = "compatible"
    REBUILD_REQUIRED = "rebuild_required"


@dataclass(frozen=True)
class ConfigChange:
    field: str
    kind: ChangeKind
    detail: str

    def describe(self) -> str:
        return f"{self.field}: {self.detail}"


def json_safe(value: Any) -> Any:
    """Normalize a dumped config into plain JSON types.

    The descriptor is persisted and read back to be compared field by field,
    so it must round-trip through JSON. `canonical_json` deliberately does not:
    it encodes types alongside values for hashing. Sets are sorted so a config
    whose ordering is incidental cannot look like a change.
    """
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, set | frozenset):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [json_safe(item) for item in value]
    return value


def descriptor_json(descriptor: dict[str, Any]) -> str:
    """Serialize a descriptor for storage beside its collection."""
    return json.dumps(json_safe(descriptor), sort_keys=True, separators=(",", ":"))


def semantic_search_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Project a dumped `SearchConfig` onto the fields that define the index.

    Everything dropped here is either execution cadence or routing identity;
    what remains is what a published row's shape and meaning depend on.
    """
    excluded = NON_SEMANTIC_FIELDS | ROUTING_FIELDS
    return {
        key: json_safe(value)
        for key, value in sorted(payload.items())
        if key not in excluded
    }


def classify_changes(
    stored: dict[str, Any], current: dict[str, Any]
) -> list[ConfigChange]:
    """Diff two semantic descriptors into named, classified changes.

    An empty result means the two describe the same index. Callers decide what
    to do with the classification; this function never decides for them.
    """
    changes: list[ConfigChange] = []
    for field in sorted(set(stored) | set(current)):
        before = stored.get(field)
        after = current.get(field)
        if before == after:
            continue
        if field == "attributes":
            changes.append(_classify_attributes(before, after))
        elif field == "vector":
            changes.append(_classify_vector(before, after))
        elif field in _ADDITIVE_FIELDS:
            changes.append(_classify_projection(field, before, after))
        else:
            changes.append(
                ConfigChange(
                    field=field,
                    kind=ChangeKind.REBUILD_REQUIRED,
                    detail=f"changed from {before!r} to {after!r}",
                )
            )
    return changes


def _classify_projection(field: str, before: Any, after: Any) -> ConfigChange:
    old = tuple(before or ())
    new = tuple(after or ())
    if set(old).issubset(new):
        return ConfigChange(
            field=field,
            kind=ChangeKind.COMPATIBLE,
            detail=f"added {sorted(set(new) - set(old))} — projection only",
        )
    return ConfigChange(
        field=field,
        kind=ChangeKind.REBUILD_REQUIRED,
        detail=f"dropped {sorted(set(old) - set(new))}",
    )


def _classify_vector(before: Any, after: Any) -> ConfigChange:
    """A vector-search strategy change is an index build, not a rebuild.

    Treating the whole `vector` mapping as opaque made switching `exact` ->
    `approximate` demand a new collection name and a full republish, when the
    vectors themselves are untouched and only an ANN index needs building. For
    a corpus large enough to need the switch, that is hours of re-embedding to
    change an index flag -- and it is precisely the corpora large enough to
    need it that make the mistake expensive to correct (issue #461).
    """
    old = dict(before or {})
    new = dict(after or {})
    moved = {
        key for key in set(old) | set(new) if old.get(key) != new.get(key)
    }
    if moved and moved <= _VECTOR_INDEX_ONLY_FIELDS:
        described = ", ".join(
            f"{key} {old.get(key)!r} -> {new.get(key)!r}" for key in sorted(moved)
        )
        return ConfigChange(
            field="vector",
            kind=ChangeKind.COMPATIBLE,
            detail=(
                f"{described} — an index build over vectors already published, "
                "not a re-embed"
            ),
        )
    return ConfigChange(
        field="vector",
        kind=ChangeKind.REBUILD_REQUIRED,
        detail=f"changed from {before!r} to {after!r}",
    )


def _classify_attributes(before: Any, after: Any) -> ConfigChange:
    """Attributes are the one field where additive and breaking changes mix.

    Adding an attribute widens the collection; changing an existing one's type
    or filter role reinterprets rows already written under the old meaning,
    which no amount of widening can fix.
    """
    old = {item["name"]: item for item in (before or ())}
    new = {item["name"]: item for item in (after or ())}
    removed = sorted(set(old) - set(new))
    if removed:
        return ConfigChange(
            field="attributes",
            kind=ChangeKind.REBUILD_REQUIRED,
            detail=f"removed {removed}",
        )
    redefined = sorted(name for name in old if old[name] != new[name])
    if redefined:
        return ConfigChange(
            field="attributes",
            kind=ChangeKind.REBUILD_REQUIRED,
            detail=(
                f"redefined {redefined} — rows already written carry the old "
                "type or filter role"
            ),
        )
    added = sorted(set(new) - set(old))
    return ConfigChange(
        field="attributes",
        kind=ChangeKind.COMPATIBLE,
        detail=f"added {added}",
    )


def rebuild_required(changes: list[ConfigChange]) -> list[ConfigChange]:
    return [change for change in changes if change.kind is ChangeKind.REBUILD_REQUIRED]


# Descriptor fields outside `search` that invalidate a collection on their own:
# the store contract and semantic implementation version, per the design doc's
# "whole-index invalidation" row.
_CONTRACT_FIELDS = ("contract_version", "store_type", "store_implementation")


def classify_descriptor_changes(
    stored: dict[str, Any], current: dict[str, Any]
) -> list[ConfigChange]:
    """Classify a whole descriptor, not just its `search` mapping.

    Comparing only `search` silently drops the contract fields, so a store
    implementation bump — which invalidates every row written under it —
    produced an empty change list and was reported as an additive change the
    existing collection could serve (Codex review, #344).
    """
    changes = [
        ConfigChange(
            field=field,
            kind=ChangeKind.REBUILD_REQUIRED,
            detail=f"changed from {stored.get(field)!r} to {current.get(field)!r}",
        )
        for field in _CONTRACT_FIELDS
        if stored.get(field) != current.get(field)
    ]
    return changes + classify_changes(
        stored.get("search", {}), current.get("search", {})
    )
