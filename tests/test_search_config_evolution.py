"""Search-config change classification and the pre-#344 re-stamp (issue #344).

Two claims are under test. First, that fields which only change execution
cadence no longer invalidate a published index — the bug this issue opened on,
where tuning `batch_size` demanded a full re-embed. Second, that a change which
*does* matter is named rather than merely detected, so the operator learns
which field forced a rebuild instead of being told "something changed".
"""
from __future__ import annotations

from typing import Any

import pytest

from stel.execution.contracts import RunError
from stel.execution.search import _verify_collection_config
from stel.retrieval import (
    ChangeKind,
    CollectionMetadata,
    CollectionSpec,
    classify_changes,
    collection_config_fingerprint,
    descriptor_json,
    legacy_collection_config_fingerprint,
)
from stel.retrieval.registry import collection_descriptor

_BASE: dict[str, Any] = {
    "access": "public",
    "store": None,
    "collection": None,
    "id_field": "chunk_id",
    "document_id_field": "document_id",
    "chunk_id_field": "chunk_id",
    "text_fields": ("text",),
    "return_text_fields": ("text",),
    "vector": {"field": "embedding", "dimensions": 2, "metric": "cosine"},
    "full_text": {"fields": ("text",)},
    "attributes": (
        {"name": "category", "data_type": "string", "filter_role": "user"},
    ),
    "display_fields": ("title",),
    "query": {"modes": {"vector", "text"}},
    "on_index_change": "fail",
    "batch_size": 1000,
    "index_options": {},
}


def _config(**overrides: Any) -> dict[str, Any]:
    return {**_BASE, **overrides}


def _fingerprint(**overrides: Any) -> str:
    return collection_config_fingerprint(_config(**overrides), store_type="lancedb")


def _search(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = collection_descriptor(
        _config(**overrides), store_type="lancedb"
    )["search"]
    return payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 500),
        ("index_options", {"nprobes": 20}),
        ("on_index_change", "rebuild"),
    ],
)
def test_cadence_fields_do_not_invalidate_a_published_index(
    field: str, value: Any
) -> None:
    """The reported bug. `batch_size` changes how many rows a publish sends per
    call and `index_options` is not read by any code at all, yet both used to
    change the fingerprint and force a blue/green rebuild plus a full re-embed.

    `on_index_change` is the sharper one: it is the field that decides how to
    react to a config change, so having it inside the fingerprint meant
    adopting a non-default policy tripped the very gate it was adopting."""
    assert _fingerprint(**{field: value}) == _fingerprint()


@pytest.mark.parametrize(
    "field,value",
    [
        ("id_field", "other_id"),
        ("vector", {"field": "embedding", "dimensions": 3, "metric": "cosine"}),
        ("vector", {"field": "embedding", "dimensions": 2, "metric": "dot"}),
        ("text_fields", ("text", "title")),
        ("access", "governed"),
    ],
)
def test_semantic_fields_still_invalidate(field: str, value: Any) -> None:
    """The complement: narrowing the descriptor must not have narrowed it so
    far that a real change slips through."""
    assert _fingerprint(**{field: value}) != _fingerprint()


def test_an_added_attribute_is_compatible_and_named() -> None:
    """Widening a collection with a new filterable attribute leaves every row
    already written valid under its own definition."""
    (change,) = classify_changes(
        _search(),
        _search(
            attributes=(
                {"name": "category", "data_type": "string", "filter_role": "user"},
                {"name": "symbol", "data_type": "string", "filter_role": "user"},
            )
        ),
    )
    assert change.field == "attributes"
    assert change.kind is ChangeKind.COMPATIBLE
    assert "symbol" in change.detail


def test_a_redefined_attribute_requires_a_rebuild() -> None:
    """Prevents the dangerous confusion: changing an existing attribute's type
    or filter role is not additive. Rows already indexed carry the old meaning,
    and widening cannot reinterpret them."""
    (change,) = classify_changes(
        _search(),
        _search(
            attributes=(
                {"name": "category", "data_type": "int", "filter_role": "user"},
            )
        ),
    )
    assert change.kind is ChangeKind.REBUILD_REQUIRED
    assert "category" in change.detail


def test_a_removed_attribute_requires_a_rebuild() -> None:
    (change,) = classify_changes(_search(), _search(attributes=()))
    assert change.kind is ChangeKind.REBUILD_REQUIRED
    assert "category" in change.detail


def test_a_changed_vector_dimension_names_the_field() -> None:
    """The whole point of classification: the operator is told which field
    forced the rebuild, not just that the digest moved."""
    (change,) = classify_changes(
        _search(),
        _search(vector={"field": "embedding", "dimensions": 3, "metric": "cosine"}),
    )
    assert change.field == "vector"
    assert change.kind is ChangeKind.REBUILD_REQUIRED


def test_a_wider_projection_is_compatible_but_dropping_one_is_not() -> None:
    (added,) = classify_changes(_search(), _search(display_fields=("title", "url")))
    assert added.kind is ChangeKind.COMPATIBLE
    (dropped,) = classify_changes(_search(), _search(display_fields=()))
    assert dropped.kind is ChangeKind.REBUILD_REQUIRED


def test_an_identical_config_classifies_as_no_change() -> None:
    assert classify_changes(_search(), _search()) == []


class _RecordingStore:
    """Captures whether the collection was re-stamped rather than rebuilt."""

    def __init__(self) -> None:
        self.restamped: list[str] = []

    def restamp_collection(self, spec: CollectionSpec) -> None:
        self.restamped.append(spec.descriptor)


def _spec(**overrides: Any) -> CollectionSpec:
    config = _config(**overrides)
    return CollectionSpec(
        logical_name="context",
        physical_name="proj__dev__context",
        id_field="chunk_id",
        text_fields=("text",),
        full_text_fields=("text",),
        attribute_fields=("category",),
        scalar_index_fields=("category",),
        display_fields=("title",),
        vector_field="embedding",
        vector_dimensions=2,
        distance_metric="cosine",
        vector_search="exact",
        config_fingerprint=collection_config_fingerprint(config, store_type="lancedb"),
        descriptor=descriptor_json(
            collection_descriptor(config, store_type="lancedb")
        ),
        legacy_config_fingerprint=legacy_collection_config_fingerprint(
            config, store_type="lancedb"
        ),
        arrow_schema=None,
    )


def _metadata(
    *, descriptor: str | None, config_fingerprint: str | None
) -> CollectionMetadata:
    return CollectionMetadata(
        physical_name="proj__dev__context",
        config_fingerprint=config_fingerprint,
        descriptor=descriptor,
        physical_generation="gen",
        row_count=2,
        schema=None,
    )


def test_a_pre_344_collection_is_restamped_not_rebuilt() -> None:
    """The migration. A collection published before the descriptor existed
    carries only the legacy digest. Recomputing that digest proves the config
    is unchanged, so the stamp is rewritten in place — no rebuild, no re-embed,
    and no operator action. Getting this wrong would inflict exactly the cost
    this issue exists to remove."""
    spec = _spec()
    store = _RecordingStore()
    existing = _metadata(
        descriptor=None, config_fingerprint=spec.legacy_config_fingerprint
    )

    _verify_collection_config(store, existing, spec)

    assert store.restamped == [spec.descriptor]


def test_a_pre_344_collection_whose_config_really_changed_still_fails() -> None:
    """The complement: the re-stamp must fire only on positive proof that
    nothing changed, never as a way to wave through a real change."""
    store = _RecordingStore()
    existing = _metadata(descriptor=None, config_fingerprint="some-other-digest")

    with pytest.raises(RunError, match="cannot be named"):
        _verify_collection_config(store, existing, _spec())

    assert store.restamped == []


def test_a_matching_descriptor_neither_fails_nor_restamps() -> None:
    spec = _spec()
    store = _RecordingStore()
    existing = _metadata(
        descriptor=spec.descriptor, config_fingerprint=spec.config_fingerprint
    )

    _verify_collection_config(store, existing, spec)

    assert store.restamped == []


def test_a_rebuild_change_names_the_field_in_the_error() -> None:
    spec = _spec(vector={"field": "embedding", "dimensions": 3, "metric": "cosine"})
    stored = _spec()
    existing = _metadata(
        descriptor=stored.descriptor, config_fingerprint=stored.config_fingerprint
    )

    with pytest.raises(RunError, match="requires a rebuild: vector"):
        _verify_collection_config(_RecordingStore(), existing, spec)


def test_an_additive_change_says_so_rather_than_demanding_a_rebuild() -> None:
    """`fail` is still the only policy, so this raises — but it must not tell
    the operator to rebuild a collection that could serve the change."""
    spec = _spec(
        attributes=(
            {"name": "category", "data_type": "string", "filter_role": "user"},
            {"name": "symbol", "data_type": "string", "filter_role": "user"},
        )
    )
    stored = _spec()
    existing = _metadata(
        descriptor=stored.descriptor, config_fingerprint=stored.config_fingerprint
    )

    with pytest.raises(RunError, match="additive") as error:
        _verify_collection_config(_RecordingStore(), existing, spec)
    assert "rebuild" not in str(error.value).split("on_index_change")[0]
