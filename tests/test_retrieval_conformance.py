"""One conformance suite every retrieval store must satisfy (issues #133, #371).

`RetrievalStore` had exactly one implementation until DuckDB (#371), which
made the abstraction a guess: nothing forced LanceDB's behavior to be the
*contract's* behavior rather than LanceDB's. This suite is where that gets
settled. Every test here runs against every registered store.

Two rules keep it honest as stores are added:

- **Test the contract, not an implementation.** If a behavior is legitimately
  store-specific (DuckDB's HNSW persistence opt-in, LanceDB's cloud URIs), it
  belongs in that store's own test module, not here.
- **Gate on declared capabilities, never on the store's name.** A test that
  needs `ARRAY_CONTAINMENT_FILTERS` skips for a store that does not advertise
  it. Branching on `store_type()` instead would let a store quietly opt out of
  a contract it does claim to implement.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from stel.retrieval.base import (
    CollectionSpec,
    IndexedRow,
    RetrievalError,
    RetrievalFeature,
    RetrievalPredicate,
    RetrievalPredicateOperator,
    RetrievalStore,
)
from stel.retrieval.duckdb import DuckDBConfig, DuckDBStore
from stel.retrieval.lancedb import LanceDBConfig, LanceDBStore

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("body", pa.string()),
        pa.field("tenant_id", pa.string()),
        pa.field("rank", pa.int64()),
        pa.field("access_groups", pa.list_(pa.string())),
        pa.field("embedding", pa.list_(pa.float32(), 3)),
    ]
)

ROWS = (
    ("a", "inflation rose sharply", "acme", 1, ["analysts", "ops"], [1.0, 0.0, 0.0]),
    ("b", "the labor market cooled", "acme", 2, ["admins"], [0.0, 1.0, 0.0]),
    ("c", "tariffs and trade policy", "globex", 3, [], [0.0, 0.0, 1.0]),
)


def _build_store(kind: str, tmp_path: Path) -> RetrievalStore:
    if kind == "duckdb":
        return DuckDBStore(
            DuckDBConfig(type="duckdb", path=str(tmp_path / "store.duckdb")),
            project_name="proj",
            target_name="dev",
            alias="default",
        )
    return LanceDBStore(
        LanceDBConfig(type="lancedb", path=str(tmp_path / "lance")),
        project_name="proj",
        target_name="dev",
        alias="default",
    )


@pytest.fixture(params=["duckdb", "lancedb"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[RetrievalStore]:
    built = _build_store(str(request.param), tmp_path)
    with built:
        yield built


def _spec(store: RetrievalStore, name: str) -> CollectionSpec:
    return CollectionSpec(
        logical_name="ctx",
        physical_name=name,
        id_field="id",
        text_fields=("body",),
        full_text_fields=("body",),
        attribute_fields=("tenant_id", "rank", "access_groups"),
        scalar_index_fields=(),
        display_fields=("body",),
        vector_field="embedding",
        vector_dimensions=3,
        distance_metric="cosine",
        vector_search="exact",
        config_fingerprint="cfg1",
        descriptor='{"distance_metric": "cosine"}',
        legacy_config_fingerprint="legacy1",
        arrow_schema=SCHEMA,
    )


def _rows() -> list[IndexedRow]:
    return [
        IndexedRow(
            row[0],
            {
                "id": row[0],
                "body": row[1],
                "tenant_id": row[2],
                "rank": row[3],
                "access_groups": row[4],
                "embedding": row[5],
            },
            f"fp-{row[0]}",
        )
        for row in ROWS
    ]


def _populate(store: RetrievalStore) -> str:
    name = store.physical_collection("ctx")
    spec = _spec(store, name)
    store.create_collection(spec)
    store.upsert(name, _rows(), id_field="id", mutation_digest="d1")
    store.ensure_indexes(spec)
    return name


def _requires(store: RetrievalStore, feature: RetrievalFeature) -> None:
    if feature not in store.capabilities().features:
        pytest.skip(f"{store.store_type()} does not advertise {feature.value}")


def _ids(table: pa.Table) -> list[Any]:
    return table.column("id").to_pylist()


# ─── identity ───────────────────────────────────────────────────────────────


def test_store_type_is_registered_and_stable(store: RetrievalStore) -> None:
    from stel.retrieval import list_store_types

    assert store.store_type() in list_store_types()


def test_safe_descriptor_carries_no_raw_path(store: RetrievalStore) -> None:
    """The descriptor is fingerprinted into state scopes and artifacts, so it
    must identify the target without disclosing where it lives."""
    descriptor = store.safe_descriptor()

    assert descriptor.store_type == store.store_type()
    assert descriptor.safe_target_identity
    assert "/" not in descriptor.safe_target_identity
    assert "\\" not in descriptor.safe_target_identity


def test_physical_collection_is_deterministic(store: RetrievalStore) -> None:
    assert store.physical_collection("ctx") == store.physical_collection("ctx")


def test_generation_suffix_is_distinct_from_the_base_name(
    store: RetrievalStore,
) -> None:
    base = store.physical_collection("ctx")
    generation = store.physical_collection("ctx", generation="abc123")

    assert generation != base
    assert generation.startswith(base)


# ─── lifecycle ──────────────────────────────────────────────────────────────


def test_create_then_inspect_round_trips_the_stamp(store: RetrievalStore) -> None:
    name = _populate(store)

    metadata = store.inspect_collection(name)

    assert metadata is not None
    assert metadata.physical_name == name
    assert metadata.config_fingerprint == "cfg1"
    assert metadata.row_count == len(ROWS)


def test_inspecting_a_missing_collection_returns_none(store: RetrievalStore) -> None:
    assert store.inspect_collection(store.physical_collection("ctx")) is None


def test_created_collection_is_listed(store: RetrievalStore) -> None:
    name = _populate(store)

    assert name in store.list_collections()


def test_dropping_reports_whether_anything_was_removed(
    store: RetrievalStore,
) -> None:
    name = _populate(store)

    assert store.drop_collection(name) is True
    assert store.drop_collection(name) is False
    assert store.inspect_collection(name) is None


def test_upsert_is_keyed_by_id_not_appended(store: RetrievalStore) -> None:
    name = _populate(store)
    revised = IndexedRow(
        "a",
        {
            "id": "a",
            "body": "revised",
            "tenant_id": "acme",
            "rank": 1,
            "access_groups": ["analysts"],
            "embedding": [1.0, 0.0, 0.0],
        },
        "fp-a2",
    )

    store.upsert(name, [revised], id_field="id", mutation_digest="d2")

    metadata = store.inspect_collection(name)
    assert metadata is not None
    assert metadata.row_count == len(ROWS)


def test_delete_removes_only_the_named_ids(store: RetrievalStore) -> None:
    name = _populate(store)

    store.delete(name, ["b"], id_field="id", mutation_digest="d3")

    metadata = store.inspect_collection(name)
    assert metadata is not None
    assert metadata.row_count == len(ROWS) - 1


def test_empty_mutations_are_accepted_as_no_ops(store: RetrievalStore) -> None:
    """Publication computes a change set that is often empty. A store that
    errored on it would make "nothing changed" an exceptional path."""
    name = _populate(store)

    upserted = store.upsert(name, [], id_field="id", mutation_digest="d4")
    deleted = store.delete(name, [], id_field="id", mutation_digest="d5")

    assert upserted.outcomes == ()
    assert deleted.outcomes == ()


def test_schema_evolution_preserves_rows(store: RetrievalStore) -> None:
    _requires(store, RetrievalFeature.ONLINE_SCHEMA_EVOLUTION)
    name = _populate(store)
    widened = pa.schema([*list(SCHEMA), pa.field("classification", pa.string())])
    base = _spec(store, name)
    spec = CollectionSpec(**{**base.__dict__, "arrow_schema": widened})

    store.evolve_collection(spec, ["classification"])

    metadata = store.inspect_collection(name)
    assert metadata is not None
    assert "classification" in metadata.schema.names
    assert metadata.row_count == len(ROWS)


# ─── queries ────────────────────────────────────────────────────────────────


def test_vector_search_orders_by_similarity(store: RetrievalStore) -> None:
    name = _populate(store)

    table = store.vector_search(
        name, [1.0, 0.0, 0.0], vector_field="embedding", limit=3
    )

    assert _ids(table)[0] == "a"


def test_vector_search_honors_the_limit(store: RetrievalStore) -> None:
    name = _populate(store)

    table = store.vector_search(
        name, [1.0, 0.0, 0.0], vector_field="embedding", limit=1
    )

    assert table.num_rows == 1


def test_vector_search_rejects_a_nonpositive_limit(store: RetrievalStore) -> None:
    name = _populate(store)

    with pytest.raises(RetrievalError):
        store.vector_search(name, [1.0, 0.0, 0.0], vector_field="embedding", limit=0)


def test_text_search_matches_terms(store: RetrievalStore) -> None:
    _requires(store, RetrievalFeature.FULL_TEXT_SEARCH)
    name = _populate(store)

    table = store.text_search(name, "inflation", text_field="body", limit=5)

    assert _ids(table) == ["a"]


def test_text_search_returns_no_rows_when_nothing_matches(
    store: RetrievalStore,
) -> None:
    _requires(store, RetrievalFeature.FULL_TEXT_SEARCH)
    name = _populate(store)

    table = store.text_search(name, "zzzznotpresent", text_field="body", limit=5)

    assert table.num_rows == 0


# ─── predicate translation ──────────────────────────────────────────────────


def test_equality_predicate_filters(store: RetrievalStore) -> None:
    _requires(store, RetrievalFeature.METADATA_FILTERING)
    name = _populate(store)

    table = store.vector_search(
        name,
        [1.0, 0.0, 0.0],
        vector_field="embedding",
        limit=5,
        predicates=[
            RetrievalPredicate("tenant_id", RetrievalPredicateOperator.EQUAL, "globex")
        ],
    )

    assert _ids(table) == ["c"]


def test_in_predicate_filters(store: RetrievalStore) -> None:
    _requires(store, RetrievalFeature.METADATA_FILTERING)
    name = _populate(store)

    table = store.vector_search(
        name,
        [1.0, 0.0, 0.0],
        vector_field="embedding",
        limit=5,
        predicates=[
            RetrievalPredicate(
                "tenant_id", RetrievalPredicateOperator.IN, ("globex", "nobody")
            )
        ],
    )

    assert _ids(table) == ["c"]


def test_numeric_comparison_predicate_filters(store: RetrievalStore) -> None:
    _requires(store, RetrievalFeature.METADATA_FILTERING)
    name = _populate(store)

    table = store.vector_search(
        name,
        [1.0, 0.0, 0.0],
        vector_field="embedding",
        limit=5,
        predicates=[
            RetrievalPredicate(
                "rank", RetrievalPredicateOperator.GREATER_THAN_OR_EQUAL, 2
            )
        ],
    )

    assert sorted(_ids(table)) == ["b", "c"]


def test_several_predicates_compose_as_conjunction(store: RetrievalStore) -> None:
    """The contract's predicate list is a flat AND. There is deliberately no
    OR or nesting: a policy prefilter that could be widened by an OR elsewhere
    in the expression would not be a prefilter."""
    _requires(store, RetrievalFeature.METADATA_FILTERING)
    name = _populate(store)

    table = store.vector_search(
        name,
        [1.0, 0.0, 0.0],
        vector_field="embedding",
        limit=5,
        predicates=[
            RetrievalPredicate("tenant_id", RetrievalPredicateOperator.EQUAL, "acme"),
            RetrievalPredicate("rank", RetrievalPredicateOperator.EQUAL, 2),
        ],
    )

    assert _ids(table) == ["b"]


def test_array_overlap_predicate_filters(store: RetrievalStore) -> None:
    _requires(store, RetrievalFeature.ARRAY_CONTAINMENT_FILTERS)
    name = _populate(store)

    table = store.vector_search(
        name,
        [0.0, 0.0, 1.0],
        vector_field="embedding",
        limit=5,
        predicates=[
            RetrievalPredicate(
                "access_groups",
                RetrievalPredicateOperator.ARRAY_CONTAINS_ANY,
                ("analysts", "nobody"),
            )
        ],
    )

    assert _ids(table) == ["a"]


def test_predicates_apply_to_the_text_leg_too(store: RetrievalStore) -> None:
    """A policy prefilter that held on one leg and not the other would leak
    through whichever mode skipped it."""
    _requires(store, RetrievalFeature.FULL_TEXT_SEARCH)
    _requires(store, RetrievalFeature.METADATA_FILTERING)
    name = _populate(store)

    table = store.text_search(
        name,
        "inflation",
        text_field="body",
        limit=5,
        predicates=[
            RetrievalPredicate("tenant_id", RetrievalPredicateOperator.EQUAL, "globex")
        ],
    )

    assert _ids(table) == []


# ─── publisher fencing ──────────────────────────────────────────────────────


# These two build their own pair of stores rather than taking the `store`
# fixture: contention needs two independent handles, and the fixture yields a
# single already-entered one. They stay parametrized over every store so the
# suite keeps its rule against branching on a store's name.
@pytest.mark.parametrize("kind", ["duckdb", "lancedb"])
def test_publisher_fence_excludes_a_second_holder(kind: str, tmp_path: Path) -> None:
    """The single-host guarantee, exercised through two independent store
    objects the way two publisher processes would contend."""
    first = _build_store(kind, tmp_path)
    second = _build_store(kind, tmp_path)
    with first, second:
        _requires(first, RetrievalFeature.SINGLE_HOST_PUBLISHER_LOCK)
        collection = first.physical_collection("ctx")

        with first.publisher_fence(collection):
            with pytest.raises(RetrievalError, match="publisher_lock_held"):
                with second.publisher_fence(collection):
                    pass


@pytest.mark.parametrize("kind", ["duckdb", "lancedb"])
def test_publisher_fence_is_released_on_exit(kind: str, tmp_path: Path) -> None:
    first = _build_store(kind, tmp_path)
    second = _build_store(kind, tmp_path)
    with first, second:
        _requires(first, RetrievalFeature.SINGLE_HOST_PUBLISHER_LOCK)
        collection = first.physical_collection("ctx")

        with first.publisher_fence(collection):
            pass

        with second.publisher_fence(collection):
            pass
