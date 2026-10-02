"""Dropping a private generation's indices before it is written (issue #616).

A resumed generation arrives carrying whatever indices its earlier attempt
built, and then every page pays to maintain them. A fresh generation never
does, because `ensure_indexes` runs after the page loop. `drop_indexes` is
how a resume gets the fresh path's shape back.

The round-trip test is the one that makes the fix safe: whatever this removes,
`ensure_indexes` has to put back.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from stel.retrieval import (
    CollectionSpec,
    DuckDBStore,
    IndexedRow,
    LanceDBStore,
    StoreRole,
    parse_store_config,
)

PHYSICAL = "demo__dev__context"
pytestmark = pytest.mark.e2e


def _spec(*, vector_search: str = "exact") -> CollectionSpec:
    return CollectionSpec(
        logical_name="context",
        physical_name=PHYSICAL,
        id_field="chunk_id",
        text_fields=("text",),
        full_text_fields=("text",),
        attribute_fields=("category",),
        scalar_index_fields=("category",),
        display_fields=(),
        vector_field="embedding",
        vector_dimensions=2,
        distance_metric="cosine",
        vector_search=vector_search,
        vector_index=None,
        config_fingerprint="fingerprint",
        descriptor="{}",
        legacy_config_fingerprint="legacy",
        row_fingerprint="row-fp",
        arrow_schema=pa.schema(
            [
                pa.field("chunk_id", pa.string(), nullable=False),
                pa.field("text", pa.string()),
                pa.field("category", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), 2)),
            ]
        ),
    )


def _rows(count: int = 40) -> list[IndexedRow]:
    return [
        IndexedRow(
            f"c{i}",
            {
                "chunk_id": f"c{i}",
                "text": f"alpha beta gamma document {i}",
                "category": f"cat{i % 3}",
                "embedding": [0.1 * (i % 7), 0.9],
            },
            f"f{i}",
        )
        for i in range(count)
    ]


def _lancedb(tmp_path: Path) -> LanceDBStore:
    return LanceDBStore(
        parse_store_config({"type": "lancedb", "path": str(tmp_path / "lance")}),
        project_name="demo",
        target_name="dev",
        alias="primary",
        role=StoreRole.PUBLISH,
    )


def _published(store: LanceDBStore, spec: CollectionSpec) -> None:
    store.create_collection(spec)
    store.upsert(PHYSICAL, _rows(), id_field="chunk_id", mutation_digest="d")
    store.ensure_indexes(spec)


def _index_count(store: LanceDBStore) -> int:
    table = store._open_owned_table(PHYSICAL)
    return len(list(table.list_indices()))


# ─── lancedb ────────────────────────────────────────────────────────────────


def test_every_index_is_dropped_and_counted(tmp_path: Path) -> None:
    with _lancedb(tmp_path) as store:
        _published(store, _spec())
        before = _index_count(store)
        assert before > 0, "precondition: ensure_indexes built something to drop"

        dropped = store.drop_indexes(PHYSICAL)

        assert dropped == before
        assert _index_count(store) == 0


def test_dropping_twice_is_idempotent(tmp_path: Path) -> None:
    # The caller is resuming and cannot know what the earlier attempt got as
    # far as building, so a second drop has to be a no-op rather than an error.
    with _lancedb(tmp_path) as store:
        _published(store, _spec())
        store.drop_indexes(PHYSICAL)
        assert store.drop_indexes(PHYSICAL) == 0


def test_a_collection_with_no_indices_is_not_an_error(tmp_path: Path) -> None:
    # The common case for a generation that died before its index step --
    # which is exactly the generation a resume adopts.
    with _lancedb(tmp_path) as store:
        store.create_collection(_spec())
        store.upsert(PHYSICAL, _rows(), id_field="chunk_id", mutation_digest="d")
        assert store.drop_indexes(PHYSICAL) == 0


def test_ensure_indexes_puts_back_what_the_drop_removed(tmp_path: Path) -> None:
    """The round trip the fix depends on.

    Dropping is only safe because the publish rebuilds afterwards. If
    `ensure_indexes` did not restore the full set, a resume would quietly
    publish a less-indexed collection than a fresh build does.
    """
    spec = _spec()
    with _lancedb(tmp_path) as store:
        _published(store, spec)
        before = {
            (tuple(index.columns), index.index_type)
            for index in store._open_owned_table(PHYSICAL).list_indices()
        }

        store.drop_indexes(PHYSICAL)
        store.ensure_indexes(spec)

        after = {
            (tuple(index.columns), index.index_type)
            for index in store._open_owned_table(PHYSICAL).list_indices()
        }
    assert after == before


# ─── duckdb ─────────────────────────────────────────────────────────────────


def _duckdb(tmp_path: Path) -> DuckDBStore:
    return DuckDBStore(
        parse_store_config(
            {"type": "duckdb", "path": str(tmp_path / "retrieval.duckdb")}
        ),
        project_name="demo",
        target_name="dev",
        alias="primary",
        role=StoreRole.PUBLISH,
    )


def test_duckdb_drops_its_full_text_index_and_is_idempotent(tmp_path: Path) -> None:
    # The other in-tree store implements the same contract: a resume on
    # DuckDB-backed retrieval gets the same treatment, and a second call is
    # still a no-op.
    spec = _spec()
    with _duckdb(tmp_path) as store:
        store.create_collection(spec)
        store.upsert(PHYSICAL, _rows(), id_field="chunk_id", mutation_digest="d")
        store.ensure_indexes(spec)

        assert store.drop_indexes(PHYSICAL) >= 1
        assert store.drop_indexes(PHYSICAL) == 0


def test_duckdb_drop_on_an_unindexed_collection_is_not_an_error(
    tmp_path: Path,
) -> None:
    # `PRAGMA drop_fts_index` has no `IF EXISTS` and raises on a table that
    # was never indexed, so the guard is load-bearing rather than tidiness.
    with _duckdb(tmp_path) as store:
        store.create_collection(_spec())
        assert store.drop_indexes(PHYSICAL) == 0
