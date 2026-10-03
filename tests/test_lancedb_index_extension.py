"""An index behind on rows is extended, not retrained (issue #619).

`create_index` replaces. An incremental publish that added tens of thousands
of rows to a collection of millions was therefore retraining every index over
the whole corpus, which is both the slow half of the publish and the half that
exhausts Lance's memory pool (issue #636). `Table.optimize()` adds the new rows
to the indices that already exist -- measured 8-14x cheaper for a 10,000-row
increment onto bases of 50,000 to 200,000 rows.

What these pin is the boundary. Extension is for an index that is merely
behind; an index that has to change *shape* -- a vector index of a newly
declared type, or an ANN index under `exact`, which is implemented by its
absence (issue #461) -- must still reach the build path. And the extension is
an optimization, so a failure falls back to the rebuild rather than taking the
publish with it, without leaking native text on the way (issue #490).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from stel.retrieval import (
    CollectionSpec,
    IndexedRow,
    LanceDBConfig,
    LanceDBStore,
    RetrievalError,
    StoreRole,
)

SENTINEL = "gs://distinctive-bucket/prefix?token=distinctive-native-secret"
POOL_REFUSAL = (
    "Resources exhausted: Failed to allocate additional 4194304 bytes for "
    "ExternalSorterMerge, 1048576 bytes remain available for the total pool"
)
PHYSICAL = "demo__dev__context"


def _config(tmp_path: Path) -> LanceDBConfig:
    return LanceDBConfig.model_validate({"type": "lancedb", "path": str(tmp_path / "lance")})


def _store(tmp_path: Path) -> LanceDBStore:
    return LanceDBStore(
        _config(tmp_path),
        project_name="demo",
        target_name="dev",
        alias="primary",
        role=StoreRole.PUBLISH,
    )


def _spec(
    *,
    scalar_index_fields: tuple[str, ...] = ("category",),
    full_text_fields: tuple[str, ...] = (),
    vector_search: str = "exact",
    vector_index: str | None = None,
) -> CollectionSpec:
    return CollectionSpec(
        logical_name="context",
        physical_name=PHYSICAL,
        id_field="chunk_id",
        text_fields=("text",),
        full_text_fields=full_text_fields,
        attribute_fields=("category",),
        scalar_index_fields=scalar_index_fields,
        display_fields=(),
        vector_field="embedding",
        vector_dimensions=2,
        distance_metric="cosine",
        vector_search=vector_search,
        vector_index=vector_index,
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


def _rows(start: int, count: int) -> list[IndexedRow]:
    return [
        IndexedRow(
            f"c{index}",
            {
                "chunk_id": f"c{index}",
                "text": f"document body {index}",
                "category": f"cat-{index % 3}",
                "embedding": [index / 100, 1 - index / 100],
            },
            f"f{index}",
        )
        for index in range(start, start + count)
    ]


class _SpyTable:
    """A real table that records the maintenance calls made against it."""

    def __init__(self, table: Any, *, optimize_raises: Exception | None = None) -> None:
        self._table = table
        self._optimize_raises = optimize_raises
        self.built: list[str] = []
        self.dropped: list[str] = []
        self.optimized = 0

    def create_index(self, column: str, *args: Any, **kwargs: Any) -> Any:
        self.built.append(column)
        return self._table.create_index(column, *args, **kwargs)

    def drop_index(self, name: str) -> Any:
        self.dropped.append(name)
        return self._table.drop_index(name)

    def optimize(self, *args: Any, **kwargs: Any) -> Any:
        self.optimized += 1
        if self._optimize_raises is not None:
            raise self._optimize_raises
        return self._table.optimize(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._table, name)


def _spy(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> dict[str, _SpyTable]:
    original = LanceDBStore._open_owned_table
    seen: dict[str, _SpyTable] = {}

    def open_spy(self: LanceDBStore, name: str) -> Any:
        if "table" not in seen:
            seen["table"] = _SpyTable(original(self, name), **kwargs)
        return seen["table"]

    monkeypatch.setattr(LanceDBStore, "_open_owned_table", open_spy)
    return seen


def _unindexed(store: LanceDBStore) -> int:
    table = store._open_owned_table(PHYSICAL)
    return sum(index.num_unindexed_rows or 0 for index in table.list_indices())


# ─── the extension ──────────────────────────────────────────────────────────


def test_an_index_behind_on_rows_is_extended_rather_than_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="stel.retrieval.lancedb")
    spec = _spec(full_text_fields=("text",))
    with _store(tmp_path) as store:
        store.create_collection(spec)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(spec)
        store.append(PHYSICAL, _rows(4, 4), id_field="chunk_id", mutation_digest="d1")
        assert _unindexed(store) == 8  # four new rows, two indices

        seen = _spy(monkeypatch)
        metadata = store.ensure_indexes(spec)

        assert metadata.row_count == 8
        assert _unindexed(store) == 0
        # The whole point: the rows were absorbed without retraining either
        # index over the collection.
        assert seen["table"].built == []
        assert seen["table"].optimized == 1

    assert any(
        "extended 2 index(es) over the rows added" in record.getMessage()
        for record in caplog.records
    )


def test_an_unchanged_rerun_does_not_optimize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun that wrote nothing stays the metadata check it was (PR #486).

    `optimize()` compacts and prunes as well as indexing, so calling it where
    no index is behind would add minutes of object-store work to a publish
    that had nothing to do.
    """
    spec = _spec()
    with _store(tmp_path) as store:
        store.create_collection(spec)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(spec)

        seen = _spy(monkeypatch)
        store.ensure_indexes(spec)

        assert seen["table"].optimized == 0
        assert seen["table"].built == []


def test_a_missing_index_is_built_while_a_behind_one_is_extended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    with _store(tmp_path) as store:
        store.create_collection(spec)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(spec)
        store.append(PHYSICAL, _rows(4, 4), id_field="chunk_id", mutation_digest="d1")

        seen = _spy(monkeypatch)
        # The same collection, now also declaring a full-text index.
        store.ensure_indexes(_spec(full_text_fields=("text",)))

        assert seen["table"].optimized == 1
        # Only the index that did not exist yet; the BTree was extended.
        assert seen["table"].built == ["text"]
        assert _unindexed(store) == 0


def test_an_approximate_vector_index_is_extended_over_the_new_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index the publish actually spends its time on, and the one whose
    retrain exhausts the memory pool (issue #636)."""
    spec = _spec(vector_search="approximate", vector_index="ivf_hnsw_flat")
    with _store(tmp_path) as store:
        store.create_collection(spec)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(spec)
        store.append(PHYSICAL, _rows(4, 4), id_field="chunk_id", mutation_digest="d1")

        seen = _spy(monkeypatch)
        store.ensure_indexes(spec)

        assert seen["table"].built == []
        assert seen["table"].optimized == 1
        assert _unindexed(store) == 0


# ─── the boundary: shape changes still build ────────────────────────────────


def test_a_newly_declared_vector_index_type_is_rebuilt_not_extended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _spec(vector_search="approximate", vector_index="ivf_hnsw_flat")
    with _store(tmp_path) as store:
        store.create_collection(built)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(built)
        store.append(PHYSICAL, _rows(4, 4), id_field="chunk_id", mutation_digest="d1")

        seen = _spy(monkeypatch)
        declared = _spec(vector_search="approximate", vector_index="ivf_hnsw_sq")
        store.ensure_indexes(declared)

        # Extension cannot change an index's type, so this one must not have
        # been absorbed into the old shape.
        assert seen["table"].optimized == 0
        assert seen["table"].built == ["category", "embedding"]


def test_switching_to_exact_still_drops_the_ann_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`exact` is the absence of an ANN index, so extending one would serve
    approximate results under a configuration promising exact ones (#461)."""
    approximate = _spec(vector_search="approximate", vector_index="ivf_hnsw_flat")
    with _store(tmp_path) as store:
        store.create_collection(approximate)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(approximate)
        store.append(PHYSICAL, _rows(4, 4), id_field="chunk_id", mutation_digest="d1")

        seen = _spy(monkeypatch)
        store.ensure_indexes(_spec(vector_search="exact"))

        assert seen["table"].optimized == 0
        assert [name for name in seen["table"].dropped] == ["embedding_idx"]


# ─── failure ────────────────────────────────────────────────────────────────


def test_a_pool_exhausted_extension_is_refused_without_falling_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rebuild it would fall back to sorts strictly more rows through the
    same pool, so there is nothing to fall back to (issue #636)."""
    spec = _spec()
    with _store(tmp_path) as store:
        store.create_collection(spec)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(spec)
        store.append(PHYSICAL, _rows(4, 4), id_field="chunk_id", mutation_digest="d1")

        seen = _spy(monkeypatch, optimize_raises=RuntimeError(POOL_REFUSAL))
        with pytest.raises(RetrievalError) as refused:
            store.ensure_indexes(spec)

    message = str(refused.value)
    assert "code=lancedb_index_pool_exhausted" in message
    assert "index extension" in message
    assert "LANCE_MEM_POOL_SIZE" in message
    assert seen["table"].built == []
    assert "ExternalSorterMerge" not in message


def test_a_failed_extension_rebuilds_instead_and_leaks_no_native_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="stel.retrieval.lancedb")
    spec = _spec()
    with _store(tmp_path) as store:
        store.create_collection(spec)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(spec)
        store.append(PHYSICAL, _rows(4, 4), id_field="chunk_id", mutation_digest="d1")

        seen = _spy(monkeypatch, optimize_raises=RuntimeError(SENTINEL))
        metadata = store.ensure_indexes(spec)

        assert metadata.row_count == 8
        # The publish is not lost: the index is retrained the old way.
        assert seen["table"].built == ["category"]
        assert _unindexed(store) == 0

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert [record.getMessage() for record in warnings] == [
        "LanceDB could not extend the indices on context [RuntimeError]; "
        "they are rebuilt instead"
    ]
    for record in caplog.records:
        if record.levelno >= logging.INFO:
            assert "distinctive" not in record.getMessage()
    # The native text is reachable only through the DEBUG record's exc_info.
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG and r.exc_info]
    assert any(SENTINEL in str(record.exc_info[1]) for record in debug if record.exc_info)
