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

The cost of extending is drift, so the store also records what the vector
index was trained on and how many vectors extension has placed since, and the
publish says so once that passes the largest extension measured (issue #683).
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
from stel.retrieval.servability import index_extension_advisory

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


# ─── the training record (issue #683) ───────────────────────────────────────


def _training(store: LanceDBStore) -> tuple[int | None, int | None]:
    metadata = store.inspect_collection(PHYSICAL)
    assert metadata is not None
    return metadata.vector_index_trained_rows, metadata.vector_index_extended_rows


def test_a_build_records_its_training_and_each_extension_adds_to_it(
    tmp_path: Path,
) -> None:
    """An updated vector counts as well as a new one: an upsert rewrites the
    row, and extension assigns it to centroids the index already had."""
    spec = _spec(vector_search="approximate", vector_index="ivf_hnsw_flat")
    with _store(tmp_path) as store:
        store.create_collection(spec)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(spec)
        assert _training(store) == (4, 0)

        store.upsert(PHYSICAL, _rows(0, 2), id_field="chunk_id", mutation_digest="d1")
        store.append(PHYSICAL, _rows(4, 2), id_field="chunk_id", mutation_digest="d2")
        store.ensure_indexes(spec)
        assert _training(store) == (4, 4)

        store.append(PHYSICAL, _rows(6, 3), id_field="chunk_id", mutation_digest="d3")
        store.ensure_indexes(spec)
        assert _training(store) == (4, 7)


def test_a_rebuild_starts_the_record_over(tmp_path: Path) -> None:
    built = _spec(vector_search="approximate", vector_index="ivf_hnsw_flat")
    with _store(tmp_path) as store:
        store.create_collection(built)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(built)
        store.append(PHYSICAL, _rows(4, 4), id_field="chunk_id", mutation_digest="d1")
        store.ensure_indexes(built)
        assert _training(store) == (4, 4)

        store.append(PHYSICAL, _rows(8, 2), id_field="chunk_id", mutation_digest="d2")
        store.ensure_indexes(_spec(vector_search="approximate", vector_index="ivf_hnsw_sq"))

        assert _training(store) == (10, 0)


def test_an_index_trained_before_the_record_counts_from_its_first_extension(
    tmp_path: Path,
) -> None:
    """A collection published before #683 carries no stamp. Waiting for its
    next full build would leave exactly the collections that only ever extend
    unmeasured, so the baseline is what the index held before this extension."""
    spec = _spec(vector_search="approximate", vector_index="ivf_hnsw_flat")
    with _store(tmp_path) as store:
        store.create_collection(spec)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(spec)
        store._open_owned_table(PHYSICAL).update_field_metadata(
            {
                "path": "chunk_id",
                "metadata": {
                    "stel.vector_index_trained_rows": "",
                    "stel.vector_index_extended_rows": "",
                },
            }
        )
        assert _training(store) == (None, None)

        store.append(PHYSICAL, _rows(4, 2), id_field="chunk_id", mutation_digest="d1")
        store.ensure_indexes(spec)

        assert _training(store) == (4, 2)


def test_a_restamp_keeps_the_record(tmp_path: Path) -> None:
    """Every publish that completes its rows re-stamps the id field (#508).
    Lance merges field metadata rather than replacing it; if that ever
    changed, the record would vanish on the publish after the build."""
    spec = _spec(vector_search="approximate", vector_index="ivf_hnsw_flat")
    with _store(tmp_path) as store:
        store.create_collection(spec)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(spec)

        store.restamp_collection(spec)

        assert _training(store) == (4, 0)


def test_exact_search_carries_no_record(tmp_path: Path) -> None:
    """`exact` is the absence of an ANN index (#461); a record left behind
    would describe an index that no longer exists."""
    approximate = _spec(vector_search="approximate", vector_index="ivf_hnsw_flat")
    exact = _spec(vector_search="exact")
    with _store(tmp_path) as store:
        store.create_collection(approximate)
        store.append(PHYSICAL, _rows(0, 4), id_field="chunk_id", mutation_digest="d0")
        store.ensure_indexes(approximate)
        store.append(PHYSICAL, _rows(4, 4), id_field="chunk_id", mutation_digest="d1")
        store.ensure_indexes(exact)
        assert _training(store) == (None, None)

        # The BTree alone is extended from here on, which is no vector's training.
        store.append(PHYSICAL, _rows(8, 2), id_field="chunk_id", mutation_digest="d2")
        store.ensure_indexes(exact)
        assert _training(store) == (None, None)


# ─── the advisory (issue #683) ──────────────────────────────────────────────


def test_extension_within_the_measured_range_says_nothing() -> None:
    """30% is the largest extension measured on the prod corpus, and recall
    held there. The line is inclusive of what was measured."""
    assert (
        index_extension_advisory(
            model="m", collection="c", trained_rows=1_000, extended_rows=300
        )
        is None
    )


def test_extension_past_the_measured_range_names_the_retrain() -> None:
    advisory = index_extension_advisory(
        model="sec_chunk_search",
        collection="sec_chunks",
        trained_rows=3_667_730,
        extended_rows=1_200_000,
    )
    assert advisory is not None
    assert "trained on 3,667,730 rows" in advisory
    assert "1,200,000 more vectors" in advisory
    assert "(33% of that sample)" in advisory
    assert "issue #683" in advisory
    assert "`stel run --select sec_chunk_search --full-refresh`" in advisory


@pytest.mark.parametrize(
    ("trained", "extended"), [(None, None), (None, 10), (10, None), (0, 10)]
)
def test_no_record_means_no_advisory(trained: int | None, extended: int | None) -> None:
    assert (
        index_extension_advisory(
            model="m", collection="c", trained_rows=trained, extended_rows=extended
        )
        is None
    )
