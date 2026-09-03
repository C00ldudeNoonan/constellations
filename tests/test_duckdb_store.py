"""DuckDB-native retrieval store over the vss and fts extensions (issue #371).

These run against a real DuckDB file rather than a mock. The whole point of
this store is what the engine does with the SQL it is handed — a test that
asserts on generated statement text would pass while the store returned
nothing.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from stel.retrieval.base import (
    CollectionSpec,
    IndexedRow,
    RetrievalError,
    RetrievalFeature,
    RetrievalPredicate,
    RetrievalPredicateOperator,
)
from stel.retrieval.duckdb import DuckDBConfig, DuckDBStore

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("body", pa.string()),
        pa.field("tenant_id", pa.string()),
        pa.field("access_groups", pa.list_(pa.string())),
        pa.field("embedding", pa.list_(pa.float32(), 3)),
    ]
)


def _spec(name: str, *, vector_search: str = "exact") -> CollectionSpec:
    return CollectionSpec(
        logical_name="ctx",
        physical_name=name,
        id_field="id",
        text_fields=("body",),
        full_text_fields=("body",),
        attribute_fields=("tenant_id", "access_groups"),
        scalar_index_fields=(),
        display_fields=("body",),
        vector_field="embedding",
        vector_dimensions=3,
        distance_metric="cosine",
        vector_search=vector_search,
        config_fingerprint="cfg1",
        descriptor='{"distance_metric": "cosine"}',
        legacy_config_fingerprint="legacy1",
        arrow_schema=SCHEMA,
    )


def _rows() -> list[IndexedRow]:
    return [
        IndexedRow(
            "a",
            {
                "id": "a",
                "body": "inflation rose sharply in the third quarter",
                "tenant_id": "acme",
                "access_groups": ["analysts", "ops"],
                "embedding": [1.0, 0.0, 0.0],
            },
            "fp-a",
        ),
        IndexedRow(
            "b",
            {
                "id": "b",
                "body": "the labor market cooled",
                "tenant_id": "acme",
                "access_groups": ["admins"],
                "embedding": [0.0, 1.0, 0.0],
            },
            "fp-b",
        ),
        IndexedRow(
            "c",
            {
                "id": "c",
                "body": "tariffs and trade policy",
                "tenant_id": "globex",
                "access_groups": [],
                "embedding": [0.0, 0.0, 1.0],
            },
            "fp-c",
        ),
    ]


@pytest.fixture
def store(tmp_path: Path) -> DuckDBStore:
    config = DuckDBConfig(path=str(tmp_path / "store.duckdb"))
    return DuckDBStore(
        config, project_name="proj", target_name="dev", alias="default"
    )


def _populate(store: DuckDBStore, *, vector_search: str = "exact") -> str:
    name = store.physical_collection("ctx")
    spec = _spec(name, vector_search=vector_search)
    store.create_collection(spec)
    store.upsert(name, _rows(), id_field="id", mutation_digest="d1")
    store.ensure_indexes(spec)
    return name


# ─── lifecycle ──────────────────────────────────────────────────────────────


def test_create_upsert_and_inspect_round_trip(store: DuckDBStore) -> None:
    with store:
        name = _populate(store)

        metadata = store.inspect_collection(name)

        assert metadata is not None
        assert metadata.row_count == 3
        assert metadata.config_fingerprint == "cfg1"
        assert store.list_collections() == (name,)


def test_upsert_is_keyed_not_appending(store: DuckDBStore) -> None:
    with store:
        name = _populate(store)
        changed = IndexedRow(
            "a",
            {
                "id": "a",
                "body": "revised text",
                "tenant_id": "acme",
                "access_groups": ["analysts"],
                "embedding": [1.0, 0.0, 0.0],
            },
            "fp-a2",
        )

        store.upsert(name, [changed], id_field="id", mutation_digest="d2")

        metadata = store.inspect_collection(name)
        assert metadata is not None
        assert metadata.row_count == 3


def test_delete_removes_by_id(store: DuckDBStore) -> None:
    with store:
        name = _populate(store)

        receipt = store.delete(name, ["b"], id_field="id", mutation_digest="d3")

        assert receipt.acknowledged
        metadata = store.inspect_collection(name)
        assert metadata is not None
        assert metadata.row_count == 2


def test_an_unowned_table_is_never_adopted(store: DuckDBStore) -> None:
    """The stamp is an adoption gate. A table stel did not create must be
    refused rather than published into, because publishing overwrites."""
    with store:
        conn = store._connection()
        conn.execute("CREATE TABLE someone_elses(id VARCHAR)")

        assert "someone_elses" not in store.list_collections()
        with pytest.raises(RetrievalError, match="not stel-owned"):
            store.inspect_collection("someone_elses")


def test_schema_evolution_adds_a_column_in_place(store: DuckDBStore) -> None:
    with store:
        name = _populate(store)
        widened = pa.schema([*list(SCHEMA), pa.field("classification", pa.string())])
        spec = _spec(name).__class__(
            **{**_spec(name).__dict__, "arrow_schema": widened}
        )

        store.evolve_collection(spec, ["classification"])

        metadata = store.inspect_collection(name)
        assert metadata is not None
        assert "classification" in metadata.schema.names
        assert metadata.row_count == 3


# ─── queries ────────────────────────────────────────────────────────────────


def test_vector_search_ranks_by_distance(store: DuckDBStore) -> None:
    with store:
        name = _populate(store)

        table = store.vector_search(
            name, [1.0, 0.0, 0.0], vector_field="embedding", limit=2
        )

        assert table.column("id").to_pylist()[0] == "a"
        assert "_distance" in table.schema.names


def test_text_search_matches_bm25(store: DuckDBStore) -> None:
    with store:
        name = _populate(store)

        table = store.text_search(name, "inflation", text_field="body", limit=5)

        assert table.column("id").to_pylist() == ["a"]
        assert "_score" in table.schema.names


def test_text_search_returns_nothing_rather_than_everything_on_no_match(
    store: DuckDBStore,
) -> None:
    """BM25 scores NULL for a non-matching row, so the NOT NULL test is the
    match filter. Without it every row comes back with a null score."""
    with store:
        name = _populate(store)

        table = store.text_search(name, "zzzznotpresent", text_field="body", limit=5)

        assert table.num_rows == 0


# ─── policy filters, including the #397 array case ──────────────────────────


def test_scalar_policy_predicate_filters_both_legs(store: DuckDBStore) -> None:
    with store:
        name = _populate(store)
        predicate = RetrievalPredicate(
            "tenant_id", RetrievalPredicateOperator.EQUAL, "globex"
        )

        vector = store.vector_search(
            name,
            [1.0, 0.0, 0.0],
            vector_field="embedding",
            limit=5,
            predicates=[predicate],
        )

        assert vector.column("id").to_pylist() == ["c"]


def test_array_containment_predicate_filters_by_overlap(store: DuckDBStore) -> None:
    """The capability declared for issue #397, exercised against the engine."""
    with store:
        name = _populate(store)
        predicate = RetrievalPredicate(
            "access_groups",
            RetrievalPredicateOperator.ARRAY_CONTAINS_ANY,
            ("analysts", "nobody"),
        )

        table = store.vector_search(
            name,
            [0.0, 0.0, 1.0],
            vector_field="embedding",
            limit=5,
            predicates=[predicate],
        )

        assert table.column("id").to_pylist() == ["a"]


def test_store_declares_array_containment(store: DuckDBStore) -> None:
    assert (
        RetrievalFeature.ARRAY_CONTAINMENT_FILTERS in DuckDBStore.capabilities().features
    )


def test_hybrid_comes_from_both_legs_not_a_native_operator() -> None:
    """DuckDB has no operator blending vss and fts ranking, so hybrid is core's
    RRF over two legs. What the store owes that arrangement is both legs; there
    is no native-hybrid feature in the contract to claim or decline."""
    features = DuckDBStore.capabilities().features

    assert RetrievalFeature.EXACT_VECTOR_SEARCH in features
    assert RetrievalFeature.FULL_TEXT_SEARCH in features


# ─── the HNSW persistence decision ──────────────────────────────────────────


def test_approximate_index_is_refused_without_the_experimental_opt_in(
    store: DuckDBStore,
) -> None:
    """DuckDB will not persist an HNSW index without an experimental flag,
    because it is not WAL-covered. Refuse with a way forward rather than
    setting the flag on the operator's behalf."""
    with store:
        name = store.physical_collection("ctx")
        spec = _spec(name, vector_search="approximate")
        store.create_collection(spec)
        store.upsert(name, _rows(), id_field="id", mutation_digest="d1")

        with pytest.raises(RetrievalError, match="hnsw_persistence_disabled"):
            store.ensure_indexes(spec)


def test_exact_vector_search_needs_no_index_and_still_ranks(
    store: DuckDBStore,
) -> None:
    """The reason refusing the index is acceptable: correctness does not
    depend on it. Exact search returns the same rows, only slower."""
    with store:
        name = _populate(store, vector_search="exact")

        table = store.vector_search(
            name, [0.0, 1.0, 0.0], vector_field="embedding", limit=1
        )

        assert table.column("id").to_pylist() == ["b"]


def test_approximate_index_builds_when_the_opt_in_is_given(tmp_path: Path) -> None:
    config = DuckDBConfig(
        path=str(tmp_path / "hnsw.duckdb"), hnsw_experimental_persistence=True
    )
    opted_in = DuckDBStore(
        config, project_name="proj", target_name="dev", alias="default"
    )
    with opted_in:
        _populate(opted_in, vector_search="approximate")

        indexes = opted_in._connection().execute(
            "SELECT index_name FROM duckdb_indexes()"
        ).fetchall()

        assert any("hnsw" in str(row[0]) for row in indexes)


def test_switching_back_to_exact_takes_the_index_away(tmp_path: Path) -> None:
    """`exact` is implemented by the *absence* of the index — the planner uses
    one whenever it exists. Leaving a stale HNSW index behind after a switch
    back would keep answering approximately under a config that promises exact
    results, and nothing in the query path would say so (issue #461)."""
    config = DuckDBConfig(
        path=str(tmp_path / "hnsw.duckdb"), hnsw_experimental_persistence=True
    )
    opted_in = DuckDBStore(
        config, project_name="proj", target_name="dev", alias="default"
    )
    with opted_in:
        name = _populate(opted_in, vector_search="approximate")

        opted_in.ensure_indexes(_spec(name, vector_search="exact"))

        indexes = opted_in._connection().execute(
            "SELECT index_name FROM duckdb_indexes()"
        ).fetchall()

        assert not any("hnsw" in str(row[0]) for row in indexes)


# ─── configuration ──────────────────────────────────────────────────────────


def test_memory_path_is_refused() -> None:
    """An in-memory database cannot be published to and then read back by a
    separate serving process, and that failure would only surface at query
    time."""
    with pytest.raises(ValueError, match="cannot be served"):
        DuckDBConfig(path=":memory:")


def test_closing_the_store_leaves_a_shared_warehouse_file_usable(
    tmp_path: Path,
) -> None:
    """The case this store exists for: warehouse and search in one file. The
    store must never close the database out from under the warehouse adapter.
    """
    import duckdb

    path = tmp_path / "shared.duckdb"
    warehouse = duckdb.connect(str(path))
    warehouse.execute("CREATE TABLE canonical(x INT)")
    warehouse.execute("INSERT INTO canonical VALUES (1)")

    store = DuckDBStore(
        DuckDBConfig(path=str(path)),
        project_name="proj",
        target_name="dev",
        alias="default",
    )
    with store:
        _populate(store)

    assert warehouse.execute("SELECT count(*) FROM canonical").fetchone() == (1,)
    warehouse.execute("INSERT INTO canonical VALUES (2)")
    assert warehouse.execute("SELECT count(*) FROM canonical").fetchone() == (2,)
    warehouse.close()


# ─── concurrency ────────────────────────────────────────────────────────────


def test_a_database_held_by_another_process_is_a_distinct_error(
    tmp_path: Path,
) -> None:
    """DuckDB is single-writer per file across processes, and a concurrent
    publisher is an ordinary operational condition rather than a
    misconfiguration. It must not surface as a generic connect failure that
    sends the operator to check their profile.

    Uses a real second process: two connections inside one process share
    DuckDB's cached instance and do not contend at all, so an in-process test
    would assert nothing.
    """
    import subprocess
    import sys
    import textwrap

    database = tmp_path / "held.duckdb"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import duckdb, sys, time
                conn = duckdb.connect(r"{database}")
                conn.execute("CREATE TABLE t(x INT)")
                sys.stdout.write("ready")
                sys.stdout.flush()
                time.sleep(30)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.read(5) == "ready"
        store = DuckDBStore(
            DuckDBConfig(path=str(database)),
            project_name="proj",
            target_name="dev",
            alias="default",
        )

        with pytest.raises(RetrievalError, match="duckdb_database_locked"):
            with store:
                pass
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_error_text_never_carries_the_database_path(tmp_path: Path) -> None:
    """DuckDB's native message embeds the file path. Errors reach logs and
    artifacts, so the path must not ride along."""
    store = DuckDBStore(
        DuckDBConfig(path=str(tmp_path / "store.duckdb")),
        project_name="proj",
        target_name="dev",
        alias="default",
    )
    with store:
        name = _populate(store)
        # A predicate on a column that does not exist fails inside DuckDB, so
        # this exercises the wrapped-native-error path rather than one of the
        # store's own precondition checks.
        with pytest.raises(RetrievalError) as caught:
            store.vector_search(
                name,
                [1.0, 0.0, 0.0],
                vector_field="embedding",
                limit=1,
                predicates=[
                    RetrievalPredicate(
                        "no_such_column", RetrievalPredicateOperator.EQUAL, "x"
                    )
                ],
            )

    assert "duckdb_vector_search_failed" in str(caught.value)
    assert str(tmp_path) not in str(caught.value)
    assert "no_such_column" not in str(caught.value)


# ─── review follow-ups (PR #400) ────────────────────────────────────────────

MULTI_FIELD_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("body", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 3)),
    ]
)


def _multi_field_spec(name: str) -> CollectionSpec:
    return CollectionSpec(
        logical_name="ctx",
        physical_name=name,
        id_field="id",
        text_fields=("title", "body"),
        full_text_fields=("title", "body"),
        attribute_fields=(),
        scalar_index_fields=(),
        display_fields=("title", "body"),
        vector_field="embedding",
        vector_dimensions=3,
        distance_metric="cosine",
        vector_search="exact",
        config_fingerprint="cfg1",
        descriptor='{"distance_metric": "cosine"}',
        legacy_config_fingerprint="legacy1",
        arrow_schema=MULTI_FIELD_SCHEMA,
    )


def test_bm25_scores_only_the_requested_field(tmp_path: Path) -> None:
    """Core calls text_search once per declared full-text field and fuses each
    as its own RRF leg. Scoring the combined index on every call would enter
    one ranking under several labels and let RRF count a single match twice,
    so each leg has to be scoped to its own field.

    The single-field fixture elsewhere cannot catch this: with one field the
    combined index and the per-field index are the same index.
    """
    store = DuckDBStore(
        DuckDBConfig(path=str(tmp_path / "multi.duckdb")),
        project_name="proj",
        target_name="dev",
        alias="default",
    )
    with store:
        name = store.physical_collection("ctx")
        spec = _multi_field_spec(name)
        store.create_collection(spec)
        store.upsert(
            name,
            [
                IndexedRow(
                    "a",
                    {
                        "id": "a",
                        "title": "inflation report",
                        "body": "the labor market cooled",
                        "embedding": [1.0, 0.0, 0.0],
                    },
                    "fp-a",
                ),
                IndexedRow(
                    "b",
                    {
                        "id": "b",
                        "title": "labor digest",
                        "body": "tariffs rose",
                        "embedding": [0.0, 1.0, 0.0],
                    },
                    "fp-b",
                ),
            ],
            id_field="id",
            mutation_digest="d1",
        )
        store.ensure_indexes(spec)

        by_title = store.text_search(name, "labor", text_field="title", limit=5)
        by_body = store.text_search(name, "labor", text_field="body", limit=5)

        # "labor" is in b's title and a's body. Each leg must see only its own.
        assert by_title.column("id").to_pylist() == ["b"]
        assert by_body.column("id").to_pylist() == ["a"]


def test_lock_directory_does_not_move_when_the_parent_appears(
    tmp_path: Path,
) -> None:
    """The lock path must come from configuration alone.

    Entering the store creates the database's parent directory. If the lock
    location depended on whether that parent existed, the first publisher and
    the second would resolve different directories, lock different inodes, and
    both believe they held the fence — during first publication, exactly when
    a recovery is most likely to be running concurrently.
    """
    database = tmp_path / "not-created-yet" / "store.duckdb"
    store = DuckDBStore(
        DuckDBConfig(path=str(database)),
        project_name="proj",
        target_name="dev",
        alias="default",
    )

    before = store._lock_dir()
    with store:
        during = store._lock_dir()
    after = store._lock_dir()

    assert before == during == after


def test_two_publishers_contend_even_on_a_fresh_directory(tmp_path: Path) -> None:
    database = tmp_path / "fresh" / "store.duckdb"
    first = DuckDBStore(
        DuckDBConfig(path=str(database)),
        project_name="proj",
        target_name="dev",
        alias="default",
    )
    second = DuckDBStore(
        DuckDBConfig(path=str(database)),
        project_name="proj",
        target_name="dev",
        alias="default",
    )
    with first, second:
        collection = first.physical_collection("ctx")

        with first.publisher_fence(collection):
            with pytest.raises(RetrievalError, match="publisher_lock_held"):
                with second.publisher_fence(collection):
                    pass


def test_an_unknown_template_placeholder_is_a_configuration_error() -> None:
    """Rejected at the profile boundary rather than as a raw KeyError raised
    from inside a publish."""
    with pytest.raises(ValueError, match="only"):
        DuckDBConfig(
            path="./x.duckdb",
            collection_template="{project}__{target}__{collection}__{unknown}",
        )


def test_configured_hnsw_search_effort_reaches_the_connection(
    tmp_path: Path,
) -> None:
    """An accepted option that is never read leaves an operator tuning recall
    against a value with no effect."""
    store = DuckDBStore(
        DuckDBConfig(path=str(tmp_path / "ef.duckdb"), hnsw_ef_search=99),
        project_name="proj",
        target_name="dev",
        alias="default",
    )
    with store:
        row = store._connection().execute(
            "SELECT current_setting('hnsw_ef_search')"
        ).fetchone()

    assert row is not None and int(row[0]) == 99
