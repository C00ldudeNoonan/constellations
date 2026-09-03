from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

from stel.adapters import (
    AdapterCapabilityError,
    AdapterConfigError,
    AdapterError,
    StateRecord,
    StateScope,
    StateValue,
    UnknownAdapterError,
    WarehouseCapability,
    adapter_capabilities,
    create_adapter,
    list_adapter_types,
    parse_warehouse_config,
)
from stel.config.profile import WarehouseConfig


def _wh(path: Path, schema: str = "testns") -> WarehouseConfig:
    return parse_warehouse_config(
        {"type": "duckdb", "path": str(path), "schema": schema}
    )


def _scope(
    model_name: str = "m1",
    *,
    stage: str = "materialization",
    target_identity: str = "warehouse-v1",
) -> StateScope:
    return StateScope(model_name, stage, target_identity)


def _state(record_key: str, fingerprint: str, version: str) -> StateRecord:
    return StateRecord(record_key, fingerprint, version)


def test_registered_types() -> None:
    assert "duckdb" in list_adapter_types()


def test_duckdb_declares_core_workflow_capabilities() -> None:
    capabilities = adapter_capabilities("duckdb")

    assert {
        WarehouseCapability.SQL_QUERIES,
        WarehouseCapability.TABULAR_READS,
        WarehouseCapability.SQL_SCHEMA_TESTS,
        WarehouseCapability.ATOMIC_FULL_REPLACE,
        WarehouseCapability.ATOMIC_KEYED_UPSERT,
        WarehouseCapability.TRANSACTIONS,
        WarehouseCapability.TYPED_EMPTY_RELATIONS,
        WarehouseCapability.CHUNKED_WRITES,
        WarehouseCapability.SCHEMA_EVOLUTION,
        WarehouseCapability.STREAMING_TABULAR_READS,
        WarehouseCapability.TABULAR_PREDICATE_PUSHDOWN,
    } <= capabilities


def test_unknown_type_raises(tmp_path: Path) -> None:
    with pytest.raises(UnknownAdapterError, match="no_such_warehouse"):
        parse_warehouse_config(
            {"type": "no_such_warehouse", "path": str(tmp_path / "x"), "schema": "s"}
        )


@pytest.mark.parametrize(
    "raw",
    [
        {
            "type": "bigquery",
            "project": "p",
            "token": "distinctive-protected-input-secret",
        },
        {
            "type": "no_such_warehouse",
            "token": "distinctive-protected-input-secret",
        },
    ],
)
def test_adapter_entrypoint_errors_scrub_raw_traceback_inputs(
    raw: dict[str, str],
) -> None:
    sentinel = "distinctive-protected-input-secret"

    with pytest.raises(AdapterError) as exc_info:
        parse_warehouse_config(raw)

    error = exc_info.value
    assert sentinel not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        if "/src/stel/" in traceback.tb_frame.f_code.co_filename:
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def test_config_validation_error_retains_only_sanitized_details(
    tmp_path: Path,
) -> None:
    sentinel = "distinctive-invalid-warehouse-secret"
    raw = {
        "type": "duckdb",
        "path": str(tmp_path / "x.duckdb"),
        "pth_typo": sentinel,
    }

    with pytest.raises(AdapterConfigError) as exc_info:
        parse_warehouse_config(raw)

    error = exc_info.value
    rendered = repr((error, error.validation_details))
    assert sentinel not in rendered
    assert raw["pth_typo"] == sentinel
    assert error.validation_details == (
        {
            "loc": ("pth_typo",),
            "msg": "Extra inputs are not permitted",
            "type": "extra_forbidden",
        },
    )
    assert error.__cause__ is None
    assert error.__context__ is None

    traceback = error.__traceback__
    while traceback is not None:
        if "/src/stel/" in traceback.tb_frame.f_code.co_filename:
            assert sentinel not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def test_duckdb_creates_schema_and_state(tmp_path: Path) -> None:
    cfg = _wh(tmp_path / "t.duckdb")
    with create_adapter(cfg) as adapter:
        # state table is in the configured schema
        cnt = adapter.scalar(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'testns' AND table_name = 'stel_state'"
        )
        assert cnt == 1
        columns = adapter.rows(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'testns' AND table_name = 'stel_state' "
            "ORDER BY ordinal_position"
        )
        assert [row[0] for row in columns] == [
            "model_name",
            "state_scope",
            "target_identity",
            "record_key",
            "input_fingerprint",
            "code_version",
            "last_run_at",
        ]


def test_list_tables_excludes_failures_tables(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full("model_a", pl.DataFrame({"x": [1]}))
        adapter.materialize_full(
            "stel_test_failures__model_a__not_null__x", pl.DataFrame({"x": [1]})
        )
        tables = adapter.list_tables()
        assert "model_a" in tables
        assert all(not t.startswith("stel_test_failures__") for t in tables)


def test_typed_table_reads_support_limits_and_counts(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full("widgets", pl.DataFrame({"x": [1, 2, 3]}))

        assert adapter.read_table("widgets", limit=2).to_dicts() == [
            {"x": 1},
            {"x": 2},
        ]
        assert adapter.row_count("widgets") == 3


def test_typed_table_read_reports_missing_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = create_adapter(_wh(tmp_path / "t.duckdb"))
    monkeypatch.setattr(type(adapter), "capabilities", classmethod(lambda cls: frozenset()))

    with pytest.raises(AdapterCapabilityError, match="tabular_reads"):
        adapter.read_table("widgets")


def test_state_upsert_and_fetch(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.upsert_state(
            _scope(),
            [_state("doc-1", "hash-a", "v1"), _state("doc-2", "hash-b", "v1")],
        )
        assert adapter.fetch_state(_scope()) == {
            "doc-1": StateValue("hash-a", "v1"),
            "doc-2": StateValue("hash-b", "v1"),
        }
        adapter.upsert_state(_scope(), [_state("doc-1", "hash-a2", "v2")])
        s = adapter.fetch_state(_scope())
        assert s["doc-1"] == StateValue("hash-a2", "v2")
        assert len(s) == 2


def test_state_persists_across_sessions(tmp_path: Path) -> None:
    cfg = _wh(tmp_path / "t.duckdb")
    with create_adapter(cfg) as adapter:
        adapter.upsert_state(_scope(), [_state("doc-1", "h", "v")])
    with create_adapter(cfg) as adapter:
        assert adapter.fetch_state(_scope()) == {"doc-1": StateValue("h", "v")}


def test_state_scope_and_target_isolation(tmp_path: Path) -> None:
    materialized = _scope()
    published_a = _scope(stage="search-publication", target_identity="target-a")
    published_b = _scope(stage="search-publication", target_identity="target-b")
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.upsert_state(materialized, [_state("shared", "warehouse", "v1")])
        adapter.upsert_state(published_a, [_state("shared", "index-a", "v1")])
        adapter.upsert_state(published_b, [_state("shared", "index-b", "v1")])

        assert adapter.fetch_state(materialized)["shared"].input_fingerprint == ("warehouse")
        assert adapter.fetch_state(published_a)["shared"].input_fingerprint == ("index-a")
        assert adapter.fetch_state(published_b)["shared"].input_fingerprint == ("index-b")


def test_target_descriptor_scope_is_stable_across_mapping_order() -> None:
    first = StateScope.for_target_descriptor(
        "chunks",
        stage="search-publication",
        descriptor={
            "index": {"name": "economic-data", "dimensions": 1536},
            "filters": ["tenant", "access_groups"],
        },
    )
    reordered = StateScope.for_target_descriptor(
        "chunks",
        stage="search-publication",
        descriptor={
            "filters": ["tenant", "access_groups"],
            "index": {"dimensions": 1536, "name": "economic-data"},
        },
    )

    assert first == reordered


def test_non_materialization_scope_requires_explicit_target_identity() -> None:
    with pytest.raises(ValueError, match="explicit target_identity"):
        StateScope("chunks", stage="search-publication")


def test_target_descriptor_change_isolates_state(tmp_path: Path) -> None:
    first = StateScope.for_target_descriptor(
        "chunks",
        stage="search-publication",
        descriptor={"index": "economic-data", "dimensions": 1536},
    )
    changed = StateScope.for_target_descriptor(
        "chunks",
        stage="search-publication",
        descriptor={"index": "economic-data", "dimensions": 3072},
    )
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.upsert_state(first, [_state("chunk-1", "first", "v1")])
        adapter.upsert_state(changed, [_state("chunk-1", "changed", "v1")])

        assert first.target_identity != changed.target_identity
        assert adapter.fetch_state(first) == {"chunk-1": StateValue("first", "v1")}
        assert adapter.fetch_state(changed) == {"chunk-1": StateValue("changed", "v1")}


def test_target_descriptor_scope_stores_only_fingerprint(tmp_path: Path) -> None:
    raw_values = ("semantic-target-canary", "tenant-namespace-canary")
    scope = StateScope.for_target_descriptor(
        "chunks",
        stage="search-publication",
        descriptor={"index": raw_values[0], "namespace": raw_values[1]},
    )
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.upsert_state(scope, [_state("chunk-1", "input", "v1")])
        stored = adapter.rows(f"SELECT target_identity FROM {adapter.table_ref('stel_state')}")

    assert stored == [(scope.target_identity,)]
    assert len(scope.target_identity) == 32
    int(scope.target_identity, 16)
    assert all(value not in scope.target_identity for value in raw_values)


def test_replace_state_replaces_only_exact_scope(tmp_path: Path) -> None:
    first = _scope(target_identity="target-a")
    other = _scope(target_identity="target-b")
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.upsert_state(
            first,
            [_state("a", "ha", "v1"), _state("b", "hb", "v1")],
        )
        adapter.upsert_state(other, [_state("a", "other", "v1")])

        adapter.replace_state(first, [_state("c", "hc", "v2")])

        assert adapter.fetch_state(first) == {"c": StateValue("hc", "v2")}
        assert adapter.fetch_state(other) == {"a": StateValue("other", "v1")}


def test_duckdb_migrates_v1_state_without_discarding_rows(tmp_path: Path) -> None:
    path = tmp_path / "legacy.duckdb"
    connection = duckdb.connect(str(path))
    try:
        connection.execute("CREATE SCHEMA testns")
        connection.execute(
            """
            CREATE TABLE testns.stel_state (
                model_name VARCHAR NOT NULL,
                document_id VARCHAR NOT NULL,
                content_hash VARCHAR NOT NULL,
                code_version VARCHAR NOT NULL,
                last_run_at TIMESTAMP NOT NULL,
                PRIMARY KEY (model_name, document_id)
            )
            """
        )
        connection.execute(
            "INSERT INTO testns.stel_state VALUES "
            "('m1', 'doc-1', 'hash-a', 'v1', TIMESTAMP '2026-07-01 12:00:00'), "
            "('m1', 'doc-2', 'hash-b', 'v1', TIMESTAMP '2026-07-02 12:00:00')"
        )
        connection.execute(
            "CREATE TABLE testns.stel_state__migration_v2 (note VARCHAR)"
        )
        connection.execute(
            "INSERT INTO testns.stel_state__migration_v2 VALUES ('user-owned')"
        )
    finally:
        connection.close()

    with create_adapter(_wh(path)) as adapter:
        assert adapter.fetch_state(_scope()) == {
            "doc-1": StateValue("hash-a", "v1"),
            "doc-2": StateValue("hash-b", "v1"),
        }
        timestamps = adapter.rows(
            f"SELECT record_key, last_run_at FROM {adapter.table_ref('stel_state')} "
            "ORDER BY record_key"
        )
        assert [row[1].isoformat() for row in timestamps] == [
            "2026-07-01T12:00:00",
            "2026-07-02T12:00:00",
        ]
        assert adapter.rows(
            f"SELECT note FROM {adapter.table_ref('stel_state__migration_v2')}"
        ) == [("user-owned",)]
        staging = adapter.rows(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'testns' "
            "AND table_name LIKE 'stel_staging__state_migration_v2__%'"
        )
        assert staging == []


def test_duckdb_rejects_unknown_state_schema(tmp_path: Path) -> None:
    path = tmp_path / "unknown.duckdb"
    connection = duckdb.connect(str(path))
    try:
        connection.execute("CREATE SCHEMA testns")
        connection.execute("CREATE TABLE testns.stel_state (model_name VARCHAR NOT NULL)")
    finally:
        connection.close()

    adapter = create_adapter(_wh(path))
    with pytest.raises(AdapterError, match="Unsupported stel_state schema"):
        adapter.__enter__()
    with pytest.raises(AdapterError, match="context manager"):
        adapter.scalar("SELECT 1")

    reopened = duckdb.connect(str(path))
    try:
        assert reopened.execute("SELECT COUNT(*) FROM testns.stel_state").fetchone() == (
            0,
        )
    finally:
        reopened.close()


def test_clear_state_is_scoped(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.upsert_state(_scope("m1"), [_state("doc-1", "h", "v")])
        adapter.upsert_state(_scope("m2"), [_state("doc-1", "h", "v")])
        adapter.clear_state(_scope("m1"))
        assert adapter.fetch_state(_scope("m1")) == {}
        assert adapter.fetch_state(_scope("m2")) == {"doc-1": StateValue("h", "v")}


def test_catalog_schema_collision(tmp_path: Path) -> None:
    """Filename matches schema name — used to break in v0.1."""
    with create_adapter(_wh(tmp_path / "collide.duckdb", schema="collide")) as adapter:
        adapter.upsert_state(_scope(), [_state("doc-1", "h", "v")])
        assert adapter.fetch_state(_scope()) == {"doc-1": StateValue("h", "v")}


def test_materialize_full(tmp_path: Path) -> None:
    df = pl.DataFrame({"document_id": ["a", "b"], "x": [1, 2]})
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        n = adapter.materialize_full("widgets", df)
        assert n == 2
        rows = adapter.rows(
            f"SELECT document_id, x FROM {adapter.table_ref('widgets')} "
            "ORDER BY document_id"
        )
        assert rows == [("a", 1), ("b", 2)]


def test_materialize_incremental_upserts(tmp_path: Path) -> None:
    cfg = _wh(tmp_path / "t.duckdb")
    with create_adapter(cfg) as adapter:
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["a", "b"], "x": [1, 2]}),
            key_col="document_id",
        )
        # Re-upsert doc 'a' with a different x; doc 'b' unchanged
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["a"], "x": [99]}),
            key_col="document_id",
        )
        rows = adapter.rows(
            f"SELECT document_id, x FROM {adapter.table_ref('widgets')} ORDER BY document_id"
        )
        assert rows == [("a", 99), ("b", 2)]


def test_update_when_changed_skips_unchanged_rows(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "docs",
            pl.DataFrame(
                {"document_id": ["a", "b"], "fp": ["v1", "v1"], "payload": ["A", "B"]}
            ),
            key_col="document_id",
        )
        # Same fingerprint for 'a' but a different payload: the no-op guard must
        # leave the original payload untouched. 'b' vanishes from this batch and
        # must remain. 'c' is new and inserted.
        written = adapter.materialize_incremental(
            "docs",
            pl.DataFrame(
                {"document_id": ["a", "c"], "fp": ["v1", "v9"], "payload": ["X", "C"]}
            ),
            key_col="document_id",
            update_when_changed=["fp"],
        )
        rows = adapter.rows(
            f"SELECT document_id, fp, payload FROM {adapter.table_ref('docs')} "
            "ORDER BY document_id"
        )
        # 'a' unchanged fingerprint → keeps A (not X); 'b' retained; 'c' inserted.
        assert rows == [("a", "v1", "A"), ("b", "v1", "B"), ("c", "v9", "C")]
        assert written == 2


def test_update_when_changed_rewrites_changed_rows(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "docs",
            pl.DataFrame({"document_id": ["a"], "fp": ["v1"], "payload": ["A"]}),
            key_col="document_id",
        )
        adapter.materialize_incremental(
            "docs",
            pl.DataFrame({"document_id": ["a"], "fp": ["v2"], "payload": ["Z"]}),
            key_col="document_id",
            update_when_changed=["fp"],
        )
        rows = adapter.rows(
            f"SELECT document_id, fp, payload FROM {adapter.table_ref('docs')}"
        )
        assert rows == [("a", "v2", "Z")]


def test_update_when_changed_is_null_safe(tmp_path: Path) -> None:
    null_fp = pl.Series("fp", [None], dtype=pl.String)
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "docs",
            pl.DataFrame({"document_id": ["a"], "payload": ["A"]}).with_columns(null_fp),
            key_col="document_id",
        )
        # NULL fp unchanged → no-op (NULL IS NOT DISTINCT FROM NULL).
        adapter.materialize_incremental(
            "docs",
            pl.DataFrame({"document_id": ["a"], "payload": ["X"]}).with_columns(null_fp),
            key_col="document_id",
            update_when_changed=["fp"],
        )
        assert adapter.rows(
            f"SELECT payload FROM {adapter.table_ref('docs')}"
        ) == [("A",)]
        # NULL → value counts as changed → rewrite.
        adapter.materialize_incremental(
            "docs",
            pl.DataFrame({"document_id": ["a"], "fp": ["v1"], "payload": ["Z"]}),
            key_col="document_id",
            update_when_changed=["fp"],
        )
        assert adapter.rows(
            f"SELECT payload FROM {adapter.table_ref('docs')}"
        ) == [("Z",)]


def test_update_when_changed_rejects_unknown_column(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "docs",
            pl.DataFrame({"document_id": ["a"], "payload": ["A"]}),
            key_col="document_id",
        )
        with pytest.raises(AdapterError, match="update_when_changed column 'fp'"):
            adapter.materialize_incremental(
                "docs",
                pl.DataFrame({"document_id": ["a"], "payload": ["B"]}),
                key_col="document_id",
                update_when_changed=["fp"],
            )


@pytest.mark.parametrize(
    ("df", "message"),
    [
        (pl.DataFrame({"x": [1]}), "missing required key"),
        (pl.DataFrame({"document_id": [None], "x": [1]}), "contains 1 NULL"),
        (
            pl.DataFrame({"document_id": ["a", "a"], "x": [1, 2]}),
            "contains 1 duplicate",
        ),
    ],
)
def test_incremental_rejects_invalid_keys(
    tmp_path: Path, df: pl.DataFrame, message: str
) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        with pytest.raises(AdapterError, match=message):
            adapter.materialize_incremental("widgets", df, key_col="document_id")
        assert "widgets" not in adapter.list_tables()


@pytest.mark.parametrize("policy", ["fail", "ignore", "append_new_columns"])
def test_incremental_rejects_existing_target_without_key(
    tmp_path: Path, policy: str
) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full("widgets", pl.DataFrame({"x": [1]}))

        with pytest.raises(AdapterError, match=r"target.*missing key"):
            adapter.materialize_incremental(
                "widgets",
                pl.DataFrame({"document_id": ["a"], "x": [2]}),
                key_col="document_id",
                on_schema_change=policy,
            )

        assert adapter.rows(f"SELECT * FROM {adapter.table_ref('widgets')}") == [
            (1,)
        ]


def test_incremental_failure_rolls_back_delete(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["a", "b"], "x": [1, 2]}),
            key_col="document_id",
        )

        with pytest.raises(duckdb.ConversionException):
            adapter.materialize_incremental(
                "widgets",
                pl.DataFrame({"document_id": ["a"], "x": ["not-an-int"]}),
                key_col="document_id",
            )

        rows = adapter.rows(
            f"SELECT document_id, x FROM {adapter.table_ref('widgets')} "
            "ORDER BY document_id"
        )
        assert rows == [("a", 1), ("b", 2)]


def test_delete_rows_and_state_is_atomic_and_scoped(tmp_path: Path) -> None:
    scope = _scope(target_identity="target-a")
    other = _scope(target_identity="target-b")
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["a", "b"], "x": [1, 2]}),
            key_col="document_id",
        )
        adapter.upsert_state(
            scope,
            [_state("a", "ha", "v1"), _state("b", "hb", "v1")],
        )
        adapter.upsert_state(other, [_state("a", "other", "v1")])

        deleted = adapter.delete_rows_and_state(
            "widgets",
            key_col="document_id",
            keys=["a"],
            state_scope=scope,
        )

        assert deleted == 1
        assert adapter.rows(
            f"SELECT document_id, x FROM {adapter.table_ref('widgets')}"
        ) == [("b", 2)]
        assert adapter.fetch_state(scope) == {"b": StateValue("hb", "v1")}
        assert adapter.fetch_state(other) == {"a": StateValue("other", "v1")}


def test_delete_rows_and_state_validates_before_mutation(tmp_path: Path) -> None:
    scope = _scope()
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["a"], "x": [1]}),
            key_col="document_id",
        )
        adapter.upsert_state(scope, [_state("a", "ha", "v1")])

        with pytest.raises(AdapterError, match="duplicate"):
            adapter.delete_rows_and_state(
                "widgets",
                key_col="document_id",
                keys=["a", "a"],
                state_scope=scope,
            )

        assert adapter.row_count("widgets") == 1
        assert adapter.fetch_state(scope) == {"a": StateValue("ha", "v1")}


def test_delete_rows_and_state_rolls_back_target_when_state_delete_fails(
    tmp_path: Path,
) -> None:
    scope = _scope()
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["a"], "x": [1]}),
            key_col="document_id",
        )
        adapter.upsert_state(scope, [_state("a", "ha", "v1")])
        adapter.execute(
            f"ALTER TABLE {adapter.table_ref('stel_state')} "
            "RENAME TO stel_state_unavailable"
        )

        with pytest.raises(duckdb.CatalogException):
            adapter.delete_rows_and_state(
                "widgets",
                key_col="document_id",
                keys=["a"],
                state_scope=scope,
            )

        assert adapter.row_count("widgets") == 1


def test_list_tables_excludes_state(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full(
            "first", pl.DataFrame({"x": [1]})
        )
        adapter.materialize_full(
            "second", pl.DataFrame({"x": [1]})
        )
        names = adapter.list_tables()
        assert "stel_state" not in names
        assert set(names) == {"first", "second"}


def test_drop_table(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full("x", pl.DataFrame({"a": [1]}))
        assert "x" in adapter.list_tables()
        adapter.drop_table("x")
        assert "x" not in adapter.list_tables()


def test_adapter_test_reset_deletes_duckdb_file(tmp_path: Path) -> None:
    cfg = _wh(tmp_path / "t.duckdb")
    with create_adapter(cfg) as adapter:
        adapter.materialize_full("x", pl.DataFrame({"a": [1]}))
    adapter2: Any = create_adapter(cfg)
    assert hasattr(adapter2, "_reset_storage_for_test")
    out = adapter2._reset_storage_for_test()
    assert "t.duckdb" in out
    assert not (tmp_path / "t.duckdb").exists()


def test_outside_context_raises(tmp_path: Path) -> None:
    adapter: Any = create_adapter(_wh(tmp_path / "t.duckdb"))
    with pytest.raises(AdapterError):
        adapter.connection  # noqa: B018


def test_incremental_insert_matches_columns_by_name(tmp_path: Path) -> None:
    """Same columns in a different order must land by name, not position."""
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["a"], "x": [1], "y": ["one"]}),
            key_col="document_id",
        )
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"y": ["two"], "x": [2], "document_id": ["b"]}),
            key_col="document_id",
        )
        rows = adapter.rows(
            f"SELECT document_id, x, y FROM {adapter.table_ref('widgets')} "
            "ORDER BY document_id"
        )
        assert rows == [("a", 1, "one"), ("b", 2, "two")]


def test_schema_change_fails_by_default(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["a"], "x": [1]}),
            key_col="document_id",
        )
        with pytest.raises(AdapterError, match="full-refresh"):
            adapter.materialize_incremental(
                "widgets",
                pl.DataFrame({"document_id": ["b"], "x": [2], "extra": ["?"]}),
                key_col="document_id",
            )
        with pytest.raises(AdapterError, match="removed columns"):
            adapter.materialize_incremental(
                "widgets",
                pl.DataFrame({"document_id": ["b"]}),
                key_col="document_id",
            )
        # failed attempts wrote nothing
        rows = adapter.rows(f"SELECT * FROM {adapter.table_ref('widgets')}")
        assert rows == [("a", 1)]


def test_schema_change_append_new_columns(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["a"], "x": [1]}),
            key_col="document_id",
        )
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["b"], "x": [2], "extra": ["new"]}),
            key_col="document_id",
            on_schema_change="append_new_columns",
        )
        rows = adapter.rows(
            f"SELECT document_id, x, extra FROM {adapter.table_ref('widgets')} "
            "ORDER BY document_id"
        )
        assert rows == [("a", 1, None), ("b", 2, "new")]


def test_schema_change_ignore_drops_new_columns(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["a"], "x": [1]}),
            key_col="document_id",
        )
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["b"], "x": [2], "extra": ["dropped"]}),
            key_col="document_id",
            on_schema_change="ignore",
        )
        rows = adapter.rows(
            f"SELECT document_id, x FROM {adapter.table_ref('widgets')} "
            "ORDER BY document_id"
        )
        assert rows == [("a", 1), ("b", 2)]
        cols = [
            r[0]
            for r in adapter.rows(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'testns' AND table_name = 'widgets'"
            )
        ]
        assert "extra" not in cols


def test_schema_change_unknown_policy_rejected(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_incremental(
            "widgets",
            pl.DataFrame({"document_id": ["a"], "x": [1]}),
            key_col="document_id",
        )
        with pytest.raises(AdapterError, match="Unknown on_schema_change"):
            adapter.materialize_incremental(
                "widgets",
                pl.DataFrame({"document_id": ["b"], "x": [2], "extra": ["?"]}),
                key_col="document_id",
                on_schema_change="sync_all",
            )


# ─── materialize_full_chunks (issue #77) ────────────────────────────────────


def test_full_chunks_concatenates_and_replaces(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full("widgets", pl.DataFrame({"old": ["stale"]}))
        total = adapter.materialize_full_chunks(
            "widgets",
            iter(
                [
                    pl.DataFrame({"document_id": ["a", "b"], "x": [1, 2]}),
                    pl.DataFrame({"document_id": ["c"], "x": [3]}),
                ]
            ),
        )
        assert total == 3
        rows = adapter.rows(
            f"SELECT document_id, x FROM {adapter.table_ref('widgets')} "
            "ORDER BY document_id"
        )
        assert rows == [("a", 1), ("b", 2), ("c", 3)]
        assert "stel_staging__widgets" not in adapter.rows(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'testns'"
        )


def test_full_chunks_unions_intra_run_drift(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full_chunks(
            "widgets",
            iter(
                [
                    pl.DataFrame({"document_id": ["a"], "x": [1]}),
                    # chunk 2 adds a column and drops one
                    pl.DataFrame({"document_id": ["b"], "extra": ["?"]}),
                ]
            ),
        )
        rows = adapter.rows(
            f"SELECT document_id, x, extra FROM {adapter.table_ref('widgets')} "
            "ORDER BY document_id"
        )
        assert rows == [("a", 1, None), ("b", None, "?")]


def test_full_chunks_failure_drops_staging_keeps_target(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full("widgets", pl.DataFrame({"x": [1]}))

        def _chunks():
            yield pl.DataFrame({"x": [2]})
            raise RuntimeError("extraction blew up mid-stream")

        with pytest.raises(RuntimeError, match="mid-stream"):
            adapter.materialize_full_chunks("widgets", _chunks())

        # target untouched, staging cleaned up
        assert adapter.rows(f"SELECT x FROM {adapter.table_ref('widgets')}") == [(1,)]
        staging = adapter.rows(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'testns' AND table_name LIKE 'stel_staging%'"
        )
        assert staging == []


def test_full_chunks_swap_failure_rolls_back_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full("widgets", pl.DataFrame({"x": [1]}))

        def fail_rename(staging_ref: str, table: str) -> None:
            raise RuntimeError(f"cannot rename {staging_ref} to {table}")

        monkeypatch.setattr(adapter, "_rename_staging", fail_rename)
        with pytest.raises(RuntimeError, match="cannot rename"):
            adapter.materialize_full_chunks(
                "widgets", iter([pl.DataFrame({"x": [2]})])
            )

        assert adapter.rows(f"SELECT x FROM {adapter.table_ref('widgets')}") == [(1,)]
        staging = adapter.rows(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'testns' AND table_name LIKE 'stel_staging%'"
        )
        assert staging == []


def test_full_chunks_empty_iterator_drops_target(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full("widgets", pl.DataFrame({"x": [1]}))
        total = adapter.materialize_full_chunks("widgets", iter([]))
        assert total == 0
        assert "widgets" not in adapter.list_tables()


def test_full_chunks_typed_empty_frame_creates_relation(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        frame = pl.DataFrame(
            schema={
                "document_id": pl.String,
                "count": pl.Int64,
                "score": pl.Float64,
                "active": pl.Boolean,
            }
        )
        assert adapter.materialize_full_chunks("widgets", iter([frame])) == 0
        assert adapter.scalar(
            f"SELECT COUNT(*) FROM {adapter.table_ref('widgets')}"
        ) == 0
        assert adapter.rows(
            f"DESCRIBE {adapter.table_ref('widgets')}"
        )[:4] == [
            ("document_id", "VARCHAR", "YES", None, None, None),
            ("count", "BIGINT", "YES", None, None, None),
            ("score", "DOUBLE", "YES", None, None, None),
            ("active", "BOOLEAN", "YES", None, None, None),
        ]


def test_list_tables_excludes_staging_tables(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full("model_a", pl.DataFrame({"x": [1]}))
        adapter.materialize_full("stel_staging__model_a", pl.DataFrame({"x": [1]}))
        assert adapter.list_tables() == ["model_a"]


# ─── DuckDB memory bounding (issue #412) ────────────────────────────────────
#
# DuckDB sizes its buffer pool from *host* RAM. Inside a container that is the
# wrong number in the dangerous direction: the cgroup ceiling is invisible to
# it, so it grows past the limit the kernel actually kills at, and a read that
# is bounded on stel's side still OOMs the process.


def _duckdb_setting(adapter: Any, name: str) -> str:
    return str(
        adapter.connection.execute(f"SELECT current_setting('{name}')").fetchone()[0]
    )


def _duckdb_cursor_setting(adapter: Any, name: str) -> str:
    """Read a setting from a fresh cursor session rather than the connection."""
    cursor = adapter.connection.cursor()
    row = cursor.execute(f"SELECT current_setting('{name}')").fetchone()
    assert row is not None
    return str(row[0])


def test_an_explicit_memory_limit_reaches_duckdb(tmp_path: Path) -> None:
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "w.duckdb"), "memory_limit": "512MB"}
    )
    with create_adapter(config) as adapter:
        assert "MiB" in _duckdb_setting(adapter, "memory_limit")


def test_the_memory_limit_reaches_cursors_too(tmp_path: Path) -> None:
    """`connection.cursor()` opens a fresh session, which is how the TimeZone
    pin was missed once already (#339). `memory_limit` is GLOBAL so it carries
    — asserted rather than assumed, because the streaming reads that most need
    the bound all run on cursors."""
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "w.duckdb"), "memory_limit": "512MB"}
    )
    with create_adapter(config) as adapter:
        parent = _duckdb_setting(adapter, "memory_limit")
        found = _duckdb_cursor_setting(adapter, "memory_limit")
    assert found == parent
    assert "MiB" in found


def test_memory_limit_none_leaves_duckdb_unbounded(tmp_path: Path) -> None:
    """The explicit opt-out: an operator who wants DuckDB's own sizing, and no
    detection either, has to be able to say so.

    Compared against a raw `duckdb.connect()` rather than against another
    adapter: on a cgroup-constrained test runner the other adapter would detect
    the runner's own ceiling and legitimately differ, making this pass or fail
    on the host rather than on the behavior.
    """
    unbounded = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "a.duckdb"), "memory_limit": "none"}
    )
    with create_adapter(unbounded) as adapter:
        opted_out = _duckdb_setting(adapter, "memory_limit")
    raw = duckdb.connect()
    try:
        row = raw.execute("SELECT current_setting('memory_limit')").fetchone()
        assert row is not None
    finally:
        raw.close()
    assert opted_out == str(row[0])


def test_a_malformed_memory_limit_is_rejected_at_config_time(tmp_path: Path) -> None:
    """Not at connect, from inside the native driver, partway into a run that
    already resolved credentials."""
    with pytest.raises(AdapterConfigError, match="memory_limit"):
        parse_warehouse_config(
            {
                "type": "duckdb",
                "path": str(tmp_path / "w.duckdb"),
                "memory_limit": "as much as it takes",
            }
        )


def test_temp_directory_reaches_duckdb(tmp_path: Path) -> None:
    """A bounded DuckDB spills; it needs somewhere to spill to."""
    spill = tmp_path / "spill"
    config = parse_warehouse_config(
        {
            "type": "duckdb",
            "path": str(tmp_path / "w.duckdb"),
            "memory_limit": "512MB",
            "temp_directory": str(spill),
        }
    )
    with create_adapter(config) as adapter:
        assert _duckdb_setting(adapter, "temp_directory") == str(spill)


def _fake_cgroup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    contents: str,
    v2: bool = True,
    physical: int | None = 64 * 1024**3,
) -> None:
    """Stand in for the cgroup mount, which no test can create for real."""
    from stel import memory as memory_module

    present = tmp_path / ("memory.max" if v2 else "memory.limit_in_bytes")
    present.write_text(contents, encoding="utf-8")
    absent = tmp_path / "absent"
    monkeypatch.setattr(memory_module, "_CGROUP_V2_MAX", present if v2 else absent)
    monkeypatch.setattr(memory_module, "_CGROUP_V1_MAX", absent if v2 else present)
    monkeypatch.setattr(memory_module, "physical_memory_bytes", lambda: physical)


def test_a_container_ceiling_bounds_duckdb_without_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The incident shape: an operator who never heard of `memory_limit` gets a
    DuckDB that respects the cgroup instead of the host."""
    from stel.adapters.duckdb import _detected_memory_limit

    _fake_cgroup(monkeypatch, tmp_path, contents=str(4 * 1024**3))
    # 75% of a 4GiB ceiling, leaving the rest for the Python process.
    assert _detected_memory_limit() == "3072MiB"


def test_cgroup_v1_is_read_when_v2_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from stel.adapters.duckdb import _detected_memory_limit

    _fake_cgroup(monkeypatch, tmp_path, contents=str(8 * 1024**3), v2=False)
    assert _detected_memory_limit() == "6144MiB"


@pytest.mark.parametrize(
    ("contents", "physical", "why"),
    [
        ("max", 64 * 1024**3, "v2 spells unlimited as a word"),
        ("9223372036854771712", 64 * 1024**3, "v1 spells it as a sentinel"),
        (str(64 * 1024**3), 64 * 1024**3, "a ceiling at physical RAM is no ceiling"),
        (str(128 * 1024**3), 64 * 1024**3, "nor is one above it"),
        ("0", 64 * 1024**3, "a zero ceiling is not a real one"),
        (str(4 * 1024**3), None, "no way to tell whether it is a constraint"),
    ],
)
def test_detection_declines_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contents: str,
    physical: int | None,
    why: str,
) -> None:
    """Every case where the ceiling is absent, unlimited, or not actually a
    constraint leaves DuckDB's own sizing alone — detection is advisory and
    must never invent a limit."""
    from stel.adapters.duckdb import _detected_memory_limit

    _fake_cgroup(monkeypatch, tmp_path, contents=contents, physical=physical)
    assert _detected_memory_limit() is None, why


@pytest.mark.parametrize(
    ("ceiling", "expected"),
    [
        (4 * 1024**3, "3072MiB"),
        (512 * 1024**2, "384MiB"),
        (256 * 1024**2, "192MiB"),
        (32 * 1024**2, "24MiB"),
    ],
)
def test_a_small_container_is_still_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ceiling: int, expected: str
) -> None:
    """The containers that need this most must not be the ones that miss out.

    An earlier revision declined below ~683MiB on the theory that a tiny limit
    would only make DuckDB thrash — which left a 256MiB cgroup with the
    host-sized default, i.e. exactly the OOM this exists to prevent. Slow beats
    killed, and DuckDB accepts limits down to 16MiB.
    """
    from stel.adapters.duckdb import _detected_memory_limit

    _fake_cgroup(monkeypatch, tmp_path, contents=str(ceiling))
    assert _detected_memory_limit() == expected


def test_an_explicit_limit_wins_over_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_cgroup(monkeypatch, tmp_path, contents=str(4 * 1024**3))
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "w.duckdb"), "memory_limit": "700MB"}
    )
    with create_adapter(config) as adapter:
        # 700MB, not the 3072MiB detection would have chosen.
        assert _duckdb_setting(adapter, "memory_limit").startswith("667")


def test_none_opts_out_of_detection_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`none` means "leave DuckDB alone", not "fall through to detection" —
    otherwise there would be no way to get host-sized behavior in a container."""
    _fake_cgroup(monkeypatch, tmp_path, contents=str(4 * 1024**3))
    opted_out = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "a.duckdb"), "memory_limit": "none"}
    )
    with create_adapter(opted_out) as adapter:
        assert "3072" not in _duckdb_setting(adapter, "memory_limit")


def test_detection_applies_at_connect_with_no_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole point of #412: an unconfigured project in a 4GiB container gets
    a DuckDB bounded by the cgroup rather than by 80% of the host's RAM."""
    _fake_cgroup(monkeypatch, tmp_path, contents=str(4 * 1024**3))
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "w.duckdb")}
    )
    with create_adapter(config) as adapter:
        assert _duckdb_setting(adapter, "memory_limit").startswith("3.0 GiB")


def test_no_container_means_no_change_in_behavior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Outside a container nothing is detected, so an ordinary workstation run
    keeps exactly the DuckDB sizing it had before this change."""
    from stel import memory as memory_module

    absent = tmp_path / "absent"
    monkeypatch.setattr(memory_module, "_CGROUP_V2_MAX", absent)
    monkeypatch.setattr(memory_module, "_CGROUP_V1_MAX", absent)
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "w.duckdb")}
    )
    with create_adapter(config) as adapter:
        bounded = _duckdb_setting(adapter, "memory_limit")
    raw = duckdb.connect()
    try:
        row = raw.execute("SELECT current_setting('memory_limit')").fetchone()
        assert row is not None
        native = str(row[0])
    finally:
        raw.close()
    assert bounded == native


def test_a_relative_temp_directory_resolves_against_the_project(tmp_path: Path) -> None:
    """The same rule `path` follows. DuckDB creates the directory lazily — only
    on an actual spill — so a CWD-relative reading would not surface until a
    large run finally needed it, somewhere the operator did not put it."""
    project = tmp_path / "project"
    project.mkdir()
    from stel.adapters.duckdb import DuckDBWarehouseConfig

    config = parse_warehouse_config(
        {
            "type": "duckdb",
            "path": "./target/w.duckdb",
            "memory_limit": "512MB",
            "temp_directory": "./target/spill",
        }
    ).absolutize(project)
    assert isinstance(config, DuckDBWarehouseConfig)
    assert config.temp_directory == (project / "target" / "spill").resolve()
    with create_adapter(config) as adapter:
        assert _duckdb_setting(adapter, "temp_directory") == str(config.temp_directory)
