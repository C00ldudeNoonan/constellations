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
            "WHERE table_schema = 'testns' AND table_name = 'dbt_ml_state'"
        )
        assert cnt == 1
        columns = adapter.rows(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'testns' AND table_name = 'dbt_ml_state' "
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
            "dbt_ml_test_failures__model_a__not_null__x", pl.DataFrame({"x": [1]})
        )
        tables = adapter.list_tables()
        assert "model_a" in tables
        assert all(not t.startswith("dbt_ml_test_failures__") for t in tables)


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
        stored = adapter.rows(f"SELECT target_identity FROM {adapter.table_ref('dbt_ml_state')}")

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
            CREATE TABLE testns.dbt_ml_state (
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
            "INSERT INTO testns.dbt_ml_state VALUES "
            "('m1', 'doc-1', 'hash-a', 'v1', TIMESTAMP '2026-07-01 12:00:00'), "
            "('m1', 'doc-2', 'hash-b', 'v1', TIMESTAMP '2026-07-02 12:00:00')"
        )
        connection.execute(
            "CREATE TABLE testns.dbt_ml_state__migration_v2 (note VARCHAR)"
        )
        connection.execute(
            "INSERT INTO testns.dbt_ml_state__migration_v2 VALUES ('user-owned')"
        )
    finally:
        connection.close()

    with create_adapter(_wh(path)) as adapter:
        assert adapter.fetch_state(_scope()) == {
            "doc-1": StateValue("hash-a", "v1"),
            "doc-2": StateValue("hash-b", "v1"),
        }
        timestamps = adapter.rows(
            f"SELECT record_key, last_run_at FROM {adapter.table_ref('dbt_ml_state')} "
            "ORDER BY record_key"
        )
        assert [row[1].isoformat() for row in timestamps] == [
            "2026-07-01T12:00:00",
            "2026-07-02T12:00:00",
        ]
        assert adapter.rows(
            f"SELECT note FROM {adapter.table_ref('dbt_ml_state__migration_v2')}"
        ) == [("user-owned",)]
        staging = adapter.rows(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'testns' "
            "AND table_name LIKE 'dbt_ml_staging__state_migration_v2__%'"
        )
        assert staging == []


def test_duckdb_rejects_unknown_state_schema(tmp_path: Path) -> None:
    path = tmp_path / "unknown.duckdb"
    connection = duckdb.connect(str(path))
    try:
        connection.execute("CREATE SCHEMA testns")
        connection.execute("CREATE TABLE testns.dbt_ml_state (model_name VARCHAR NOT NULL)")
    finally:
        connection.close()

    adapter = create_adapter(_wh(path))
    with pytest.raises(AdapterError, match="Unsupported dbt_ml_state schema"):
        adapter.__enter__()
    with pytest.raises(AdapterError, match="context manager"):
        adapter.scalar("SELECT 1")

    reopened = duckdb.connect(str(path))
    try:
        assert reopened.execute("SELECT COUNT(*) FROM testns.dbt_ml_state").fetchone() == (
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
            f"ALTER TABLE {adapter.table_ref('dbt_ml_state')} "
            "RENAME TO dbt_ml_state_unavailable"
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
        assert "dbt_ml_state" not in names
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
        assert "dbt_ml_staging__widgets" not in adapter.rows(
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
        adapter.materialize_full("dbt_ml_staging__model_a", pl.DataFrame({"x": [1]}))
        assert adapter.list_tables() == ["model_a"]
