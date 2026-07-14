from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest

from dbt_ml.adapters import (
    AdapterCapabilityError,
    AdapterError,
    UnknownAdapterError,
    WarehouseCapability,
    adapter_capabilities,
    create_adapter,
    list_adapter_types,
    parse_warehouse_config,
)
from dbt_ml.config.profile import WarehouseConfig


def _wh(path: Path, schema: str = "testns") -> WarehouseConfig:
    return parse_warehouse_config(
        {"type": "duckdb", "path": str(path), "schema": schema}
    )


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
    } <= capabilities


def test_unknown_type_raises(tmp_path: Path) -> None:
    with pytest.raises(UnknownAdapterError, match="no_such_warehouse"):
        parse_warehouse_config(
            {"type": "no_such_warehouse", "path": str(tmp_path / "x"), "schema": "s"}
        )


def test_duckdb_creates_schema_and_state(tmp_path: Path) -> None:
    cfg = _wh(tmp_path / "t.duckdb")
    with create_adapter(cfg) as adapter:
        # state table is in the configured schema
        cnt = adapter.scalar(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'testns' AND table_name = 'dbt_ml_state'"
        )
        assert cnt == 1


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
            "m1",
            [("doc-1", "hash-a", "v1"), ("doc-2", "hash-b", "v1")],
        )
        assert adapter.fetch_state("m1") == {
            "doc-1": ("hash-a", "v1"),
            "doc-2": ("hash-b", "v1"),
        }
        adapter.upsert_state("m1", [("doc-1", "hash-a2", "v2")])
        s = adapter.fetch_state("m1")
        assert s["doc-1"] == ("hash-a2", "v2")
        assert len(s) == 2


def test_state_persists_across_sessions(tmp_path: Path) -> None:
    cfg = _wh(tmp_path / "t.duckdb")
    with create_adapter(cfg) as adapter:
        adapter.upsert_state("m1", [("doc-1", "h", "v")])
    with create_adapter(cfg) as adapter:
        assert adapter.fetch_state("m1") == {"doc-1": ("h", "v")}


def test_clear_model_state(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.upsert_state("m1", [("doc-1", "h", "v")])
        adapter.upsert_state("m2", [("doc-1", "h", "v")])
        adapter.clear_model_state("m1")
        assert adapter.fetch_state("m1") == {}
        assert adapter.fetch_state("m2") == {"doc-1": ("h", "v")}


def test_catalog_schema_collision(tmp_path: Path) -> None:
    """Filename matches schema name (both 'dbt_ml') — used to break in v0.1."""
    with create_adapter(_wh(tmp_path / "dbt_ml.duckdb", schema="dbt_ml")) as adapter:
        adapter.upsert_state("m1", [("doc-1", "h", "v")])
        assert adapter.fetch_state("m1") == {"doc-1": ("h", "v")}


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
    adapter2 = create_adapter(cfg)
    assert hasattr(adapter2, "_reset_storage_for_test")
    out = adapter2._reset_storage_for_test()  # type: ignore[attr-defined]
    assert "t.duckdb" in out
    assert not (tmp_path / "t.duckdb").exists()


def test_outside_context_raises(tmp_path: Path) -> None:
    adapter = create_adapter(_wh(tmp_path / "t.duckdb"))
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
            "WHERE table_schema = 'testns' AND table_name LIKE 'dbt_ml_staging%'"
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
            "WHERE table_schema = 'testns' AND table_name LIKE 'dbt_ml_staging%'"
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
