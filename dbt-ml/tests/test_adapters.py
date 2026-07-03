from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from dbt_ml.adapters import (
    AdapterError,
    UnknownAdapterError,
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


def test_clean_deletes_duckdb_file(tmp_path: Path) -> None:
    cfg = _wh(tmp_path / "t.duckdb")
    with create_adapter(cfg) as adapter:
        adapter.materialize_full("x", pl.DataFrame({"a": [1]}))
    adapter2 = create_adapter(cfg)
    out = adapter2.clean()
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
