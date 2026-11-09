"""`replace_children` adds columns and reprocesses parents in one transaction.

Reprocessing a parent deletes its child rows; `on_schema_change:
append_new_columns` adds the columns the new rows carry. DuckDB refuses to alter
a table that the same transaction has already modified, so the delete has to
come after the alter. The failure surfaced as `Attempting to modify table ...
but another transaction has altered this table` at commit, and only when a run
both reprocessed an existing parent and grew the schema, which is why a run of
brand-new documents never hit it.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from stel.adapters import (
    AdapterError,
    StateRecord,
    StateScope,
    create_adapter,
    parse_warehouse_config,
)
from stel.adapters.duckdb import DuckDBAdapter

_SCOPE = StateScope("words")


def _adapter(tmp_path: Path) -> DuckDBAdapter:
    adapter = create_adapter(
        parse_warehouse_config(
            {"type": "duckdb", "path": str(tmp_path / "w.duckdb"), "schema": "main"}
        )
    )
    assert isinstance(adapter, DuckDBAdapter)
    return adapter


def _children(*rows: tuple[str, str, str], extra: str | None = None) -> pl.DataFrame:
    """(word_id, document_id, word); `extra` adds a column carrying that value."""
    frame = pl.DataFrame(
        {
            "word_id": [row[0] for row in rows],
            "document_id": [row[1] for row in rows],
            "word": [row[2] for row in rows],
        }
    )
    if extra is not None:
        frame = frame.with_columns(pl.lit(extra).alias("shout"))
    return frame


def _replace(
    adapter: DuckDBAdapter,
    parents: list[str],
    rows: pl.DataFrame,
    on_schema_change: str,
) -> int:
    return adapter.replace_children(
        "words",
        parent_key="document_id",
        parent_ids=parents,
        child_key="word_id",
        new_rows=rows,
        state_scope=_SCOPE,
        state_records=[StateRecord(parent, "fingerprint", "code") for parent in parents],
        on_schema_change=on_schema_change,
    )


def _seed(adapter: DuckDBAdapter) -> None:
    """Two documents, three words, no `shout` column."""
    _replace(
        adapter,
        [],
        _children(("a1", "a", "alpha"), ("a2", "a", "again"), ("b1", "b", "beta")),
        "fail",
    )


def test_reprocessing_a_parent_while_adding_a_column_commits(tmp_path: Path) -> None:
    """The reprocessed parent's rows are replaced and carry the new column; the
    parent that was not reprocessed keeps its rows, with null in the new one."""
    with _adapter(tmp_path) as adapter:
        _seed(adapter)

        _replace(
            adapter,
            ["a"],
            _children(("a1", "a", "ALPHA"), extra="yes"),
            "append_new_columns",
        )

        table = adapter.read_table("words").sort("word_id")
        assert table.columns == ["word_id", "document_id", "word", "shout"]
        assert table.select("word_id", "word", "shout").rows() == [
            ("a1", "ALPHA", "yes"),
            ("b1", "beta", None),
        ]


def test_the_refused_schema_change_leaves_the_table_as_it_was(tmp_path: Path) -> None:
    """`fail` still refuses, and the parent delete that precedes the refusal in
    the transaction rolls back with it: no rows lost, no column added."""
    with _adapter(tmp_path) as adapter:
        _seed(adapter)

        with pytest.raises(AdapterError, match="Schema change"):
            _replace(adapter, ["a"], _children(("a1", "a", "ALPHA"), extra="yes"), "fail")

        table = adapter.read_table("words").sort("word_id")
        assert table.columns == ["word_id", "document_id", "word"]
        assert table["word"].to_list() == ["alpha", "again", "beta"]
