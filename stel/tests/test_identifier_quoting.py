"""Identifier quoting + name validation (issue #64).

Model/source names are validated to a conservative charset at config load;
column names stay free-form (extraction schemas produce them) and must be
safely quoted wherever they reach SQL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from stel.adapters import create_adapter, parse_warehouse_config
from stel.checks.schema import evaluate_test_spec
from stel.config.model import ModelConfig
from stel.config.profile import WarehouseConfig
from stel.config.source import SourceConfig


def _wh(path: Path) -> WarehouseConfig:
    return parse_warehouse_config(
        {"type": "duckdb", "path": str(path), "schema": "testns"}
    )


# ─── quote_ident ────────────────────────────────────────────────────────────


def test_quote_ident_plain(tmp_path: Path) -> None:
    adapter = create_adapter(_wh(tmp_path / "t.duckdb"))
    assert adapter.quote_ident("order") == '"order"'


def test_quote_ident_escapes_embedded_quotes(tmp_path: Path) -> None:
    adapter = create_adapter(_wh(tmp_path / "t.duckdb"))
    assert adapter.quote_ident('a"b') == '"a""b"'
    # An injection-shaped name stays a single identifier
    assert adapter.quote_ident('x"; DROP TABLE y; --') == '"x""; DROP TABLE y; --"'


def test_table_ref_quotes_table_name(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        assert adapter.table_ref("order").endswith('."order"')


# ─── reserved words round-trip through the warehouse ────────────────────────


def test_reserved_word_table_and_column_round_trip(tmp_path: Path) -> None:
    """A model named `order` with a column named `select` must materialize,
    upsert, delete, and query cleanly."""
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        df = pl.DataFrame({"select": ["a", "b"], "value": [1, 2]})
        assert adapter.materialize_full("order", df) == 2

        adapter.materialize_incremental(
            "order",
            pl.DataFrame({"select": ["a", "c"], "value": [99, 3]}),
            key_col="select",
        )
        rows = adapter.rows(
            f'SELECT "select", value FROM {adapter.table_ref("order")} ORDER BY "select"'
        )
        assert rows == [("a", 99), ("b", 2), ("c", 3)]

        assert adapter.delete_rows("order", key_col="select", keys=["b"]) == 1
        assert "order" in adapter.list_tables()


def test_schema_tests_with_reserved_column(tmp_path: Path) -> None:
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full(
            "order",
            pl.DataFrame({"select": ["a", "b", None], "value": [1, 2, 3]}),
        )
        adapter.materialize_full(
            "group", pl.DataFrame({"select": ["a", "b"]})
        )
        table_ref = adapter.table_ref("order")

        def run(spec: object) -> list[Any]:
            return evaluate_test_spec(
                spec, model_name="order", table_ref=table_ref, adapter=adapter
            )

        assert run({"not_null": ["select"]})[0].status == "fail"
        assert run({"not_null": ["value"]})[0].status == "pass"
        assert run({"unique": "select"})[0].status == "pass"
        assert run({"accepted_values": {"column": "select", "values": ["a", "b"]}})[
            0
        ].status == "pass"
        assert run({"accepted_range": {"column": "value", "min": 0, "max": 10}})[
            0
        ].status == "pass"
        assert run({"null_rate": {"column": "select", "max": 0.5}})[0].status == "pass"
        assert run(
            {"relationships": {"column": "select", "to": "ref('group')", "field": "select"}}
        )[0].status == "pass"


def test_injection_shaped_column_does_not_execute(tmp_path: Path) -> None:
    """A hostile column name in a test spec is quoted into a (nonexistent)
    identifier: the query errors, nothing else runs."""
    with create_adapter(_wh(tmp_path / "t.duckdb")) as adapter:
        adapter.materialize_full("victims", pl.DataFrame({"x": [1]}))
        table_ref = adapter.table_ref("victims")
        with pytest.raises(Exception, match=r"(?i)column|binder"):
            evaluate_test_spec(
                {"not_null": ['x"; DROP TABLE victims; --']},
                model_name="victims",
                table_ref=table_ref,
                adapter=adapter,
            )
        assert "victims" in adapter.list_tables()


# ─── config-level name validation ───────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_name",
    ["has space", "semi;colon", 'quo"te', "dash-ed", "1leading_digit", "dot.ted", ""],
)
def test_model_name_charset_rejected(bad_name: str) -> None:
    with pytest.raises(ValidationError, match="letters, digits, and underscores"):
        ModelConfig(name=bad_name)


@pytest.mark.parametrize("reserved", ["dbt_ml_state", "dbt_ml_staging", "DBT_ML_thing"])
def test_model_name_reserved_prefix_rejected(reserved: str) -> None:
    with pytest.raises(ValidationError, match="reserved"):
        ModelConfig(name=reserved)


def test_model_name_reserved_word_allowed() -> None:
    # SQL reserved words are fine — quoting handles them downstream.
    assert ModelConfig(name="order").name == "order"


def test_source_name_charset_rejected() -> None:
    with pytest.raises(ValidationError, match="letters, digits, and underscores"):
        SourceConfig(name="bad name", path="data")


def test_source_name_valid() -> None:
    assert SourceConfig(name="raw_docs", path="data").name == "raw_docs"
