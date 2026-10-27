"""DuckDB relation reads across supported driver versions (issue #715)."""

from pathlib import Path

import polars as pl

from stel.adapters import create_adapter, parse_warehouse_config


def test_query_df_preserves_parameters_and_empty_result_types(tmp_path: Path) -> None:
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "warehouse.duckdb"), "schema": "main"}
    )

    with create_adapter(config) as adapter:
        result = adapter.query_df(
            "SELECT ?::BIGINT AS id, ?::VARCHAR AS label", [7, "first"]
        )
        empty = adapter.query_df("SELECT 1::BIGINT AS id WHERE FALSE")

    assert result.to_dict(as_series=False) == {"id": [7], "label": ["first"]}
    assert result.schema == {"id": pl.Int64, "label": pl.String}
    assert empty.schema == {"id": pl.Int64}
    assert empty.is_empty()
