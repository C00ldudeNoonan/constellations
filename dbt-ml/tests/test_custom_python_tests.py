from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

from dbt_ml.adapters import WarehouseAdapter, create_adapter, parse_warehouse_config
from dbt_ml.checks.schema import evaluate_test_spec


@pytest.fixture
def db(tmp_path: Path) -> Iterator[WarehouseAdapter]:
    """Adapter with an `items` table — passed to custom Python tests as `con`
    via the adapter's `raw_connection`."""
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "t.duckdb"), "schema": "main"}
    )
    with create_adapter(cfg) as adapter:
        adapter.materialize_full(
            "items",
            pl.DataFrame({"id": [1, 2, 3], "total": [100.0, 200.0, -5.0]}),
        )
        yield adapter


def _write_test_module(project_dir: Path, name: str, body: str) -> None:
    tests_dir = project_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / f"{name}.py").write_text(body)


def test_python_test_pass(tmp_path: Path, db: WarehouseAdapter) -> None:
    _write_test_module(
        tmp_path,
        "all_positive",
        "def run(con, table_ref):\n"
        "    row = con.execute(f'SELECT COUNT(*) FROM {table_ref} WHERE total < 0').fetchone()\n"
        "    return None if row[0] == 0 else f'{row[0]} rows with negative total'\n",
    )
    results = evaluate_test_spec(
        {"python": "tests.all_positive"},
        model_name="items",
        table_ref=db.table_ref("items"),
        adapter=db,
        project_dir=tmp_path,
    )
    assert results[0].status == "fail"
    assert "1 rows with negative" in results[0].message


def test_python_test_pass_when_no_failures(
    tmp_path: Path, db: WarehouseAdapter
) -> None:
    _write_test_module(
        tmp_path,
        "row_count_ok",
        "def run(con, table_ref):\n"
        "    return None  # nothing to complain about\n",
    )
    results = evaluate_test_spec(
        {"python": "tests.row_count_ok"},
        model_name="items",
        table_ref=db.table_ref("items"),
        adapter=db,
        project_dir=tmp_path,
    )
    assert results[0].status == "pass"


def test_python_test_with_severity_warn(
    tmp_path: Path, db: WarehouseAdapter
) -> None:
    _write_test_module(
        tmp_path,
        "fails_always",
        "def run(con, table_ref):\n    return 'always fails'\n",
    )
    results = evaluate_test_spec(
        {"python": "tests.fails_always", "severity": "warn"},
        model_name="items",
        table_ref=db.table_ref("items"),
        adapter=db,
        project_dir=tmp_path,
    )
    assert results[0].status == "warn"
    assert not results[0].is_hard_failure


def test_python_test_missing_module(
    tmp_path: Path, db: WarehouseAdapter
) -> None:
    results = evaluate_test_spec(
        {"python": "tests.nonexistent"},
        model_name="items",
        table_ref=db.table_ref("items"),
        adapter=db,
        project_dir=tmp_path,
    )
    assert results[0].status == "fail"
    assert "not found" in results[0].message


def test_python_test_missing_run_function(
    tmp_path: Path, db: WarehouseAdapter
) -> None:
    _write_test_module(
        tmp_path,
        "no_run",
        "x = 1\n",  # module defines no `run`
    )
    results = evaluate_test_spec(
        {"python": "tests.no_run"},
        model_name="items",
        table_ref=db.table_ref("items"),
        adapter=db,
        project_dir=tmp_path,
    )
    assert results[0].status == "fail"
    assert "run" in results[0].message
