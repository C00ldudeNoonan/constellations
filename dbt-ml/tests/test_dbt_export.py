from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from dbt_ml.dbt_export import build_dbt_sources, write_dbt_sources


@pytest.fixture
def fresh_project(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


def _table(payload: dict, name: str) -> dict:
    src = payload["sources"][0]
    return next(t for t in src["tables"] if t["name"] == name)


def _col(table: dict, name: str) -> dict:
    return next(c for c in table["columns"] if c["name"] == name)


def test_basic_shape(fresh_project: Path) -> None:
    payload = build_dbt_sources(fresh_project)
    assert payload["version"] == 2
    assert len(payload["sources"]) == 1
    src = payload["sources"][0]
    assert src["name"] == "dbt_ml_invoice_pipeline"
    assert src["database"] == "dbt_ml"  # from duckdb path stem
    assert src["schema"] == "dbt_ml"
    table_names = {t["name"] for t in src["tables"]}
    assert table_names == {"raw_invoices", "invoice_summary", "monthly_totals"}


def test_custom_source_name(fresh_project: Path) -> None:
    payload = build_dbt_sources(fresh_project, source_name="my_dbt_ml")
    assert payload["sources"][0]["name"] == "my_dbt_ml"


def test_not_null_translated_to_column_tests(fresh_project: Path) -> None:
    payload = build_dbt_sources(fresh_project)
    raw = _table(payload, "raw_invoices")
    vendor_col = _col(raw, "vendor")
    assert "not_null" in vendor_col["tests"]
    total_col = _col(raw, "total")
    assert "not_null" in total_col["tests"]


def test_declared_data_type_propagates_to_dbt_columns(fresh_project: Path) -> None:
    raw = _table(build_dbt_sources(fresh_project), "raw_invoices")
    assert _col(raw, "invoice_id")["data_type"] == "string"
    assert _col(raw, "total")["data_type"] == "float"
    assert _col(raw, "line_items")["data_type"] == "string"


def test_single_column_unique_on_column(fresh_project: Path) -> None:
    payload = build_dbt_sources(fresh_project)
    raw = _table(payload, "raw_invoices")
    invoice_id = _col(raw, "invoice_id")
    assert "unique" in invoice_id["tests"]


def test_select_filters_tables(fresh_project: Path) -> None:
    payload = build_dbt_sources(fresh_project, select="raw_invoices")
    src = payload["sources"][0]
    assert [t["name"] for t in src["tables"]] == ["raw_invoices"]


def test_exclude_by_tag(fresh_project: Path) -> None:
    payload = build_dbt_sources(fresh_project, exclude="tag:monthly")
    table_names = {t["name"] for t in payload["sources"][0]["tables"]}
    assert "monthly_totals" not in table_names


def test_tags_propagate(fresh_project: Path) -> None:
    payload = build_dbt_sources(fresh_project)
    raw = _table(payload, "raw_invoices")
    assert set(raw["tags"]) == {"raw", "invoices"}


def test_dagster_meta_off_by_default(fresh_project: Path) -> None:
    payload = build_dbt_sources(fresh_project)
    assert "meta" not in _table(payload, "raw_invoices")


def test_dagster_meta_stamps_asset_key(fresh_project: Path) -> None:
    payload = build_dbt_sources(fresh_project, dagster_meta=True)
    raw = _table(payload, "raw_invoices")
    assert raw["meta"]["dagster"]["asset_key"] == [
        "dbt_ml_invoice_pipeline",
        "raw_invoices",
    ]


def test_dagster_meta_uses_custom_source_name(fresh_project: Path) -> None:
    payload = build_dbt_sources(
        fresh_project, source_name="my_ml", dagster_meta=True
    )
    raw = _table(payload, "raw_invoices")
    assert raw["meta"]["dagster"]["asset_key"] == ["my_ml", "raw_invoices"]


def test_write_creates_yaml(fresh_project: Path) -> None:
    path = write_dbt_sources(fresh_project)
    assert path.exists()
    assert path.name == "sources.yml"
    parsed = yaml.safe_load(path.read_text())
    assert parsed["sources"][0]["name"] == "dbt_ml_invoice_pipeline"


def test_composite_unique_emits_dbt_utils_macro(tmp_path: Path) -> None:
    """Project with a composite-unique test should emit dbt_utils macro."""
    project_dir = tmp_path / "p"
    project_dir.mkdir()
    (project_dir / "dbt_ml_project.yml").write_text(
        "name: p\nduckdb:\n  path: ./target/p.duckdb\n  schema: p\n"
    )
    (project_dir / "models").mkdir()
    (project_dir / "models" / "x.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: x\n"
        "    extraction:\n      backend: json\n"
        "    source: ref('s')\n"
        "    fields:\n      - name: a\n      - name: b\n"
        "    tests:\n      - unique: [a, b]\n"
    )
    (project_dir / "sources").mkdir()
    (project_dir / "sources" / "s.yml").write_text(
        "version: 2\nsources:\n  - name: s\n    path: ./data/\n"
    )

    payload = build_dbt_sources(project_dir)
    table = payload["sources"][0]["tables"][0]
    assert any(
        isinstance(t, dict) and "dbt_utils.unique_combination_of_columns" in t
        for t in table.get("tests", [])
    )


def test_min_rows_dropped(tmp_path: Path) -> None:
    """min_rows has no dbt source equivalent — it should not appear."""
    project_dir = tmp_path / "p"
    project_dir.mkdir()
    (project_dir / "dbt_ml_project.yml").write_text(
        "name: p\nduckdb:\n  path: ./target/p.duckdb\n  schema: p\n"
    )
    (project_dir / "models").mkdir()
    (project_dir / "models" / "x.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: x\n"
        "    extraction:\n      backend: json\n"
        "    source: ref('s')\n"
        "    tests:\n      - min_rows: 5\n"
    )
    (project_dir / "sources").mkdir()
    (project_dir / "sources" / "s.yml").write_text(
        "version: 2\nsources:\n  - name: s\n    path: ./data/\n"
    )

    payload = build_dbt_sources(project_dir)
    table = payload["sources"][0]["tables"][0]
    assert "min_rows" not in yaml.safe_dump(table)
