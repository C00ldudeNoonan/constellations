from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

from stel.adapters import WarehouseAdapter, create_adapter, parse_warehouse_config
from stel.checks import run_model_tests, run_project_tests
from stel.checks.schema import TestResult, UnknownTestError, evaluate_test_spec
from stel.config import ConfigError
from stel.config.model import ModelConfig
from stel.runner import run_project
from stel.synth import generate_invoices


@pytest.fixture
def fresh_project(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


@pytest.fixture
def populated_db(tmp_path: Path) -> Iterator[WarehouseAdapter]:
    """Adapter with a small `items` table — used to exercise schema tests."""
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "t.duckdb"), "schema": "main"}
    )
    with create_adapter(cfg) as adapter:
        adapter.materialize_full(
            "items",
            pl.DataFrame(
                {
                    "id": [1, 2, 3, 3],
                    "vendor": ["A", "B", "C", "D"],
                    "total": [10.0, 20.0, None, 30.0],
                }
            ),
        )
        yield adapter


def _by(results: list[TestResult], **filters: str | None) -> TestResult:
    return next(
        r
        for r in results
        if all(getattr(r, k) == v for k, v in filters.items())
    )


def test_not_null_pass_and_fail(populated_db: WarehouseAdapter) -> None:
    results = evaluate_test_spec(
        {"not_null": ["id", "total"]},
        model_name="items",
        table_ref=populated_db.table_ref("items"),
        adapter=populated_db,
    )
    assert _by(results, column="id").passed
    assert not _by(results, column="total").passed
    assert "1 rows" in _by(results, column="total").message


def test_unique_detects_dups(populated_db: WarehouseAdapter) -> None:
    results = evaluate_test_spec(
        {"unique": "id"},
        model_name="items",
        table_ref=populated_db.table_ref("items"),
        adapter=populated_db,
    )
    assert len(results) == 1
    assert not results[0].passed


def test_unique_composite(populated_db: WarehouseAdapter) -> None:
    results = evaluate_test_spec(
        {"unique": ["id", "vendor"]},
        model_name="items",
        table_ref=populated_db.table_ref("items"),
        adapter=populated_db,
    )
    assert results[0].passed


def test_relationships_pass_and_fail(populated_db: WarehouseAdapter) -> None:
    # parent 'items' has id in {1, 2, 3}; build two child tables referencing it.
    populated_db.materialize_full(
        "orders_ok", pl.DataFrame({"item_id": [1, 2, 3, None]})
    )
    populated_db.materialize_full(
        "orders_bad", pl.DataFrame({"item_id": [1, 2, 99]})
    )

    ok = evaluate_test_spec(
        {"relationships": {"column": "item_id", "to": "ref('items')", "field": "id"}},
        model_name="orders_ok",
        table_ref=populated_db.table_ref("orders_ok"),
        adapter=populated_db,
    )
    assert ok[0].passed

    bad = evaluate_test_spec(
        {"relationships": {"column": "item_id", "to": "ref('items')", "field": "id"}},
        model_name="orders_bad",
        table_ref=populated_db.table_ref("orders_bad"),
        adapter=populated_db,
    )
    assert not bad[0].passed
    assert "missing from items.id" in bad[0].message


def test_store_failures_sql_test_persists_rows(populated_db: WarehouseAdapter) -> None:
    # items.total has one NULL (row id=3, vendor=C)
    results = evaluate_test_spec(
        {"not_null": "total"},
        model_name="items",
        table_ref=populated_db.table_ref("items"),
        adapter=populated_db,
        store_failures=True,
    )
    r = results[0]
    assert not r.passed
    assert r.failures_table == "dbt_ml_test_failures__items__not_null__total"
    assert r.failure_count == 1
    stored = populated_db.query_df(
        f"SELECT * FROM {populated_db.table_ref(r.failures_table)}"
    )
    assert stored.height == 1
    assert stored["vendor"].to_list() == ["C"]


def test_store_failures_in_memory_test_persists_rows(
    populated_db: WarehouseAdapter,
) -> None:
    # vendor in {A, B, C, D}; only "A" matches ^A$, so 3 rows fail.
    results = evaluate_test_spec(
        {"matches_regex": {"column": "vendor", "pattern": "^A$"}},
        model_name="items",
        table_ref=populated_db.table_ref("items"),
        adapter=populated_db,
        store_failures=True,
    )
    r = results[0]
    assert not r.passed
    assert r.failure_count == 3
    assert r.failures_table is not None
    assert r.failures_table.endswith("matches_regex__vendor")
    stored = populated_db.query_df(
        f"SELECT * FROM {populated_db.table_ref(r.failures_table)}"
    )
    assert set(stored["vendor"].to_list()) == {"B", "C", "D"}


def test_store_failures_grounded_in(populated_db: WarehouseAdapter) -> None:
    populated_db.materialize_full(
        "grounding",
        pl.DataFrame(
            {
                "value": ["Acme", "Hallucinated", "Beta"],
                "source": [
                    "Invoice from Acme Corp",
                    "totally unrelated text",
                    "Beta line item",
                ],
            }
        ),
    )
    results = evaluate_test_spec(
        {"grounded_in": {"value": "value", "source": "source"}},
        model_name="grounding",
        table_ref=populated_db.table_ref("grounding"),
        adapter=populated_db,
        store_failures=True,
    )
    r = results[0]
    assert not r.passed
    assert r.failure_count == 1
    assert r.failures_table is not None
    stored = populated_db.query_df(
        f"SELECT * FROM {populated_db.table_ref(r.failures_table)}"
    )
    assert stored["value"].to_list() == ["Hallucinated"]


def test_store_failures_noop_when_passing(populated_db: WarehouseAdapter) -> None:
    results = evaluate_test_spec(
        {"not_null": "id"},
        model_name="items",
        table_ref=populated_db.table_ref("items"),
        adapter=populated_db,
        store_failures=True,
    )
    assert results[0].passed
    assert results[0].failures_table is None
    assert results[0].failure_count is None


def test_relationships_requires_to_and_field(populated_db: WarehouseAdapter) -> None:
    with pytest.raises(UnknownTestError, match="relationships requires"):
        evaluate_test_spec(
            {"relationships": {"column": "item_id"}},
            model_name="items",
            table_ref=populated_db.table_ref("items"),
            adapter=populated_db,
        )


def test_min_rows(populated_db: WarehouseAdapter) -> None:
    ref = populated_db.table_ref("items")
    ok = evaluate_test_spec(
        {"min_rows": 1}, model_name="items", table_ref=ref, adapter=populated_db
    )
    assert ok[0].passed
    bad = evaluate_test_spec(
        {"min_rows": 100}, model_name="items", table_ref=ref, adapter=populated_db
    )
    assert not bad[0].passed


def test_not_empty_bare_string(populated_db: WarehouseAdapter) -> None:
    ref = populated_db.table_ref("items")
    results = evaluate_test_spec(
        "not_empty", model_name="items", table_ref=ref, adapter=populated_db
    )
    assert results[0].test_name == "not_empty"
    assert results[0].passed


def test_unknown_test_raises(populated_db: WarehouseAdapter) -> None:
    with pytest.raises(UnknownTestError):
        evaluate_test_spec(
            {"nonsense": "foo"},
            model_name="items",
            table_ref=populated_db.table_ref("items"),
            adapter=populated_db,
        )


def test_severity_warn_downgrades_failure(
    populated_db: WarehouseAdapter,
) -> None:
    # `total` has a NULL → would be a "fail" by default
    results = evaluate_test_spec(
        {"not_null": "total", "severity": "warn"},
        model_name="items",
        table_ref=populated_db.table_ref("items"),
        adapter=populated_db,
    )
    assert results[0].status == "warn"
    assert results[0].severity == "warn"
    assert not results[0].passed
    assert not results[0].is_hard_failure


def test_severity_error_is_default(populated_db: WarehouseAdapter) -> None:
    results = evaluate_test_spec(
        {"not_null": "total"},
        model_name="items",
        table_ref=populated_db.table_ref("items"),
        adapter=populated_db,
    )
    assert results[0].status == "fail"
    assert results[0].severity == "error"
    assert results[0].is_hard_failure


def test_severity_warn_keeps_passing_results(
    populated_db: WarehouseAdapter,
) -> None:
    # Passing test should still report "pass" even with severity: warn
    results = evaluate_test_spec(
        {"not_null": "id", "severity": "warn"},
        model_name="items",
        table_ref=populated_db.table_ref("items"),
        adapter=populated_db,
    )
    assert results[0].status == "pass"


def test_unknown_severity_raises(populated_db: WarehouseAdapter) -> None:
    with pytest.raises(UnknownTestError, match="Unknown severity"):
        evaluate_test_spec(
            {"not_null": "id", "severity": "loud"},
            model_name="items",
            table_ref=populated_db.table_ref("items"),
            adapter=populated_db,
        )


def test_missing_model_relation_is_a_clean_test_failure(
    populated_db: WarehouseAdapter,
) -> None:
    results = run_model_tests(
        ModelConfig(name="missing_model", tests=["not_empty"]),
        populated_db,
    )

    assert len(results) == 1
    assert results[0].test_name == "relation_exists"
    assert results[0].status == "fail"
    assert results[0].is_hard_failure
    assert results[0].message == (
        "Materialized relation for model 'missing_model' is missing. "
        "Run the model before testing it; a model that matched zero documents "
        "may not have created a relation."
    )


def test_model_without_tests_does_not_require_a_relation(
    populated_db: WarehouseAdapter,
) -> None:
    assert run_model_tests(ModelConfig(name="missing_model"), populated_db) == []


def test_zero_match_model_creates_testable_typed_relation(
    fresh_project: Path,
) -> None:
    run_result = run_project(fresh_project, select="raw_invoices")
    assert run_result[0].rows_written == 0
    assert run_result[0].warnings == {
        "Source 'vendor_invoices' matched zero documents; verify its path and "
        "file_pattern.": 1
    }

    results = run_project_tests(fresh_project, select="raw_invoices")

    assert results
    assert all(result.status == "pass" for result in results)
    assert all(result.test_name != "relation_exists" for result in results)


def test_project_tests_validate_effective_options_before_adapter_connection(
    fresh_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (fresh_project / "profiles.yml").write_text(
        "invoice_pipeline:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: bigquery\n"
        "        project: econ\n"
        "        dataset: documents_dev\n"
    )
    model_path = fresh_project / "models" / "raw_invoices.yml"
    model_path.write_text(
        model_path.read_text().replace(
            "    materialization: incremental",
            "    materialization: incremental\n"
            "    warehouse_options:\n"
            "      inherit: not-a-boolean",
            1,
        )
    )

    def unexpected_connection(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid options must fail before the adapter connects")

    monkeypatch.setattr(
        "stel.adapters.bigquery.BigQueryAdapter._connect", unexpected_connection
    )

    with pytest.raises(ConfigError, match="inherit"):
        run_project_tests(fresh_project, select="raw_invoices")


def test_end_to_end_passes(fresh_project: Path) -> None:
    generate_invoices(15, fresh_project / "data" / "invoices", seed=99)
    run_project(fresh_project)
    results = run_project_tests(fresh_project)
    assert len(results) > 0
    failed = [r for r in results if not r.passed]
    assert failed == []


def test_end_to_end_fail_when_unique_violated(fresh_project: Path) -> None:
    """Two files with identical invoice_id violate raw_invoices' unique test."""
    invoices_dir = fresh_project / "data" / "invoices"
    invoices_dir.mkdir(parents=True)
    body = (
        '{"invoice_id":"DUP","vendor":"V","issue_date":"2025-01-01",'
        '"currency":"USD","line_items":[],"total":1.0}'
    )
    (invoices_dir / "a.json").write_text(body)
    (invoices_dir / "b.json").write_text(body)
    run_project(fresh_project)
    results = run_project_tests(fresh_project)
    fails = [r for r in results if not r.passed]
    assert any(r.test_name == "unique" for r in fails)
