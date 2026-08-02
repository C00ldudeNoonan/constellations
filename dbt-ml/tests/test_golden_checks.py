"""Golden-set checks (issue #10): compare a model's rows to a checked-in
expected model joined on a key, with optional per-column numeric tolerance."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dbt_ml.adapters import WarehouseAdapter, create_adapter, parse_warehouse_config
from dbt_ml.checks.schema import evaluate_test_spec
from dbt_ml.test_specs import TestSpecError as SpecError
from dbt_ml.test_specs import parse_test_spec, relationship_test_targets

_GOLDEN = pl.DataFrame(
    {"id": ["1", "2", "3"], "vendor": ["A", "B", "C"], "total": [10.0, 20.0, 30.0]}
)


@pytest.fixture
def adapter(tmp_path: Path) -> Any:
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "g.duckdb"), "schema": "main"}
    )
    with create_adapter(cfg) as ad:
        ad.materialize_full("golden", _GOLDEN)
        yield ad


def _golden(ad: WarehouseAdapter, actual: pl.DataFrame, spec: dict[str, Any], **kw: Any) -> Any:
    ad.materialize_full("model", actual)
    return evaluate_test_spec(
        {"golden": {"to": "ref('golden')", "key": "id", **spec}},
        model_name="model", table_ref=ad.table_ref("model"), adapter=ad, **kw,
    )[0]


def test_exact_match_passes(adapter: WarehouseAdapter) -> None:
    assert _golden(adapter, _GOLDEN, {}).status == "pass"


def test_value_mismatch_fails(adapter: WarehouseAdapter) -> None:
    bad = _GOLDEN.with_columns(pl.Series("vendor", ["A", "X", "C"]))
    r = _golden(adapter, bad, {})
    assert r.status == "fail"
    assert "1 mismatched" in r.message


def test_numeric_tolerance(adapter: WarehouseAdapter) -> None:
    near = _GOLDEN.with_columns(pl.Series("total", [10.0, 20.05, 30.0]))
    assert _golden(adapter, near, {"tolerance": {"total": 0.1}}).status == "pass"
    assert _golden(adapter, near, {"tolerance": {"total": 0.01}}).status == "fail"


def test_missing_golden_key_fails(adapter: WarehouseAdapter) -> None:
    short = _GOLDEN.head(2)
    r = _golden(adapter, short, {})
    assert r.status == "fail"
    assert "1 missing" in r.message


def test_extra_rows_ignored_unless_exhaustive(adapter: WarehouseAdapter) -> None:
    extra = pl.concat(
        [_GOLDEN, pl.DataFrame({"id": ["4"], "vendor": ["D"], "total": [40.0]})]
    )
    assert _golden(adapter, extra, {}).status == "pass"
    r = _golden(adapter, extra, {"exhaustive": True})
    assert r.status == "fail"
    assert "1 unexpected" in r.message


def test_columns_restricts_comparison(adapter: WarehouseAdapter) -> None:
    # vendor differs but is not compared when columns=[total].
    bad_vendor = _GOLDEN.with_columns(pl.Series("vendor", ["A", "X", "C"]))
    assert _golden(adapter, bad_vendor, {"columns": ["total"]}).status == "pass"


def test_store_failures_persists_offending_keys(adapter: WarehouseAdapter) -> None:
    bad = _GOLDEN.with_columns(pl.Series("vendor", ["A", "X", "C"]))
    ad = adapter
    ad.materialize_full("model", bad)
    r = evaluate_test_spec(
        {"golden": {"to": "ref('golden')", "key": "id"}},
        model_name="model", table_ref=ad.table_ref("model"), adapter=ad,
        store_failures=True,
    )[0]
    assert r.failure_count == 1
    assert r.failures_table is not None
    stored = ad.query_df(f"SELECT * FROM {ad.table_ref(r.failures_table)}")
    assert stored["key"].to_list() == ["2"]
    assert "mismatch:vendor" in stored["issue"][0]


def test_missing_key_column_fails_actionably(adapter: WarehouseAdapter) -> None:
    with pytest.raises(SpecError, match="key 'id' not found"):
        _golden(adapter, _GOLDEN.rename({"id": "other"}), {})


def test_golden_target_is_a_dag_dependency() -> None:
    targets = relationship_test_targets(
        [{"golden": {"to": "ref('expected_snapshot')", "key": "id"}}]
    )
    assert targets == {"expected_snapshot"}


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"golden": {"key": "id"}}, "missing required options"),
        ({"golden": {"to": "ref('g')"}}, "missing required options"),
        ({"golden": {"to": "ref('g')", "key": "id", "columns": []}}, "non-empty list"),
        (
            {"golden": {"to": "ref('g')", "key": "id", "tolerance": {"total": -1}}},
            "non-negative",
        ),
        (
            {"golden": {"to": "ref('g')", "key": "id", "exhaustive": "yes"}},
            "must be a boolean",
        ),
    ],
)
def test_golden_specs_are_strict(spec: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        parse_test_spec(spec)
