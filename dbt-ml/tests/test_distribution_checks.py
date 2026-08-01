"""Deterministic distribution / statistical quality checks (issue #10). All run
against a real DuckDB adapter over a column — no LLM, no sampling."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dbt_ml.adapters import create_adapter, parse_warehouse_config
from dbt_ml.checks.schema import TestResult, evaluate_test_spec
from dbt_ml.test_specs import TestSpecError as SpecError
from dbt_ml.test_specs import parse_test_spec


def _adapter(tmp_path: Path) -> Any:
    return create_adapter(
        parse_warehouse_config(
            {"type": "duckdb", "path": str(tmp_path / "d.duckdb"), "schema": "main"}
        )
    )


def _check(tmp_path: Path, frame: pl.DataFrame, spec: dict[str, Any], **kw: Any) -> TestResult:
    with _adapter(tmp_path) as adapter:
        adapter.materialize_full("t", frame)
        return evaluate_test_spec(
            spec, model_name="t", table_ref=adapter.table_ref("t"), adapter=adapter, **kw
        )[0]


# 1..10 with 100 as an extreme outlier; three categories.
_FRAME = pl.DataFrame(
    {
        "id": [str(i) for i in range(10)],
        "n": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        "cat": ["x", "x", "y", "y", "z", "z", "x", "y", "z", "x"],
    }
)


# ─── column_stat ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("spec", "status"),
    [
        ({"column_stat": {"column": "n", "stat": "mean", "max": 20}}, "pass"),
        ({"column_stat": {"column": "n", "stat": "mean", "max": 10}}, "fail"),
        ({"column_stat": {"column": "n", "stat": "min", "min": 1, "max": 1}}, "pass"),
        ({"column_stat": {"column": "n", "stat": "max", "max": 50}}, "fail"),
        ({"column_stat": {"column": "n", "stat": "median", "min": 5, "max": 6}}, "pass"),
        (
            {"column_stat": {"column": "n", "stat": "quantile", "quantile": 0.5, "max": 6}},
            "pass",
        ),
    ],
)
def test_column_stat(tmp_path: Path, spec: dict[str, Any], status: str) -> None:
    assert _check(tmp_path, _FRAME, spec).status == status


def test_column_stat_median_is_interpolated(tmp_path: Path) -> None:
    r = _check(tmp_path, _FRAME, {"column_stat": {"column": "n", "stat": "median", "max": 5}})
    # median of 1..9,100 = mean(5,6) = 5.5
    assert r.status == "fail"
    assert "median=5.5" in r.message


def test_column_stat_empty_passes(tmp_path: Path) -> None:
    empty = pl.DataFrame({"n": []}, schema={"n": pl.Float64})
    r = _check(tmp_path, empty, {"column_stat": {"column": "n", "stat": "mean", "max": 1}})
    assert r.status == "pass"


def test_column_stat_non_numeric_column_fails_actionably(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="must be numeric"):
        _check(tmp_path, _FRAME, {"column_stat": {"column": "cat", "stat": "mean", "max": 1}})


# ─── cardinality ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("spec", "status"),
    [
        ({"cardinality": {"column": "cat", "min": 3, "max": 3}}, "pass"),
        ({"cardinality": {"column": "cat", "min": 4}}, "fail"),
        ({"cardinality": {"column": "cat", "max_ratio": 0.2}}, "fail"),
        ({"cardinality": {"column": "cat", "max_ratio": 0.5}}, "pass"),
        ({"cardinality": {"column": "id", "min_ratio": 1.0}}, "pass"),  # all unique
    ],
)
def test_cardinality(tmp_path: Path, spec: dict[str, Any], status: str) -> None:
    assert _check(tmp_path, _FRAME, spec).status == status


# ─── outlier_rate ────────────────────────────────────────────────────────────


def test_outlier_rate_iqr_flags_extreme(tmp_path: Path) -> None:
    r = _check(tmp_path, _FRAME, {"outlier_rate": {"column": "n", "method": "iqr"}})
    assert r.status == "fail"
    assert "1/10 outliers" in r.message


def test_outlier_rate_tolerance(tmp_path: Path) -> None:
    r = _check(
        tmp_path, _FRAME, {"outlier_rate": {"column": "n", "method": "iqr", "max_rate": 0.2}}
    )
    assert r.status == "pass"


def test_outlier_rate_zscore(tmp_path: Path) -> None:
    r = _check(
        tmp_path, _FRAME, {"outlier_rate": {"column": "n", "method": "zscore", "k": 2.5}}
    )
    assert r.status == "fail"


def test_outlier_rate_needs_four_values(tmp_path: Path) -> None:
    small = pl.DataFrame({"n": [1, 2, 99]})
    r = _check(tmp_path, small, {"outlier_rate": {"column": "n"}})
    assert r.status == "pass"
    assert "need >= 4" in r.message


def test_outlier_rate_store_failures_persists_rows(tmp_path: Path) -> None:
    with _adapter(tmp_path) as adapter:
        adapter.materialize_full("t", _FRAME)
        r = evaluate_test_spec(
            {"outlier_rate": {"column": "n", "method": "iqr"}},
            model_name="t",
            table_ref=adapter.table_ref("t"),
            adapter=adapter,
            store_failures=True,
        )[0]
        assert r.status == "fail"
        assert r.failure_count == 1
        stored = adapter.query_df(f"SELECT * FROM {adapter.table_ref(r.failures_table)}")
        assert stored["n"].to_list() == [100]


# ─── numeric-type robustness (Codex review) ──────────────────────────────────


def test_column_stat_accepts_decimal_column(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {"p": [Decimal("1.5"), Decimal("2.5"), Decimal("3.5")]},
        schema={"p": pl.Decimal(10, 2)},
    )
    r = _check(
        tmp_path, frame, {"column_stat": {"column": "p", "stat": "mean", "min": 2, "max": 3}}
    )
    assert r.status == "pass"


def test_column_stat_preserves_large_integer_precision(tmp_path: Path) -> None:
    # 2**53 + 1 is not exactly representable as float64; an exact max must not
    # round down to 2**53 and pass a bound of 2**53.
    big = 9007199254740993
    frame = pl.DataFrame({"n": [big, 1, 2]}, schema={"n": pl.Int64})
    r = _check(tmp_path, frame, {"column_stat": {"column": "n", "stat": "max", "max": big - 1}})
    assert r.status == "fail"
    assert str(big) in r.message


def test_non_numeric_empty_column_fails_not_passes(tmp_path: Path) -> None:
    # An all-null / empty non-numeric column must still fail the numeric contract
    # rather than slip through as "no non-null numeric values".
    empty = pl.DataFrame({"c": []}, schema={"c": pl.String})
    with pytest.raises(SpecError, match="must be numeric"):
        _check(tmp_path, empty, {"column_stat": {"column": "c", "stat": "mean", "max": 1}})
    with pytest.raises(SpecError, match="must be numeric"):
        _check(tmp_path, empty, {"outlier_rate": {"column": "c"}})


def test_cardinality_collapses_repeated_nans(tmp_path: Path) -> None:
    nan = float("nan")
    frame = pl.DataFrame({"v": [1.0, nan, nan, nan, 1.0]}, schema={"v": pl.Float64})
    # distinct values are {1.0, NaN} = 2, not five separate NaN objects.
    assert _check(tmp_path, frame, {"cardinality": {"column": "v", "max": 2}}).status == "pass"
    assert _check(tmp_path, frame, {"cardinality": {"column": "v", "max": 1}}).status == "fail"


# ─── compile-time validation ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"column_stat": {"column": "n"}}, "missing required options"),
        ({"column_stat": {"column": "n", "stat": "bogus", "max": 1}}, "stat must be one of"),
        ({"column_stat": {"column": "n", "stat": "mean"}}, "at least one of: min, max"),
        (
            {"column_stat": {"column": "n", "stat": "mean", "min": 5, "max": 1}},
            "min <= max",
        ),
        (
            {"column_stat": {"column": "n", "stat": "quantile", "max": 1}},
            "requires a quantile",
        ),
        (
            {"column_stat": {"column": "n", "stat": "quantile", "quantile": 2, "max": 1}},
            "quantile must be between 0 and 1",
        ),
        (
            {"column_stat": {"column": "n", "stat": "mean", "quantile": 0.5, "max": 1}},
            "only applies when stat is 'quantile'",
        ),
        ({"cardinality": {"column": "c"}}, "at least one of: min, max"),
        ({"cardinality": {"column": "c", "min": -1}}, "non-negative integer"),
        ({"cardinality": {"column": "c", "max_ratio": 2}}, "between 0 and 1"),
        ({"outlier_rate": {"column": "n", "method": "mad"}}, "must be 'iqr' or 'zscore'"),
        ({"outlier_rate": {"column": "n", "k": 0}}, "k must be a positive"),
        ({"outlier_rate": {"column": "n", "max_rate": 1.5}}, "between 0 and 1"),
    ],
)
def test_distribution_specs_are_strict(spec: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        parse_test_spec(spec)
