"""Run-over-run drift checks (issue #10): PSI / KS / Jensen-Shannon of a column
against a baseline model referenced by `to: ref(...)`. Deterministic, offline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel.adapters import WarehouseAdapter, create_adapter, parse_warehouse_config
from stel.checks.schema import evaluate_test_spec
from stel.test_specs import TestSpecError as SpecError
from stel.test_specs import parse_test_spec, relationship_test_targets


@pytest.fixture
def adapter(tmp_path: Path) -> Any:
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "d.duckdb"), "schema": "main"}
    )
    with create_adapter(cfg) as ad:
        yield ad


def _materialize(ad: WarehouseAdapter, name: str, frame: pl.DataFrame) -> None:
    ad.materialize_full(name, frame)


def _drift(ad: WarehouseAdapter, model: str, spec: dict[str, Any]) -> Any:
    return evaluate_test_spec(
        {"drift": spec}, model_name=model, table_ref=ad.table_ref(model), adapter=ad
    )[0]


# ─── numeric drift ───────────────────────────────────────────────────────────


def test_identical_numeric_has_zero_drift(adapter: WarehouseAdapter) -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    _materialize(adapter, "baseline", frame)
    _materialize(adapter, "current", frame)
    for metric in ("psi", "ks", "jensen_shannon"):
        r = _drift(adapter, "current", {"column": "x", "to": "ref('baseline')",
                                        "metric": metric, "max": 0.0})
        assert r.status == "pass", (metric, r.message)


def test_ks_is_one_for_disjoint_ranges(adapter: WarehouseAdapter) -> None:
    _materialize(adapter, "baseline", pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}))
    _materialize(adapter, "current", pl.DataFrame({"x": [5.0, 6.0, 7.0, 8.0]}))
    r = _drift(adapter, "current", {"column": "x", "to": "ref('baseline')",
                                    "metric": "ks", "max": 0.5})
    assert r.status == "fail"
    assert "ks=1" in r.message


def test_numeric_shift_trips_psi(adapter: WarehouseAdapter) -> None:
    import random

    random.seed(0)
    _materialize(adapter, "baseline", pl.DataFrame({"x": [random.gauss(0, 1) for _ in range(400)]}))
    _materialize(adapter, "current", pl.DataFrame({"x": [random.gauss(3, 1) for _ in range(400)]}))
    r = _drift(adapter, "current", {"column": "x", "to": "ref('baseline')", "max": 0.2})
    assert r.status == "fail"
    assert r.test_name == "drift"


# ─── categorical drift ───────────────────────────────────────────────────────


def test_identical_categorical_zero_psi(adapter: WarehouseAdapter) -> None:
    frame = pl.DataFrame({"c": ["a", "a", "b", "b", "c"]})
    _materialize(adapter, "baseline", frame)
    _materialize(adapter, "current", frame)
    r = _drift(adapter, "current", {"column": "c", "to": "ref('baseline')", "max": 0.0})
    assert r.status == "pass"
    assert "psi=0" in r.message


def test_categorical_proportion_shift_fails(adapter: WarehouseAdapter) -> None:
    _materialize(adapter, "baseline", pl.DataFrame({"c": ["a", "a", "a", "b", "b"]}))
    _materialize(adapter, "current", pl.DataFrame({"c": ["a", "b", "b", "b", "b"]}))
    r = _drift(adapter, "current", {"column": "c", "to": "ref('baseline')",
                                    "metric": "psi", "max": 0.1})
    assert r.status == "fail"


def test_ks_on_categorical_fails_actionably(adapter: WarehouseAdapter) -> None:
    frame = pl.DataFrame({"c": ["a", "b", "c", "d"]})
    _materialize(adapter, "baseline", frame)
    _materialize(adapter, "current", frame)
    r = _drift(adapter, "current", {"column": "c", "to": "ref('baseline')",
                                    "metric": "ks", "max": 0.5})
    assert r.status == "fail"
    assert "requires numeric" in r.message


def test_type_mismatch_fails(adapter: WarehouseAdapter) -> None:
    _materialize(adapter, "baseline", pl.DataFrame({"x": [1.0, 2.0, 3.0]}))
    _materialize(adapter, "current", pl.DataFrame({"x": ["a", "b", "c"]}))
    r = _drift(adapter, "current", {"column": "x", "to": "ref('baseline')", "max": 0.2})
    assert r.status == "fail"
    assert "both" in r.message


def test_field_option_maps_to_baseline_column(adapter: WarehouseAdapter) -> None:
    _materialize(adapter, "baseline", pl.DataFrame({"legacy_x": [1.0, 2.0, 3.0, 4.0]}))
    _materialize(adapter, "current", pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}))
    r = _drift(adapter, "current", {"column": "x", "to": "ref('baseline')",
                                    "field": "legacy_x", "metric": "ks", "max": 0.1})
    assert r.status == "pass"


def test_empty_baseline_is_insufficient_data(adapter: WarehouseAdapter) -> None:
    _materialize(adapter, "baseline", pl.DataFrame({"x": []}, schema={"x": pl.Float64}))
    _materialize(adapter, "current", pl.DataFrame({"x": [1.0, 2.0, 3.0]}))
    r = _drift(adapter, "current", {"column": "x", "to": "ref('baseline')", "max": 0.2})
    assert r.status == "pass"
    assert "insufficient data" in r.message


# ─── chi_squared ─────────────────────────────────────────────────────────────


def test_chi_squared_zero_for_identical(adapter: WarehouseAdapter) -> None:
    frame = pl.DataFrame({"c": ["a", "a", "b", "b"]})
    _materialize(adapter, "baseline", frame)
    _materialize(adapter, "current", frame)
    r = _drift(adapter, "current", {"column": "c", "to": "ref('baseline')",
                                    "metric": "chi_squared", "max": 0.001})
    assert r.status == "pass"
    assert "chi_squared=0" in r.message


def test_chi_squared_trips_on_categorical_shift(adapter: WarehouseAdapter) -> None:
    _materialize(adapter, "baseline", pl.DataFrame({"c": ["a", "a", "a", "b", "b"]}))
    _materialize(adapter, "current", pl.DataFrame({"c": ["a", "b", "b", "b", "b"]}))
    r = _drift(adapter, "current", {"column": "c", "to": "ref('baseline')",
                                    "metric": "chi_squared", "max": 1.0})
    assert r.status == "fail"


# ─── robustness (Codex review) ───────────────────────────────────────────────


def test_constant_baseline_still_detects_shift(adapter: WarehouseAdapter) -> None:
    # A single-valued baseline collapses quantile bins; drift must still fire for
    # a current column that has moved entirely off that constant.
    _materialize(adapter, "baseline", pl.DataFrame({"x": [0.0, 0.0, 0.0]}))
    _materialize(adapter, "current", pl.DataFrame({"x": [100.0, 100.0, 100.0]}))
    for metric in ("psi", "jensen_shannon"):
        r = _drift(adapter, "current", {"column": "x", "to": "ref('baseline')",
                                        "metric": metric, "max": 0.2})
        assert r.status == "fail", (metric, r.message)


def test_constant_baseline_no_shift_passes(adapter: WarehouseAdapter) -> None:
    _materialize(adapter, "baseline", pl.DataFrame({"x": [5.0, 5.0]}))
    _materialize(adapter, "current", pl.DataFrame({"x": [5.0, 5.0, 5.0]}))
    r = _drift(adapter, "current", {"column": "x", "to": "ref('baseline')", "max": 0.2})
    assert r.status == "pass"
    assert "psi=0" in r.message


def test_nonscalar_categorical_column_fails_actionably(adapter: WarehouseAdapter) -> None:
    # A list/struct column classifies as categorical; the unhashable values must
    # produce a failed check, not an uncaught TypeError that aborts the run.
    frame = pl.DataFrame({"tags": [["a"], ["b"]]}, schema={"tags": pl.List(pl.String)})
    _materialize(adapter, "baseline", frame)
    _materialize(adapter, "current", frame)
    with pytest.raises(SpecError, match="unhashable"):
        _drift(adapter, "current", {"column": "tags", "to": "ref('baseline')", "max": 0.2})


# ─── dependency wiring + validation ──────────────────────────────────────────


def test_drift_target_is_a_dag_dependency() -> None:
    targets = relationship_test_targets(
        [{"drift": {"column": "x", "to": "ref('baseline_snapshot')", "max": 0.2}}]
    )
    assert targets == {"baseline_snapshot"}


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"drift": {"column": "x", "max": 0.2}}, "missing required options"),
        ({"drift": {"column": "x", "to": "ref('b')"}}, "missing required options"),
        (
            {"drift": {"column": "x", "to": "ref('b')", "metric": "wasserstein", "max": 0.2}},
            "metric must be one of",
        ),
        ({"drift": {"column": "x", "to": "ref('b')", "max": -1}}, "non-negative"),
        ({"drift": {"column": "x", "to": "ref('b')", "max": 0.2, "bins": 1}}, "bins must be"),
    ],
)
def test_drift_specs_are_strict(spec: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        parse_test_spec(spec)
