"""Deterministic embedding-quality checks (issue #10). All run against a real
DuckDB adapter over a vector column — no provider, no sampling."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel.adapters import create_adapter, parse_warehouse_config
from stel.checks.schema import TestResult, evaluate_test_spec
from stel.test_specs import TestSpecError as SpecError
from stel.test_specs import parse_test_spec


def _check(tmp_path: Path, vectors: list[list[float] | None], spec: dict[str, Any]) -> TestResult:
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "e.duckdb"), "schema": "main"}
    )
    with create_adapter(cfg) as adapter:
        adapter.materialize_full(
            "emb",
            pl.DataFrame(
                {"id": [str(i) for i in range(len(vectors))], "vec": vectors},
                schema={"id": pl.String, "vec": pl.List(pl.Float64)},
            ),
        )
        return evaluate_test_spec(
            spec, model_name="emb", table_ref=adapter.table_ref("emb"), adapter=adapter
        )[0]


# ─── embedding_valid ────────────────────────────────────────────────────────


def test_embedding_valid_passes_clean_unit_vectors(tmp_path: Path) -> None:
    r = _check(tmp_path, [[1.0, 0.0], [0.0, 1.0]], {"embedding_valid": {"column": "vec"}})
    assert r.status == "pass"


def test_embedding_valid_flags_zero_vector_by_default(tmp_path: Path) -> None:
    r = _check(
        tmp_path, [[1.0, 0.0], [0.0, 0.0]], {"embedding_valid": {"column": "vec"}}
    )
    assert r.status == "fail"
    assert "zero-vector rate 0.500" in r.message


def test_embedding_valid_zero_rate_tolerance(tmp_path: Path) -> None:
    r = _check(
        tmp_path,
        [[1.0, 0.0], [0.0, 0.0]],
        {"embedding_valid": {"column": "vec", "max_zero_rate": 0.5}},
    )
    assert r.status == "pass"


def test_embedding_valid_flags_wrong_dimension(tmp_path: Path) -> None:
    r = _check(
        tmp_path,
        [[1.0, 0.0, 0.0]],
        {"embedding_valid": {"column": "vec", "dimensions": 2}},
    )
    assert r.status == "fail"
    assert "1 wrong-dimension" in r.message


def test_embedding_valid_flags_out_of_norm_range(tmp_path: Path) -> None:
    # norms are 5 and 13; max_norm 10 rejects the second.
    r = _check(
        tmp_path,
        [[3.0, 4.0], [5.0, 12.0]],
        {"embedding_valid": {"column": "vec", "min_norm": 1.0, "max_norm": 10.0}},
    )
    assert r.status == "fail"
    assert "1 out-of-norm-range" in r.message


def test_embedding_valid_skips_nulls(tmp_path: Path) -> None:
    r = _check(tmp_path, [[1.0, 0.0], None], {"embedding_valid": {"column": "vec"}})
    assert r.status == "pass"


def test_embedding_valid_missing_column_fails_actionably(tmp_path: Path) -> None:
    # A missing column raises UnknownTestError, which the test runner turns into a
    # fail result (see run_model_tests); here we assert the actionable message.
    with pytest.raises(SpecError, match="not found"):
        _check(tmp_path, [[1.0, 0.0]], {"embedding_valid": {"column": "nope"}})


# ─── embedding_variance ─────────────────────────────────────────────────────


def test_embedding_variance_passes_spread_vectors(tmp_path: Path) -> None:
    r = _check(
        tmp_path,
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
        {"embedding_variance": {"column": "vec", "min_variance": 0.1}},
    )
    assert r.status == "pass"


def test_embedding_variance_detects_collapse(tmp_path: Path) -> None:
    # Every vector identical → mean per-dimension variance is exactly 0.
    r = _check(
        tmp_path,
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        {"embedding_variance": {"column": "vec", "min_variance": 0.01}},
    )
    assert r.status == "fail"
    assert "variance=0" in r.message


def test_embedding_variance_hand_computed(tmp_path: Path) -> None:
    # dim0 = {0,2} var=1.0, dim1 = {0,0} var=0.0 → mean per-dim variance = 0.5.
    r = _check(
        tmp_path,
        [[0.0, 0.0], [2.0, 0.0]],
        {"embedding_variance": {"column": "vec", "min_variance": 0.5}},
    )
    assert r.status == "pass"
    assert "variance=0.5" in r.message


# ─── embedding_duplicates ───────────────────────────────────────────────────


def test_embedding_duplicates_flags_exact_copies(tmp_path: Path) -> None:
    r = _check(
        tmp_path,
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        {"embedding_duplicates": {"column": "vec"}},
    )
    assert r.status == "fail"
    assert "1/3 duplicate vector copies" in r.message


def test_embedding_duplicates_tolerates_within_rate(tmp_path: Path) -> None:
    r = _check(
        tmp_path,
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        {"embedding_duplicates": {"column": "vec", "max_rate": 0.5}},
    )
    assert r.status == "pass"


# ─── embedding_outliers ─────────────────────────────────────────────────────


def test_embedding_outliers_flags_far_vector(tmp_path: Path) -> None:
    r = _check(
        tmp_path,
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [50.0, 50.0]],
        {"embedding_outliers": {"column": "vec", "z": 1.0}},
    )
    assert r.status == "fail"
    assert "beyond 1σ" in r.message


def test_embedding_outliers_pass_when_tight(tmp_path: Path) -> None:
    r = _check(
        tmp_path,
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]],
        {"embedding_outliers": {"column": "vec", "z": 3.0}},
    )
    assert r.status == "pass"


def test_embedding_outliers_needs_three_vectors(tmp_path: Path) -> None:
    r = _check(
        tmp_path, [[0.0, 0.0], [9.0, 9.0]], {"embedding_outliers": {"column": "vec"}}
    )
    assert r.status == "pass"
    assert "need >= 3" in r.message


# ─── robustness: non-vector columns and non-finite input ────────────────────


def _adapter(tmp_path: Path) -> Any:
    return create_adapter(
        parse_warehouse_config(
            {"type": "duckdb", "path": str(tmp_path / "e.duckdb"), "schema": "main"}
        )
    )


@pytest.mark.parametrize("bad_column", [[1, 2, 3], ["10", "20", "30"]])
def test_non_vector_column_raises_unknown_test_error(
    tmp_path: Path, bad_column: list[Any]
) -> None:
    # A scalar/string column (typo or schema drift) must raise UnknownTestError —
    # which run_model_tests turns into a failed check — not a TypeError that aborts
    # the run, and a string must never be silently read as a character vector.
    with _adapter(tmp_path) as adapter:
        adapter.materialize_full("emb", pl.DataFrame({"vec": bad_column}))
        with pytest.raises(SpecError, match="must contain vectors"):
            evaluate_test_spec(
                {"embedding_valid": {"column": "vec"}},
                model_name="emb",
                table_ref=adapter.table_ref("emb"),
                adapter=adapter,
            )


@pytest.mark.parametrize("check", ["embedding_variance", "embedding_outliers"])
def test_nonfinite_vectors_fail_without_crashing(tmp_path: Path, check: str) -> None:
    # Opposing infinities make math.fsum raise; the aggregate checks must report a
    # failure themselves instead of letting the ValueError abort the run.
    spec: dict[str, Any] = {check: {"column": "vec"}}
    if check == "embedding_variance":
        spec[check]["min_variance"] = 0.01
    r = _check(
        tmp_path,
        [[float("inf")], [float("-inf")], [0.0]],
        spec,
    )
    assert r.status == "fail"
    assert "non-finite" in r.message


# ─── --store-failures persistence ────────────────────────────────────────────


def test_store_failures_persists_duplicate_rows(tmp_path: Path) -> None:
    with _adapter(tmp_path) as adapter:
        adapter.materialize_full(
            "emb",
            pl.DataFrame(
                {"id": ["a", "b", "c"], "vec": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]},
                schema={"id": pl.String, "vec": pl.List(pl.Float64)},
            ),
        )
        r = evaluate_test_spec(
            {"embedding_duplicates": {"column": "vec"}},
            model_name="emb",
            table_ref=adapter.table_ref("emb"),
            adapter=adapter,
            store_failures=True,
        )[0]
        assert r.status == "fail"
        assert r.failures_table is not None
        assert r.failure_count == 2  # both members of the duplicate group
        stored = adapter.query_df(f"SELECT * FROM {adapter.table_ref(r.failures_table)}")
        assert stored.height == 2
        assert set(stored["id"].to_list()) == {"a", "b"}


def test_store_failures_persists_zero_vector_rows(tmp_path: Path) -> None:
    with _adapter(tmp_path) as adapter:
        adapter.materialize_full(
            "emb",
            pl.DataFrame(
                {"id": ["a", "b"], "vec": [[1.0, 0.0], [0.0, 0.0]]},
                schema={"id": pl.String, "vec": pl.List(pl.Float64)},
            ),
        )
        r = evaluate_test_spec(
            {"embedding_valid": {"column": "vec"}},
            model_name="emb",
            table_ref=adapter.table_ref("emb"),
            adapter=adapter,
            store_failures=True,
        )[0]
        assert r.status == "fail"
        assert r.failure_count == 1  # the single zero vector
        assert r.failures_table is not None
        stored = adapter.query_df(f"SELECT * FROM {adapter.table_ref(r.failures_table)}")
        assert stored["id"].to_list() == ["b"]


# ─── compile-time validation ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"embedding_valid": {}}, "missing required options"),
        ({"embedding_valid": {"column": "v", "dimensions": 0}}, "positive integer"),
        ({"embedding_valid": {"column": "v", "min_norm": -1}}, "non-negative"),
        (
            {"embedding_valid": {"column": "v", "min_norm": 5, "max_norm": 1}},
            "min_norm <= max_norm",
        ),
        ({"embedding_valid": {"column": "v", "max_zero_rate": 2}}, "between 0 and 1"),
        ({"embedding_valid": {"column": "v", "bogus": 1}}, "unknown options"),
        ({"embedding_variance": {"column": "v"}}, "missing required options"),
        (
            {"embedding_variance": {"column": "v", "min_variance": -1}},
            "non-negative",
        ),
        ({"embedding_duplicates": {"column": "v", "max_rate": 1.5}}, "between 0 and 1"),
        ({"embedding_outliers": {"column": "v", "z": 0}}, "z must be a positive"),
    ],
)
def test_embedding_check_specs_are_strict(spec: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        parse_test_spec(spec)


def test_embedding_check_specs_accept_valid_configuration() -> None:
    for spec in (
        {"embedding_valid": {"column": "embedding", "dimensions": 8, "max_norm": 2.0}},
        {"embedding_variance": {"column": "embedding", "min_variance": 0.001}},
        {"embedding_duplicates": {"column": "embedding", "max_rate": 0.01}},
        {"embedding_outliers": {"column": "embedding", "z": 4.0, "max_rate": 0.02}},
    ):
        assert parse_test_spec(spec).name.startswith("embedding_")
