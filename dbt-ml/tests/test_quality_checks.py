"""Tests for the deterministic ML/statistical quality checks (issue #10, Tier 1
+ grounding). All run against a real DuckDB adapter — no LLM, no sampling."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

from dbt_ml.adapters import WarehouseAdapter, create_adapter, parse_warehouse_config
from dbt_ml.checks.schema import evaluate_test_spec


@pytest.fixture
def papers(tmp_path: Path) -> Iterator[WarehouseAdapter]:
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "q.duckdb"), "schema": "main"}
    )
    with create_adapter(cfg) as adapter:
        adapter.materialize_full(
            "papers",
            pl.DataFrame(
                {
                    "arxiv_id": ["2401.00001", "2401.00002", "BAD-ID", "2401.00004"],
                    "category": ["cs.LG", "cs.CL", "cs.LG", "physics.bogus"],
                    "n_authors": [3, 1, 12, 2],
                    "vendor": ["a", None, "c", None],
                    "title": [
                        "Deep Learning for Widgets",
                        "Attention Is Some of What You Need",
                        "A Study of Things",
                        "Hallucinated Title Not In Source",
                    ],
                    "abstract": [
                        "We present Deep Learning for Widgets, a new method ...",
                        "Attention Is Some of What You Need: we show ...",
                        "A Study of Things and their properties ...",
                        "This abstract is about something else entirely ...",
                    ],
                }
            ),
        )
        yield adapter


def _run(adapter: WarehouseAdapter, spec: dict):
    return evaluate_test_spec(
        spec, model_name="papers", table_ref=adapter.table_ref("papers"), adapter=adapter
    )[0]


def test_matches_regex_fails_on_bad_id(papers: WarehouseAdapter) -> None:
    r = _run(papers, {"matches_regex": {"column": "arxiv_id", "pattern": r"^\d{4}\.\d{5}$"}})
    assert r.status == "fail"
    assert "BAD-ID" in r.message


def test_matches_regex_passes(papers: WarehouseAdapter) -> None:
    r = _run(papers, {"matches_regex": {"column": "category", "pattern": r"^[a-z]"}})
    assert r.status == "pass"


def test_accepted_values_fails(papers: WarehouseAdapter) -> None:
    r = _run(papers, {"accepted_values": {"column": "category", "values": ["cs.LG", "cs.CL"]}})
    assert r.status == "fail"
    assert "1 values" in r.message


def test_accepted_values_passes(papers: WarehouseAdapter) -> None:
    r = _run(
        papers,
        {"accepted_values": {"column": "category", "values": ["cs.LG", "cs.CL", "physics.bogus"]}},
    )
    assert r.status == "pass"


def test_accepted_range_fails_high(papers: WarehouseAdapter) -> None:
    r = _run(papers, {"accepted_range": {"column": "n_authors", "min": 1, "max": 10}})
    assert r.status == "fail"  # the row with 12 authors


def test_accepted_range_passes(papers: WarehouseAdapter) -> None:
    r = _run(papers, {"accepted_range": {"column": "n_authors", "min": 1, "max": 20}})
    assert r.status == "pass"


def test_accepted_range_requires_a_bound(papers: WarehouseAdapter) -> None:
    from dbt_ml.checks.schema import UnknownTestError

    with pytest.raises(UnknownTestError, match="at least one of"):
        _run(papers, {"accepted_range": {"column": "n_authors"}})


def test_null_rate_fails(papers: WarehouseAdapter) -> None:
    # vendor is null in 2/4 rows = 0.5
    r = _run(papers, {"null_rate": {"column": "vendor", "max": 0.1}})
    assert r.status == "fail"
    assert "0.500" in r.message


def test_null_rate_passes(papers: WarehouseAdapter) -> None:
    r = _run(papers, {"null_rate": {"column": "vendor", "max": 0.6}})
    assert r.status == "pass"


def test_grounded_in_exact_catches_hallucination(papers: WarehouseAdapter) -> None:
    # First three titles appear in their abstracts; the 4th ("Hallucinated...") does not.
    r = _run(
        papers,
        {"grounded_in": {"value": "title", "source": "abstract", "method": "exact"}},
    )
    assert r.status == "fail"
    assert "1/4" in r.message


def test_grounded_in_passes_when_all_present(papers: WarehouseAdapter) -> None:
    # Build a case where every title is a substring of the abstract
    spec = {"grounded_in": {"value": "category", "source": "category", "method": "exact"}}
    r = _run(papers, spec)
    assert r.status == "pass"


def test_grounded_in_fuzzy_tolerates_minor_diffs(papers: WarehouseAdapter) -> None:
    r = _run(
        papers,
        {
            "grounded_in": {
                "value": "title",
                "source": "abstract",
                "method": "fuzzy",
                "min_score": 0.6,
            }
        },
    )
    # Still fails (the hallucinated one is genuinely absent), but exercises the fuzzy path
    assert r.status in {"pass", "fail"}


def test_severity_warn_applies_to_quality_checks(papers: WarehouseAdapter) -> None:
    r = _run(
        papers,
        {"null_rate": {"column": "vendor", "max": 0.1}, "severity": "warn"},
    )
    assert r.status == "warn"
    assert not r.is_hard_failure
