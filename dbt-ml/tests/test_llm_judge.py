"""Optional sampled LLM-as-judge check (issue #10). Runs against the offline
deterministic inference provider — no credentials, no network."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dbt_ml.adapters import WarehouseAdapter, create_adapter, parse_warehouse_config
from dbt_ml.checks.schema import evaluate_test_spec
from dbt_ml.config.profile import LLMConfig
from dbt_ml.profile import ResolvedProfile
from dbt_ml.test_specs import TestSpecError as SpecError
from dbt_ml.test_specs import parse_test_spec


@pytest.fixture
def adapter(tmp_path: Path) -> Any:
    cfg = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "j.duckdb"), "schema": "main"}
    )
    with create_adapter(cfg) as ad:
        ad.materialize_full(
            "docs",
            pl.DataFrame({"body": [f"document {i} about widgets" for i in range(8)]}),
        )
        yield ad


def _resolved(warehouse: Any, *, llm: LLMConfig | None) -> ResolvedProfile:
    return ResolvedProfile(
        profile_name="t", target_name="dev", warehouse=warehouse, llm=llm,
        source_paths={}, profiles_path=None,
    )


_DUMMY_WH = parse_warehouse_config({"type": "duckdb", "path": "./x.duckdb", "schema": "main"})


def _judge(ad: WarehouseAdapter, spec: dict[str, Any], *, llm: LLMConfig | None) -> Any:
    # llm_judge only reads resolved.llm; the warehouse field is unused here.
    return evaluate_test_spec(
        {"llm_judge": spec}, model_name="docs", table_ref=ad.table_ref("docs"),
        adapter=ad, resolved=_resolved(_DUMMY_WH, llm=llm),
    )[0]


def _det() -> LLMConfig:
    return LLMConfig(provider="deterministic")


def test_min_pass_rate_zero_always_passes(adapter: WarehouseAdapter) -> None:
    r = _judge(
        adapter,
        {"column": "body", "criterion": "mentions a product", "sample_size": 5,
         "seed": 0, "min_pass_rate": 0.0},
        llm=_det(),
    )
    assert r.status == "pass"
    assert "sampled 5 of 8" in r.message


def test_min_pass_rate_one_fails_when_not_all_pass(adapter: WarehouseAdapter) -> None:
    # The deterministic provider yields a mixed verdict set, so requiring 100%
    # pass fails — exercising the fail path without a real model.
    r = _judge(
        adapter,
        {"column": "body", "criterion": "mentions a product", "sample_size": 8,
         "seed": 0, "min_pass_rate": 1.0},
        llm=_det(),
    )
    assert r.status == "fail"


def test_same_seed_is_deterministic(adapter: WarehouseAdapter) -> None:
    spec = {"column": "body", "criterion": "is about widgets", "sample_size": 4,
            "seed": 7, "min_pass_rate": 0.0}
    first = _judge(adapter, spec, llm=_det())
    second = _judge(adapter, spec, llm=_det())
    assert first.message == second.message


def test_requires_llm_profile(adapter: WarehouseAdapter) -> None:
    with pytest.raises(SpecError, match="requires an LLM profile"):
        _judge(adapter, {"column": "body", "criterion": "x"}, llm=None)


def test_missing_column_fails_actionably(adapter: WarehouseAdapter) -> None:
    with pytest.raises(SpecError, match="not found"):
        _judge(adapter, {"column": "nope", "criterion": "x"}, llm=_det())


def test_provider_error_becomes_failed_check_not_crash(adapter: WarehouseAdapter) -> None:
    # An unknown provider must fail the check (ProviderError caught), not abort
    # the whole test run.
    r = _judge(
        adapter,
        {"column": "body", "criterion": "x", "sample_size": 2, "min_pass_rate": 0.0},
        llm=LLMConfig(provider="no_such_provider"),
    )
    assert r.status == "fail"
    assert "could not run" in r.message


def test_empty_text_passes_without_calls(adapter: WarehouseAdapter) -> None:
    adapter.materialize_full(
        "docs", pl.DataFrame({"body": [None, "  "]}, schema={"body": pl.String})
    )
    r = _judge(adapter, {"column": "body", "criterion": "x"}, llm=_det())
    assert r.status == "pass"
    assert "no non-null text" in r.message


def test_run_budget_caps_judge_calls(adapter: WarehouseAdapter) -> None:
    from dbt_ml.budget import BudgetLedger, LLMBudgetConfig

    budget = BudgetLedger(LLMBudgetConfig(max_api_calls=2), scope="run")
    r = evaluate_test_spec(
        {"llm_judge": {"column": "body", "criterion": "x", "sample_size": 5,
                       "seed": 0, "min_pass_rate": 0.0}},
        model_name="docs", table_ref=adapter.table_ref("docs"), adapter=adapter,
        resolved=_resolved(_DUMMY_WH, llm=LLMConfig(provider="deterministic")),
        run_budget=budget,
    )[0]
    assert r.status == "fail"
    assert "run budget exceeded" in r.message
    assert budget.snapshot()["api_calls"] == 2


def test_sampling_is_order_independent(adapter: WarehouseAdapter) -> None:
    # Same data, different physical row order -> same seeded sample -> same verdict.
    spec = {"column": "body", "criterion": "x", "sample_size": 3, "seed": 1,
            "min_pass_rate": 0.0}
    forward = _judge(adapter, spec, llm=_det())
    adapter.materialize_full(
        "docs", pl.DataFrame({"body": [f"document {i} about widgets" for i in reversed(range(8))]})
    )
    reverse = _judge(adapter, spec, llm=_det())
    assert forward.message == reverse.message


def test_preflight_requires_llm_profile() -> None:
    from types import SimpleNamespace
    from typing import cast

    from dbt_ml.checks.runner import validate_test_requirements
    from dbt_ml.config.model import ModelConfig

    models = cast(
        list[ModelConfig],
        [SimpleNamespace(name="m", tests=[{"llm_judge": {"column": "c", "criterion": "x"}}])],
    )
    with pytest.raises(SpecError, match="llm_judge"):
        validate_test_requirements(models, _resolved(_DUMMY_WH, llm=None))
    # With an llm: profile the preflight passes.
    validate_test_requirements(models, _resolved(_DUMMY_WH, llm=_det()))


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"llm_judge": {"column": "body"}}, "missing required options"),
        ({"llm_judge": {"criterion": "x"}}, "missing required options"),
        (
            {"llm_judge": {"column": "b", "criterion": "x", "sample_size": 0}},
            "positive integer",
        ),
        (
            {"llm_judge": {"column": "b", "criterion": "x", "min_pass_rate": 1.5}},
            "between 0 and 1",
        ),
        ({"llm_judge": {"column": "b", "criterion": "x", "seed": "z"}}, "seed must be"),
    ],
)
def test_llm_judge_specs_are_strict(spec: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        parse_test_spec(spec)
