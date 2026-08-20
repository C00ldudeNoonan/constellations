from __future__ import annotations

from pathlib import Path

from ..adapters import WarehouseAdapter, WarehouseCapability, create_adapter
from ..budget import BudgetLedger
from ..compiler import validate_project_contract, validate_warehouse_capabilities
from ..config import load_project
from ..config.model import ModelConfig
from ..profile import ResolvedProfile, resolve_profile
from ..test_specs import uses_llm_judge
from .schema import TestResult, UnknownTestError, evaluate_test_spec


def validate_test_requirements(
    models: list[ModelConfig], resolved: ResolvedProfile
) -> None:
    """Preflight: fail before discovery/build/materialization when a selected
    model declares an `llm_judge` test but no `llm:` profile is configured."""
    if resolved.llm is not None:
        return
    offenders = sorted(
        model.name
        for model in models
        if model.tests and uses_llm_judge(model.tests)
    )
    if offenders:
        raise UnknownTestError(
            f"models {offenders} declare an `llm_judge` test but no `llm:` profile "
            "is configured; add an `llm:` block to the active profile"
        )


def _test_run_budget(resolved: ResolvedProfile) -> BudgetLedger | None:
    """One run-scope budget ledger shared across every model's tests, so
    `llm_judge` calls honor `llm.budget` caps just like `llm:` models."""
    if resolved.llm is None or resolved.llm.budget is None:
        return None
    return BudgetLedger(resolved.llm.budget, scope="run")


def run_project_tests(
    project_dir: Path,
    *,
    select: str | None = None,
    exclude: str | None = None,
    target: str | None = None,
    profiles_dir: Path | None = None,
    store_failures: bool = False,
    state: Path | None = None,
) -> list[TestResult]:
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    modified: set[str] | None = None
    if state is not None:
        # Local import: manifest.py imports from runner.py at module level.
        from ..manifest import compute_modified_models

        modified = compute_modified_models(
            models,
            project_dir,
            state,
            project=project,
            resolved=resolved,
        )
    selected_names = set(
        dag.select_models(select=select, exclude=exclude, modified=modified)
    )
    selected = [model for model in models if model.name in selected_names]
    adapter = create_adapter(resolved.warehouse, project_dir=project_dir)
    validate_warehouse_capabilities(selected, adapter)
    validate_test_requirements(selected, resolved)
    run_budget = _test_run_budget(resolved)

    results: list[TestResult] = []
    with adapter:
        for model in models:
            if model.name not in selected_names:
                continue
            if not model.tests:
                continue
            results.extend(
                run_model_tests(
                    model, adapter, project_dir=project_dir,
                    store_failures=store_failures,
                    resolved=resolved,
                    run_budget=run_budget,
                )
            )
    return results


def run_model_tests(
    model: ModelConfig,
    adapter: WarehouseAdapter,
    *,
    project_dir: Path | None = None,
    store_failures: bool = False,
    resolved: ResolvedProfile | None = None,
    run_budget: BudgetLedger | None = None,
) -> list[TestResult]:
    if not model.tests:
        return []
    adapter.require_capability(
        WarehouseCapability.SQL_SCHEMA_TESTS,
        operation=f"running tests for model '{model.name}'",
    )
    if model.name not in set(adapter.list_tables()):
        return [
            TestResult(
                test_name="relation_exists",
                model_name=model.name,
                column=None,
                status="fail",
                message=(
                    f"Materialized relation for model '{model.name}' is missing. "
                    "Run the model before testing it; a model that matched zero "
                    "documents may not have created a relation."
                ),
            )
        ]

    table_ref = adapter.table_ref(model.name)
    out: list[TestResult] = []
    for spec in model.tests:
        try:
            out.extend(
                evaluate_test_spec(
                    spec,
                    model_name=model.name,
                    table_ref=table_ref,
                    adapter=adapter,
                    project_dir=project_dir,
                    store_failures=store_failures,
                    resolved=resolved,
                    run_budget=run_budget,
                )
            )
        except UnknownTestError as e:
            out.append(
                TestResult(
                    test_name=str(spec),
                    model_name=model.name,
                    column=None,
                    status="fail",
                    message=str(e),
                )
            )
    return out
