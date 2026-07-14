from __future__ import annotations

from pathlib import Path

from ..adapters import WarehouseAdapter, WarehouseCapability, create_adapter
from ..compiler import validate_project_contract, validate_warehouse_capabilities
from ..config import load_project
from ..config.model import ModelConfig
from ..profile import resolve_profile
from .schema import TestResult, UnknownTestError, evaluate_test_spec


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
    validate_warehouse_capabilities(
        [model for model in models if model.name in selected_names],
        resolved.warehouse.type,
    )

    results: list[TestResult] = []
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        for model in models:
            if model.name not in selected_names:
                continue
            if not model.tests:
                continue
            results.extend(
                run_model_tests(
                    model, adapter, project_dir=project_dir,
                    store_failures=store_failures,
                )
            )
    return results


def run_model_tests(
    model: ModelConfig,
    adapter: WarehouseAdapter,
    *,
    project_dir: Path | None = None,
    store_failures: bool = False,
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
