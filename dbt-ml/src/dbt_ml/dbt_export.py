"""Translate a dbt-ml project into a dbt-compatible sources.yml.

A dbt project using the matching warehouse adapter can consume dbt-ml tables
via `{{ source('dbt_ml_<project>', '<model>') }}`. This module emits the
sources.yml declaration that makes that work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import load_project
from .config.model import ModelConfig
from .config.profile import WarehouseConfig
from .dag import ProjectDAG
from .profile import resolve_profile

DEFAULT_OUTPUT_FILENAME = "sources.yml"


def build_dbt_sources(
    project_dir: Path,
    *,
    source_name: str | None = None,
    select: str | None = None,
    exclude: str | None = None,
    target: str | None = None,
    profiles_dir: Path | None = None,
    dagster_meta: bool = False,
) -> dict[str, Any]:
    project, sources_cfg, models = load_project(project_dir)
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    dag = ProjectDAG(sources_cfg, models)
    selected_names = set(dag.select_models(select=select, exclude=exclude))
    selected_models = [m for m in models if m.name in selected_names]

    name = source_name or f"dbt_ml_{project.name}"
    catalog = _derive_catalog(resolved.warehouse)

    return {
        "version": 2,
        "sources": [
            {
                "name": name,
                "description": (
                    f"Tables materialized by dbt-ml project '{project.name}'."
                ),
                "database": catalog,
                "schema": resolved.warehouse.schema_name,
                "tables": [
                    _table_for_model(m, source_name=name, dagster_meta=dagster_meta)
                    for m in selected_models
                ],
            }
        ],
    }


def write_dbt_sources(
    project_dir: Path,
    *,
    source_name: str | None = None,
    select: str | None = None,
    exclude: str | None = None,
    output: Path | None = None,
    target: str | None = None,
    profiles_dir: Path | None = None,
    dagster_meta: bool = False,
) -> Path:
    project, _, _ = load_project(project_dir)
    payload = build_dbt_sources(
        project_dir,
        source_name=source_name,
        select=select,
        exclude=exclude,
        target=target,
        profiles_dir=profiles_dir,
        dagster_meta=dagster_meta,
    )

    if output is None:
        target_dir = (project_dir / project.target_path).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        output = target_dir / DEFAULT_OUTPUT_FILENAME
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
    return output


def _derive_catalog(warehouse: WarehouseConfig) -> str:
    """Adapter-specific catalog: DuckDB file stem, BigQuery project, ..."""
    return warehouse.catalog_name()


def _table_for_model(
    model: ModelConfig, *, source_name: str, dagster_meta: bool = False
) -> dict[str, Any]:
    columns_by_name: dict[str, dict[str, Any]] = {}

    for field in model.fields:
        columns_by_name[field.name] = {"name": field.name}
        if field.description:
            columns_by_name[field.name]["description"] = field.description
        if field.data_type:
            # Adapters receive nested values after they are flattened to JSON text.
            columns_by_name[field.name]["data_type"] = (
                "string" if field.data_type == "json" else field.data_type
            )

    table_tests: list[Any] = []
    for spec in model.tests:
        _apply_test_spec(spec, columns_by_name, table_tests)

    table: dict[str, Any] = {"name": model.name}
    if model.description:
        table["description"] = model.description
    if model.tags:
        table["tags"] = model.tags

    if dagster_meta:
        # Pin the Dagster asset key to dagster-dbt's default source key
        # ([source, table]) so a dbt-ml producer asset and the dbt models that
        # {{ source(...) }} it agree on one key without hand-copying. dbt itself
        # ignores unknown `meta`, so pure-dbt consumers are unaffected.
        table["meta"] = {"dagster": {"asset_key": [source_name, model.name]}}

    if columns_by_name:
        table["columns"] = list(columns_by_name.values())
    if table_tests:
        table["tests"] = table_tests
    return table


def _apply_test_spec(
    spec: Any,
    columns_by_name: dict[str, dict[str, Any]],
    table_tests: list[Any],
) -> None:
    if isinstance(spec, str):
        _attach_table_test(spec, None, table_tests)
        return
    if not isinstance(spec, dict) or len(spec) != 1:
        return
    ((name, arg),) = spec.items()

    if name == "not_null":
        cols = arg if isinstance(arg, list) else [arg]
        for col in cols:
            _ensure_col(columns_by_name, col).setdefault("tests", []).append("not_null")
        return

    if name == "unique":
        cols = arg if isinstance(arg, list) else [arg]
        if len(cols) == 1:
            _ensure_col(columns_by_name, cols[0]).setdefault("tests", []).append(
                "unique"
            )
        else:
            # dbt has no native composite-unique on a source table; emit the
            # dbt_utils macro test so the file is still valid if dbt_utils
            # is installed in the consuming project.
            table_tests.append(
                {
                    "dbt_utils.unique_combination_of_columns": {
                        "combination_of_columns": list(cols)
                    }
                }
            )
        return

    # min_rows / not_empty / has_text don't map cleanly to dbt source tests in v1.
    # Silently drop them; the user can re-express in dbt-side tests if needed.


def _ensure_col(
    columns_by_name: dict[str, dict[str, Any]], name: str
) -> dict[str, Any]:
    if name not in columns_by_name:
        columns_by_name[name] = {"name": name}
    return columns_by_name[name]


def _attach_table_test(name: str, _arg: Any, table_tests: list[Any]) -> None:
    # Reserved for future named string tests; nothing to emit for v1's bare strings.
    return
