"""Translate a stel project into a dbt-compatible sources.yml.

A dbt project using the matching warehouse adapter can consume stel tables
via `{{ source('dbt_ml_<project>', '<model>') }}`. This module emits the
sources.yml declaration that makes that work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .agent_context import contract_relation
from .config import load_project
from .config.model import ModelConfig
from .config.profile import WarehouseConfig
from .dag import ProjectDAG, parse_ref
from .profile import resolve_profile

DEFAULT_OUTPUT_FILENAME = "sources.yml"

# `meta:` namespace in the emitted sources.yml. The generated file is
# committed into the consumer's dbt project and read from there, so this key
# is part of what stel hands to another tool — renaming it makes the
# agent-context block invisible to anything already reading it.
DBT_META_NAMESPACE = "dbt_ml"

# Default `sources:` name in the emitted sources.yml: `dbt_ml_<project>`.
#
# Frozen, and for the same reason as the namespace above: the generated file is
# committed into the consumer's dbt project and referenced by their `source()`
# calls, so a new default silently breaks their models. It kept the pre-#313
# spelling deliberately.
#
# Every producer of this name must go through the helper. Three call sites
# spelled it inline, which is exactly how `concept-cloud` drifted: it
# reconstructed the default and ignored `--source-name` entirely, so a project
# that had overridden the name got a linking node that silently matched
# nothing.
DEFAULT_SOURCE_NAME_PREFIX = "dbt_ml_"


def default_dbt_source_name(project_name: str) -> str:
    """The `sources:` name a project's emitted dbt sources use by default."""
    return f"{DEFAULT_SOURCE_NAME_PREFIX}{project_name}"


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
    selected_names = set(dag.select_models(select=select))
    excluded_names = set(dag.select_models(select=exclude)) if exclude else set()
    selected_names -= excluded_names
    models_by_name = {model.name: model for model in models}
    projected_names: set[str] = set()
    for selected_name in selected_names:
        selected_model = models_by_name[selected_name]
        if selected_model.search is None:
            projected_names.add(selected_name)
            continue
        upstream = parse_ref((selected_model.depends_on or [""])[0])
        if upstream not in excluded_names:
            projected_names.add(upstream)
    selected_models = [
        model
        for model in models
        if model.name in projected_names and model.search is None
    ]

    name = source_name or default_dbt_source_name(project.name)
    catalog = _derive_catalog(resolved.warehouse)

    return {
        "version": 2,
        "sources": [
            {
                "name": name,
                "description": (
                    f"Tables materialized by stel project '{project.name}'."
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

    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return output


def _derive_catalog(warehouse: WarehouseConfig) -> str:
    """Adapter-specific catalog: DuckDB file stem, BigQuery project, ..."""
    return warehouse.catalog_name()


def _table_for_model(
    model: ModelConfig, *, source_name: str, dagster_meta: bool = False
) -> dict[str, Any]:
    columns_by_name: dict[str, dict[str, Any]] = {}

    if model.agent_context is not None:
        relation = contract_relation(model.agent_context.grain)
        for contract_field in relation.fields:
            column = columns_by_name.setdefault(
                contract_field.name, {"name": contract_field.name}
            )
            column["description"] = contract_field.description
            column["data_type"] = (
                "string"
                if contract_field.data_type in {"json", "array[string]"}
                else contract_field.data_type
            )
            column["meta"] = {
                DBT_META_NAMESPACE: {
                    "agent_context": {
                        "nullable": contract_field.nullable,
                    }
                }
            }

    for model_field in model.fields:
        column = columns_by_name.setdefault(
            model_field.name, {"name": model_field.name}
        )
        if model_field.description:
            column["description"] = model_field.description
        if model_field.data_type:
            # Adapters receive nested values after they are flattened to JSON
            # text, and an `enum` field materializes as the string it
            # constrains — `enum` is stel's declaration, not a column type any
            # warehouse or dbt adapter would recognize (issue #304).
            column["data_type"] = (
                "string"
                if model_field.data_type in {"json", "enum"}
                else model_field.data_type
            )

    table_tests: list[Any] = []
    for spec in model.tests:
        _apply_test_spec(spec, columns_by_name, table_tests)

    table: dict[str, Any] = {"name": model.name}
    if model.description:
        table["description"] = model.description
    if model.tags:
        table["tags"] = model.tags

    meta: dict[str, Any] = {}
    if model.agent_context is not None:
        relation = contract_relation(model.agent_context.grain)
        meta[DBT_META_NAMESPACE] = {
            "agent_context": {
                "contract": model.agent_context.contract,
                "grain": relation.grain.value,
                "primary_key": list(relation.primary_key),
                "foreign_keys": dict(relation.foreign_keys),
            }
        }
    if dagster_meta:
        # Pin the Dagster asset key to dagster-dbt's default source key
        # ([source, table]) so a stel producer asset and the dbt models that
        # {{ source(...) }} it agree on one key without hand-copying. dbt itself
        # ignores unknown `meta`, so pure-dbt consumers are unaffected.
        meta["dagster"] = {"asset_key": [source_name, model.name]}
    if meta:
        table["meta"] = meta

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
