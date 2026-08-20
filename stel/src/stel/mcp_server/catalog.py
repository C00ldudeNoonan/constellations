from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ..config import load_project
from .authorization import PolicyAttribute
from .contracts import (
    ContextField,
    ContextModelSummary,
    FilterCapability,
    FilterOperator,
    RetrievalCapability,
)


class ArtifactCatalogError(Exception):
    pass


def _access(value: object) -> Literal["public", "governed"]:
    if value == "public":
        return "public"
    if value == "governed":
        return "governed"
    raise ArtifactCatalogError("context resource access must be public or governed")


@dataclass(frozen=True, slots=True)
class ContextResource:
    name: str
    unique_id: str
    description: str | None
    access: Literal["public", "governed"]
    context_relation: str
    registry_relation: str
    entity_relations: tuple[str, ...]
    schema_fields: tuple[ContextField, ...]
    business_filters: tuple[FilterCapability, ...]
    policy_attributes: tuple[PolicyAttribute, ...]
    modes: tuple[str, ...]
    consistency: str
    id_field: str
    store_type: str
    lineage_resources: tuple[str, ...]
    last_successful_materialization: datetime | None

    def summary(self, *, entity_types: Sequence[str] = ()) -> ContextModelSummary:
        return ContextModelSummary(
            name=self.name,
            description=self.description,
            contract="agent_context/v1",
            grain="document_chunks",
            access=self.access,
            schema_fields=self.schema_fields,
            retrieval=RetrievalCapability(
                modes=self.modes,
                consistency=self.consistency,
                filter_fields=self.business_filters,
            ),
            freshness="unknown",
            last_successful_materialization=self.last_successful_materialization,
            entity_types=tuple(sorted(set(entity_types))),
        )


class ArtifactCatalog:
    def __init__(self, resources: Sequence[ContextResource]) -> None:
        self._resources = {resource.name: resource for resource in resources}

    @classmethod
    def load(
        cls,
        project_dir: Path,
        *,
        expected_target: str | None = None,
    ) -> ArtifactCatalog:
        project, _, _ = load_project(project_dir)
        target_dir = (project_dir / project.target_path).resolve()
        manifest = _read_json(target_dir / "manifest.json", required=True)
        run_results = _read_json(target_dir / "run_results.json", required=False)
        if manifest is None:
            raise AssertionError("required manifest read returned no payload")
        return cls.from_payloads(
            manifest,
            run_results=run_results,
            expected_target=expected_target,
        )

    @classmethod
    def from_payloads(
        cls,
        manifest: Mapping[str, Any],
        *,
        run_results: Mapping[str, Any] | None = None,
        expected_target: str | None = None,
    ) -> ArtifactCatalog:
        if manifest.get("manifest_version") != 2:
            raise ArtifactCatalogError(
                "stel MCP requires a manifest v2 artifact; run `stel compile`"
            )
        target = manifest.get("target")
        if not isinstance(target, Mapping):
            raise ArtifactCatalogError("The manifest has no safe target descriptor")
        target_name = target.get("name")
        if expected_target is not None and target_name != expected_target:
            raise ArtifactCatalogError(
                "The manifest target does not match the requested MCP target; recompile it"
            )
        models_raw = manifest.get("models")
        dag_raw = manifest.get("dag")
        if not isinstance(models_raw, list) or not isinstance(dag_raw, Mapping):
            raise ArtifactCatalogError("The manifest is missing model or DAG metadata")
        models = {
            str(model["unique_id"]): model
            for model in models_raw
            if isinstance(model, Mapping)
            and isinstance(model.get("unique_id"), str)
        }
        predecessors, successors = _graph(dag_raw)
        execution_order = tuple(
            value
            for value in dag_raw.get("execution_order", ())
            if isinstance(value, str)
        )
        last_success = _successful_materializations(run_results)
        resources: list[ContextResource] = []
        for unique_id, model in models.items():
            if model.get("resource_type") != "search_index":
                continue
            ancestors = _reachable(unique_id, predecessors)
            context_id = _closest_contract_model(
                unique_id,
                predecessors,
                models,
                grain="document_chunks",
            )
            if context_id is None:
                continue
            registry_id = _closest_contract_model(
                context_id,
                predecessors,
                models,
                grain="document_registry",
            )
            if registry_id is None:
                continue
            context_model = models[context_id]
            registry_model = models[registry_id]
            context_descriptor = context_model.get("agent_context")
            if not isinstance(context_descriptor, Mapping):
                continue
            serving = _serving_descriptor(model)
            entity_ids = sorted(
                candidate
                for candidate in _reachable(context_id, successors)
                if _contract_grain(models.get(candidate)) == "context_entity_links"
            )
            attributes = serving.get("attributes")
            if not isinstance(attributes, list):
                attributes = []
            policy_attributes = tuple(
                PolicyAttribute(str(item["name"]), str(item["data_type"]))
                for item in attributes
                if isinstance(item, Mapping)
                and item.get("filter_role") in {"policy", "user_and_policy"}
                and isinstance(item.get("name"), str)
                and isinstance(item.get("data_type"), str)
            )
            business_filters = tuple(
                _filter_capability(item)
                for item in attributes
                if isinstance(item, Mapping)
                and item.get("filter_role") in {"user", "user_and_policy"}
            )
            query = serving.get("query")
            modes = (
                tuple(sorted(str(value) for value in query.get("modes", ())))
                if isinstance(query, Mapping)
                else ()
            )
            consistency = (
                str(query.get("consistency", "strong"))
                if isinstance(query, Mapping)
                else "strong"
            )
            name = model.get("name")
            if not isinstance(name, str):
                continue
            resources.append(
                ContextResource(
                    name=name,
                    unique_id=unique_id,
                    description=(
                        str(model["description"])
                        if model.get("description") is not None
                        else None
                    ),
                    access=_access(model.get("access", "public")),
                    context_relation=_relation_name(context_model),
                    registry_relation=_relation_name(registry_model),
                    entity_relations=tuple(
                        _relation_name(models[entity_id]) for entity_id in entity_ids
                    ),
                    schema_fields=_schema_fields(context_descriptor),
                    business_filters=business_filters,
                    policy_attributes=policy_attributes,
                    modes=modes,
                    consistency=consistency,
                    id_field=str(serving.get("id_field", "context_id")),
                    store_type=str(serving.get("store_type", "unknown")),
                    lineage_resources=tuple(
                        resource_id
                        for resource_id in execution_order
                        if resource_id in ancestors or resource_id == unique_id
                    ),
                    last_successful_materialization=last_success.get(name),
                )
            )
        return cls(resources)

    def all(self) -> tuple[ContextResource, ...]:
        return tuple(sorted(self._resources.values(), key=lambda item: item.name))

    def get(self, name: str) -> ContextResource | None:
        return self._resources.get(name)


def _read_json(path: Path, *, required: bool) -> Mapping[str, Any] | None:
    if not path.is_file():
        if required:
            raise ArtifactCatalogError(
                f"No {path.name} artifact is available; run `stel compile`"
            )
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        raise ArtifactCatalogError(f"The {path.name} artifact is not readable JSON") from None
    if not isinstance(value, dict):
        raise ArtifactCatalogError(f"The {path.name} artifact must contain an object")
    return value


def _graph(
    dag: Mapping[str, Any],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    predecessors: dict[str, set[str]] = {}
    successors: dict[str, set[str]] = {}
    edges = dag.get("edges", ())
    if not isinstance(edges, list):
        return predecessors, successors
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(isinstance(value, str) for value in edge)
        ):
            continue
        predecessor, successor = edge
        predecessors.setdefault(successor, set()).add(predecessor)
        successors.setdefault(predecessor, set()).add(successor)
    return predecessors, successors


def _reachable(start: str, graph: Mapping[str, set[str]]) -> set[str]:
    found: set[str] = set()
    pending = list(graph.get(start, ()))
    while pending:
        value = pending.pop()
        if value in found:
            continue
        found.add(value)
        pending.extend(graph.get(value, ()))
    return found


def _closest_contract_model(
    start: str,
    graph: Mapping[str, set[str]],
    models: Mapping[str, Mapping[str, Any]],
    *,
    grain: str,
) -> str | None:
    pending = deque(graph.get(start, ()))
    seen: set[str] = set()
    while pending:
        level = tuple(pending)
        pending.clear()
        matches = sorted(
            value for value in level if _contract_grain(models.get(value)) == grain
        )
        if matches:
            return matches[0]
        for value in level:
            if value in seen:
                continue
            seen.add(value)
            pending.extend(graph.get(value, ()))
    return None


def _contract_grain(model: Mapping[str, Any] | None) -> str | None:
    if model is None:
        return None
    descriptor = model.get("agent_context")
    return str(descriptor.get("grain")) if isinstance(descriptor, Mapping) else None


def _serving_descriptor(model: Mapping[str, Any]) -> Mapping[str, Any]:
    output = model.get("output")
    serving = output.get("serving_resource") if isinstance(output, Mapping) else None
    if not isinstance(serving, Mapping):
        raise ArtifactCatalogError("A search index is missing its serving descriptor")
    return serving


def _relation_name(model: Mapping[str, Any]) -> str:
    output = model.get("output")
    relation = output.get("relation") if isinstance(output, Mapping) else None
    name = relation.get("name") if isinstance(relation, Mapping) else None
    if not isinstance(name, str) or not name:
        raise ArtifactCatalogError("An agent-context model is missing its relation name")
    return name


def _schema_fields(descriptor: Mapping[str, Any]) -> tuple[ContextField, ...]:
    fields = descriptor.get("fields")
    if not isinstance(fields, list):
        raise ArtifactCatalogError("An agent-context descriptor is missing its schema")
    return tuple(
        ContextField(
            name=str(field["name"]),
            data_type=str(field["data_type"]),
            nullable=bool(field["nullable"]),
            description=str(field["description"]),
        )
        for field in fields
        if isinstance(field, Mapping)
    )


def _filter_capability(attribute: Mapping[str, Any]) -> FilterCapability:
    data_type = str(attribute.get("data_type", "string"))
    operators = (
        (FilterOperator.EQUAL, FilterOperator.NOT_EQUAL, FilterOperator.IN)
        if data_type in {"boolean", "array[string]"}
        else tuple(FilterOperator)
    )
    return FilterCapability(
        field=str(attribute.get("name", "")),
        data_type=data_type,
        operators=operators,
    )


def _successful_materializations(
    run_results: Mapping[str, Any] | None,
) -> dict[str, datetime]:
    if run_results is None:
        return {}
    metadata = run_results.get("metadata")
    generated = metadata.get("generated_at") if isinstance(metadata, Mapping) else None
    if not isinstance(generated, str):
        return {}
    try:
        generated_at = datetime.fromisoformat(generated)
    except ValueError:
        return {}
    rows = run_results.get("results")
    if not isinstance(rows, list):
        return {}
    return {
        str(row["model_name"]): generated_at
        for row in rows
        if isinstance(row, Mapping)
        and row.get("status") == "success"
        and isinstance(row.get("model_name"), str)
    }
