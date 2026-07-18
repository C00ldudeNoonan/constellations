from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from .adapters import StateScope
from .compiler import validate_project_contract, validate_retrieval_capabilities
from .config import load_project
from .config.model import ModelConfig
from .config.project import ProjectConfig
from .dag import NodeKind, ProjectDAG, parse_ref
from .hashing import canonical_fingerprint
from .profile import (
    ProfileError,
    ResolvedProfile,
    apply_source_path_overrides,
    resolve_profile,
)
from .retrieval import collection_config_fingerprint, create_store, store_class
from .runner import ModelRunResult
from .versioning import (
    compute_model_code_version,
    describe_model_embedding,
    describe_model_inference,
)

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
RUN_RESULTS_FILENAME = "run_results.json"


def build_manifest(
    project_dir: Path,
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
) -> dict[str, Any]:
    project, sources, models = load_project(project_dir)
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    sources = apply_source_path_overrides(sources, resolved)
    has_search = any(model.search is not None for model in models)
    dag = (
        validate_project_contract(project, sources, models, project_dir)
        if has_search
        else ProjectDAG(sources, models)
    )
    if has_search:
        validate_retrieval_capabilities(models, project, resolved)
        return _build_manifest_v2(project, sources, models, dag, project_dir, resolved)

    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": _now(),
        "project": {
            "name": project.name,
            "version": project.version,
            "profile": resolved.profile_name,
            "target": resolved.target_name,
            "duckdb_path": resolved.warehouse.storage_location(),
            "duckdb_schema": resolved.warehouse.schema_name,
        },
        "sources": [
            {
                "name": s.name,
                "description": s.description,
                "path": s.path,
                "file_pattern": s.file_pattern,
                "recursive": s.recursive,
                "external": s.external,
                "tags": s.tags,
            }
            for s in sources
        ],
        "models": [
            _model_dict(m, project, project_dir, resolved) for m in models
        ],
        "dag": {
            "execution_order": dag.execution_order(),
            "nodes": [
                {"name": n, "kind": dag.nodes[n].kind.value}
                for n in dag.nodes
            ],
            "edges": [
                [pred, succ]
                for succ, preds in dag.predecessors.items()
                for pred in preds
            ],
        },
    }


def write_manifest(
    project_dir: Path,
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
) -> Path:
    project, _, _ = load_project(project_dir)
    target_dir = (project_dir / project.target_path).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / MANIFEST_FILENAME
    out.write_text(
        json.dumps(
            build_manifest(project_dir, target=target, profiles_dir=profiles_dir),
            indent=2,
        )
    )
    return out


def build_run_results(
    project_dir: Path,
    results: list[ModelRunResult],
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
    invocation: str = "run",
    skipped: list[str] | None = None,
    elapsed_seconds: float | None = None,
    test_failures: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Assemble the run_results.json payload: run-level metadata (target,
    counts, status) plus per-model status and the fully-qualified relation each
    model materialized into. This is the contract Dagster reads to attach
    materialization metadata (issue #87).

    `test_failures` maps model_name → hard-failed test labels (from `build`). A
    model that materialized fine but failed a test has empty `errors`, so its
    status must be derived here too — otherwise a failing test on a leaf model
    (no skipped descendants) would report `success` while the command exits 1."""
    skipped = skipped or []
    test_failures = test_failures or {}
    target_block = _target_block(project_dir, target=target, profiles_dir=profiles_dir)
    catalog = target_block["catalog"] if target_block else None
    schema = target_block["schema"] if target_block else None
    _, sources, models = load_project(project_dir)
    dag = ProjectDAG(sources, models)
    model_names = {model.name for model in models}
    considered_models = [
        name
        for name in [*(r.model_name for r in results), *skipped]
        if name in model_names
    ]
    sources_considered = dag.required_sources(considered_models)

    result_rows: list[dict[str, Any]] = []
    n_error = 0
    for r in results:
        failed = test_failures.get(r.model_name, [])
        status = "error" if (r.errors or failed) else "success"
        n_error += status == "error"
        row = asdict(r)
        row["status"] = status
        row["test_failures"] = failed
        model = next((item for item in models if item.name == r.model_name), None)
        row["relation"] = (
            None
            if model is not None and model.search is not None
            else _relation(catalog, schema, r.model_name)
        )
        result_rows.append(row)

    for name in skipped:
        row = asdict(ModelRunResult(model_name=name, materialization="", kind="skipped"))
        row["status"] = "skipped"
        row["test_failures"] = []
        model = next((item for item in models if item.name == name), None)
        row["relation"] = (
            None
            if model is not None and model.search is not None
            else _relation(catalog, schema, name)
        )
        result_rows.append(row)

    overall = "error" if (n_error or skipped) else "success"
    metadata: dict[str, Any] = {
        "dbt_ml_version": _dbt_ml_version(),
        "generated_at": _now(),
        "invocation": invocation,
        "status": overall,
        "elapsed_seconds": elapsed_seconds,
        "target": target_block,
        "sources_considered": sources_considered,
        "counts": {
            "total": len(results) + len(skipped),
            "success": len(results) - n_error,
            "error": n_error,
            "skipped": len(skipped),
            "warnings": sum(sum(r.warnings.values()) for r in results),
        },
    }
    return {"metadata": metadata, "results": result_rows}


def write_run_results(
    project_dir: Path,
    results: list[ModelRunResult],
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
    invocation: str = "run",
    skipped: list[str] | None = None,
    elapsed_seconds: float | None = None,
    test_failures: dict[str, list[str]] | None = None,
) -> Path:
    project, _, _ = load_project(project_dir)
    target_dir = (project_dir / project.target_path).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = build_run_results(
        project_dir,
        results,
        target=target,
        profiles_dir=profiles_dir,
        invocation=invocation,
        skipped=skipped,
        elapsed_seconds=elapsed_seconds,
        test_failures=test_failures,
    )
    out = target_dir / RUN_RESULTS_FILENAME
    out.write_text(json.dumps(payload, indent=2))
    return out


def _target_block(
    project_dir: Path,
    *,
    target: str | None,
    profiles_dir: Path | None,
) -> dict[str, Any] | None:
    """Warehouse target descriptor for run_results. Returns None if the profile
    can't be resolved (never fail a completed run just to describe its target)."""
    project, _, _ = load_project(project_dir)
    try:
        resolved = resolve_profile(
            project, project_dir, target=target, profiles_dir=profiles_dir
        )
    except ProfileError:
        return None
    w = resolved.warehouse
    return {
        "profile": resolved.profile_name,
        "name": resolved.target_name,
        "adapter_type": w.type,
        "schema": w.schema_name,
        "catalog": w.catalog_name(),
        "location": w.storage_location(),
    }


def _relation(
    catalog: str | None, schema: str | None, name: str
) -> dict[str, Any]:
    parts = [p for p in (catalog, schema, name) if p]
    return {
        "catalog": catalog,
        "schema": schema,
        "name": name,
        "fully_qualified": ".".join(parts),
    }


def _dbt_ml_version() -> str:
    try:
        return _pkg_version("dbt-ml")
    except PackageNotFoundError:
        return "unknown"


def _model_dict(
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    resolved: ResolvedProfile,
) -> dict[str, Any]:
    if model.extraction is not None:
        kind = "extraction"
    elif model.ml is not None:
        kind = "ml"
    elif model.transform is not None:
        kind = "transform"
    elif model.chunk is not None:
        kind = "chunk"
    elif model.embed is not None:
        kind = "embed"
    else:
        kind = "unknown"

    model_dict = {
        "name": model.name,
        "description": model.description,
        "kind": kind,
        "materialization": model.materialization,
        "tags": model.tags,
        "source": model.source,
        "depends_on": model.depends_on or [],
        "extraction": model.extraction.model_dump() if model.extraction else None,
        "transform": model.transform.model_dump() if model.transform else None,
        "ml": model.ml.model_dump(mode="json") if model.ml else None,
        "chunk": model.chunk.model_dump() if model.chunk else None,
        "embed": model.embed.model_dump() if model.embed else None,
        "fields": [f.model_dump() for f in model.fields],
        "tests": model.tests,
        "code_version": compute_model_code_version(
            model,
            project,
            project_dir,
            resolved=resolved,
        ),
    }
    inference = describe_model_inference(model, project, resolved=resolved)
    if inference is not None:
        model_dict["inference"] = inference
    embedding = describe_model_embedding(model)
    if embedding is not None:
        model_dict["embedding"] = embedding
    return model_dict


def _build_manifest_v2(
    project: ProjectConfig,
    sources: list[Any],
    models: list[ModelConfig],
    dag: ProjectDAG,
    project_dir: Path,
    resolved: ResolvedProfile,
) -> dict[str, Any]:
    used_aliases = {
        (model.search.store or resolved.retrieval.default)
        for model in models
        if model.search is not None and resolved.retrieval is not None
    }
    warehouse_identity = canonical_fingerprint(
        {
            "type": resolved.warehouse.type,
            "catalog": resolved.warehouse.catalog_name(),
            "schema": resolved.warehouse.schema_name,
            "location": resolved.warehouse.storage_location(),
        },
        domain="dbt-ml-safe-warehouse-target",
    )
    retrieval_targets: list[dict[str, Any]] = []
    if resolved.retrieval is not None:
        for alias in sorted(used_aliases):
            config = resolved.retrieval.stores[alias]
            store = create_store(
                config,
                project_name=project.name,
                target_name=resolved.target_name,
                alias=alias,
            )
            safe = store.safe_descriptor()
            retrieval_targets.append(
                {
                    "alias": alias,
                    "store_type": config.type,
                    "safe_target_identity": safe.safe_target_identity,
                }
            )
    return {
        "manifest_version": 2,
        "generated_at": _now(),
        "project": {"name": project.name, "version": project.version},
        "target": {
            "profile": resolved.profile_name,
            "name": resolved.target_name,
            "warehouse": {
                "adapter_type": resolved.warehouse.type,
                "safe_target_identity": warehouse_identity,
                "catalog": resolved.warehouse.catalog_name(),
                "schema": resolved.warehouse.schema_name,
            },
            "retrieval": retrieval_targets,
        },
        "sources": [
            {
                "name": source.name,
                "unique_id": f"source.{project.name}.{source.name}",
                "description": source.description,
                "path": source.path,
                "file_pattern": source.file_pattern,
                "recursive": source.recursive,
                "external": source.external,
                "tags": source.tags,
            }
            for source in sources
        ],
        "models": [
            _model_dict_v2(model, project, project_dir, resolved) for model in models
        ],
        "dag": {
            "execution_order": [
                _unique_id(dag.nodes[name].kind, project.name, name)
                for name in dag.execution_order()
            ],
            "nodes": [
                {
                    "name": node.name,
                    "kind": node.kind.value,
                    "unique_id": _unique_id(node.kind, project.name, node.name),
                }
                for node in dag.all_nodes_in_order()
            ],
            "edges": [
                [
                    _unique_id(dag.nodes[pred].kind, project.name, pred),
                    _unique_id(dag.nodes[succ].kind, project.name, succ),
                ]
                for succ, predecessors in dag.predecessors.items()
                for pred in predecessors
            ],
        },
    }


def _model_dict_v2(
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    resolved: ResolvedProfile,
) -> dict[str, Any]:
    if model.search is None:
        row = _model_dict(model, project, project_dir, resolved)
        row["unique_id"] = f"model.{project.name}.{model.name}"
        row["resource_type"] = "model"
        row["output"] = {
            "type": "warehouse_relation",
            "relation": _relation(
                resolved.warehouse.catalog_name(),
                resolved.warehouse.schema_name,
                model.name,
            ),
        }
        return row

    assert resolved.retrieval is not None
    search = model.search
    alias = search.store or resolved.retrieval.default
    config = resolved.retrieval.stores[alias]
    store = create_store(
        config,
        project_name=project.name,
        target_name=resolved.target_name,
        alias=alias,
    )
    logical = search.collection or model.name
    state_target = store.state_descriptor(logical)
    scope = StateScope.for_target_descriptor(
        model.name,
        stage="retrieval_publish",
        descriptor=state_target.descriptor(),
    )
    capabilities = store_class(config.type).capabilities()
    required = [
        "keyed_upsert",
        "keyed_delete",
        "index_readiness",
        "durable_write_ack",
        "atomic_batch_mutation",
    ]
    if search.vector is not None:
        required.append(f"{search.vector.search}_vector_search")
    if search.full_text is not None:
        required.append("full_text_search")
    if any(attribute.filter_role != "none" for attribute in search.attributes):
        required.append("metadata_filtering")
    upstream = parse_ref((model.depends_on or [""])[0])
    return {
        "name": model.name,
        "unique_id": f"search_index.{project.name}.{model.name}",
        "resource_type": "search_index",
        "kind": "search",
        "description": model.description,
        "access": search.access,
        "materialization": model.materialization,
        "tags": model.tags,
        "depends_on": [f"model.{project.name}.{upstream}"],
        "code_version": compute_model_code_version(
            model, project, project_dir, resolved=resolved
        ),
        "tests": [],
        "output": {
            "type": "serving_resource",
            "serving_resource": {
                "kind": "retrieval_index",
                "store_type": config.type,
                "store_implementation": store_class(config.type).implementation_identity(),
                "safe_target_identity": store.safe_descriptor().safe_target_identity,
                "logical_collection": logical,
                "physical_collection": state_target.physical_collection,
                "scope_fingerprint": scope.target_identity,
                "materialization": model.materialization,
                "schema_version": 1,
                "config_fingerprint": collection_config_fingerprint(
                    search.model_dump(mode="python"), store_type=config.type
                ),
                "id_field": search.id_field,
                "text_fields": list(search.text_fields),
                "return_text_fields": list(search.return_text_fields),
                "full_text": (
                    {"fields": list(search.full_text.fields)}
                    if search.full_text is not None
                    else None
                ),
                "vector": search.vector.model_dump(mode="json") if search.vector else None,
                "attributes": [
                    attribute.model_dump(mode="json") for attribute in search.attributes
                ],
                "display_fields": list(search.display_fields),
                "query": {
                    "modes": sorted(search.query.modes),
                    "consistency": search.query.consistency,
                },
                "capabilities": {
                    "required": sorted(required),
                    "available": sorted(feature.value for feature in capabilities.features),
                    "consistency_modes": sorted(capabilities.consistency_modes),
                    "distance_metrics": sorted(capabilities.distance_metrics),
                    "max_batch_size": capabilities.max_batch_size,
                },
                "upstream": f"model.{project.name}.{upstream}",
            },
        },
    }


def _unique_id(kind: NodeKind, project_name: str, name: str) -> str:
    prefix = {
        NodeKind.SOURCE: "source",
        NodeKind.MODEL: "model",
        NodeKind.SEARCH_INDEX: "search_index",
    }[kind]
    return f"{prefix}.{project_name}.{name}"


class StateError(Exception):
    pass


def read_state_code_versions(state_path: Path) -> dict[str, str]:
    """Read {model_name: code_version} from a previous manifest.

    `state_path` may be the manifest.json itself or a directory containing
    one (dbt's --state convention).
    """
    path = state_path / MANIFEST_FILENAME if state_path.is_dir() else state_path
    if not path.exists():
        raise StateError(
            f"No manifest found at {path}. Point --state at a manifest.json "
            "(or its directory) written by a previous `compile` or `run`."
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise StateError(f"{path} is not valid JSON: {e}") from e
    models = data.get("models")
    if not isinstance(models, list):
        raise StateError(f"{path} has no `models` list; is it a dbt-ml manifest?")
    return {
        m["name"]: m.get("code_version", "")
        for m in models
        if isinstance(m, dict) and "name" in m
    }


def compute_modified_models(
    models: list[ModelConfig],
    project_dir: Path,
    state_path: Path,
    *,
    project: ProjectConfig | None = None,
    resolved: ResolvedProfile | None = None,
    target: str | None = None,
    profiles_dir: Path | None = None,
) -> set[str]:
    """Models whose code_version differs from the state manifest, or that
    the state manifest has never seen."""
    previous = read_state_code_versions(state_path)
    if project is None:
        project, _, _ = load_project(project_dir)
    if resolved is None:
        resolved = resolve_profile(
            project,
            project_dir,
            target=target,
            profiles_dir=profiles_dir,
        )
    modified: set[str] = set()
    for model in models:
        current = compute_model_code_version(
            model,
            project,
            project_dir,
            resolved=resolved,
        )
        if previous.get(model.name) != current:
            modified.add(model.name)
    return modified


def _now() -> str:
    return datetime.now(UTC).isoformat()
