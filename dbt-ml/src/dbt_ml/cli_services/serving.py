"""Serving-readiness operations (issue #190, Workstream D).

Scope resolution and the status/recover operations behind the `serving`
commands, factored out of `cli.py` so they run — and are tested — without
Click. Each returns the publication-ledger entry as data; the command edge
formats it. Retrieval imports stay lazy so importing this module never pulls a
vector-store backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..adapters import create_adapter
from ..compiler import validate_project_contract
from ..config import load_project
from ..profile import resolve_profile
from .context import ConfigClickError

if TYPE_CHECKING:
    from ..profile import ResolvedProfile
    from ..retrieval import ServingLedgerEntry


def resolve_serving_scope(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
) -> tuple[Any, ResolvedProfile]:
    """Resolve the retrieval-publish state scope for one search index.

    Domain failures (unknown index, no retrieval config, unavailable store)
    raise ConfigClickError (exit 2); configuration/profile errors propagate for
    the edge to translate."""
    from ..adapters.base import StateScope
    from ..retrieval import create_store

    project_config, sources, models = load_project(project_dir)
    validate_project_contract(project_config, sources, models, project_dir)
    model = next((item for item in models if item.name == model_name), None)
    if model is None or model.search is None:
        raise ConfigClickError(f"Search index '{model_name}' was not found")
    resolved = resolve_profile(
        project_config, project_dir, target=target, profiles_dir=profiles_dir
    )
    if resolved.retrieval is None:
        raise ConfigClickError("The active profile has no retrieval configuration")
    alias = model.search.store or resolved.retrieval.default
    store_config = resolved.retrieval.stores.get(alias)
    if store_config is None:
        raise ConfigClickError(
            f"Search index '{model_name}' selects an unavailable retrieval store"
        )
    store = create_store(
        store_config,
        project_name=project_config.name,
        target_name=resolved.target_name,
        alias=alias,
    )
    logical = model.search.collection or model.name
    scope = StateScope.for_target_descriptor(
        model.name,
        stage="retrieval_publish",
        descriptor=store.state_descriptor(logical).descriptor(),
    )
    return scope, resolved


def serving_status(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
) -> ServingLedgerEntry:
    """Read the publication ledger for one search index."""
    from ..retrieval import ServingCoordinator

    scope, resolved = resolve_serving_scope(
        project_dir, profiles_dir=profiles_dir, target=target, model_name=model_name
    )
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        return ServingCoordinator(adapter).status(scope)


def serving_recover(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
    owner_terminated: bool,
) -> ServingLedgerEntry:
    """Reassign serving authority, advancing the fencing token and clearing
    leases. Refused unless the caller confirms the old owner was terminated."""
    from ..retrieval import ServingCoordinator

    scope, resolved = resolve_serving_scope(
        project_dir, profiles_dir=profiles_dir, target=target, model_name=model_name
    )
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        return ServingCoordinator(adapter).recover(
            scope, owner_terminated=owner_terminated
        )
