"""Serving-readiness operations (issue #190, Workstream D).

Scope resolution and the status/recover operations behind the `serving`
commands, factored out of `cli.py` so they run — and are tested — without
Click. Each returns the publication-ledger entry as data; the command edge
formats it. Retrieval imports stay lazy so importing this module never pulls a
vector-store backend.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ServingReport:
    """A ledger entry plus which target and store it was read from.

    The entry alone is ambiguous: `status=unpublished` is equally true of the
    index you meant and of a target that has never heard of it, and
    "Recovered serving scope for 'x'" is equally true of dev and prod. Naming
    the resolution is what makes acting on the wrong one self-evident
    (issue #511).
    """

    entry: ServingLedgerEntry
    target: str
    warehouse: str
    store_alias: str
    store_type: str
    store_location: str
    # Whether a ledger row existed *before* this command ran. False alongside a
    # `status=unpublished` entry means this warehouse has no record of the
    # index at all, which is the shape a wrong-target lookup takes.
    had_ledger_row: bool


def _resolve_serving_scopes(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
) -> tuple[Any, Any, ResolvedProfile, tuple[str, str, str]]:
    """Resolve the current and pre-#355 retrieval-publish scopes for an index.

    Domain failures (unknown index, no retrieval config, unavailable store)
    raise ConfigClickError (exit 2); configuration/profile errors propagate for
    the edge to translate."""
    from ..adapters.base import StateScope
    from ..retrieval import StoreRole, create_store

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
        # Ledger admin: reads the descriptor, never an index.
        role=StoreRole.INSPECT,
    )
    logical = model.search.collection or model.name
    state_target = store.state_descriptor(logical)
    context = (alias, store_config.type, store_config.storage_location())
    scope = StateScope.for_target_descriptor(
        model.name,
        stage="retrieval_publish",
        descriptor=state_target.descriptor(),
    )
    legacy_scope = StateScope.for_target_descriptor(
        model.name,
        stage="retrieval_publish",
        descriptor=state_target.legacy_descriptor(),
    )
    return scope, legacy_scope, resolved, context


def resolve_serving_scope(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
) -> tuple[Any, ResolvedProfile]:
    """The current (logical-keyed) serving scope for one search index."""
    scope, _legacy, resolved, _context = _resolve_serving_scopes(
        project_dir,
        profiles_dir=profiles_dir,
        target=target,
        model_name=model_name,
    )
    return scope, resolved


def serving_status(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
) -> ServingReport:
    """Read the publication ledger for one search index."""
    from ..retrieval import ServingCoordinator

    scope, _legacy, resolved, context = _resolve_serving_scopes(
        project_dir, profiles_dir=profiles_dir, target=target, model_name=model_name
    )
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        coordinator = ServingCoordinator(adapter)
        return _report(
            coordinator.status(scope),
            resolved=resolved,
            context=context,
            had_ledger_row=coordinator.scope_exists(scope),
        )


def serving_recover(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
    owner_terminated: bool,
) -> ServingReport:
    """Reassign serving authority, advancing the fencing token and clearing
    leases. Refused unless the caller confirms the old owner was terminated,
    and unless they named the target explicitly."""
    from ..retrieval import ServingCoordinator

    scope, _legacy, resolved, context = _resolve_serving_scopes(
        project_dir, profiles_dir=profiles_dir, target=target, model_name=model_name
    )
    if target is None:
        # Resolution above is a read, so this refuses before anything moves
        # -- and it can name the default the caller would otherwise have
        # got. `--owner-terminated` already treats this as an operation
        # worth confirming; inferring *which store* to confirm it against
        # undoes that care, and did (issue #511).
        raise ConfigClickError(
            "'stel serving recover' requires an explicit --target: it "
            "advances the fencing token and marks the scope failed, so it "
            "must not act on a target nobody named. This profile would "
            f"have used '{resolved.target_name}' (store {context[0]}: "
            f"{context[1]} {context[2]}). Re-run with --target "
            f"{resolved.target_name} to confirm that is the one you mean."
        )
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        coordinator = ServingCoordinator(adapter)
        had_row = coordinator.scope_exists(scope)
        entry = coordinator.recover(scope, owner_terminated=owner_terminated)
        return _report(
            entry, resolved=resolved, context=context, had_ledger_row=had_row
        )


def _report(
    entry: ServingLedgerEntry,
    *,
    resolved: ResolvedProfile,
    context: tuple[str, str, str],
    had_ledger_row: bool,
) -> ServingReport:
    alias, store_type, location = context
    return ServingReport(
        entry=entry,
        target=resolved.target_name,
        warehouse=(
            f"{resolved.warehouse.type} "
            f"{resolved.warehouse.storage_location()}".strip()
        ),
        store_alias=alias,
        store_type=store_type,
        store_location=location,
        had_ledger_row=had_ledger_row,
    )


def serving_migrate_scope(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
) -> dict[str, int | str]:
    """Move an index's serving scope from the pre-#355 physical-collection key
    onto the logical-collection key.

    Issue #355 re-keys the retrieval serving scope so the ledger stays
    readable once a logical collection can have more than one physical
    generation behind it. Indexes published before that change keep their
    state and ledger row under the old identity, where nothing looks for it —
    and an unreachable publication state means the next run re-embeds an index
    that is already published. This moves both, or reports that there is
    nothing to move.
    """
    from ..retrieval import ServingCoordinator

    scope, legacy_scope, resolved, _context = _resolve_serving_scopes(
        project_dir, profiles_dir=profiles_dir, target=target, model_name=model_name
    )
    if scope.target_identity == legacy_scope.target_identity:
        return {"model": model_name, "state_rows": 0, "ledger_rows": 0}
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        coordinator = ServingCoordinator(adapter)
        # Ledger first: it is the row that decides whether an index is
        # considered published at all. If the state move fails after it, a
        # re-run finds the ledger already moved and finishes the state.
        ledger_rows = coordinator.rekey_scope(legacy_scope, scope)
        state_rows = adapter.rekey_state_scope(legacy_scope, scope)
    return {
        "model": model_name,
        "state_rows": state_rows,
        "ledger_rows": ledger_rows,
    }
