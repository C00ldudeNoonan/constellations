"""Serving-readiness operations (issue #190, Workstream D).

Scope resolution and the status/recover operations behind the `serving`
commands, factored out of `cli.py` so they run — and are tested — without
Click. Each returns the publication-ledger entry as data; the command edge
formats it. Retrieval imports stay lazy so importing this module never pulls a
vector-store backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..adapters import create_adapter
from ..compiler import validate_project_contract
from ..config import load_project
from ..profile import resolve_profile
from .context import ConfigClickError

if TYPE_CHECKING:
    from ..adapters.base import StateScope
    from ..profile import ResolvedProfile
    from ..retrieval import RetrievalStoreConfig, ServingLedgerEntry


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


@dataclass(frozen=True)
class _ServingScopes:
    """What a serving command resolves from a project directory and a model name.

    One named value rather than a tuple because `migrate-scope` needs the store
    config and the logical collection to re-derive this index's scope for a
    store that has since moved (issue #666), and an unnamed sixth tuple slot is
    how a caller ends up re-keying state against the wrong scope.
    """

    scope: StateScope
    legacy_scope: StateScope
    resolved: ResolvedProfile
    # (store alias, store type, physical location) -- for `_report`, which
    # names the resolution in output so acting on the wrong one is evident.
    context: tuple[str, str, str]
    store_config: RetrievalStoreConfig
    project_name: str
    logical_collection: str
    model_name: str

    def scope_at(self, location: str) -> StateScope:
        """This index's scope as it would be with the store at `location`.

        A scope is keyed on the store's identity fingerprint, and that is
        derived from where the store is unless the profile declares an
        `identity:`. A store that moved -- or one that has just declared an
        identity so that it *can* move -- therefore left its ledger row and
        publication state under the fingerprint of the location it used to
        have, where nothing looks for them; stel reads an index whose state is
        unreachable as unpublished, and re-embeds the corpus (issue #666).

        The copy keeps this store's routing options, because the fingerprint
        the old location produced folded them in. Only the path and any
        declared identity are replaced.
        """
        from ..adapters.base import StateScope
        from ..retrieval import StoreRole, create_store

        alias, _store_type, _location = self.context
        # `model_copy` rather than re-validating a dump: a dump redacts the
        # credential references a store config may hold, so rebuilding from one
        # would either lose them or fail validation. Nothing here needs a
        # credential -- a descriptor is computed from the location and the
        # non-secret routing alone, and this store is never opened.
        source = self.store_config.model_copy(
            update={"path": location, "identity": None}
        )
        store = create_store(
            source,
            project_name=self.project_name,
            target_name=self.resolved.target_name,
            alias=alias,
            role=StoreRole.INSPECT,
        )
        return StateScope.for_target_descriptor(
            self.model_name,
            stage="retrieval_publish",
            descriptor=store.state_descriptor(self.logical_collection).descriptor(),
        )


def _resolve_serving_scopes(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
) -> _ServingScopes:
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
    return _ServingScopes(
        scope=scope,
        legacy_scope=legacy_scope,
        resolved=resolved,
        context=context,
        store_config=store_config,
        project_name=project_config.name,
        logical_collection=logical,
        model_name=model.name,
    )


def resolve_serving_scope(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
) -> tuple[StateScope, ResolvedProfile]:
    """The current (logical-keyed) serving scope for one search index."""
    resolution = _resolve_serving_scopes(
        project_dir,
        profiles_dir=profiles_dir,
        target=target,
        model_name=model_name,
    )
    return resolution.scope, resolution.resolved


def serving_status(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
) -> ServingReport:
    """Read the publication ledger for one search index."""
    from ..retrieval import ServingCoordinator

    resolution = _resolve_serving_scopes(
        project_dir, profiles_dir=profiles_dir, target=target, model_name=model_name
    )
    scope, resolved, context = resolution.scope, resolution.resolved, resolution.context
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        coordinator = ServingCoordinator(adapter, ensure_schema=True)
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

    resolution = _resolve_serving_scopes(
        project_dir, profiles_dir=profiles_dir, target=target, model_name=model_name
    )
    scope, resolved, context = resolution.scope, resolution.resolved, resolution.context
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
        coordinator = ServingCoordinator(adapter, ensure_schema=True)
        had_row = coordinator.scope_exists(scope)
        entry = coordinator.recover(scope, owner_terminated=owner_terminated)
        return _report(
            entry, resolved=resolved, context=context, had_ledger_row=had_row
        )


@dataclass(frozen=True)
class ActivationReport:
    """What `serving activate` did, plus the ledger it left behind."""

    report: ServingReport
    # Deferred import target; typed loosely so this module stays importable
    # without the execution stack.
    activation: Any


def serving_activate(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
    generation: str,
    rows_verified: bool,
) -> ActivationReport:
    """Make a physically complete generation the served one (issue #615).

    Refused without an explicit target, for the same reason `recover` is: it
    changes what a store serves, and a defaulted target is how that lands on
    the wrong one (#511). Refused without `rows_verified`, because re-stamping
    the generation's state at the current code version is an assertion about
    its rows that only the operator can make.
    """
    from ..execution.activation import activate_search_generation
    from ..retrieval import ServingCoordinator

    resolution = _resolve_serving_scopes(
        project_dir, profiles_dir=profiles_dir, target=target, model_name=model_name
    )
    scope, resolved, context = resolution.scope, resolution.resolved, resolution.context
    if target is None:
        raise ConfigClickError(
            "'stel serving activate' requires an explicit --target: it changes "
            "which generation a store serves, so it must not act on a target "
            f"nobody named. This profile would have used '{resolved.target_name}' "
            f"(store {context[0]}: {context[1]} {context[2]}). Re-run with "
            f"--target {resolved.target_name} to confirm that is the one you mean."
        )
    if not rows_verified:
        raise ConfigClickError(
            "'stel serving activate' re-stamps the generation's rows as current "
            "under this code version, which it cannot verify. Confirm with "
            "--rows-verified that the collection's rows are the current corpus "
            "-- for example, after a release that changed only what the "
            "code_version hash reads."
        )
    project_config, sources, models = load_project(project_dir)
    validate_project_contract(project_config, sources, models, project_dir)
    model = next(item for item in models if item.name == model_name)
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        activation = activate_search_generation(
            model=model,
            models_by_name={item.name: item for item in models},
            project=project_config,
            project_dir=project_dir,
            adapter=adapter,
            resolved=resolved,
            physical_collection=generation,
            rows_verified=rows_verified,
        )
        coordinator = ServingCoordinator(adapter, ensure_schema=False)
        report = _report(
            coordinator.status(scope),
            resolved=resolved,
            context=context,
            had_ledger_row=True,
        )
    return ActivationReport(report=report, activation=activation)


def describe_serving(entry: ServingLedgerEntry) -> str:
    """One line saying what a reader of this index gets right now.

    The ledger fields answer it only to someone who knows the admission rule:
    `active_generation: -` beside `status: failed` meant three weeks of refused
    queries read as an idle index (issue #617). So the answer is spelled out,
    and "nothing" is said in words rather than left as two dashes.
    """
    from ..retrieval.coordination import (
        SERVABLE_STATUSES,
        STATUS_DEGRADED,
        STATUS_PUBLISHING_IN_PLACE,
        STATUS_UNPUBLISHED,
    )

    if entry.status == STATUS_UNPUBLISHED:
        return "nothing; the index has never been published"
    if entry.status == STATUS_PUBLISHING_IN_PLACE:
        return (
            "nothing while an in-place publish holds the index; readers are "
            "told to retry after it completes"
        )
    if entry.status not in SERVABLE_STATUSES or entry.active_generation is None:
        return (
            "nothing; no generation is active, and queries are refused until a "
            "publish succeeds"
        )
    collection = entry.active_collection or "the default collection"
    served = f"generation {entry.active_generation} from {collection}"
    if entry.status == STATUS_DEGRADED:
        return (
            f"{served}, degraded: the last publish failed "
            f"({entry.safe_error_code or 'no error code recorded'}), so readers "
            "get the generation published before it"
        )
    return served


def describe_publisher_claim(entry: ServingLedgerEntry, *, now_epoch: int) -> str:
    """One line saying who holds the publish claim and when they were last
    heard from, or "-" when nobody does (issue #621).

    `publisher: active` beside a process dead for forty minutes was what the
    line used to say; the ledger now records enough to say which process, on
    which host, and how long since its last page.
    """
    from ..retrieval.publisher_identity import describe_publisher

    return describe_publisher(
        entry.publisher, heartbeat_epoch=entry.publisher_heartbeat_epoch, now_epoch=now_epoch
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


def _sources_to_rekey(
    destination: StateScope, candidates: Sequence[StateScope]
) -> list[StateScope]:
    """The candidates worth re-keying onto `destination`, in the order given.

    A candidate equal to the destination is dropped, and so is a repeat: a
    re-key onto the same scope is not a no-op at the adapter -- it matches
    every row and reports them as moved -- so leaving one in would answer
    "migrated 20,115 rows" to a command that moved nothing. `--from-path`
    naming the store's current location is the obvious operator mistake, and
    it is the one that would look most like success (issue #666).
    """
    seen = {destination.target_identity}
    pending: list[StateScope] = []
    for candidate in candidates:
        if candidate.target_identity in seen:
            continue
        seen.add(candidate.target_identity)
        pending.append(candidate)
    return pending


def serving_migrate_scope(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    model_name: str,
    from_path: str | None,
) -> dict[str, int | str]:
    """Move an index's serving scope onto the scope its profile resolves today.

    Two things move a scope out from under an index. Issue #355 re-keyed it so
    the ledger stays readable once a logical collection can have more than one
    physical generation behind it, which stranded every index published before
    that change. Issue #666 added a declared store `identity:`, so a store can
    move — off a gs:// URI onto local disk, say, to stop paying egress on every
    read — and `--from-path` is how the rows it published at the old location
    come with it.

    Either way the state and ledger row sit under a key nothing looks for, and
    an unreachable publication state means the next run re-embeds an index that
    is already published. This moves both, or reports that there is nothing to
    move. It is idempotent: a second run finds nothing and reports zero.
    """
    from ..retrieval import ServingCoordinator

    resolution = _resolve_serving_scopes(
        project_dir, profiles_dir=profiles_dir, target=target, model_name=model_name
    )
    scope = resolution.scope
    candidates = [resolution.legacy_scope]
    if from_path is not None:
        candidates.append(resolution.scope_at(from_path))
    pending = _sources_to_rekey(scope, candidates)
    if not pending:
        return {"model": model_name, "state_rows": 0, "ledger_rows": 0}
    state_rows = 0
    ledger_rows = 0
    with create_adapter(resolution.resolved.warehouse, project_dir=project_dir) as adapter:
        coordinator = ServingCoordinator(adapter, ensure_schema=True)
        for source in pending:
            # Ledger first: it is the row that decides whether an index is
            # considered published at all. If the state move fails after it, a
            # re-run finds the ledger already moved and finishes the state.
            ledger_rows += coordinator.rekey_scope(source, scope)
            state_rows += adapter.rekey_state_scope(source, scope)
    return {
        "model": model_name,
        "state_rows": state_rows,
        "ledger_rows": ledger_rows,
    }
