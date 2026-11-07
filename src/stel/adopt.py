"""`stel adopt`: restamp published rows to the current code_version (issue #750).

A release can move a model's `code_version` without changing a single output
row. `stel plan` shows what that would cost; `--accept-reprocess` pays it.
This is the third answer, for an operator who has read the plan and checked the
output is unchanged: record the current version against rows already
published, and rewrite nothing.

It touches `stel_state` only. No source is discovered, no provider is called,
and no model table is read or written, so a model's own `code_version` column
(one that `update_when_changed` watches, say) keeps the old value until its
row is next rewritten for another reason. Every restamp is appended to
`stel_adoption_log` with who, when, and the old and new versions.
"""
from __future__ import annotations

import getpass
from dataclasses import dataclass
from pathlib import Path

from .adapters import StateScope, create_adapter
from .compiler import validate_project_contract, validate_warehouse_capabilities
from .config.loader import load_project
from .plan import state_scope_for
from .profile import resolve_profile
from .versioning import compute_model_code_identity
from .vocabulary_sources import resolve_warehouse_vocabularies


@dataclass(frozen=True)
class AdoptionResult:
    model: str
    #: `adopted`, `current` (nothing stale), `empty` (no published state),
    #: `full` (keeps no state to restamp).
    status: str
    code_version: str
    #: Rows moved from each old code_version; empty unless `adopted`.
    adopted_rows: dict[str, int]
    dry_run: bool

    @property
    def rows(self) -> int:
        return sum(self.adopted_rows.values())


def adopt_project(
    project_dir: Path,
    *,
    select: str,
    exclude: str | None,
    target: str | None,
    profiles_dir: Path | None,
    dry_run: bool,
) -> list[AdoptionResult]:
    """Restamp each selected model's published rows to its current version.

    Same preflight as `plan`: a bad configuration fails before the warehouse
    is opened."""
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)
    resolved = resolve_profile(project, project_dir, target=target, profiles_dir=profiles_dir)
    adapter = create_adapter(resolved.warehouse, project_dir=project_dir)
    models_by_name = {model.name: model for model in models}
    planned = [
        models_by_name[name] for name in dag.select_models(select=select, exclude=exclude)
    ]
    validate_warehouse_capabilities(planned, adapter)
    resolve_warehouse_vocabularies(project, sources, models, project_dir, adapter)
    who = getpass.getuser()
    results: list[AdoptionResult] = []
    with adapter:
        for model in planned:
            identity = compute_model_code_identity(
                model, project, project_dir, resolved=resolved
            )
            if model.search is None and model.materialization == "full":
                results.append(
                    AdoptionResult(model.name, "full", identity.code_version, {}, dry_run)
                )
                continue
            scope: StateScope = state_scope_for(model, project=project, resolved=resolved)
            counts = adapter.state_code_version_counts(scope)
            stale = {v: n for v, n in counts.items() if v != identity.code_version}
            if not counts:
                status = "empty"
            elif not stale:
                status = "current"
            else:
                status = "adopted"
            moved: dict[str, int] = {}
            if status == "adopted":
                moved = (
                    stale
                    if dry_run
                    else adapter.adopt_code_version(
                        scope, identity.code_version, identity.components, adopted_by=who
                    )
                )
            results.append(
                AdoptionResult(model.name, status, identity.code_version, moved, dry_run)
            )
    return results

