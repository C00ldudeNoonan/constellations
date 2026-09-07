"""What an experiment leaves behind (issue #531).

A renamed model, a dropped `for_each` variant, or a deleted model file
orphans two things in the warehouse: its table, and the rows `stel_state`
holds for its scope. Nothing points at either any more, nothing prunes them,
and `stel clean` correctly never touches the warehouse. This module lists
them so they are noticed rather than discovered in a storage bill.

Listing only. Deleting is a separate, explicit act outside this module, per
the cleanup invariant: a familiar command must never hide a warehouse-wide
drop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import StateScopeSummary, WarehouseAdapter, create_adapter
from .compiler import validate_project_contract
from .config.loader import load_project
from .config.model import ModelConfig
from .profile import resolve_profile


@dataclass(frozen=True)
class OrphanTable:
    name: str
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "table", "name": self.name, "rows": self.rows}


@dataclass(frozen=True)
class OrphanStateScope:
    summary: StateScopeSummary

    @property
    def name(self) -> str:
        return self.summary.scope.model_name

    def to_dict(self) -> dict[str, Any]:
        scope = self.summary.scope
        return {
            "kind": "state_scope",
            "name": scope.model_name,
            "stage": scope.stage,
            "target_identity": scope.target_identity,
            "rows": self.summary.rows,
            "code_versions": self.summary.code_versions,
            "last_run_at": self.summary.last_run_at,
        }


@dataclass(frozen=True)
class OrphanReport:
    tables: tuple[OrphanTable, ...]
    state_scopes: tuple[OrphanStateScope, ...]
    schema: str

    def is_empty(self) -> bool:
        return not self.tables and not self.state_scopes

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tables": [table.to_dict() for table in self.tables],
            "state_scopes": [scope.to_dict() for scope in self.state_scopes],
        }


def find_orphans(
    project_dir: Path,
    *,
    target: str | None,
    profiles_dir: Path | None,
) -> OrphanReport:
    """Tables and state scopes in the target schema that no model claims.

    A model claims its own table and every state scope carrying its name;
    `for_each` variants are expanded before this runs, so each variant claims
    its own. Everything else in the schema is reported, which includes any
    table something other than stel put there -- the report says where it
    looked, and deletion is not its job."""
    project, sources, models = load_project(project_dir)
    validate_project_contract(project, sources, models, project_dir)
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    adapter = create_adapter(resolved.warehouse, project_dir=project_dir)
    with adapter:
        report = _collect(adapter, models, schema=resolved.warehouse.schema_name)
    return report


def _collect(
    adapter: WarehouseAdapter, models: list[ModelConfig], *, schema: str
) -> OrphanReport:
    claimed = {model.name for model in models}
    tables = tuple(
        OrphanTable(name=name, rows=adapter.row_count(name))
        for name in sorted(adapter.list_tables())
        if name not in claimed
    )
    scopes = tuple(
        OrphanStateScope(summary)
        for summary in adapter.list_state_scopes()
        if summary.scope.model_name not in claimed
    )
    return OrphanReport(tables=tables, state_scopes=scopes, schema=schema)


def format_orphans(report: OrphanReport) -> list[str]:
    if report.is_empty():
        return [f"No orphans in schema {report.schema}: every table and state scope is claimed."]
    lines = [f"{'name':<32}{'kind':<13}{'rows':>10}  detail"]
    lines.append("-" * len(lines[0]))
    for table in report.tables:
        lines.append(f"{table.name:<32}{'table':<13}{table.rows:>10}")
    for orphan in report.state_scopes:
        summary = orphan.summary
        detail = (
            f"stage={summary.scope.stage} code_versions={summary.code_versions}"
            f" last_run_at={summary.last_run_at or '-'}"
        )
        lines.append(f"{orphan.name:<32}{'state_scope':<13}{summary.rows:>10}  {detail}")
    lines.append("")
    lines.append(
        f"{len(report.tables)} table(s) and {len(report.state_scopes)} state scope(s) in "
        f"schema {report.schema} are claimed by no model in this project. Nothing was "
        "removed; drop what you no longer need explicitly."
    )
    return lines
