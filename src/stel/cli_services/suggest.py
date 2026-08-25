"""`stel suggest` operations (issue #361).

Reads candidate suggestions from a warehouse relation the analysis project
produced, renders them as a patch against a dbt project, and returns the diff
as data. The command edge formats and decides whether to write.

Retrieval and adapter imports stay lazy so importing this module never pulls a
warehouse driver.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import load_project
from ..profile import resolve_profile
from ..suggest import (
    DEFAULT_MIN_EVIDENCE,
    SuggestionError,
    SuggestionOutcome,
    SuggestionRow,
    plan_suggestions,
)
from .context import ConfigClickError

if TYPE_CHECKING:
    pass

# Columns the analysis project must produce. Named here rather than inferred
# so a relation that drifted fails with the missing column, not with a
# suggestion silently built from the wrong field.
REQUIRED_COLUMNS = (
    "dbt_model",
    "suggested_description",
    "evidence_count",
    "evidence_sessions",
)


def _split_sessions(value: Any) -> tuple[str, ...]:
    """Provenance arrives as a list or a delimited string, depending on adapter."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item).strip())
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def read_suggestions(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    relation: str,
) -> list[SuggestionRow]:
    """Load candidate suggestions from `relation` through the active adapter."""
    from ..adapters import create_adapter
    from ..config.source import validate_relation_name

    try:
        validate_relation_name(relation)
    except ValueError as error:
        raise ConfigClickError(str(error)) from error

    project_config, _sources, _models = load_project(project_dir)
    resolved = resolve_profile(
        project_config, project_dir, target=target, profiles_dir=profiles_dir
    )
    with create_adapter(resolved.warehouse, project_dir=project_dir) as adapter:
        quoted = ".".join(adapter.quote_ident(part) for part in relation.split("."))
        columns = ", ".join(adapter.quote_ident(name) for name in REQUIRED_COLUMNS)
        rows = adapter.rows(
            f"SELECT {columns}, "
            f"{adapter.quote_ident('dbt_column')} FROM {quoted} "
            f"ORDER BY {adapter.quote_ident('evidence_count')} DESC, "
            f"{adapter.quote_ident('dbt_model')}"
        )
    return [
        SuggestionRow(
            dbt_model=str(row[0]),
            suggested_description=str(row[1]),
            evidence_count=int(row[2]),
            evidence_sessions=_split_sessions(row[3]),
            dbt_column=None if row[4] is None else str(row[4]),
        )
        for row in rows
    ]


def suggest_dbt(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    relation: str,
    dbt_project_dir: Path,
    min_evidence: int = DEFAULT_MIN_EVIDENCE,
    write: bool = False,
) -> tuple[str, list[SuggestionOutcome]]:
    """Return the unified diff for a dbt project, applying it only if asked.

    Nothing is written unless `write` is set, and nothing is ever committed:
    the artifact is a patch a human reads.
    """
    rows = read_suggestions(
        project_dir,
        profiles_dir=profiles_dir,
        target=target,
        relation=relation,
    )
    try:
        pending, outcomes = plan_suggestions(
            rows, dbt_project_dir, min_evidence=min_evidence
        )
    except SuggestionError as error:
        raise ConfigClickError(str(error)) from error

    diff_lines: list[str] = []
    for path in sorted(pending):
        before = path.read_text(encoding="utf-8")
        relative = path.relative_to(dbt_project_dir).as_posix()
        diff_lines.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                pending[path].splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    if write:
        for path, content in pending.items():
            path.write_text(content, encoding="utf-8")
    return "".join(diff_lines), outcomes
