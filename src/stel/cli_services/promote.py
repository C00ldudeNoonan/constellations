"""`stel promote` operations (issue #380, #329 phase 3).

Reads candidate judgments from the warehouse relation the transcript project
produced and renders them as a golden-set file for review. The command edge
formats and decides whether to write; nothing here promotes anything.

Retrieval and adapter imports stay lazy so importing this module never pulls a
warehouse driver.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from ..config import load_project
from ..profile import resolve_profile
from ..promotion.contract import UNCONFIRMED_QUERY_TEXT, PromotionError
from ..promotion.draft import CandidateRow, Draft, draft_golden_set
from .context import ConfigClickError

# Columns the candidates model must produce, named rather than inferred so a
# relation that drifted fails on the missing column instead of drafting a
# golden set from the wrong field.
REQUIRED_COLUMNS = (
    "session_id",
    "harness",
    "query_fingerprint",
    "query_text",
    "id_space",
    "context_id",
    "judgment",
)

_HEADER = f"""\
# Promoted golden set - DRAFTED, NOT PROMOTED.
#
# `stel promote` proposed these rows from candidate judgments. Nothing here is
# a promotion until a human has read it and merged it: review this file like
# any other change.
#
# Before merging:
#   - confirm every `query_text` is the question the golden should ask. Text
#     transcribed from the corpus is shown as captured; any row still reading
#     "{UNCONFIRMED_QUERY_TEXT}"
#     has none and will be refused until you write it.
#   - confirm `id_space` matches the target index's `id_field`. A mismatch is
#     a hard error at run time, not a silent zero recall.
#   - `relevant_ids` hold only ids an answer actually cited. Ids that were
#     returned and not cited were deliberately left out: that is absence of
#     evidence, not evidence of irrelevance.
"""


def read_candidates(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    relation: str,
) -> list[CandidateRow]:
    """Load candidate judgments from `relation` through the active adapter."""
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
            f"SELECT {columns} FROM {quoted} "
            f"ORDER BY {adapter.quote_ident('query_fingerprint')}, "
            f"{adapter.quote_ident('context_id')}"
        )
    return [
        CandidateRow(
            session_id="" if row[0] is None else str(row[0]),
            harness=None if row[1] is None else str(row[1]),
            query_fingerprint="" if row[2] is None else str(row[2]),
            query_text=None if row[3] is None else str(row[3]),
            id_space="" if row[4] is None else str(row[4]),
            context_id=None if row[5] is None else str(row[5]),
            judgment="" if row[6] is None else str(row[6]),
        )
        for row in rows
    ]


def render_golden_set(draft: Draft) -> str:
    """The YAML a reviewer reads, header comment included."""
    document = draft.golden_set.model_dump(mode="json")
    # Tuples render as YAML sequences only after model_dump; sort_keys off so
    # the field order stays the one the contract declares and a re-draft
    # produces a reviewable diff rather than a reordering.
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    return _HEADER + body


def promote_from_candidates(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    relation: str,
    output: Path,
    promoted_by: str,
    write: bool = False,
    force: bool = False,
    today: date | None = None,
) -> tuple[str, Draft]:
    """Return the drafted golden set, writing it only if asked.

    The file is human-owned, so an existing one is never overwritten without
    `force`: a re-draft would discard the reviewer's own edits, which are the
    entire value of the artifact.
    """
    rows = read_candidates(
        project_dir,
        profiles_dir=profiles_dir,
        target=target,
        relation=relation,
    )
    try:
        draft = draft_golden_set(
            rows,
            promoted_by=promoted_by,
            promoted_at=today or date.today(),
        )
    except PromotionError as error:
        raise ConfigClickError(str(error)) from error

    rendered = render_golden_set(draft)
    if not write:
        return rendered, draft
    if output.exists() and not force:
        raise ConfigClickError(
            f"{output} already exists; re-drafting would discard the review "
            "it already carries. Pass --force to overwrite it deliberately."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return rendered, draft
