"""Context improvements proposed as a reviewable diff (issue #361).

The feedback loop in #329 ends at measurement. Measurement tells you the
context is bad; it does not fix it, and the fix is almost always a small edit
to a file in a repo — a missing `description:` on the dbt model an agent had
to read the SQL of, three sessions running.

**The output is a diff, not a warehouse row.** dbt projects keep their context
as files in git, so the acceptance mechanism is a patch a human reads and
merges. That is #329's rule 2 — candidates, never auto-promotion — expressed
as the thing engineers already do all day, and it means the proposal lands
where the context actually lives instead of in a table nobody opens.

**Where the analysis lives.** Not here. Deciding *which* models are
under-documented and *what* their descriptions should say is a stel project:
models over the transcript corpus, with the provider, prompt provenance, and
incremental machinery that already exists. This module reads the relation
those models produce and renders it as a patch. The split keeps #329's rule 3
— reuse the pipeline, don't grow a sidecar — and it keeps prompt output inside
the artifact surfaces that already govern it.

The contract between the two is `SuggestionRow` below.

**What this will not do**, because each is a way for a well-meaning suggestion
to destroy work:

- overwrite a description that already exists — only absent ones are filled;
- touch any key other than `description:`;
- write outside the dbt project directory;
- apply anything without `--write`, and never commit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config.identifiers import validate_node_name

# Minimum distinct sessions before a gap is worth proposing a change for. One
# session is an anecdote: an agent reads a model's SQL for all sorts of
# reasons, most of them not a documentation gap. Repetition across sessions is
# what distinguishes "someone looked at this once" from "this keeps costing
# people time".
DEFAULT_MIN_EVIDENCE = 3


class SuggestionError(Exception):
    """Artifact-safe suggestion failure."""


@dataclass(frozen=True)
class SuggestionRow:
    """One proposed description, as produced by the analysis project.

    `evidence_sessions` is the provenance a reviewer needs first: the question
    they will ask is "where did this come from?", and a suggestion that cannot
    answer it should not be proposed.
    """

    dbt_model: str
    suggested_description: str
    evidence_count: int
    evidence_sessions: tuple[str, ...]
    dbt_column: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("dbt_model", "suggested_description"):
            if not str(getattr(self, field_name)).strip():
                raise SuggestionError(f"Suggestion {field_name} must not be empty")
        # The model and column names cross into a path lookup and a YAML key,
        # so they are validated on the same identifier charset stel uses for
        # its own nodes rather than trusted from a relation.
        validate_node_name(self.dbt_model, kind="dbt model")
        if self.dbt_column is not None:
            validate_node_name(self.dbt_column, kind="dbt column")
        if self.evidence_count < 1:
            raise SuggestionError("Suggestion evidence_count must be positive")

    @property
    def target(self) -> str:
        """Human-readable target, for diff headers and refusal messages."""
        return (
            self.dbt_model
            if self.dbt_column is None
            else f"{self.dbt_model}.{self.dbt_column}"
        )


@dataclass(frozen=True)
class SuggestionOutcome:
    """What happened to one suggestion, so the caller can report honestly."""

    target: str
    applied: bool
    reason: str
    path: Path | None = None


def schema_files(dbt_project_dir: Path) -> list[Path]:
    """Every `models/**/*.yml` in a dbt project, in a stable order.

    Restricted to `models/` deliberately: `description:` for a model lives
    there, and widening the search to the whole project would put seeds,
    snapshots, and `dbt_project.yml` itself in reach of an automated edit.
    """
    models_dir = dbt_project_dir / "models"
    if not models_dir.is_dir():
        raise SuggestionError(
            f"No models/ directory under {dbt_project_dir}; this does not look "
            "like a dbt project"
        )
    if models_dir.is_symlink():
        raise SuggestionError(
            f"{models_dir} is a symlink; refusing to edit through it"
        )
    return sorted(
        path
        for pattern in ("*.yml", "*.yaml")
        for path in models_dir.rglob(pattern)
        if path.is_file() and _contained_without_symlinks(path, models_dir)
    )


def _contained_without_symlinks(path: Path, root: Path) -> bool:
    """Whether `path` sits under `root` with no symlink anywhere between.

    `rglob` follows symlinked directories, and `is_symlink()` on the file it
    yields sees only the regular destination — so a symlinked `models/sub`
    would put files outside the dbt project in reach of `--write`, breaking
    the containment this module promises (Codex review). Every component from
    `root` down is checked instead.

    Deliberately lexical: `resolve()` would follow the very links being
    guarded against, so ancestry is compared on the unresolved path.
    """
    if path.is_symlink():
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return False
    return True


def _model_entry(document: Any, model_name: str) -> dict[str, Any] | None:
    """The `models:` entry for `model_name`, or None if this file lacks it."""
    if not isinstance(document, dict):
        return None
    models = document.get("models")
    if not isinstance(models, list):
        return None
    for entry in models:
        if isinstance(entry, dict) and entry.get("name") == model_name:
            return entry
    return None


def _column_entry(entry: dict[str, Any], column: str) -> dict[str, Any] | None:
    columns = entry.get("columns")
    if not isinstance(columns, list):
        return None
    for item in columns:
        if isinstance(item, dict) and item.get("name") == column:
            return item
    return None


def _entry_block(lines: list[str], start: int) -> int:
    """Index one past the last line of the list entry beginning at `start`."""
    marker = len(lines[start]) - len(lines[start].lstrip())
    index = start + 1
    while index < len(lines):
        line = lines[index]
        if line.strip() and not line.lstrip().startswith("#"):
            indent = len(line) - len(line.lstrip())
            if indent <= marker:
                break
        index += 1
    return index


def _sequence_item_indent(lines: list[str], span: range) -> int | None:
    """Indent of the first `- ` item in `span` — the depth every direct entry
    of that sequence sits at."""
    for index in span:
        stripped = lines[index].lstrip()
        if stripped.startswith("- "):
            return len(lines[index]) - len(stripped)
    return None


def _find_entry_line(
    lines: list[str], name: str, *, within: range | None, depth: int | None = None
) -> int | None:
    """Line index of the `- name: <name>` entry, optionally inside a block.

    `depth` restricts the match to entries of one sequence. Without it a
    *column* named like the model it belongs beside would match first and be
    documented instead of the model (Codex review, extended): scoping to the
    `models:` block is not enough, because columns are nested inside it.
    """
    span = within if within is not None else range(len(lines))
    for index in span:
        line = lines[index]
        stripped = line.strip()
        if stripped not in (f"- name: {name}", f"- name: '{name}'", f'- name: "{name}"'):
            continue
        if depth is not None and len(line) - len(line.lstrip()) != depth:
            continue
        return index
    return None


def _top_level_block(lines: list[str], key: str) -> range | None:
    """Line range covered by a top-level `key:` block, or None.

    A schema file may declare `sources:` beside `models:`, and a source table
    can share a name with a model. Searching the whole file for `- name: <x>`
    would then find the source entry first and document the wrong object
    (Codex review), so the search is scoped to the block that owns models.
    """
    for index, line in enumerate(lines):
        if line.strip() != f"{key}:" or line[:1].isspace():
            continue
        end = index + 1
        while end < len(lines):
            candidate = lines[end]
            if (
                candidate.strip()
                and not candidate.lstrip().startswith("#")
                and not candidate[:1].isspace()
            ):
                break
            end += 1
        return range(index + 1, end)
    return None


def _documented_reason(entry: dict[str, Any]) -> str | None:
    """Why this entry must not receive an inserted description, or None.

    Presence of the key decides, not truthiness: inserting a second
    `description:` beside an empty or null one produces a duplicate mapping
    key, which loaders resolve to the *existing* empty value — the suggestion
    would be silently discarded while the command reported it applied, and
    stricter loaders reject the file outright (Codex review).
    """
    if "description" not in entry:
        return None
    if entry.get("description"):
        return "already documented"
    return "description key already present but empty; edit it by hand"


def _columns_block(lines: list[str], model_start: int, model_end: int) -> range | None:
    for index in range(model_start, model_end):
        if lines[index].strip() == "columns:":
            return range(index + 1, model_end)
    return None


def render_suggestion(
    text: str, row: SuggestionRow
) -> tuple[str | None, str]:
    """Return (new file text, reason). `None` means this file was not changed.

    Text-level insertion rather than a YAML round-trip: dumping a parsed
    document back would strip the project's comments and reflow every key,
    burying a one-line proposal in a whole-file diff. A reviewer has to be able
    to see what is being suggested.
    """
    document = yaml.safe_load(text)
    entry = _model_entry(document, row.dbt_model)
    if entry is None:
        return None, "model not declared in this file"

    lines = text.splitlines(keepends=True)
    models_block = _top_level_block(lines, "models")
    if models_block is None:
        return None, "models block is not a plain block sequence; edit it by hand"
    model_line = _find_entry_line(
        lines,
        row.dbt_model,
        within=models_block,
        depth=_sequence_item_indent(lines, models_block),
    )
    if model_line is None:
        # Parsed but not locatable as a plain `- name:` line — a flow mapping
        # or an anchor. Refusing beats guessing at an insertion point.
        return None, "model entry is not a plain block mapping; edit it by hand"
    model_end = _entry_block(lines, model_line)

    if row.dbt_column is None:
        documented = _documented_reason(entry)
        if documented is not None:
            return None, documented
        insert_at, indent_from = model_line + 1, lines[model_line]
    else:
        column = _column_entry(entry, row.dbt_column)
        if column is None:
            return None, f"column '{row.dbt_column}' not declared in this file"
        documented = _documented_reason(column)
        if documented is not None:
            return None, documented
        columns = _columns_block(lines, model_line, model_end)
        if columns is None:
            return None, "columns block is not a plain block sequence"
        column_line = _find_entry_line(
            lines,
            row.dbt_column,
            within=columns,
            depth=_sequence_item_indent(lines, columns),
        )
        if column_line is None:
            return None, "column entry is not a plain block mapping"
        insert_at, indent_from = column_line + 1, lines[column_line]

    # Align with the key the entry opens with, not the dash: `- name: x` puts
    # `name` two columns right of the dash, and `description` is its sibling.
    dash = len(indent_from) - len(indent_from.lstrip())
    indent = " " * (dash + 2)
    # `splitlines(keepends=True)` leaves a final line without a trailing
    # newline unterminated; inserting after it would splice both onto one line
    # and emit malformed YAML (Codex review).
    if insert_at > 0 and not lines[insert_at - 1].endswith(chr(10)):
        lines[insert_at - 1] += chr(10)
    lines.insert(insert_at, f"{indent}description: {_yaml_scalar(row.suggested_description)}\n")
    return "".join(lines), "applied"


def _yaml_scalar(value: str) -> str:
    """A safely quoted one-line scalar.

    Round-tripped through the YAML dumper rather than hand-quoted: a
    description carrying a colon, a quote, or a leading `>` is ordinary prose
    and must not be able to restructure the document it lands in.
    """
    dumped = yaml.safe_dump(
        value, default_flow_style=True, width=10**6, allow_unicode=True
    ).strip()
    return dumped.removesuffix("...").strip()


def plan_suggestions(
    rows: list[SuggestionRow],
    dbt_project_dir: Path,
    *,
    min_evidence: int = DEFAULT_MIN_EVIDENCE,
) -> tuple[dict[Path, str], list[SuggestionOutcome]]:
    """Resolve suggestions into edited file contents, without writing anything.

    Returns the proposed contents keyed by path, and one outcome per
    suggestion — including the skipped ones, because "nothing to do here" is
    the answer a reviewer most needs to trust.
    """
    files = schema_files(dbt_project_dir)
    pending: dict[Path, str] = {}
    outcomes: list[SuggestionOutcome] = []
    for row in rows:
        if row.evidence_count < min_evidence:
            outcomes.append(
                SuggestionOutcome(
                    row.target,
                    False,
                    f"below the evidence threshold ({row.evidence_count} < {min_evidence})",
                )
            )
            continue
        outcome = _apply_to_files(row, files, pending)
        outcomes.append(outcome)
    return pending, outcomes


def _apply_to_files(
    row: SuggestionRow, files: list[Path], pending: dict[Path, str]
) -> SuggestionOutcome:
    reason = "model not declared in any models/**/*.yml"
    for path in files:
        current = pending.get(path, path.read_text(encoding="utf-8"))
        updated, why = render_suggestion(current, row)
        if updated is not None:
            pending[path] = updated
            return SuggestionOutcome(row.target, True, why, path)
        if why != "model not declared in this file":
            # The model was found here and deliberately left alone; that is a
            # more useful answer than continuing to search for it elsewhere.
            return SuggestionOutcome(row.target, False, why, path)
    return SuggestionOutcome(row.target, False, reason)
