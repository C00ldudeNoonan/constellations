"""Validity intervals on extracted relations (issue #766, ADR-0056).

Three pieces share one vocabulary of columns:

- `SupersedeOptions` is the options block of the `supersede_relations`
  transform, kept here so the compiler's project-level checks can parse it
  without importing a transform module.
- `supersede` closes the interval of a functional relation that a later,
  different fact replaced. It never deletes a row.
- `relations_as_of` reads a relation frame at an instant.

All are pure functions over frames, so they are exercised without a warehouse.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, NamedTuple

import polars as pl
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# One dtype for every instant a relation row carries, so the extractor's output,
# the supersession step's and `relations_as_of` agree on what a timestamp is.
INSTANT_DTYPE = pl.Datetime("us", "UTC")

SUPERSEDE_TRANSFORM_MODULE = "stel.text.transforms.supersede_relations"

# Bump when the closing rule changes: a built-in transform has no project file
# for `compute_model_code_version` to hash, so this is what re-keys a model
# whose intervals would now come out differently.
SUPERSESSION_VERSION = "1"


class SupersedeOptions(BaseModel):
    """`transform.options` for `supersede_relations`.

    Both inputs are named models the project builds: the relation table
    `extract_relations` writes and the link table `link_entities` writes. Their
    columns are fixed by those transforms, so there are no field-name options.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    relations: str
    entity_links: str

    @field_validator("relations", "entity_links")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @model_validator(mode="after")
    def _distinct_models(self) -> SupersedeOptions:
        if self.relations == self.entity_links:
            raise ValueError(
                "`relations` and `entity_links` must name different models"
            )
        return self


def parse_supersede_options(options: Mapping[str, Any]) -> SupersedeOptions:
    return SupersedeOptions.model_validate(dict(options))


def is_temporal(dtype: pl.DataType) -> bool:
    """A column that can carry an instant: a date, or a datetime of any zone."""
    return dtype == pl.Date or isinstance(dtype, pl.Datetime)


def to_utc_instant(column: str, dtype: pl.DataType) -> pl.Expr:
    """`column` as `INSTANT_DTYPE`. A date reads as midnight UTC and a naive
    datetime as UTC; an aware one is converted. The caller checks
    `is_temporal` first so it can say which column was refused."""
    expr = pl.col(column)
    if dtype == pl.Date:
        expr = expr.cast(pl.Datetime("us"))
    if getattr(dtype, "time_zone", None) is None:
        expr = expr.dt.replace_time_zone("UTC")
    else:
        expr = expr.dt.convert_time_zone("UTC")
    return expr.dt.cast_time_unit("us")


_RELATIONS_HINT = (
    "`extract_relations` writes `valid_from` and `valid_to` when it is given "
    "`effective_date_field`, so name the mentions column that carries each "
    "document's date and rebuild the model"
)
_LINKS_HINT = "`link_entities` writes them, so this is not its output"


def _require_columns(
    frame: pl.DataFrame, required: Sequence[str], model: str, hint: str
) -> None:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"the {model} is missing column(s) {', '.join(missing)}; {hint}")


@dataclass(frozen=True)
class SupersessionResult:
    """The relation rows with intervals closed, and what the run did to them.

    `skipped` counts rows of a functional type that took no part, by reason, so
    a short count of closures is explained rather than discovered.
    `simultaneous_conflicts` counts (subject, instant) pairs where different
    objects were asserted at the same `valid_from`: nothing orders them, so they
    stay open together.
    """

    frame: pl.DataFrame
    closed: int
    skipped: dict[str, int]
    simultaneous_conflicts: int


_RELATION_COLUMNS = (
    "relation_id",
    "subject_mention_id",
    "object_mention_id",
    "relation_type",
    "directed",
    "status",
    "valid_from",
    "valid_to",
)
_LINK_COLUMNS = ("mention_id", "entity_namespace", "canonical_id", "status")

# (entity_namespace, canonical_id): the same canonical id in two namespaces is
# two entities, as it is for `link_entities` and the agent-context entity id.
_Entity = tuple[str, str]


class _Fact(NamedTuple):
    valid_from: datetime
    relation_id: str
    obj: _Entity


def supersede(
    relations: pl.DataFrame,
    links: pl.DataFrame,
    *,
    functional: frozenset[str],
) -> SupersessionResult:
    """Close each functional fact that a later, different fact replaced.

    A fact is keyed by `(relation_type, subject entity)`, the entity being the
    subject mention's matched link. It closes at the earliest strictly later
    `valid_from` among facts of the same key asserting a different object entity,
    and records that fact's `relation_id` in `superseded_by`. Restating the same
    object closes nothing. No row is removed, and rows come back sorted by
    `relation_id`, so the output does not depend on the order they arrived in.

    Only `asserted` rows whose two mentions both resolve to a matched entity take
    part; the rest pass through unchanged. A mention matched to two entities, a
    functional row with no `valid_from`, and an undirected functional row are
    errors: each would let row order, or a guess, decide an answer.
    """
    _require_columns(relations, _RELATION_COLUMNS, "relation model", _RELATIONS_HINT)
    _require_columns(links, _LINK_COLUMNS, "link model", _LINKS_HINT)
    if "superseded_by" in relations.columns:
        raise ValueError(
            "the relation model already has a `superseded_by` column; "
            "`supersede_relations` writes it, so it cannot read its own output"
        )
    repeated = relations.filter(pl.col("relation_id").is_duplicated())
    if repeated.height:
        raise ValueError(
            f"relation_id '{repeated['relation_id'][0]}' appears more than once in "
            "the relation model; it must be unique"
        )

    entities = _mention_entities(links)
    frame = relations.with_columns(
        to_utc_instant("valid_from", relations.schema["valid_from"]).alias("valid_from"),
        to_utc_instant("valid_to", relations.schema["valid_to"]).alias("valid_to"),
    ).sort("relation_id")

    rows = frame.filter(pl.col("relation_type").is_in(sorted(functional)))
    _reject_undirected(rows)
    skipped = {"not_asserted": 0, "unlinked_mention": 0}
    groups: dict[tuple[str, _Entity], list[_Fact]] = defaultdict(list)
    already_ended: dict[str, datetime] = {}
    undated: dict[str, int] = defaultdict(int)
    for row in rows.iter_rows(named=True):
        if row["status"] != "asserted":
            skipped["not_asserted"] += 1
            continue
        subject = entities.get(str(row["subject_mention_id"]))
        obj = entities.get(str(row["object_mention_id"]))
        if subject is None or obj is None:
            skipped["unlinked_mention"] += 1
            continue
        if row["valid_from"] is None:
            undated[str(row["relation_type"])] += 1
            continue
        groups[(str(row["relation_type"]), subject)].append(
            _Fact(row["valid_from"], str(row["relation_id"]), obj)
        )
        if row["valid_to"] is not None:
            already_ended[str(row["relation_id"])] = row["valid_to"]
    if undated:
        relation_type = sorted(undated)[0]
        raise ValueError(
            f"functional relation '{relation_type}' has {undated[relation_type]} "
            "asserted row(s) with no valid_from, so a later fact cannot be ordered "
            "against them; name the "
            "mentions column holding each document's date as `effective_date_field` "
            "on the relation extractor"
        )

    closings: dict[str, tuple[datetime, str]] = {}
    simultaneous = 0
    for facts in groups.values():
        group_closings, group_simultaneous = _close_group(facts)
        closings.update(group_closings)
        simultaneous += group_simultaneous
    # An interval the input already ended before the next fact began was not
    # closed by it: crediting that fact in `superseded_by`, or counting a
    # closure, would say something the data does not.
    closings = {
        relation_id: (closes_at, closer)
        for relation_id, (closes_at, closer) in closings.items()
        if relation_id not in already_ended or closes_at <= already_ended[relation_id]
    }

    return SupersessionResult(
        frame=_apply_closings(frame, closings),
        closed=len(closings),
        skipped=skipped,
        simultaneous_conflicts=simultaneous,
    )


def _reject_undirected(rows: pl.DataFrame) -> None:
    undirected = rows.filter(~pl.col("directed").fill_null(False))
    if undirected.height:
        raise ValueError(
            f"relation type '{undirected['relation_type'][0]}' is declared functional "
            "but the relation model holds undirected rows of it; a functional "
            "relation needs a subject and an object, which co-occurrence does not have"
        )


def _mention_entities(links: pl.DataFrame) -> dict[str, _Entity]:
    """Mention id -> its entity, for matched links only.

    An ambiguous or unmatched mention names no single entity. A mention that two
    matched links resolve to different entities is a defect upstream, so it fails
    instead of letting row order pick a side.
    """
    matched = (
        links.filter(
            (pl.col("status") == "matched")
            & pl.col("canonical_id").is_not_null()
            & pl.col("entity_namespace").is_not_null()
            & (pl.col("canonical_id").cast(pl.String) != "")
        )
        .select(
            pl.col("mention_id").cast(pl.String),
            pl.col("entity_namespace").cast(pl.String),
            pl.col("canonical_id").cast(pl.String),
        )
        .unique()
    )
    split = matched.filter(pl.col("mention_id").is_duplicated())
    if split.height:
        raise ValueError(
            f"mention '{split.sort('mention_id')['mention_id'][0]}' is matched to two "
            "different entities; the linking model must resolve a mention to one"
        )
    return {
        mention: (namespace, canonical)
        for mention, namespace, canonical in matched.iter_rows()
    }


def _first_different(
    best: Sequence[tuple[_Entity, datetime, str]], obj: _Entity
) -> tuple[datetime, str] | None:
    """The soonest of the (at most two) candidate closers naming another object."""
    for candidate, closes_at, closer in best:
        if candidate != obj:
            return closes_at, closer
    return None


def _close_group(facts: list[_Fact]) -> tuple[dict[str, tuple[datetime, str]], int]:
    """Closings within one `(relation_type, subject)`, and its simultaneous count.

    Walks the instants from latest to earliest. After each, `best` holds the two
    objects whose earliest assertion in the suffix is soonest, which is all a
    fact needs: the soonest of them that differs from its own object is the
    closer. A fact only ever looks at instants strictly after its own, so facts
    that share a `valid_from` never close one another.
    """
    facts = sorted(facts, key=lambda fact: (fact.valid_from, fact.relation_id))
    closings: dict[str, tuple[datetime, str]] = {}
    simultaneous = 0
    best: list[tuple[_Entity, datetime, str]] = []
    end = len(facts)
    while end > 0:
        instant = facts[end - 1].valid_from
        start = end
        while start > 0 and facts[start - 1].valid_from == instant:
            start -= 1
        same_instant = facts[start:end]
        for fact in same_instant:
            closer = _first_different(best, fact.obj)
            if closer is not None:
                closings[fact.relation_id] = closer
        # Rows are in relation_id order, so the first per object is its lowest.
        lowest: dict[_Entity, str] = {}
        for fact in same_instant:
            lowest.setdefault(fact.obj, fact.relation_id)
        if len(lowest) > 1:
            simultaneous += 1
        entries = [
            (obj, instant, relation_id)
            for relation_id, obj in sorted((rid, obj) for obj, rid in lowest.items())
        ]
        best = (entries + [entry for entry in best if entry[0] not in lowest])[:2]
        end = start
    return closings, simultaneous


def _apply_closings(
    frame: pl.DataFrame, closings: Mapping[str, tuple[datetime, str]]
) -> pl.DataFrame:
    closed = pl.DataFrame(
        {
            "relation_id": list(closings),
            "_closes_at": [closes_at for closes_at, _ in closings.values()],
            "superseded_by": [closer for _, closer in closings.values()],
        },
        schema={
            "relation_id": pl.String,
            "_closes_at": INSTANT_DTYPE,
            "superseded_by": pl.String,
        },
    )
    return (
        frame.join(closed, on="relation_id", how="left")
        # `closings` holds only closers at or before an existing end, so the
        # closing instant, when there is one, is the end.
        .with_columns(pl.coalesce("_closes_at", "valid_to").alias("valid_to"))
        .drop("_closes_at")
    )


def relations_as_of(
    relations: pl.DataFrame,
    instant: datetime,
    *,
    include_unknown_validity: bool,
) -> pl.DataFrame:
    """The rows whose interval holds at `instant`.

    An interval is half-open, `[valid_from, valid_to)`: at the instant a newer
    fact begins, only the newer fact holds. A row with no `valid_from` does not
    claim to hold at any instant, so it is left out unless the caller asks for
    it. Status is not filtered: combine this with `status == 'asserted'` for
    only the facts an extractor stood behind.
    """
    if not isinstance(instant, datetime) or instant.tzinfo is None:
        raise ValueError("instant must be a timezone-aware datetime")
    _require_columns(relations, ("valid_from", "valid_to"), "relation model", _RELATIONS_HINT)
    at = pl.lit(instant.astimezone(UTC), dtype=INSTANT_DTYPE)
    start = to_utc_instant("valid_from", relations.schema["valid_from"])
    end = to_utc_instant("valid_to", relations.schema["valid_to"])
    holds = (start <= at) & (end.is_null() | (at < end))
    if include_unknown_validity:
        holds = holds | start.is_null()
    return relations.filter(holds)
