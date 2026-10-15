"""Built-in lenses: colorings every concept cloud gets for free (issue #345).

Each lens bands a field the bundle already carries on every concept -- mention
count, document spread, link status, best match score -- into a `derived`
dimension, so the picker can switch between them live. The payoff is where two
lenses disagree: a concept named in a thousand documents that nothing ever
linked confidently is invisible in either coloring alone.

Two rules, both from the design notes on #345:

- **Fixed bands, never per-corpus quantiles.** A quantile ramp manufactures a
  hot node in a corpus where everything is cold, and makes one color mean two
  counts on two corpora. Every band is declared on every bundle, populated or
  not, so a band keeps its color from one export to the next.
- **Refuse rather than draw zeros.** A lens with no data would paint every
  concept the same band, which claims they are equal when the truth is that
  nothing was measured. Such a lens is named as unavailable, with the reason.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from .schema import Concept, DimensionDef, UnavailableDimension

MENTIONS = "mentions"
DOCUMENTS = "documents"
LINK_STATUS = "link status"
LINK_SCORE = "link score"
LENS_NAMES = (MENTIONS, DOCUMENTS, LINK_STATUS, LINK_SCORE)

# Decades: a count's order of magnitude is what reads at a glance, and decade
# edges mean the same thing on a corpus of a hundred documents or a million.
_COUNT_BANDS: tuple[tuple[str, int], ...] = (
    ("1", 1),
    ("2–9", 2),
    ("10–99", 10),
    ("100–999", 100),
    ("1,000+", 1000),
)
# Lower bounds, ascending, over the [0, 1] the schema holds a score to. A link
# only exists above its resolver's configured threshold, so the bands resolve
# the top of the range.
_SCORE_BANDS: tuple[tuple[str, float], ...] = (
    ("below 0.80", 0.0),
    ("0.80–0.90", 0.80),
    ("0.90–0.95", 0.90),
    ("0.95 and up", 0.95),
)

LensValues = dict[str, dict[str, str]]


def _band(value: float, bands: Sequence[tuple[str, float]]) -> str:
    """The label of the highest band whose lower bound `value` reaches."""
    label = bands[0][0]
    for name, lower in bands:
        if value >= lower:
            label = name
    return label


def built_in_lenses(
    concepts: Sequence[Concept],
) -> tuple[tuple[DimensionDef, ...], LensValues, tuple[UnavailableDimension, ...]]:
    """The lens definitions, each concept's value per lens, and the lenses
    this corpus cannot fill.

    A concept missing from a lens it cannot be placed on -- a match score on
    an exact match in an otherwise scored corpus -- omits the key, which the
    viewer draws as "no value" rather than as the lowest band. An empty cloud
    gets no lenses at all: there is nothing to color, and a reason claiming
    the corpus lacks a field would be about rows that were never read.
    """
    if not concepts:
        return (), {}, ()
    definitions: list[DimensionDef] = []
    unavailable: list[UnavailableDimension] = []
    values: LensValues = {c.canonical_id: {} for c in concepts}

    def add(
        definition: DimensionDef, value_of: Callable[[Concept], str | None]
    ) -> None:
        definitions.append(definition)
        for concept in concepts:
            value = value_of(concept)
            if value is not None:
                values[concept.canonical_id][definition.name] = value

    add(
        DimensionDef(
            name=MENTIONS,
            values=tuple(name for name, _ in _COUNT_BANDS),
            source="derived",
            ordinal=True,
            description="Mentions linked to the concept, in fixed decade bands",
        ),
        lambda c: _band(c.frequency, _COUNT_BANDS),
    )
    if any(c.provenance.documents > 0 for c in concepts):
        add(
            DimensionDef(
                name=DOCUMENTS,
                values=tuple(name for name, _ in _COUNT_BANDS),
                source="derived",
                ordinal=True,
                description=(
                    "Distinct documents naming the concept, in fixed decade "
                    "bands: how widely the corpus uses it"
                ),
            ),
            lambda c: (
                _band(c.provenance.documents, _COUNT_BANDS)
                if c.provenance.documents > 0 else None
            ),
        )
    else:
        unavailable.append(UnavailableDimension(
            name=DOCUMENTS, reason="the linking rows carry no document_id",
        ))
    add(
        DimensionDef(
            name=LINK_STATUS,
            values=("matched", "ambiguous"),
            source="derived",
            description=(
                "Ambiguous when the resolver could not choose between "
                "candidates for at least one of the concept's mentions"
            ),
        ),
        lambda c: c.link_status,
    )
    if any(c.match_score is not None for c in concepts):
        add(
            DimensionDef(
                name=LINK_SCORE,
                values=tuple(name for name, _ in _SCORE_BANDS),
                source="derived",
                ordinal=True,
                description=(
                    "The best match score among the concept's links: a low "
                    "band means no mention of it linked confidently"
                ),
            ),
            lambda c: (
                _band(c.match_score, _SCORE_BANDS)
                if c.match_score is not None else None
            ),
        )
    else:
        unavailable.append(UnavailableDimension(
            name=LINK_SCORE,
            reason="no link carries a match score (an alias table matches exactly)",
        ))
    return tuple(definitions), values, tuple(unavailable)


def with_built_in_lenses(
    concepts: Sequence[Concept], dimensions: Sequence[DimensionDef]
) -> tuple[tuple[Concept, ...], tuple[DimensionDef, ...], tuple[UnavailableDimension, ...]]:
    """`concepts` and `dimensions` with the built-in lenses added after the
    declared dimensions, which keep their picker order ahead of them. A
    declared dimension named like a lens fails the bundle's unique-name check;
    the export refuses that name before it reads anything (`LENS_NAMES`)."""
    lens_defs, lens_values, unavailable = built_in_lenses(concepts)
    concepts = tuple(
        c.model_copy(update={
            "dimensions": {**c.dimensions, **lens_values[c.canonical_id]},
        })
        for c in concepts
    )
    return concepts, (*dimensions, *lens_defs), unavailable
