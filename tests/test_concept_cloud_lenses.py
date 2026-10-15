"""Built-in lenses and coverage counts on the concept cloud (issue #345).

The lenses band fields every concept already carries; these tests pin the two
rules the design notes on #345 set for them -- fixed bands rather than
per-corpus quantiles, and a refusal rather than a lens full of zeros -- and the
counts that let the caption say what the export left out.
"""
from __future__ import annotations

import polars as pl
import pytest

from stel.concept_cloud import (
    Concept,
    ConceptCloudExportError,
    DagPlane,
    Provenance,
    build_concept_cloud,
)
from stel.concept_cloud.lenses import (
    DOCUMENTS,
    LINK_SCORE,
    LINK_STATUS,
    MENTIONS,
    built_in_lenses,
)


def _concept(
    cid: str, *, frequency: int = 1, documents: int = 1, score: float | None = None
) -> Concept:
    return Concept(
        canonical_id=cid,
        display=cid,
        frequency=frequency,
        match_score=score,
        provenance=Provenance(model="link_entities", documents=documents),
    )


def _lens(concepts: list[Concept], name: str) -> dict[str, str | None]:
    _, values, _ = built_in_lenses(concepts)
    return {cid: by_name.get(name) for cid, by_name in values.items()}


def test_counts_fall_in_fixed_decade_bands() -> None:
    concepts = [
        _concept(f"c{n}", frequency=n, documents=n)
        for n in (1, 2, 9, 10, 99, 100, 999, 1000, 52_000)
    ]
    expected = {
        "c1": "1", "c2": "2–9", "c9": "2–9", "c10": "10–99", "c99": "10–99",
        "c100": "100–999", "c999": "100–999", "c1000": "1,000+",
        "c52000": "1,000+",
    }

    assert _lens(concepts, MENTIONS) == expected
    assert _lens(concepts, DOCUMENTS) == expected


def test_a_band_does_not_depend_on_the_rest_of_the_corpus() -> None:
    """A quantile ramp would put the biggest concept of a uniformly small
    corpus in the top band, and give one count two colors on two corpora."""
    small = [_concept(f"c{n}", frequency=n) for n in (1, 2, 3)]
    large = [*small, _concept("big", frequency=5000)]

    assert _lens(small, MENTIONS)["c3"] == _lens(large, MENTIONS)["c3"] == "2–9"


def test_every_band_is_declared_whether_or_not_a_concept_falls_in_it() -> None:
    """The viewer colors a band by its position in the declared values, so
    the set has to be the same on every bundle for a color to mean one band."""
    definitions, _, _ = built_in_lenses([_concept("only", frequency=3, score=0.97)])
    by_name = {d.name: d for d in definitions}

    assert by_name[MENTIONS].values == ("1", "2–9", "10–99", "100–999", "1,000+")
    assert by_name[LINK_SCORE].values == (
        "below 0.80", "0.80–0.90", "0.90–0.95", "0.95 and up",
    )
    assert by_name[LINK_STATUS].values == ("matched", "ambiguous")
    assert by_name[MENTIONS].ordinal and by_name[LINK_SCORE].ordinal
    assert not by_name[LINK_STATUS].ordinal


def test_match_scores_band_at_their_lower_bounds() -> None:
    concepts = [
        _concept("low", score=0.79), _concept("edge80", score=0.80),
        _concept("edge90", score=0.90), _concept("edge95", score=0.95),
        _concept("zero", score=0.0),
    ]

    assert _lens(concepts, LINK_SCORE) == {
        "low": "below 0.80", "edge80": "0.80–0.90", "edge90": "0.90–0.95",
        "edge95": "0.95 and up", "zero": "below 0.80",
    }


def test_an_unscored_concept_in_a_scored_corpus_has_no_value_not_the_lowest_band() -> None:
    concepts = [_concept("scored", score=0.91), _concept("exact")]

    assert _lens(concepts, LINK_SCORE) == {"scored": "0.90–0.95", "exact": None}


def test_a_lens_the_corpus_cannot_fill_is_unavailable_with_its_reason() -> None:
    """An alias-table corpus has no match scores by design; linking rows with
    no document_id give every concept zero documents. Drawn, either lens would
    paint every concept the same band and read as "all equal"."""
    definitions, values, unavailable = built_in_lenses(
        [_concept("a", documents=0), _concept("b", documents=0)]
    )

    reasons = {u.name: u.reason for u in unavailable}
    assert set(reasons) == {LINK_SCORE, DOCUMENTS}
    assert "alias table" in reasons[LINK_SCORE]
    assert "document_id" in reasons[DOCUMENTS]
    assert {d.name for d in definitions} == {MENTIONS, LINK_STATUS}
    assert all(set(v) == {MENTIONS, LINK_STATUS} for v in values.values())


def test_an_empty_cloud_gets_no_lenses_and_no_reasons() -> None:
    assert built_in_lenses([]) == ((), {}, ())


def _links() -> pl.DataFrame:
    # Acme twice, NY once (ambiguous: two candidate rows for one mention),
    # Globex once; m5 and m6 unmatched; m7 linked but rejected by status.
    return pl.DataFrame({
        "mention_id": ["m1", "m2", "m3", "m3", "m4", "m5", "m6", "m7"],
        "canonical_id": [
            "org:acme", "org:acme", "gpe:ny", "gpe:nyc", "org:globex",
            None, None, "org:rejected",
        ],
        "document_id": ["d1", "d2", "d1", "d1", "d3", "d1", "d2", "d3"],
        "status": [
            "matched", "matched", "ambiguous", "ambiguous", "matched",
            "unmatched", "unmatched", "rejected",
        ],
        "match_score": [None] * 8,
    }, schema_overrides={"match_score": pl.Float64})


def test_coverage_counts_what_the_cap_and_the_resolver_left_out() -> None:
    export = build_concept_cloud(
        project="p", links=_links(), dag_plane=DagPlane(nodes=()), top_n=2,
    )

    assert len(export.concepts) == 2
    assert export.coverage is not None
    # Four canonical ids in an accepted status; the rejected row is not one.
    assert export.coverage.concepts_total == 4
    assert export.coverage.unmatched_mentions == 2


def test_an_export_carries_the_lenses_after_its_declared_dimensions() -> None:
    tone = pl.DataFrame({"canonical_id": ["org:acme"], "tone": ["positive"]})
    export = build_concept_cloud(
        project="p", links=_links(), dag_plane=DagPlane(nodes=()),
        dimension_columns={"tone": (tone, "tone")},
    )

    assert [d.name for d in export.dimensions] == [
        "tone", MENTIONS, DOCUMENTS, LINK_STATUS,
    ]
    assert [u.name for u in export.unavailable_dimensions] == [LINK_SCORE]
    acme = next(c for c in export.concepts if c.canonical_id == "org:acme")
    assert acme.dimensions == {
        "tone": "positive", MENTIONS: "2–9", DOCUMENTS: "2–9",
        LINK_STATUS: "matched",
    }


def test_a_declared_dimension_may_not_take_a_lens_name() -> None:
    frame = pl.DataFrame({"canonical_id": ["org:acme"], "v": ["x"]})

    with pytest.raises(ConceptCloudExportError, match="built-in lens"):
        build_concept_cloud(
            project="p", links=_links(), dag_plane=DagPlane(nodes=()),
            dimension_columns={MENTIONS: (frame, "v")},
        )
