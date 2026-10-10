"""Who names a concept: document counts by a document-level field (issue #555
item 5). "FERC: named by 300 filings, 71% Utilities, 18% Energy." """
from __future__ import annotations

import re

import polars as pl
import pytest

from stel.concept_cloud import (
    Concept,
    ConceptCloudExport,
    ConceptCloudExportError,
    DagPlane,
    Provenance,
    build_concept_cloud,
    demo_export,
    render_concept_cloud,
)
from stel.concept_cloud.schema import BreakdownDef


def _links() -> pl.DataFrame:
    # Acme is named twice in d1, once in d2, once in d3 (which has no sector).
    # New York is named in d1, and in d4 only through an ambiguous match.
    return pl.DataFrame(
        {
            "mention_id": ["m1", "m2", "m3", "m4", "m5", "m6"],
            "canonical_id": [
                "org:acme", "org:acme", "org:acme", "org:acme", "gpe:ny", "gpe:ny",
            ],
            "document_id": ["d1", "d1", "d2", "d3", "d1", "d4"],
            "status": [
                "matched", "matched", "matched", "matched", "matched", "ambiguous",
            ],
        }
    )


def _sectors() -> pl.DataFrame:
    # d9 is a document nothing links to; its value must not be declared.
    return pl.DataFrame(
        {
            "document_id": ["d1", "d2", "d4", "d9"],
            "sector": ["Utilities", "Energy", "Energy", "Technology"],
        }
    )


def _build(
    links: pl.DataFrame,
    sectors: pl.DataFrame,
    *,
    statuses: tuple[str, ...] = ("matched", "ambiguous"),
) -> ConceptCloudExport:
    return build_concept_cloud(
        project="p",
        links=links,
        dag_plane=DagPlane(nodes=()),
        generated_at="2026-10-09T00:00:00Z",
        statuses=statuses,
        breakdown_columns={"sector": (sectors, "sector")},
    )


def _concept(export: ConceptCloudExport, canonical_id: str) -> Concept:
    return next(c for c in export.concepts if c.canonical_id == canonical_id)


def test_a_concept_counts_its_documents_by_the_field() -> None:
    """Documents, not mentions: Acme named twice in d1 is one Utilities filing.
    d3 has no sector, so it is in `documents` and in no count -- the card's
    "no sector" share, rather than a value invented for it."""
    export = _build(_links(), _sectors())

    acme = _concept(export, "org:acme")
    assert acme.provenance.documents == 3
    assert acme.breakdowns == {"sector": {"Energy": 1, "Utilities": 1}}
    assert _concept(export, "gpe:ny").breakdowns == {
        "sector": {"Energy": 1, "Utilities": 1}
    }
    # Declared from what the concepts carry, not from every row of the model.
    assert export.breakdowns == (
        BreakdownDef(name="sector", values=("Energy", "Utilities")),
    )


def test_a_breakdown_counts_only_the_documents_the_concept_is_built_from() -> None:
    """New York's d4 link is ambiguous. With ambiguous links excluded, d4 is
    not one of its documents, so it cannot be one of its Energy filings either
    -- counting it would put a share over 100% on the card."""
    export = _build(_links(), _sectors(), statuses=("matched",))

    ny = _concept(export, "gpe:ny")
    assert ny.provenance.documents == 1
    assert ny.breakdowns == {"sector": {"Utilities": 1}}


def test_a_document_given_two_values_is_refused() -> None:
    """A warehouse read has no row order, so picking one would let a card's
    shares change between two exports of unchanged data."""
    sectors = pl.concat(
        [_sectors(), pl.DataFrame({"document_id": ["d1"], "sector": ["Energy"]})]
    )

    with pytest.raises(
        ConceptCloudExportError,
        match=r"breakdown 'sector' gives more than one `sector` to 1 document\(s\): d1",
    ):
        _build(_links(), sectors)


def test_a_repeated_row_with_the_same_value_is_not_a_conflict() -> None:
    """A SQL join that fans a document out to the same sector twice states one
    value, and refusing it would send the operator hunting for a conflict
    that is not there."""
    sectors = pl.concat([_sectors(), _sectors()])

    assert _concept(_build(_links(), sectors), "org:acme").breakdowns == {
        "sector": {"Energy": 1, "Utilities": 1}
    }


def test_an_identifier_field_is_refused_with_the_remedy() -> None:
    documents = pl.DataFrame(
        {
            "document_id": [f"d{i}" for i in range(51)],
            "sector": [f"TICKER{i}" for i in range(51)],
        }
    )

    with pytest.raises(
        ConceptCloudExportError,
        match=re.escape(
            "has 51 distinct `sector` values; a breakdown is a categorical "
            "document field and is capped at 50. Group the values in the model."
        ),
    ):
        _build(_links(), documents)


def test_a_breakdown_model_missing_a_column_is_refused_by_name() -> None:
    with pytest.raises(
        ConceptCloudExportError,
        match="breakdown 'sector' needs `document_id` and `sector` columns",
    ):
        _build(_links(), _sectors().rename({"document_id": "filing_id"}))


def test_the_bundle_does_not_depend_on_row_order() -> None:
    """The counts are written from dicts built off a group_by, which promises
    no order; two exports of unchanged data must be byte-identical."""
    forward = _build(_links(), _sectors()).to_json()
    backward = _build(_links().reverse(), _sectors().reverse()).to_json()

    assert forward == backward


def _bundle(counts: dict[str, dict[str, int]], *, documents: int) -> dict[str, object]:
    return {
        "generated_at": "2026-10-09T00:00:00Z",
        "project": "p",
        "dag_plane": DagPlane(nodes=()),
        "breakdowns": (BreakdownDef(name="sector", values=("Energy", "Utilities")),),
        "concepts": (
            Concept(
                canonical_id="org:acme",
                display="Acme",
                frequency=5,
                provenance=Provenance(model="m", documents=documents),
                breakdowns=counts,
            ),
        ),
    }


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"industry": {"Energy": 1}}, "uses undeclared breakdown 'industry'"),
        ({"sector": {"Retail": 1}}, r"values \['Retail'\] outside its declared set"),
        ({"sector": {"Energy": 0}}, "counts must be at least 1"),
        (
            {"sector": {"Energy": 2, "Utilities": 2}},
            "counts 4 documents, more than the 3 it is named in",
        ),
    ],
)
def test_a_bundle_cannot_carry_counts_the_viewer_would_misdraw(
    counts: dict[str, dict[str, int]], message: str
) -> None:
    """Each one renders wrong rather than failing: a value with no legend
    color, a zero-width segment, or shares adding up past 100%."""
    with pytest.raises(ValueError, match=message):
        ConceptCloudExport.model_validate(_bundle(counts, documents=3))


def test_the_detail_card_says_who_names_a_concept() -> None:
    html = render_concept_cloud(demo_export())

    # The node carries what the card reads, from the bundle.
    assert "documents: c.provenance?.documents || 0," in html
    assert "breakdowns: c.breakdowns || {}," in html
    # Reached from the card, not defined and orphaned.
    assert "breakdowns(node) +" in html
    # A document with no value is said in words, not folded into a value.
    assert "no ${esc(b.name)} ${share(total - known, total)}" in html
    # The slider changes the stars, not the counts, so the card says so.
    assert 'const scope = period !== null ? " (all periods)" : "";' in html


def test_the_demo_shows_a_breakdown() -> None:
    export = demo_export()

    assert [b.name for b in export.breakdowns] == ["section"]
    fed = next(c for c in export.concepts if c.display == "Federal Reserve")
    assert sum(fed.breakdowns["section"].values()) < fed.provenance.documents
