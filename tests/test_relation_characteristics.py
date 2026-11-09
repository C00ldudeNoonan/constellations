"""Relation characteristics: `inverse_of`, `symmetric`, `transitive` (issue #775).

The relation-checks reference names the failure: nothing stopped one model
asserting `owns` and another `owned_by` for the same fact, so the fact had two
row types. A declared inverse makes them one canonical row type and records
the type that was asserted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel.adapters import parse_warehouse_config
from stel.config.project import ProjectConfig
from stel.config.relation_characteristics import (
    canonical_orientations,
    has_characteristics,
)
from stel.config.vocabulary import RelationTypeDef
from stel.text.transforms import extract_relations
from stel.transforms import TransformContext


def _rel(name: str, subject: str, obj: str, **extra: Any) -> RelationTypeDef:
    return RelationTypeDef(name=name, subject_class=subject, object_class=obj, **extra)


def _project(*relations: RelationTypeDef) -> ProjectConfig:
    return ProjectConfig(name="p", classes=("org", "asset"), relations=relations)


# --- declaration checks ------------------------------------------------------


def test_an_inverse_declared_on_one_side_is_inferred_for_the_other() -> None:
    relations = (
        _rel("owns", "org", "asset"),
        _rel("owned_by", "asset", "org", inverse_of="owns"),
    )
    _project(*relations)

    orientations = canonical_orientations(relations)
    # The first-declared name is canonical, whichever side wrote `inverse_of`.
    assert orientations["owns"].canonical == "owns" and not orientations["owns"].swap
    assert orientations["owned_by"].canonical == "owns" and orientations["owned_by"].swap


def test_canonical_name_follows_declaration_order_not_alphabet() -> None:
    relations = (
        _rel("owned_by", "asset", "org"),
        _rel("owns", "org", "asset", inverse_of="owned_by"),
    )
    assert canonical_orientations(relations)["owns"].canonical == "owned_by"


def test_an_inverse_pair_whose_pairings_do_not_mirror_fails_compile() -> None:
    """The done-when: `owns` (org -> asset) against `owned_by` (org -> asset)."""
    with pytest.raises(ValueError, match=r"mirror"):
        _project(
            _rel("owns", "org", "asset"),
            _rel("owned_by", "org", "asset", inverse_of="owns"),
        )


def test_every_pairing_of_a_polymorphic_inverse_needs_a_mirror() -> None:
    with pytest.raises(ValueError, match=r"org -> org.*mirror|mirror.*org -> org"):
        ProjectConfig(
            name="p",
            classes=("org", "asset"),
            relations=(
                _rel("owns", "org", "asset"),
                _rel("owns", "org", "org"),
                _rel("owned_by", "asset", "org", inverse_of="owns"),
            ),
        )


@pytest.mark.parametrize(
    ("relations", "message"),
    [
        ([_rel("owns", "org", "asset", inverse_of="missing")], "not declared under"),
        ([_rel("owns", "org", "asset", inverse_of="owns")], "itself"),
        (
            [
                _rel("owns", "org", "asset", inverse_of="owned_by"),
                _rel("owned_by", "asset", "org", inverse_of="holds"),
                _rel("holds", "org", "asset"),
            ],
            "at most one inverse",
        ),
        (
            [
                _rel("owns", "org", "asset", inverse_of="owned_by"),
                _rel("owned_by", "asset", "org", symmetric=True),
            ],
            "both `symmetric` and an inverse",
        ),
        (
            [
                _rel("owns", "org", "asset", inverse_of="owned_by", transitive=True),
                _rel("owned_by", "asset", "org"),
            ],
            "disagree on `transitive`",
        ),
        (
            [
                _rel("owns", "org", "asset", inverse_of="owned_by", functional=True),
                _rel("owned_by", "asset", "org"),
            ],
            "functional",
        ),
        (
            [_rel("owns", "org", "asset", symmetric=True)],
            "mirror",
        ),
        (
            [
                _rel("owns", "org", "asset", symmetric=True),
                _rel("owns", "asset", "org"),
            ],
            "`symmetric` on one pairing and not",
        ),
    ],
)
def test_inconsistent_characteristics_fail_compile(
    relations: list[RelationTypeDef], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _project(*relations)


def test_a_symmetric_relation_over_one_class_is_valid() -> None:
    ProjectConfig(
        name="p",
        classes=("org",),
        relations=(_rel("partners_with", "org", "org", symmetric=True, transitive=True),),
    )


def test_a_project_without_characteristics_dumps_as_before() -> None:
    """Absent characteristics stay out of the dump, so existing declarations
    keep their serialized shape."""
    dumped = _rel("owns", "org", "asset").model_dump()
    assert "inverse_of" not in dumped
    assert "symmetric" not in dumped
    assert "transitive" not in dumped
    assert _rel("owns", "org", "asset", transitive=True).model_dump()["transitive"] is True


# --- row orientation ---------------------------------------------------------

_MENTIONS = pl.DataFrame(
    {
        "entity_id": ["e1", "e2", "e3", "e4"],
        "document_id": ["d1", "d1", "d2", "d2"],
        "sentence_index": [0, 0, 0, 0],
        "start": [0, 10, 0, 10],
        "end": [5, 15, 5, 15],
        "label": ["ORG", "ASSET", "ORG", "ORG"],
        "entity_text": ["Acme", "Plant", "Beta", "Gamma"],
    }
)


def _ctx(
    rules: list[dict[str, str]], relations: tuple[RelationTypeDef, ...]
) -> TransformContext:
    return TransformContext(
        project_dir=Path("."),
        profile_name="test",
        target_name="dev",
        warehouse=parse_warehouse_config(
            {"type": "duckdb", "path": "./test.duckdb", "schema": "main"}
        ),
        llm=None,
        options={
            "mentions": "mentions",
            "extractor": "rule",
            "label_field": "label",
            "rules": rules,
        },
        relations=relations,
    )


_OWNS_RULES = [
    {"subject_label": "ORG", "object_label": "ASSET", "relation_type": "owns"},
    {"subject_label": "ASSET", "object_label": "ORG", "relation_type": "owned_by"},
]
_INVERSES = (
    _rel("owns", "org", "asset"),
    _rel("owned_by", "asset", "org", inverse_of="owns"),
)


def test_owns_and_owned_by_for_one_fact_are_one_canonical_row() -> None:
    """Both rules fire on the same Acme/Plant pair; the fact is stored once, as
    `owns` from Acme to Plant, and the original type is recorded."""
    out = extract_relations.run({"mentions": _MENTIONS}, _ctx(_OWNS_RULES, _INVERSES))
    fact = out.filter(pl.col("document_id") == "d1")

    assert fact.height == 1
    row = fact.row(0, named=True)
    assert (row["relation_type"], row["subject_mention_id"], row["object_mention_id"]) == (
        "owns",
        "e1",
        "e2",
    )
    assert row["asserted_type"] in {"owns", "owned_by"}


def test_an_inverse_assertion_alone_is_swapped_and_keeps_its_asserted_type() -> None:
    rules = [_OWNS_RULES[1]]
    out = extract_relations.run({"mentions": _MENTIONS}, _ctx(rules, _INVERSES))
    row = out.filter(pl.col("document_id") == "d1").row(0, named=True)

    assert row["relation_type"] == "owns"
    assert row["asserted_type"] == "owned_by"
    # Subject and object swapped: Acme (e1) owns Plant (e2).
    assert (row["subject_mention_id"], row["object_mention_id"]) == ("e1", "e2")
    assert (row["subject_label"], row["object_label"]) == ("ORG", "ASSET")


def test_the_canonical_relation_id_matches_whichever_side_was_asserted() -> None:
    forward = extract_relations.run(
        {"mentions": _MENTIONS}, _ctx([_OWNS_RULES[0]], _INVERSES)
    ).filter(pl.col("document_id") == "d1")
    inverse = extract_relations.run(
        {"mentions": _MENTIONS}, _ctx([_OWNS_RULES[1]], _INVERSES)
    ).filter(pl.col("document_id") == "d1")

    assert forward["relation_id"].to_list() == inverse["relation_id"].to_list()


def test_a_symmetric_relation_is_stored_in_document_order() -> None:
    relations = (_rel("partners_with", "org", "org", symmetric=True),)
    rules = [{"subject_label": "ORG", "object_label": "ORG", "relation_type": "partners_with"}]
    out = extract_relations.run(
        {"mentions": _MENTIONS.filter(pl.col("document_id") == "d2")},
        _ctx(rules, relations),
    )

    # Beta (e3, start 0) and Gamma (e4, start 10): earlier mention is the subject,
    # and the rule fires both ways, so there is one row, not two.
    assert out.height == 1
    row = out.row(0, named=True)
    assert (row["subject_mention_id"], row["object_mention_id"]) == ("e3", "e4")


def test_a_project_without_characteristics_gains_no_column() -> None:
    plain = (_rel("owns", "org", "asset"), _rel("owned_by", "asset", "org"))
    assert not has_characteristics(plain)
    out = extract_relations.run({"mentions": _MENTIONS}, _ctx(_OWNS_RULES, plain))

    # No re-orientation and no new column: an incremental table would otherwise
    # refuse the run as a schema change on upgrade.
    assert "asserted_type" not in out.columns
    d1 = out.filter(pl.col("document_id") == "d1")
    assert sorted(d1["relation_type"].to_list()) == ["owned_by", "owns"]
