"""Concept-cloud nodes carry their declared class, definition and broader term (issue #629).

A concept in the cloud is a canonical entity with a label and a count, and
nothing said what it *is*. A project's vocabulary states that for the concepts
it names. These tests pin that the bundle carries only what a declaration
states, omits the fields entirely where nothing declares them, and never takes
a definition from document text.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from stel.concept_cloud import (
    Concept,
    ConceptCloudExport,
    DagPlane,
    build_concept_cloud,
    parse_concept_cloud_export,
)
from stel.concept_cloud.schema import CONCEPT_CLOUD_SCHEMA_VERSION
from stel.config.project import ProjectConfig
from stel.config.vocabulary import Vocabulary, VocabularyTerm

_LINKING = "link_entities"


def _links() -> pl.DataFrame:
    # m1/m2 link to a declared term; m3 links to a namespace nothing declares.
    return pl.DataFrame(
        {
            "mention_id": ["m1", "m2", "m3"],
            "canonical_id": ["Federal Reserve", "Federal Reserve", "Acme"],
            "document_id": ["d1", "d2", "d1"],
            "status": ["matched", "matched", "matched"],
            "match_score": [None, None, None],
            "entity_namespace": ["central_banks", "central_banks", "cik"],
            "mention_text": ["the Fed", "Federal Reserve", "Acme Corp"],
        }
    )


def _central_banks() -> Vocabulary:
    return Vocabulary(
        terms=[
            VocabularyTerm(
                label="Federal Reserve",
                description="The United States central bank",
                broader="Central bank system",
                entity_class="institution",
            ),
            VocabularyTerm(label="Central bank system"),
        ]
    )


def _build(vocabularies: dict[str, Vocabulary] | None) -> ConceptCloudExport:
    return build_concept_cloud(
        project="p",
        links=_links(),
        dag_plane=DagPlane(nodes=()),
        linking_model=_LINKING,
        vocabularies=vocabularies,
    )


def _concept(export: ConceptCloudExport, canonical_id: str) -> Concept:
    return next(c for c in export.concepts if c.canonical_id == canonical_id)


def test_a_declared_concept_carries_its_class_definition_and_broader_term() -> None:
    fed = _concept(_build({"central_banks": _central_banks()}), "Federal Reserve")

    assert fed.entity_class == "institution"
    assert fed.definition == "The United States central bank"
    assert fed.broader == "Central bank system"


def test_a_definition_is_the_declared_text_never_document_text() -> None:
    # The mention text in this fixture is "the Fed", "Federal Reserve" and
    # "Acme Corp"; the definition must be the declaration's, not any of those.
    definition = _concept(
        _build({"central_banks": _central_banks()}), "Federal Reserve"
    ).definition

    assert definition == "The United States central bank"
    assert definition not in {"the Fed", "Federal Reserve", "Acme Corp"}


def test_an_undeclared_concept_has_none_of_the_fields() -> None:
    acme = _concept(_build({"central_banks": _central_banks()}), "Acme")

    assert acme.entity_class is None
    assert acme.definition is None
    assert acme.broader is None


def test_an_undeclared_concept_omits_the_keys_from_the_bundle_json() -> None:
    # Absent, not null: null would read as a declared empty value.
    bundle = json.loads(_build({"central_banks": _central_banks()}).to_json())
    by_id = {c["canonical_id"]: c for c in bundle["concepts"]}

    assert "class" not in by_id["Acme"]
    assert "definition" not in by_id["Acme"]
    assert "broader" not in by_id["Acme"]
    assert by_id["Federal Reserve"]["class"] == "institution"


def test_no_vocabularies_means_no_declared_fields_anywhere() -> None:
    export = _build(None)

    assert all(
        c.entity_class is None and c.definition is None and c.broader is None
        for c in export.concepts
    )


def test_the_same_label_declared_differently_in_two_vocabularies_is_left_undeclared() -> None:
    # Asserting either definition would guess which one the operator meant.
    other = Vocabulary(
        terms=[
            VocabularyTerm(
                label="Federal Reserve",
                description="A different definition",
                entity_class="agency",
            )
        ]
    )
    # The concept's rows must name both namespaces for the two declarations to
    # compete at all; a vocabulary none of its rows link through is not in play.
    links = pl.DataFrame(
        {
            "mention_id": ["m1", "m2"],
            "canonical_id": ["Federal Reserve", "Federal Reserve"],
            "document_id": ["d1", "d2"],
            "status": ["matched", "matched"],
            "match_score": [None, None],
            "entity_namespace": ["central_banks", "agencies"],
            "mention_text": ["the Fed", "Federal Reserve"],
        }
    )
    export = build_concept_cloud(
        project="p",
        links=links,
        dag_plane=DagPlane(nodes=()),
        linking_model=_LINKING,
        vocabularies={"central_banks": _central_banks(), "agencies": other},
    )
    fed = _concept(export, "Federal Reserve")

    assert fed.entity_class is None
    assert fed.definition is None


def test_bundle_round_trips_through_the_versioned_contract() -> None:
    bundle = json.loads(_build({"central_banks": _central_banks()}).to_json())

    parsed = parse_concept_cloud_export(bundle)

    assert parsed.schema_version == CONCEPT_CLOUD_SCHEMA_VERSION == "4"
    assert _concept(parsed, "Federal Reserve").entity_class == "institution"


def test_a_version_3_bundle_is_rejected_rather_than_read_as_current() -> None:
    bundle = json.loads(_build(None).to_json())
    bundle["schema_version"] = "3"

    with pytest.raises(ValueError, match="unsupported concept-cloud schema_version"):
        parse_concept_cloud_export(bundle)


def test_a_term_class_must_be_declared_under_classes() -> None:
    with pytest.raises(ValueError, match="not declared under `classes:`"):
        ProjectConfig(
            name="p",
            classes=("organization",),
            vocabularies={"central_banks": _central_banks()},
        )


def test_the_documented_class_example_parses_as_a_project() -> None:
    # docs/reference.md, "Shared vocabularies": the `class:` example must load.
    import yaml

    documented = """
name: p
classes: [institution]
vocabularies:
  central_banks:
    terms:
      - label: Federal Reserve
        description: The United States central bank
        class: institution
"""
    project = ProjectConfig.model_validate(yaml.safe_load(documented))

    assert project.vocabularies["central_banks"].terms[0].entity_class == "institution"


def test_a_term_class_declared_under_classes_is_accepted() -> None:
    project = ProjectConfig(
        name="p",
        classes=("institution",),
        vocabularies={"central_banks": _central_banks()},
    )

    assert project.vocabularies["central_banks"].terms[0].entity_class == "institution"
