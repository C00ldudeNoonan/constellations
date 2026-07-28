from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from dbt_ml.adapters import parse_warehouse_config
from dbt_ml.compiler import validate_project_contract
from dbt_ml.config import load_project
from dbt_ml.config.loader import ConfigError
from dbt_ml.text.transforms import nlp_document_features as features
from dbt_ml.transforms import TransformContext

_IDENTITY = {
    "nlp_provider": "spacy",
    "nlp_provider_version": "3.8.0",
    "nlp_model": "en_core_web_sm",
    "nlp_model_version": "3.8.0",
    "nlp_language": "en",
}


def _identity_columns(height: int) -> dict[str, list[str]]:
    return {key: [value] * height for key, value in _IDENTITY.items()}


# d1: three tokens across two sentences, one of them a stop word.
# d2: one token, no sentence boundaries (parser disabled).
_TOKENS = pl.DataFrame(
    {
        "document_id": ["d1", "d1", "d1", "d2"],
        "token_text": ["Fed", "held", "rates", "Apple"],
        "lemma": ["Fed", "hold", "rate", "Apple"],
        "pos": ["PROPN", "VERB", "NOUN", "PROPN"],
        "tag": ["NNP", "VBD", "NNS", "NNP"],
        "sentence_index": [0, 0, 1, None],
        "is_stop": [False, True, False, False],
        "is_alpha": [True, True, True, True],
        **_identity_columns(4),
    }
)
_ENTITIES = pl.DataFrame(
    {
        "document_id": ["d1", "d2"],
        "label": ["ORG", "ORG"],
        **_identity_columns(2),
    }
)
_LINKS = pl.DataFrame(
    {
        "document_id": ["d1", "d1", "d1", "d2"],
        "entity_namespace": ["cik", "ticker", "ticker", None],
        "canonical_id": ["0000320193", "MCY", "MERC", None],
        "status": ["matched", "ambiguous", "ambiguous", "unmatched"],
        "resolver": ["alias_table"] * 4,
        "resolver_version": ["1"] * 4,
        "alias_set_version": ["fingerprint"] * 4,
    }
)
_DOCUMENTS = pl.DataFrame(
    {
        "economic_id": ["d1", "d2", "d3"],
        "publisher": ["Federal Reserve", "SEC", "BEA"],
        "text": ["raw text", "raw text", ""],
    }
)


def _ctx(options: dict[str, object]) -> TransformContext:
    return TransformContext(
        project_dir=Path("."),
        profile_name="test",
        target_name="dev",
        warehouse=parse_warehouse_config(
            {"type": "duckdb", "path": "./test.duckdb", "schema": "main"}
        ),
        llm=None,
        options=options,
    )


def _run(
    options: dict[str, object],
    *,
    tokens: pl.DataFrame = _TOKENS,
    entities: pl.DataFrame | None = None,
    links: pl.DataFrame | None = None,
    documents: pl.DataFrame | None = None,
) -> pl.DataFrame:
    merged: dict[str, object] = {"tokens": "tok", **options}
    deps = {"tok": tokens}
    if entities is not None:
        merged.setdefault("entities", "ent")
        deps["ent"] = entities
    if links is not None:
        merged.setdefault("links", "lnk")
        deps["lnk"] = links
    if documents is not None:
        merged.setdefault("documents", "doc")
        merged.setdefault("documents_id_field", "economic_id")
        deps["doc"] = documents
    return features.run(deps, _ctx(merged))


def test_emits_one_stable_row_per_document() -> None:
    first = _run({}, entities=_ENTITIES)
    second = _run({}, entities=_ENTITIES)

    assert first["document_id"].to_list() == ["d1", "d2"]
    assert first.height == first["document_id"].n_unique()
    assert first.equals(second)


def test_base_features_match_hand_computed_values() -> None:
    output = _run({}, entities=_ENTITIES)
    d1 = output.filter(pl.col("document_id") == "d1").row(0, named=True)

    assert d1["token_count"] == 3
    assert d1["sentence_count"] == 2
    assert d1["entity_count"] == 1
    assert d1["unique_lemma_count"] == 3
    assert d1["lexical_diversity"] == pytest.approx(1.0)
    assert d1["stop_ratio"] == pytest.approx(1 / 3)
    assert d1["alpha_ratio"] == pytest.approx(1.0)


def test_repeated_lemmas_lower_lexical_diversity() -> None:
    repeated = pl.DataFrame(
        {
            "document_id": ["d1"] * 4,
            "lemma": ["rate", "rate", "rate", "hike"],
            "pos": ["NOUN"] * 4,
            "sentence_index": [0] * 4,
            "is_stop": [False] * 4,
            "is_alpha": [True] * 4,
            **_identity_columns(4),
        }
    )

    output = _run({}, tokens=repeated)

    assert output["unique_lemma_count"].to_list() == [2]
    assert output["lexical_diversity"].to_list() == [pytest.approx(0.5)]


def test_absent_sentence_boundaries_yield_null_not_zero() -> None:
    output = _run({})

    d2 = output.filter(pl.col("document_id") == "d2").row(0, named=True)
    assert d2["sentence_count"] is None
    assert d2["token_count"] == 1


def test_documents_dependency_keeps_empty_documents_with_zero_counts() -> None:
    output = _run({}, entities=_ENTITIES, documents=_DOCUMENTS)

    assert output["document_id"].to_list() == ["d1", "d2", "d3"]
    empty = output.filter(pl.col("document_id") == "d3").row(0, named=True)
    assert empty["token_count"] == 0
    assert empty["entity_count"] == 0
    assert empty["unique_lemma_count"] == 0
    # Ratios are undefined at a zero denominator, not zero.
    assert empty["lexical_diversity"] is None
    assert empty["stop_ratio"] is None
    assert empty["alpha_ratio"] is None
    assert empty["nlp_model"] is None


def test_without_documents_dependency_the_token_table_defines_the_universe() -> None:
    output = _run({})

    assert output["document_id"].to_list() == ["d1", "d2"]


def test_configured_pos_and_label_features_use_documented_column_names() -> None:
    output = _run(
        {
            "pos_counts": ["NOUN", "PROPN"],
            "pos_ratios": ["PROPN"],
            "entity_label_counts": ["ORG", "GPE"],
        },
        entities=_ENTITIES,
    )
    d1 = output.filter(pl.col("document_id") == "d1").row(0, named=True)

    assert d1["pos_noun_count"] == 1
    assert d1["pos_propn_count"] == 1
    assert d1["pos_propn_ratio"] == pytest.approx(1 / 3)
    assert d1["entity_org_count"] == 1
    # A configured label the document never uses is zero, not null.
    assert d1["entity_gpe_count"] == 0


def test_pos_ratio_does_not_require_requesting_the_count() -> None:
    output = _run({"pos_ratios": ["NOUN"]})

    assert "pos_noun_ratio" in output.columns
    assert "pos_noun_count" not in output.columns


def test_link_rollups_count_distinct_canonical_ids_and_statuses() -> None:
    output = _run(
        {
            "link_namespace_counts": ["cik", "ticker"],
            "link_status_counts": ["matched", "ambiguous", "unmatched"],
        },
        links=_LINKS,
    )
    d1 = output.filter(pl.col("document_id") == "d1").row(0, named=True)
    d2 = output.filter(pl.col("document_id") == "d2").row(0, named=True)

    assert d1["linked_cik_count"] == 1
    # One ambiguous mention resolving to two candidates counts both.
    assert d1["linked_ticker_count"] == 2
    assert d1["link_matched_count"] == 1
    assert d1["link_ambiguous_count"] == 2
    assert d1["link_unmatched_count"] == 0
    # An unmatched mention has no canonical ID, so it counts toward no namespace.
    assert d2["linked_cik_count"] == 0
    assert d2["link_unmatched_count"] == 1
    assert d1["link_resolver"] == "alias_table"
    assert d1["link_alias_set_version"] == "fingerprint"


def test_identity_passes_through_per_document() -> None:
    output = _run({})

    assert output["nlp_model"].to_list() == ["en_core_web_sm", "en_core_web_sm"]
    assert output["nlp_language"].to_list() == ["en", "en"]


def test_mixed_identity_within_a_document_is_rejected() -> None:
    mixed = _TOKENS.vstack(
        _TOKENS.head(1).with_columns(pl.lit("3.9.0").alias("nlp_model_version"))
    )

    with pytest.raises(ValueError, match="disagree on") as excinfo:
        _run({}, tokens=mixed)

    assert "d1" in str(excinfo.value)


def test_token_and_entity_identity_disagreement_is_rejected() -> None:
    other_model = _ENTITIES.with_columns(pl.lit("de_core_news_sm").alias("nlp_model"))

    with pytest.raises(ValueError, match="disagrees with the tokens model"):
        _run({}, entities=other_model)


def test_no_document_token_or_entity_text_reaches_the_output() -> None:
    output = _run(
        {"pos_counts": ["NOUN"], "entity_label_counts": ["ORG"]},
        entities=_ENTITIES,
        documents=_DOCUMENTS,
    )

    for leaked in ("token_text", "lemma", "entity_text", "text", "pos", "tag"):
        assert leaked not in output.columns


def test_include_fields_pass_through_allow_listed_parent_metadata() -> None:
    output = _run({"include_fields": ["publisher"]}, documents=_DOCUMENTS)

    assert output["publisher"].to_list() == ["Federal Reserve", "SEC", "BEA"]
    assert "text" not in output.columns


def test_emit_selects_the_base_features() -> None:
    output = _run({"emit": ["token_count", "stop_ratio"]})

    assert output.columns[:3] == ["document_id", "token_count", "stop_ratio"]
    assert "lexical_diversity" not in output.columns
    assert "alpha_ratio" not in output.columns


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "Field required"),
        ({"tokens": "tok", "unexpected": 1}, "Extra inputs are not permitted"),
        ({"tokens": "tok", "entities": "tok"}, "must name different models"),
        ({"tokens": "tok", "emit": []}, "emit must not be empty"),
        ({"tokens": "tok", "emit": ["token_count", "token_count"]}, "must be unique"),
        ({"tokens": "tok", "emit": ["not_a_feature"]}, "Input should be"),
        ({"tokens": "tok", "emit": ["entity_count"]}, "needs an `entities:`"),
        ({"tokens": "tok", "entity_label_counts": ["ORG"]}, "needs an `entities:`"),
        ({"tokens": "tok", "link_status_counts": ["matched"]}, "need a `links:`"),
        ({"tokens": "tok", "include_fields": ["publisher"]}, "needs a `documents:`"),
        (
            {"tokens": "tok", "pos_counts": ["NOUN", "noun"]},
            "collides with",
        ),
        (
            {"tokens": "tok", "pos_counts": ["not a tag"]},
            "invalid output column name",
        ),
        (
            {
                "tokens": "tok",
                "documents": "doc",
                "include_fields": ["token_count"],
            },
            "collides with",
        ),
    ],
)
def test_options_are_strict(options: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        features.validate_options(options)


@pytest.mark.parametrize(
    ("options", "deps", "message"),
    [
        (
            {"tokens": "tok"},
            {"wrong": _TOKENS},
            "expect dependencies named",
        ),
        (
            {"tokens": "tok", "emit": ["token_count"], "pos_counts": ["NOUN"]},
            {"tok": _TOKENS.drop("pos")},
            "missing configured columns",
        ),
        (
            {"tokens": "tok"},
            {"tok": _TOKENS.drop("nlp_model")},
            "missing configured columns",
        ),
    ],
)
def test_runtime_contract_violations_fail_actionably(
    options: dict[str, object],
    deps: dict[str, pl.DataFrame],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        features.run(deps, _ctx(options))


@pytest.mark.parametrize(
    ("documents", "message"),
    [
        (
            _DOCUMENTS.with_columns(pl.lit(None).alias("economic_id")),
            "contains null values",
        ),
        (
            _DOCUMENTS.head(1).vstack(_DOCUMENTS.head(1)),
            "contains duplicate values",
        ),
    ],
)
def test_document_universe_requires_unique_nonnull_ids(
    documents: pl.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _run({}, documents=documents)


def test_declared_dependencies_reports_every_configured_model() -> None:
    assert features.declared_dependencies({"tokens": "t"}) == ("t",)
    assert features.declared_dependencies(
        {"tokens": "t", "entities": "e", "links": "l", "documents": "d"}
    ) == ("t", "e", "l", "d")


def _example_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "economic_nlp"


def test_economic_nlp_example_compiles_with_document_features() -> None:
    project_dir = _example_dir()
    project, sources, models = load_project(project_dir)

    dag = validate_project_contract(project, sources, models, project_dir)

    order = dag.execution_order()
    assert order.index("document_features") > order.index("document_tokens")
    assert order.index("document_features") > order.index("entity_links")


def test_misspelled_dependency_option_fails_at_compile_time(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    shutil.copytree(
        _example_dir(), project_dir, ignore=shutil.ignore_patterns("target")
    )
    model_path = project_dir / "models" / "document_features.yml"
    model_path.write_text(
        model_path.read_text().replace("tokens: document_tokens", "tokens: document_token")
    )
    project, sources, models = load_project(project_dir)

    with pytest.raises(ConfigError, match="referenced by options but not in depends_on"):
        validate_project_contract(project, sources, models, project_dir)
