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
from dbt_ml.text import (
    ALIAS_RESOLVER_VERSION,
    VECTOR_SIMILARITY_RESOLVER_VERSION,
    alias_set_fingerprint,
    normalize_alias_text,
)
from dbt_ml.text.transforms import link_entities
from dbt_ml.transforms import TransformContext

_MENTIONS = pl.DataFrame(
    {
        "entity_id": ["m-apple", "m-fed", "m-mercury", "m-acme"],
        "document_id": ["sec-10k", "fomc-min", "sec-10k", "fomc-min"],
        "entity_text": ["Apple Inc.", "THE  fed", "Mercury", "Acme Widgets"],
        "label": ["ORG", "ORG", "ORG", "ORG"],
        "start": [0, 10, 20, 30],
        "end": [10, 17, 27, 42],
        "sensitive_notes": ["a", "b", "c", "d"],
    }
)
_ALIASES = pl.DataFrame(
    {
        "alias": ["Apple Inc.", "Apple Inc.", "The Fed", "Mercury", "Mercury"],
        "entity_namespace": ["cik", "ticker", "agency", "ticker", "ticker"],
        "canonical_id": ["0000320193", "AAPL", "FRB", "MCY", "MERC"],
    }
)


def _ctx(options: dict[str, object] | None = None) -> TransformContext:
    merged: dict[str, object] = {"mentions": "mentions", "aliases": "aliases"}
    merged.update(options or {})
    return TransformContext(
        project_dir=Path("."),
        profile_name="test",
        target_name="dev",
        warehouse=parse_warehouse_config(
            {"type": "duckdb", "path": "./test.duckdb", "schema": "main"}
        ),
        llm=None,
        options=merged,
    )


def _deps(
    mentions: pl.DataFrame = _MENTIONS,
    aliases: pl.DataFrame = _ALIASES,
) -> dict[str, pl.DataFrame]:
    return {"mentions": mentions, "aliases": aliases}


def test_exact_matches_span_namespaces_with_stable_ids() -> None:
    first = link_entities.run(_deps(), _ctx())
    second = link_entities.run(_deps(), _ctx())

    apple = first.filter(pl.col("mention_id") == "m-apple").sort("entity_namespace")
    assert apple["entity_namespace"].to_list() == ["cik", "ticker"]
    assert apple["canonical_id"].to_list() == ["0000320193", "AAPL"]
    assert apple["status"].to_list() == ["matched", "matched"]
    assert apple["match_method"].to_list() == ["exact", "exact"]
    assert apple["match_score"].to_list() == [None, None]
    assert apple["label"].to_list() == ["ORG", "ORG"]
    assert apple["start"].to_list() == [0, 0]
    assert apple["resolver"].to_list() == ["alias_table", "alias_table"]
    assert apple["resolver_version"].to_list() == [ALIAS_RESOLVER_VERSION] * 2
    assert first["entity_link_id"].to_list() == second["entity_link_id"].to_list()
    assert first["entity_link_id"].n_unique() == first.height
    assert "entity_text" not in first.columns
    assert "sensitive_notes" not in first.columns
    assert "mention_text" not in first.columns


def test_normalized_method_matches_when_exact_does_not() -> None:
    output = link_entities.run(_deps(), _ctx())

    fed = output.filter(pl.col("mention_id") == "m-fed")
    assert fed["status"].to_list() == ["matched"]
    assert fed["match_method"].to_list() == ["normalized"]
    assert fed["canonical_id"].to_list() == ["FRB"]


def test_exact_wins_per_namespace_while_normalized_adds_others() -> None:
    mentions = pl.DataFrame(
        {
            "entity_id": ["m-fed"],
            "document_id": ["fomc-min"],
            "entity_text": ["Fed"],
            "label": ["ORG"],
            "start": [0],
            "end": [3],
        }
    )
    aliases = pl.DataFrame(
        {
            "alias": ["Fed", "fed", "fed"],
            "entity_namespace": ["agency", "agency", "ticker"],
            "canonical_id": ["FRB", "FRB-normalized-only", "FDX"],
        }
    )

    output = link_entities.run(
        _deps(mentions, aliases), _ctx()
    ).sort("entity_namespace")

    assert output["entity_namespace"].to_list() == ["agency", "ticker"]
    assert output["match_method"].to_list() == ["exact", "normalized"]
    assert output["canonical_id"].to_list() == ["FRB", "FDX"]
    assert output["status"].to_list() == ["matched", "matched"]


def test_ambiguous_candidates_are_preserved_not_guessed() -> None:
    output = link_entities.run(_deps(), _ctx())

    mercury = output.filter(pl.col("mention_id") == "m-mercury")
    assert mercury["status"].to_list() == ["ambiguous", "ambiguous"]
    assert mercury["canonical_id"].to_list() == ["MCY", "MERC"]
    assert mercury["entity_link_id"].n_unique() == 2


def test_on_ambiguity_error_fails_without_leaking_mention_text() -> None:
    with pytest.raises(ValueError, match="on_ambiguity is 'error'") as excinfo:
        link_entities.run(_deps(), _ctx({"on_ambiguity": "error"}))

    assert "m-mercury" in str(excinfo.value)
    assert "Mercury" not in str(excinfo.value)


def test_unmatched_and_null_text_mentions_stay_explicit() -> None:
    mentions = pl.DataFrame(
        {
            "entity_id": ["m-acme", "m-null"],
            "document_id": ["fomc-min", "fomc-min"],
            "entity_text": ["Acme Widgets", None],
            "label": ["ORG", "ORG"],
            "start": [0, 1],
            "end": [12, 2],
        }
    )

    output = link_entities.run(_deps(mentions), _ctx())

    assert output["status"].to_list() == ["unmatched", "unmatched"]
    assert output["entity_namespace"].to_list() == [None, None]
    assert output["canonical_id"].to_list() == [None, None]
    assert output["match_method"].to_list() == [None, None]
    assert output["label"].to_list() == ["ORG", "ORG"]


def test_mention_text_requires_explicit_opt_in_and_keeps_ids_stable() -> None:
    default = link_entities.run(_deps(), _ctx())
    opted_in = link_entities.run(
        _deps(), _ctx({"include_mention_text": True})
    )

    assert "mention_text" not in default.columns
    assert "mention_text" in opted_in.columns
    assert default["entity_link_id"].to_list() == opted_in["entity_link_id"].to_list()


def test_alias_edits_change_alias_set_version_but_not_link_ids() -> None:
    original = link_entities.run(_deps(), _ctx())
    extended = link_entities.run(
        _deps(
            aliases=pl.concat(
                [
                    _ALIASES,
                    pl.DataFrame(
                        {
                            "alias": ["Acme Widgets"],
                            "entity_namespace": ["ticker"],
                            "canonical_id": ["ACME"],
                        }
                    ),
                ]
            )
        ),
        _ctx(),
    )

    assert original["alias_set_version"].n_unique() == 1
    assert extended["alias_set_version"].n_unique() == 1
    assert (
        original["alias_set_version"].to_list()[0]
        != extended["alias_set_version"].to_list()[0]
    )
    apple_before = original.filter(pl.col("mention_id") == "m-apple")
    apple_after = extended.filter(pl.col("mention_id") == "m-apple")
    assert (
        apple_before["entity_link_id"].to_list()
        == apple_after["entity_link_id"].to_list()
    )


def test_empty_mentions_preserve_schema_without_reading_aliases() -> None:
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.String(),
            "document_id": pl.String(),
            "entity_text": pl.String(),
            "label": pl.String(),
            "start": pl.Int64(),
            "end": pl.Int64(),
            "release_year": pl.Int64(),
        }
    )
    broken_aliases = pl.DataFrame({"unrelated": ["x"]})

    output = link_entities.run(
        _deps(empty, broken_aliases),
        _ctx({"include_fields": ["release_year"]}),
    )

    assert output.is_empty()
    assert output.schema["entity_link_id"] == pl.String()
    assert output.schema["start"] == pl.Int64()
    assert output.schema["release_year"] == pl.Int64()


def test_empty_alias_table_yields_all_unmatched() -> None:
    empty_aliases = pl.DataFrame(
        schema={
            "alias": pl.String(),
            "entity_namespace": pl.String(),
            "canonical_id": pl.String(),
        }
    )

    output = link_entities.run(_deps(aliases=empty_aliases), _ctx())

    assert output["status"].unique().to_list() == ["unmatched"]
    assert output.height == _MENTIONS.height


def test_include_fields_pass_through_upstream_metadata() -> None:
    mentions = _MENTIONS.rename({"sensitive_notes": "release_year"})

    output = link_entities.run(
        _deps(mentions), _ctx({"include_fields": ["release_year"]})
    )

    assert output["release_year"].null_count() == 0


@pytest.mark.parametrize(
    ("deps", "options", "message"),
    [
        (
            {"wrong": _MENTIONS, "aliases": _ALIASES},
            {},
            "expects dependencies named",
        ),
        (
            _deps(_MENTIONS.drop("entity_text")),
            {},
            "include_text: true",
        ),
        (
            _deps(aliases=_ALIASES.drop("canonical_id")),
            {},
            "missing configured columns",
        ),
        (
            _deps(
                pl.concat([_MENTIONS, _MENTIONS.head(1)]),
            ),
            {},
            "duplicate value 'm-apple'",
        ),
        (
            _deps(
                _MENTIONS.head(1).with_columns(pl.lit("").alias("entity_id"))
            ),
            {},
            "null or empty values",
        ),
        (
            _deps(
                _MENTIONS.head(1).with_columns(pl.lit(None).alias("document_id"))
            ),
            {},
            "null or empty values",
        ),
        (
            _deps(
                _MENTIONS.head(1).with_columns(pl.lit(7).alias("entity_text"))
            ),
            {},
            "strings or nulls",
        ),
        (
            _deps(
                aliases=_ALIASES.head(1).with_columns(pl.lit(" ").alias("alias"))
            ),
            {},
            "non-empty strings",
        ),
        (
            _deps(),
            {"include_fields": ["missing"]},
            "unknown columns",
        ),
        (
            _deps(_MENTIONS.rename({"sensitive_notes": "status"})),
            {"include_fields": ["status"]},
            "collides with entity-link output columns",
        ),
    ],
)
def test_runtime_contract_violations_fail_actionably(
    deps: dict[str, pl.DataFrame],
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        link_entities.run(deps, _ctx(options))


_BASE_OPTIONS: dict[str, object] = {"mentions": "mentions", "aliases": "aliases"}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"mentions": "same", "aliases": "same"}, "two different upstream models"),
        ({"mentions": "mentions"}, "aliases"),
        ({**_BASE_OPTIONS, "unexpected": True}, "Extra inputs are not permitted"),
        ({**_BASE_OPTIONS, "match_methods": []}, "must not be empty"),
        ({**_BASE_OPTIONS, "match_methods": ["exact", "exact"]}, "must be unique"),
        ({**_BASE_OPTIONS, "include_fields": ["entity_text"]}, "must not repeat"),
        ({**_BASE_OPTIONS, "include_mention_text": "yes"}, "valid boolean"),
        ({**_BASE_OPTIONS, "label_field": ""}, "use null to disable"),
        ({**_BASE_OPTIONS, "resolver": "fuzzy"}, "alias_table"),
    ],
)
def test_options_are_strict(options: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        link_entities.validate_options(options)


def test_alias_fingerprint_ignores_duplicate_rows() -> None:
    single = [{"alias": "Fed", "entity_namespace": "agency", "canonical_id": "FRB"}]
    duplicated = single * 3
    reordered = [
        {"alias": "US", "entity_namespace": "iso3166", "canonical_id": "US"},
        *single,
    ]

    assert alias_set_fingerprint(single) == alias_set_fingerprint(duplicated)
    assert alias_set_fingerprint(reordered) == alias_set_fingerprint(
        list(reversed(reordered))
    )
    assert alias_set_fingerprint(single) != alias_set_fingerprint(reordered)


def test_duplicate_alias_rows_do_not_change_links_or_version() -> None:
    duplicated = pl.concat([_ALIASES, _ALIASES])

    original = link_entities.run(_deps(), _ctx())
    with_duplicates = link_entities.run(_deps(aliases=duplicated), _ctx())

    assert original.sort("entity_link_id").equals(
        with_duplicates.sort("entity_link_id")
    )


def test_declared_dependencies_reports_the_configured_models() -> None:
    assert link_entities.declared_dependencies(
        {"mentions": "a", "aliases": "b"}
    ) == ("a", "b")


def test_normalize_alias_text_is_nfkc_casefold_and_collapsed() -> None:
    assert normalize_alias_text("  The  FED ") == "the fed"
    assert normalize_alias_text("Ｆｅｄｅｒａｌ") == "federal"
    assert normalize_alias_text("Apple  Inc.") == "apple inc."


# --- vector-similarity resolver ---------------------------------------------

_VEC_MENTIONS = pl.DataFrame(
    {
        "entity_id": ["m-apple", "m-msft", "m-far"],
        "document_id": ["d1", "d1", "d1"],
        "entity_text": ["Apple", "Microsoft", "Something else"],
        "embedding": [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]],
        "label": ["ORG", "ORG", "ORG"],
        "start": [0, 0, 0],
        "end": [5, 9, 14],
        "sensitive_notes": ["a", "b", "c"],
    }
)
_VEC_ALIASES = pl.DataFrame(
    {
        "canonical_id": ["AAPL", "0000320193", "MSFT"],
        "entity_namespace": ["ticker", "cik", "ticker"],
        "embedding": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    }
)


def _vec_ctx(options: dict[str, object] | None = None) -> TransformContext:
    merged: dict[str, object] = {"resolver": "vector_similarity", "threshold": 0.9}
    merged.update(options or {})
    return _ctx(merged)


def _vec_deps(
    mentions: pl.DataFrame = _VEC_MENTIONS,
    aliases: pl.DataFrame = _VEC_ALIASES,
) -> dict[str, pl.DataFrame]:
    return _deps(mentions, aliases)


# Disables the default label/start/end passthrough for minimal mention frames
# that only carry identity + embedding columns.
_NO_SPANS: dict[str, object] = {
    "label_field": None,
    "start_field": None,
    "end_field": None,
}


def test_vector_similarity_matches_above_threshold_with_scores() -> None:
    first = link_entities.run(_vec_deps(), _vec_ctx())
    second = link_entities.run(_vec_deps(), _vec_ctx())

    apple = first.filter(pl.col("mention_id") == "m-apple").sort("entity_namespace")
    assert apple["entity_namespace"].to_list() == ["cik", "ticker"]
    assert apple["canonical_id"].to_list() == ["0000320193", "AAPL"]
    assert apple["status"].to_list() == ["matched", "matched"]
    assert apple["match_method"].to_list() == ["cosine", "cosine"]
    assert apple["match_score"].to_list() == [pytest.approx(1.0), pytest.approx(1.0)]
    assert apple["resolver"].to_list() == ["vector_similarity", "vector_similarity"]
    assert apple["resolver_version"].to_list() == [VECTOR_SIMILARITY_RESOLVER_VERSION] * 2

    msft = first.filter(pl.col("mention_id") == "m-msft")
    assert msft["canonical_id"].to_list() == ["MSFT"]
    assert msft["match_score"].to_list() == [pytest.approx(1.0)]

    # Mention text and unlisted source columns are withheld by default.
    assert "entity_text" not in first.columns
    assert "sensitive_notes" not in first.columns
    assert first["entity_link_id"].to_list() == second["entity_link_id"].to_list()
    assert first["entity_link_id"].n_unique() == first.height


def test_vector_similarity_below_threshold_is_unmatched() -> None:
    output = link_entities.run(_vec_deps(), _vec_ctx())

    far = output.filter(pl.col("mention_id") == "m-far")
    assert far["status"].to_list() == ["unmatched"]
    assert far["entity_namespace"].to_list() == [None]
    assert far["canonical_id"].to_list() == [None]
    assert far["match_method"].to_list() == [None]
    assert far["match_score"].to_list() == [None]


def test_vector_similarity_null_mention_vector_is_unmatched() -> None:
    mentions = pl.DataFrame(
        {
            "entity_id": ["m-null"],
            "document_id": ["d1"],
            "embedding": [None],
        },
        schema_overrides={"embedding": pl.List(pl.Float64)},
    )

    output = link_entities.run(_vec_deps(mentions), _vec_ctx(_NO_SPANS))

    assert output["status"].to_list() == ["unmatched"]


def test_vector_similarity_ties_are_ambiguous_not_guessed() -> None:
    aliases = pl.DataFrame(
        {
            "canonical_id": ["AAPL", "APPL"],
            "entity_namespace": ["ticker", "ticker"],
            "embedding": [[1.0, 0.0], [1.0, 0.0]],
        }
    )

    output = link_entities.run(
        _vec_deps(aliases=aliases), _vec_ctx()
    ).filter(pl.col("mention_id") == "m-apple")

    assert output["status"].to_list() == ["ambiguous", "ambiguous"]
    assert sorted(output["canonical_id"].to_list()) == ["AAPL", "APPL"]
    assert output["entity_link_id"].n_unique() == 2


def test_vector_similarity_ambiguity_margin_widens_ties() -> None:
    mentions = _VEC_MENTIONS.head(1)  # m-apple only
    aliases = pl.DataFrame(
        {
            "canonical_id": ["EXACT", "CLOSE"],
            "entity_namespace": ["ticker", "ticker"],
            # cosine 1.0 vs ~0.99987 — distinct without a margin, tied with one.
            "embedding": [[1.0, 0.0], [0.984, 0.178]],
        }
    )

    strict = link_entities.run(_vec_deps(mentions, aliases), _vec_ctx())
    assert strict["status"].to_list() == ["matched"]
    assert strict["canonical_id"].to_list() == ["EXACT"]

    lenient = link_entities.run(
        _vec_deps(mentions, aliases), _vec_ctx({"ambiguity_margin": 0.05})
    )
    assert lenient["status"].to_list() == ["ambiguous", "ambiguous"]


def test_vector_similarity_on_ambiguity_error_names_only_ids() -> None:
    aliases = pl.DataFrame(
        {
            "canonical_id": ["AAPL", "APPL"],
            "entity_namespace": ["ticker", "ticker"],
            "embedding": [[1.0, 0.0], [1.0, 0.0]],
        }
    )

    with pytest.raises(ValueError, match="on_ambiguity is 'error'") as excinfo:
        link_entities.run(
            _vec_deps(aliases=aliases), _vec_ctx({"on_ambiguity": "error"})
        )

    assert "m-apple" in str(excinfo.value)
    assert "Apple" not in str(excinfo.value)


@pytest.mark.parametrize(
    ("metric", "threshold", "expected"),
    [
        ("dot", 1.5, "AAPL"),
        ("euclidean", -0.5, "AAPL"),
    ],
)
def test_vector_similarity_metric_variants(
    metric: str, threshold: float, expected: str
) -> None:
    mentions = pl.DataFrame(
        {
            "entity_id": ["m-apple"],
            "document_id": ["d1"],
            "embedding": [[2.0, 0.0]] if metric == "dot" else [[1.0, 0.0]],
        }
    )
    aliases = pl.DataFrame(
        {
            "canonical_id": ["AAPL", "MSFT"],
            "entity_namespace": ["ticker", "ticker"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        }
    )

    output = link_entities.run(
        _vec_deps(mentions, aliases),
        _vec_ctx({**_NO_SPANS, "metric": metric, "threshold": threshold}),
    )

    assert output["status"].to_list() == ["matched"]
    assert output["canonical_id"].to_list() == [expected]
    assert output["match_method"].to_list() == [metric]


def test_vector_similarity_dimension_mismatch_fails() -> None:
    mentions = pl.DataFrame(
        {
            "entity_id": ["m-apple"],
            "document_id": ["d1"],
            "embedding": [[1.0, 0.0, 0.0]],
        }
    )

    with pytest.raises(ValueError, match="dimensionality"):
        link_entities.run(_vec_deps(mentions), _vec_ctx(_NO_SPANS))


def test_vector_similarity_rejects_non_numeric_vectors() -> None:
    aliases = pl.DataFrame(
        {
            "canonical_id": ["AAPL"],
            "entity_namespace": ["ticker"],
            "embedding": [["not", "a", "number"]],
        }
    )

    with pytest.raises(ValueError, match="must contain only numbers"):
        link_entities.run(_vec_deps(aliases=aliases), _vec_ctx())


def test_vector_similarity_reference_version_tracks_alias_vectors() -> None:
    original = link_entities.run(_vec_deps(), _vec_ctx())
    moved = _VEC_ALIASES.with_columns(
        pl.Series("embedding", [[0.9, 0.1], [1.0, 0.0], [0.0, 1.0]])
    )
    changed = link_entities.run(_vec_deps(aliases=moved), _vec_ctx())
    unchanged = link_entities.run(_vec_deps(), _vec_ctx())

    assert original["alias_set_version"].n_unique() == 1
    assert (
        original["alias_set_version"].to_list()[0]
        == unchanged["alias_set_version"].to_list()[0]
    )
    assert (
        original["alias_set_version"].to_list()[0]
        != changed["alias_set_version"].to_list()[0]
    )


def test_vector_similarity_passthrough_and_include_fields() -> None:
    mentions = _VEC_MENTIONS.rename({"sensitive_notes": "release_year"})

    output = link_entities.run(
        _vec_deps(mentions),
        _vec_ctx({"include_fields": ["release_year"], "include_mention_text": True}),
    )

    apple = output.filter(pl.col("mention_id") == "m-apple")
    assert apple["label"].to_list() == ["ORG", "ORG"]
    assert apple["mention_text"].to_list() == ["Apple", "Apple"]
    assert apple["release_year"].to_list() == ["a", "a"]


def test_vector_similarity_empty_alias_frame_yields_unmatched() -> None:
    empty_aliases = pl.DataFrame(
        schema={
            "canonical_id": pl.String(),
            "entity_namespace": pl.String(),
            "embedding": pl.List(pl.Float64),
        }
    )

    output = link_entities.run(_vec_deps(aliases=empty_aliases), _vec_ctx())

    assert output["status"].unique().to_list() == ["unmatched"]
    assert output.height == _VEC_MENTIONS.height


def test_vector_similarity_rejects_mismatched_embedding_spaces() -> None:
    mentions = _VEC_MENTIONS.with_columns(pl.lit("cfg-A").alias("embedding_config_hash"))
    aliases = _VEC_ALIASES.with_columns(pl.lit("cfg-B").alias("embedding_config_hash"))

    with pytest.raises(ValueError, match="share one embedding space"):
        link_entities.run(_vec_deps(mentions, aliases), _vec_ctx())


def test_vector_similarity_accepts_matching_embedding_spaces() -> None:
    mentions = _VEC_MENTIONS.with_columns(pl.lit("cfg-A").alias("embedding_config_hash"))
    aliases = _VEC_ALIASES.with_columns(pl.lit("cfg-A").alias("embedding_config_hash"))

    output = link_entities.run(_vec_deps(mentions, aliases), _vec_ctx())

    assert output.filter(pl.col("mention_id") == "m-msft")["canonical_id"].to_list() == [
        "MSFT"
    ]
    assert "embedding_config_hash" not in output.columns


def test_vector_similarity_embedding_space_check_can_be_disabled() -> None:
    mentions = _VEC_MENTIONS.with_columns(pl.lit("cfg-A").alias("embedding_config_hash"))
    aliases = _VEC_ALIASES.with_columns(pl.lit("cfg-B").alias("embedding_config_hash"))

    output = link_entities.run(
        _vec_deps(mentions, aliases),
        _vec_ctx({"embedding_config_hash_field": None}),
    )

    assert output.filter(pl.col("mention_id") == "m-msft")["status"].to_list() == [
        "matched"
    ]


def test_vector_similarity_does_not_require_mention_text_by_default() -> None:
    mentions = _VEC_MENTIONS.drop("entity_text")

    output = link_entities.run(_vec_deps(mentions), _vec_ctx())

    assert output.filter(pl.col("mention_id") == "m-msft")["canonical_id"].to_list() == [
        "MSFT"
    ]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"resolver": "vector_similarity"}, "threshold"),
        (
            {"resolver": "vector_similarity", "threshold": 0.5, "match_methods": ["exact"]},
            "Extra inputs are not permitted",
        ),
        (
            {"resolver": "vector_similarity", "threshold": 0.5, "ambiguity_margin": -0.1},
            "non-negative",
        ),
        (
            {"resolver": "vector_similarity", "threshold": float("inf")},
            "finite",
        ),
    ],
)
def test_vector_similarity_options_are_strict(
    options: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        link_entities.validate_options({**_BASE_OPTIONS, **options})


def _example_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "economic_entity_links"


def test_economic_entity_links_example_compiles() -> None:
    project_dir = _example_dir()
    project, sources, models = load_project(project_dir)

    dag = validate_project_contract(project, sources, models, project_dir)

    order = dag.execution_order()
    assert order.index("entity_links") > order.index("entity_mentions")
    assert order.index("entity_links") > order.index("entity_aliases")


def test_economic_entity_links_embeddings_example_compiles() -> None:
    project_dir = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "economic_entity_links_embeddings"
    )
    project, sources, models = load_project(project_dir)

    dag = validate_project_contract(project, sources, models, project_dir)

    order = dag.execution_order()
    assert order.index("mention_embeddings") > order.index("entity_mentions")
    assert order.index("alias_embeddings") > order.index("entity_aliases")
    assert order.index("entity_links") > order.index("mention_embeddings")
    assert order.index("entity_links") > order.index("alias_embeddings")


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        (
            "mentions: entity_mentions",
            "mentions: entity_mention",
            "referenced by options but not in depends_on",
        ),
        (
            "depends_on: [ref('entity_mentions'), ref('entity_aliases')]",
            "depends_on: [ref('entity_mentions'), ref('entity_aliases'), "
            "ref('unused_extra')]",
            "in depends_on but unused by options",
        ),
    ],
)
def test_dependency_mismatch_fails_before_any_warehouse_mutation(
    tmp_path: Path,
    original: str,
    replacement: str,
    message: str,
) -> None:
    """A misspelled or stale dependency reference used to survive `compile` and
    only fail mid-`build`, after upstream models had already been materialized."""
    project_dir = tmp_path / "project"
    shutil.copytree(
        _example_dir(), project_dir, ignore=shutil.ignore_patterns("target")
    )
    if "unused_extra" in replacement:
        aliases_yml = project_dir / "models" / "entity_aliases.yml"
        (project_dir / "models" / "unused_extra.yml").write_text(
            aliases_yml.read_text().replace("name: entity_aliases", "name: unused_extra")
        )
    model_path = project_dir / "models" / "entity_links.yml"
    model_path.write_text(model_path.read_text().replace(original, replacement))
    project, sources, models = load_project(project_dir)

    with pytest.raises(ConfigError, match=message):
        validate_project_contract(project, sources, models, project_dir)
