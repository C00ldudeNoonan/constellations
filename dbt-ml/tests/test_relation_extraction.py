from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from dbt_ml.adapters import parse_warehouse_config
from dbt_ml.budget import BudgetExceededError, BudgetLedger, LLMBudgetConfig
from dbt_ml.compiler import validate_project_contract
from dbt_ml.config import load_project
from dbt_ml.config.loader import ConfigError
from dbt_ml.config.profile import LLMConfig
from dbt_ml.text.relations import (
    CO_OCCURRENCE_EXTRACTOR_VERSION,
    RULE_EXTRACTOR_VERSION,
    Mention,
    ModelAssertionExtractor,
    ModelAssertionExtractorOptions,
    Relation,
    RelationAssertion,
    RelationExtractor,
    RelationInference,
)
from dbt_ml.text.transforms import extract_relations
from dbt_ml.transforms import IncrementalContract, TransformContext

# Three mentions share sentence 0 of doc "d1"; a fourth sits alone in sentence 1.
# Doc "d2" has a single mention. Shaped like `nlp_entities` output.
_MENTIONS = pl.DataFrame(
    {
        "entity_id": ["e1", "e2", "e3", "e4", "e5"],
        "document_id": ["d1", "d1", "d1", "d1", "d2"],
        "sentence_index": [0, 0, 0, 1, 0],
        "start": [0, 10, 20, 40, 0],
        "end": [5, 15, 25, 45, 5],
        "label": ["ORG", "GPE", "MONEY", "ORG", "ORG"],
        "entity_text": ["Apple", "France", "$1M", "Acme", "Beta"],
    }
)


def _ctx(options: dict[str, object] | None = None) -> TransformContext:
    merged: dict[str, object] = {"mentions": "mentions"}
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


def _deps(mentions: pl.DataFrame = _MENTIONS) -> dict[str, pl.DataFrame]:
    return {"mentions": mentions}


# --- co-occurrence: sentence scope -------------------------------------------


def test_sentence_scope_pairs_every_mention_in_a_sentence() -> None:
    first = extract_relations.run(_deps(), _ctx())
    second = extract_relations.run(_deps(), _ctx())

    # Sentence 0 of d1 has 3 mentions → C(3,2)=3 pairs; the lone mentions produce
    # none. The output is symmetric, so each unordered pair appears once.
    d1 = first.filter(pl.col("document_id") == "d1").sort(
        ["subject_mention_id", "object_mention_id"]
    )
    assert list(
        zip(d1["subject_mention_id"], d1["object_mention_id"], strict=True)
    ) == [("e1", "e2"), ("e1", "e3"), ("e2", "e3")]
    assert first.height == 3
    assert d1["relation_type"].unique().to_list() == ["co_occurs_with"]
    assert d1["method"].unique().to_list() == ["co_occurrence"]
    assert d1["status"].unique().to_list() == ["asserted"]
    assert d1["directed"].unique().to_list() == [False]
    assert d1["confidence"].to_list() == [None, None, None]
    assert d1["sentence_index"].unique().to_list() == [0]
    assert d1["extractor"].unique().to_list() == ["co_occurrence"]
    assert d1["extractor_version"].unique().to_list() == [CO_OCCURRENCE_EXTRACTOR_VERSION]
    # Evidence text is withheld by default.
    assert "subject_text" not in first.columns
    assert "object_text" not in first.columns
    # Stable and unique ids across identical runs.
    assert first["relation_id"].to_list() == second["relation_id"].to_list()
    assert first["relation_id"].n_unique() == first.height


def test_subject_object_orientation_and_labels_are_positional() -> None:
    output = extract_relations.run(_deps(), _ctx()).filter(
        (pl.col("subject_mention_id") == "e1") & (pl.col("object_mention_id") == "e2")
    )
    row = output.row(0, named=True)
    # Subject is the earlier-positioned mention (order key start, end, id).
    assert row["subject_start"] == 0
    assert row["object_start"] == 10
    assert row["subject_label"] == "ORG"
    assert row["object_label"] == "GPE"


def test_relation_id_is_stable_regardless_of_input_row_order() -> None:
    shuffled = _MENTIONS.reverse()
    ordered = extract_relations.run(_deps(), _ctx()).sort("relation_id")
    reordered = extract_relations.run(_deps(shuffled), _ctx()).sort("relation_id")
    assert ordered["relation_id"].to_list() == reordered["relation_id"].to_list()


def test_single_mention_document_yields_no_relations() -> None:
    solo = _MENTIONS.filter(pl.col("document_id") == "d2")
    assert extract_relations.run(_deps(solo), _ctx()).height == 0


def test_label_filter_restricts_participants() -> None:
    # Only ORG mentions participate: e1 (d1 s0), e4 (d1 s1), e5 (d2) — none share
    # a sentence, so no ORG pair exists.
    org_only = extract_relations.run(_deps(), _ctx({"labels": ["ORG"]}))
    assert org_only.height == 0

    # GPE + MONEY share sentence 0 of d1 → exactly one pair.
    two = extract_relations.run(_deps(), _ctx({"labels": ["GPE", "MONEY"]}))
    assert two.height == 1
    assert sorted(
        [two["subject_label"][0], two["object_label"][0]]
    ) == ["GPE", "MONEY"]


# --- co-occurrence: window scope ---------------------------------------------


def test_window_scope_pairs_within_character_gap() -> None:
    # Gaps: e1(0-5)→e2(10-15)=5, e2→e3=5, e1→e3=15, e3→e4=15. With gap 6 only the
    # two adjacent pairs qualify, and both share sentence 0.
    output = extract_relations.run(
        _deps(), _ctx({"scope": "window", "max_char_gap": 6})
    ).sort("subject_mention_id")
    assert list(
        zip(output["subject_mention_id"], output["object_mention_id"], strict=True)
    ) == [("e1", "e2"), ("e2", "e3")]
    assert output["sentence_index"].to_list() == [0, 0]


def test_window_scope_records_null_sentence_when_pair_crosses_sentences() -> None:
    mentions = pl.DataFrame(
        {
            "entity_id": ["a", "b"],
            "document_id": ["d", "d"],
            "sentence_index": [0, 1],
            "start": [0, 8],
            "end": [5, 12],
            "label": ["ORG", "ORG"],
            "entity_text": ["x", "y"],
        }
    )
    output = extract_relations.run(
        _deps(mentions), _ctx({"scope": "window", "max_char_gap": 100})
    )
    assert output.height == 1
    assert output["sentence_index"].to_list() == [None]


def test_sentence_scope_rejects_null_sentence_index() -> None:
    mentions = _MENTIONS.with_columns(
        pl.lit(None, dtype=pl.Int64).alias("sentence_index")
    )
    with pytest.raises(ValueError, match="non-null sentence_index"):
        extract_relations.run(_deps(mentions), _ctx())


# --- evidence text opt-in ----------------------------------------------------


def test_include_mention_text_adds_subject_and_object_text() -> None:
    output = extract_relations.run(_deps(), _ctx({"include_mention_text": True}))
    pair = output.filter(
        (pl.col("subject_mention_id") == "e1") & (pl.col("object_mention_id") == "e2")
    ).row(0, named=True)
    assert pair["subject_text"] == "Apple"
    assert pair["object_text"] == "France"


def test_include_mention_text_requires_the_text_column() -> None:
    without_text = _MENTIONS.drop("entity_text")
    with pytest.raises(ValueError, match="entity_text"):
        extract_relations.run(_deps(without_text), _ctx({"include_mention_text": True}))


# --- guards and edge cases ---------------------------------------------------


def test_duplicate_mention_id_is_rejected() -> None:
    dupes = _MENTIONS.with_columns(pl.lit("dup").alias("entity_id"))
    with pytest.raises(ValueError, match="duplicate value 'dup'"):
        extract_relations.run(_deps(dupes), _ctx())


def test_empty_mentions_preserve_schema() -> None:
    empty = _MENTIONS.clear()
    output = extract_relations.run(_deps(empty), _ctx())
    assert output.height == 0
    assert "relation_id" in output.columns
    assert "directed" in output.columns


def test_max_pairs_per_document_fails_closed() -> None:
    many = pl.DataFrame(
        {
            "entity_id": [f"e{i}" for i in range(6)],
            "document_id": ["d"] * 6,
            "sentence_index": [0] * 6,
            "start": list(range(0, 60, 10)),
            "end": list(range(5, 65, 10)),
            "label": ["ORG"] * 6,
            "entity_text": ["x"] * 6,
        }
    )
    # C(6,2) = 15 pairs > cap of 5.
    with pytest.raises(ValueError, match="max_pairs_per_document"):
        extract_relations.run(_deps(many), _ctx({"max_pairs_per_document": 5}))


def test_declared_dependencies_and_incremental_contract() -> None:
    assert extract_relations.declared_dependencies({"mentions": "m"}) == ("m",)
    contract = extract_relations.declared_incremental_contract(
        {"mentions": "m", "document_id_field": "doc"}
    )
    assert contract == IncrementalContract(
        parent_key="document_id",
        child_key="relation_id",
        parent_source="m",
        parent_source_key="doc",
    )
    contract.validate_against(["m"])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "mentions"),
        ({"mentions": "m", "unexpected": True}, "Extra inputs are not permitted"),
        ({"mentions": "m", "extractor": "nonsense"}, "co_occurrence"),
        ({"mentions": "m", "labels": ["ORG", "ORG"]}, "unique"),
        ({"mentions": "m", "labels": ["ORG"], "label_field": None}, "label_field"),
        ({"mentions": "m", "relation_type": "  "}, "must not be empty"),
        ({"mentions": "m", "scope": "window", "max_char_gap": -1}, "max_char_gap"),
        ({"mentions": "m", "max_pairs_per_document": 0}, "max_pairs_per_document"),
    ],
)
def test_options_are_strict(options: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        extract_relations.validate_options(options)


# --- rule extractor: directed typed relations --------------------------------


def _rule_ctx(rules: list[dict[str, str]], extra: dict[str, object] | None = None):
    options: dict[str, object] = {"extractor": "rule", "rules": rules}
    options.update(extra or {})
    return _ctx(options)


def test_rule_extractor_emits_directed_typed_relations() -> None:
    output = extract_relations.run(
        _deps(),
        _rule_ctx(
            [
                {"subject_label": "ORG", "object_label": "GPE",
                 "relation_type": "operates_in"},
                {"subject_label": "ORG", "object_label": "MONEY",
                 "relation_type": "reports"},
            ]
        ),
    ).sort("relation_type")

    assert list(
        zip(
            output["subject_mention_id"],
            output["object_mention_id"],
            output["relation_type"],
            strict=True,
        )
    ) == [("e1", "e2", "operates_in"), ("e1", "e3", "reports")]
    assert output["method"].unique().to_list() == ["rule"]
    assert output["directed"].unique().to_list() == [True]
    assert output["status"].unique().to_list() == ["asserted"]
    assert output["extractor"].unique().to_list() == ["rule"]
    assert output["extractor_version"].unique().to_list() == [RULE_EXTRACTOR_VERSION]
    assert output["relation_id"].n_unique() == output.height


def test_rule_orientation_follows_the_rule_not_text_position() -> None:
    # e1 (ORG) precedes e2 (GPE) in text, but the rule is GPE -> ORG, so the
    # emitted subject is the GPE mention even though it is positionally later.
    mentions = pl.DataFrame(
        {
            "entity_id": ["e1", "e2"],
            "document_id": ["d", "d"],
            "sentence_index": [0, 0],
            "start": [0, 10],
            "end": [5, 15],
            "label": ["ORG", "GPE"],
            "entity_text": ["Fed", "France"],
        }
    )
    output = extract_relations.run(
        _deps(mentions),
        _rule_ctx(
            [{"subject_label": "GPE", "object_label": "ORG",
              "relation_type": "hosts"}]
        ),
    )
    assert output.height == 1
    row = output.row(0, named=True)
    assert row["subject_mention_id"] == "e2"
    assert row["object_mention_id"] == "e1"
    assert row["subject_label"] == "GPE"
    assert row["object_label"] == "ORG"
    assert row["directed"] is True


def test_rule_extractor_yields_no_rows_when_no_pair_matches() -> None:
    output = extract_relations.run(
        _deps(),
        _rule_ctx(
            [{"subject_label": "PERSON", "object_label": "GPE",
              "relation_type": "born_in"}]
        ),
    )
    assert output.height == 0


def test_rule_extractor_respects_window_scope() -> None:
    output = extract_relations.run(
        _deps(),
        _rule_ctx(
            [{"subject_label": "ORG", "object_label": "MONEY",
              "relation_type": "reports"}],
            {"scope": "window", "max_char_gap": 6},
        ),
    )
    # e1(ORG,0-5) → e3(MONEY,20-25) has a 15-char gap, beyond the window, so no
    # relation is emitted even though the labels match.
    assert output.height == 0


@pytest.mark.parametrize(
    ("rules", "extra", "message"),
    [
        ([], {}, "rules must not be empty"),
        (
            [
                {"subject_label": "ORG", "object_label": "GPE",
                 "relation_type": "x"},
                {"subject_label": "ORG", "object_label": "GPE",
                 "relation_type": "x"},
            ],
            {},
            "rules must be unique",
        ),
        (
            [{"subject_label": "ORG", "object_label": "GPE",
              "relation_type": "x"}],
            {"label_field": None},
            "label_field",
        ),
        (
            [{"subject_label": "ORG", "object_label": "GPE"}],
            {},
            "relation_type",
        ),
        (
            [{"subject_label": " ", "object_label": "GPE",
              "relation_type": "x"}],
            {},
            "must not be empty",
        ),
    ],
)
def test_rule_options_are_strict(
    rules: list[dict[str, str]], extra: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        extract_relations.validate_options(
            {"mentions": "m", "extractor": "rule", "rules": rules, **extra}
        )


# --- extractor seam: typed / directed / scored relations ---------------------


class _FakeModelExtractor(RelationExtractor):
    """A stand-in for a future learned extractor, proving the grain and driver
    carry typed, directed, scored relations and the full status vocabulary
    faithfully — without shipping a non-deterministic extractor."""

    name = "fake_model"
    version = "9"
    method = "model_assertion"

    def required_mention_columns(self, options: object) -> tuple[str, ...]:
        return ()

    def extract(
        self,
        mentions: Sequence[Mention],
        options: object,
        *,
        infer: RelationInference | None = None,
    ) -> list[Relation]:
        del infer
        if len(mentions) < 2:
            return []
        subject, obj = mentions[0], mentions[1]
        return [
            Relation(
                subject=subject,
                object=obj,
                relation_type="acquired",
                directed=True,
                status="asserted",
                confidence=0.87,
                sentence_index=subject.sentence_index,
            ),
            Relation(
                subject=subject,
                object=obj,
                relation_type="partnered_with",
                directed=True,
                status="ambiguous",
                confidence=0.42,
                sentence_index=subject.sentence_index,
            ),
        ]


def test_extractor_seam_carries_typed_directed_scored_relations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeModelExtractor()
    monkeypatch.setattr(
        "dbt_ml.text.transforms._relations.get_relation_extractor",
        lambda name: fake,
    )
    output = extract_relations.run(_deps(), _ctx()).sort("relation_type")

    assert output["method"].unique().to_list() == ["model_assertion"]
    assert output["extractor"].unique().to_list() == ["fake_model"]
    assert output["directed"].unique().to_list() == [True]
    assert output["relation_type"].to_list() == ["acquired", "partnered_with"]
    assert output["status"].to_list() == ["asserted", "ambiguous"]
    assert output["confidence"].to_list() == [pytest.approx(0.87), pytest.approx(0.42)]
    # Distinct relation_type → distinct relation_id for the same mention pair.
    assert output["relation_id"].n_unique() == 2


# --- model_assertion extractor -----------------------------------------------

_MA_MENTIONS = [
    Mention(mention_id="e1", sentence_index=0, start=0, end=5, label="ORG", text="Apple"),
    Mention(mention_id="e2", sentence_index=0, start=10, end=15, label="ORG", text="Beats"),
    Mention(mention_id="e3", sentence_index=0, start=20, end=25, label="GPE", text="US"),
    Mention(mention_id="e4", sentence_index=1, start=40, end=45, label="ORG", text="Acme"),
]


class _FakeInference:
    def __init__(self, assertions: list[RelationAssertion]) -> None:
        self._assertions = assertions

    def assert_relations(
        self,
        pairs: Sequence[tuple[Mention, Mention]],
        options: ModelAssertionExtractorOptions,
    ) -> list[RelationAssertion]:
        del pairs, options
        return list(self._assertions)


def _ma_options(**overrides: object) -> ModelAssertionExtractorOptions:
    payload: dict[str, object] = {
        "mentions": "m",
        "relation_types": ("acquired", "located_in"),
        "threshold": 0.6,
    }
    payload.update(overrides)
    return ModelAssertionExtractorOptions.model_validate(payload)


def test_model_assertion_maps_status_by_threshold_and_conflicts() -> None:
    infer = _FakeInference(
        [
            RelationAssertion("e1", "e2", "acquired", 0.9),  # asserted
            RelationAssertion("e1", "e3", "located_in", 0.4),  # below threshold
            RelationAssertion("e2", "e3", "acquired", 0.8),  # two types for one
            RelationAssertion("e2", "e3", "located_in", 0.7),  # pair -> ambiguous
        ]
    )
    relations = ModelAssertionExtractor().extract(_MA_MENTIONS, _ma_options(), infer=infer)
    by_key = {
        (r.subject.mention_id, r.object.mention_id, r.relation_type): r for r in relations
    }
    assert by_key[("e1", "e2", "acquired")].status == "asserted"
    assert by_key[("e1", "e2", "acquired")].confidence == pytest.approx(0.9)
    assert by_key[("e1", "e2", "acquired")].directed is True
    assert by_key[("e1", "e3", "located_in")].status == "no_relation"
    assert by_key[("e2", "e3", "acquired")].status == "ambiguous"
    assert by_key[("e2", "e3", "located_in")].status == "ambiguous"


def test_model_assertion_drops_ungoverned_assertions() -> None:
    infer = _FakeInference(
        [
            RelationAssertion("e1", "e2", "merged_with", 0.9),  # not in allow-list
            RelationAssertion("e1", "e9", "acquired", 0.9),  # unknown mention
            RelationAssertion("e1", "e1", "acquired", 0.9),  # self relation
            RelationAssertion("e1", "e4", "acquired", 0.9),  # not an in-scope pair
        ]
    )
    assert ModelAssertionExtractor().extract(_MA_MENTIONS, _ma_options(), infer=infer) == []


def test_model_assertion_requires_inference() -> None:
    with pytest.raises(ValueError, match="requires an inference provider"):
        ModelAssertionExtractor().extract(_MA_MENTIONS, _ma_options(), infer=None)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"extractor": "model_assertion", "mentions": "m"}, "relation_types"),
        (
            {"extractor": "model_assertion", "mentions": "m", "relation_types": []},
            "must not be empty",
        ),
        (
            {"extractor": "model_assertion", "mentions": "m", "relation_types": ["a", "a"]},
            "must be unique",
        ),
        (
            {
                "extractor": "model_assertion",
                "mentions": "m",
                "relation_types": ["a"],
                "threshold": 1.5,
            },
            "between 0 and 1",
        ),
        (
            {
                "extractor": "model_assertion",
                "mentions": "m",
                "relation_types": ["a"],
                "threshold": "0.5",
            },
            "must be a number",
        ),
    ],
)
def test_model_assertion_options_are_strict(
    options: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        extract_relations.validate_options(options)


def test_model_assertion_runs_through_deterministic_provider() -> None:
    # Full driver wiring: build_inference -> extract_fields_with_usage ->
    # deterministic provider -> mapping -> row shaping. The deterministic
    # provider emits placeholder assertions that reference no real pair or
    # allowed type, so all are dropped, leaving a well-formed empty frame.
    ctx = TransformContext(
        project_dir=Path("."),
        profile_name="test",
        target_name="dev",
        warehouse=parse_warehouse_config(
            {"type": "duckdb", "path": "./test.duckdb", "schema": "main"}
        ),
        llm=LLMConfig(provider="deterministic"),
        options={
            "mentions": "mentions",
            "extractor": "model_assertion",
            "relation_types": ["acquired", "located_in"],
        },
    )
    output = extract_relations.run(_deps(), ctx)
    assert output.schema["confidence"] == pl.Float64
    assert set(("method", "status", "relation_type")).issubset(output.columns)


def _ma_ctx(run_budget: BudgetLedger | None = None) -> TransformContext:
    return TransformContext(
        project_dir=Path("."),
        profile_name="test",
        target_name="dev",
        warehouse=parse_warehouse_config(
            {"type": "duckdb", "path": "./test.duckdb", "schema": "main"}
        ),
        llm=LLMConfig(provider="deterministic"),
        options={
            "mentions": "m",
            "extractor": "model_assertion",
            "relation_types": ["acquired", "located_in"],
        },
        run_budget=run_budget,
    )


def _ma_pairs() -> list[tuple[Mention, Mention]]:
    return [(_MA_MENTIONS[0], _MA_MENTIONS[1])]


def test_model_assertion_malformed_response_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A missing/malformed `items` must fail, not silently produce zero relations
    # (which would delete a document's children on an incremental run).
    from dbt_ml.text.transforms import _relations

    monkeypatch.setattr(
        _relations, "extract_fields_with_usage", lambda content, **kw: ({"items": "x"}, {})
    )
    infer = _relations._build_inference(_ma_ctx())
    with pytest.raises(ValueError, match="malformed"):
        infer.assert_relations(_ma_pairs(), _ma_options())


def test_model_assertion_drops_out_of_range_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dbt_ml.text.transforms import _relations

    items = [
        {"subject_mention_id": "e1", "object_mention_id": "e2",
         "relation_type": "acquired", "confidence": 1.5},
        {"subject_mention_id": "e1", "object_mention_id": "e2",
         "relation_type": "acquired", "confidence": float("nan")},
        {"subject_mention_id": "e1", "object_mention_id": "e2",
         "relation_type": "acquired", "confidence": 0.9},
    ]
    monkeypatch.setattr(
        _relations, "extract_fields_with_usage", lambda content, **kw: ({"items": items}, {})
    )
    infer = _relations._build_inference(_ma_ctx())
    result = infer.assert_relations(_ma_pairs(), _ma_options())
    assert [a.confidence for a in result] == [pytest.approx(0.9)]


def test_model_assertion_charges_and_enforces_run_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dbt_ml.text.transforms import _relations

    monkeypatch.setattr(
        _relations,
        "extract_fields_with_usage",
        lambda content, **kw: ({"items": []}, {"api_calls": 1}),
    )
    ledger = BudgetLedger(LLMBudgetConfig(max_api_calls=1), scope="run")
    infer = _relations._build_inference(_ma_ctx(run_budget=ledger))
    infer.assert_relations(_ma_pairs(), _ma_options())  # charges api_calls=1
    with pytest.raises(BudgetExceededError):
        infer.assert_relations(_ma_pairs(), _ma_options())  # budget now exhausted


_MA_MODEL_YAML = """version: 2
models:
  - name: document_relations_llm
    depends_on: [ref('document_entities')]
    transform:
      type: python
      module: dbt_ml.text.transforms.extract_relations
      {uses_llm}options:
        mentions: document_entities
        extractor: model_assertion
        relation_types: [acquired, references_geography]
    materialization: full
"""


def _write_ma_model(project_dir: Path, *, uses_llm: bool) -> None:
    (project_dir / "models" / "document_relations_llm.yml").write_text(
        _MA_MODEL_YAML.format(uses_llm="uses_llm: true\n      " if uses_llm else "")
    )


def test_model_assertion_requires_uses_llm_at_compile(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    shutil.copytree(
        _example_dir(), project_dir, ignore=shutil.ignore_patterns("target")
    )
    _write_ma_model(project_dir, uses_llm=False)
    project, sources, models = load_project(project_dir)
    with pytest.raises(ConfigError, match="uses_llm"):
        validate_project_contract(project, sources, models, project_dir)


def test_model_assertion_compiles_with_uses_llm(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    shutil.copytree(
        _example_dir(), project_dir, ignore=shutil.ignore_patterns("target")
    )
    _write_ma_model(project_dir, uses_llm=True)
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)
    assert dag.execution_order().index("document_relations_llm") > dag.execution_order().index(
        "document_entities"
    )


# --- example -----------------------------------------------------------------


def _example_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "economic_nlp"


def test_economic_nlp_relations_example_compiles() -> None:
    project, sources, models = load_project(_example_dir())
    dag = validate_project_contract(project, sources, models, _example_dir())

    order = dag.execution_order()
    assert order.index("document_relations") > order.index("document_entities")
    assert order.index("document_typed_relations") > order.index("document_entities")
    by_name = {model.name: model for model in models}
    assert by_name["document_relations"].materialization == "incremental"
    assert by_name["document_typed_relations"].materialization == "incremental"


def test_relation_dependency_mismatch_fails_at_compile(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    shutil.copytree(
        _example_dir(), project_dir, ignore=shutil.ignore_patterns("target")
    )
    model_path = project_dir / "models" / "document_relations.yml"
    model_path.write_text(
        model_path.read_text().replace(
            "mentions: document_entities", "mentions: document_entity"
        )
    )
    project, sources, models = load_project(project_dir)
    with pytest.raises(ConfigError, match="referenced by options but not in depends_on"):
        validate_project_contract(project, sources, models, project_dir)
