"""Declared domain vocabularies (issue #625).

A `type: enum` field's label set lived inline, repeated by any model
classifying into the same set with no way to notice drift between two
copies — the problem #304 solved for one field's three consumers (the
provider schema, the `accepted_values` test, the prompt fallback), one level
up. These tests pin the vocabulary declaration itself, its resolution into an
ordinary `values:` list, and the one extra thing resolution carries along: a
term's `description:`, into the prompt fallback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stel.config.loader import ConfigError, load_project
from stel.config.model import FieldConfig
from stel.config.project import ProjectConfig
from stel.config.vocabulary import Vocabulary, VocabularyTerm

# ─── the vocabulary declaration ─────────────────────────────────────────────


def test_vocabulary_preserves_declared_term_order() -> None:
    vocabulary = Vocabulary(terms=[VocabularyTerm(label="b"), VocabularyTerm(label="a")])

    assert vocabulary.labels() == ["b", "a"]


def test_vocabulary_descriptions_only_cover_terms_that_declare_one() -> None:
    vocabulary = Vocabulary(
        terms=[
            VocabularyTerm(label="fed", description="The US central bank"),
            VocabularyTerm(label="ecb"),
        ]
    )

    assert vocabulary.descriptions() == {"fed": "The US central bank"}


def test_aliases_do_not_widen_the_label_set() -> None:
    # Alternative labels expand what an entity linker matches (issue #627),
    # never what a classifier is allowed to output.
    vocabulary = Vocabulary(terms=[VocabularyTerm(label="fed", aliases=["the Fed", "FOMC"])])

    assert vocabulary.labels() == ["fed"]


def test_duplicate_term_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="declared twice"):
        Vocabulary(terms=[VocabularyTerm(label="fed"), VocabularyTerm(label="fed")])


def test_empty_term_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        VocabularyTerm(label="  ")


def test_duplicate_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="listed twice"):
        VocabularyTerm(label="fed", aliases=["the Fed", "the Fed"])


def test_broader_term_must_exist_in_the_same_vocabulary() -> None:
    with pytest.raises(ValueError, match="not a term in this vocabulary"):
        Vocabulary(terms=[VocabularyTerm(label="fed", broader="central_bank")])


def test_broader_term_cannot_be_itself() -> None:
    with pytest.raises(ValueError, match="its own `broader` term"):
        Vocabulary(terms=[VocabularyTerm(label="fed", broader="fed")])


def test_broader_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="cyclic `broader` chain"):
        Vocabulary(
            terms=[
                VocabularyTerm(label="a", broader="b"),
                VocabularyTerm(label="b", broader="a"),
            ]
        )


def test_broader_chain_resolves_without_cycling() -> None:
    # a -> b -> c terminates cleanly; must not raise.
    Vocabulary(
        terms=[
            VocabularyTerm(label="a", broader="b"),
            VocabularyTerm(label="b", broader="c"),
            VocabularyTerm(label="c"),
        ]
    )


def test_vocabulary_name_must_be_a_valid_identifier() -> None:
    with pytest.raises(ValueError, match="invalid"):
        ProjectConfig(
            name="p",
            vocabularies={
                "not-an-identifier": Vocabulary(terms=[VocabularyTerm(label="a")])
            },
        )


# ─── the field-level declaration ────────────────────────────────────────────


def test_values_from_must_look_like_vocab_dot_name() -> None:
    with pytest.raises(ValueError, match=r"must look like 'vocab\.<name>'"):
        FieldConfig(name="signal", type="enum", values_from="sector")


def test_values_and_values_from_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="use exactly one"):
        FieldConfig(name="signal", type="enum", values=["a"], values_from="vocab.sector")


def test_values_from_on_a_non_enum_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="use `type: enum`"):
        FieldConfig(name="signal", type="string", values_from="vocab.sector")


def test_enum_with_neither_values_nor_values_from_is_rejected() -> None:
    with pytest.raises(ValueError, match="declares no `values:` or `values_from:`"):
        FieldConfig(name="signal", type="enum")


def test_value_descriptions_cannot_be_authored_directly() -> None:
    with pytest.raises(ValueError, match="must not be set directly"):
        FieldConfig(
            name="signal", type="enum", values=["a"], value_descriptions={"a": "A"}
        )


# ─── resolution against a loaded project ────────────────────────────────────

_VOCAB_YAML = (
    "vocabularies:\n"
    "  sector:\n"
    "    terms:\n"
    "      - label: financials\n"
    "        description: Banks, insurers, and other financial firms\n"
    "      - label: technology\n"
)


def _field_lines(values_from: str = "vocab.sector") -> str:
    return (
        "    fields:\n"
        "      - name: sector\n"
        "        type: enum\n"
        f"        values_from: {values_from}\n"
    )


def _model_block(name: str) -> str:
    return (
        f"  - name: {name}\n"
        "    extraction:\n"
        "      backend: json\n" + _field_lines()
    )


def _write_project(tmp_path: Path, *, project_yaml: str, model_yaml: str) -> None:
    (tmp_path / "stel_project.yml").write_text(project_yaml, encoding="utf-8")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "m.yml").write_text(model_yaml, encoding="utf-8")


def test_values_from_resolves_to_the_vocabularys_labels(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        project_yaml="name: vocab_project\n" + _VOCAB_YAML,
        model_yaml="version: 2\nmodels:\n" + _model_block("classified"),
    )

    _, _, models = load_project(tmp_path)

    field = models[0].fields[0]
    assert field.values == ["financials", "technology"]
    assert field.values_from is None
    assert field.value_descriptions == {
        "financials": "Banks, insurers, and other financial firms"
    }


def test_two_models_share_one_vocabulary_without_drift(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        project_yaml="name: vocab_project\n" + _VOCAB_YAML,
        model_yaml=(
            "version: 2\nmodels:\n"
            + _model_block("classified_a")
            + _model_block("classified_b")
        ),
    )

    _, _, models = load_project(tmp_path)

    values_by_model = {m.name: m.fields[0].values for m in models}
    assert values_by_model == {
        "classified_a": ["financials", "technology"],
        "classified_b": ["financials", "technology"],
    }


def test_unknown_vocabulary_fails_before_anything_else(tmp_path: Path) -> None:
    # No `vocabularies:` block at all: the project file loads, but the
    # field's reference does not resolve.
    _write_project(
        tmp_path,
        project_yaml="name: vocab_project\n",
        model_yaml="version: 2\nmodels:\n" + _model_block("classified"),
    )

    with pytest.raises(ConfigError, match=r"vocab\.sector.*not declared"):
        load_project(tmp_path)


def test_unknown_vocabulary_error_names_whats_available(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        project_yaml=(
            "name: vocab_project\nvocabularies:\n  signal:\n    terms:\n"
            "      - label: churn_risk\n"
        ),
        model_yaml="version: 2\nmodels:\n" + _model_block("classified"),
    )

    with pytest.raises(ConfigError, match=r"Available:.*signal"):
        load_project(tmp_path)


def test_a_field_without_values_from_is_unaffected(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        project_yaml="name: plain_project\n",
        model_yaml=(
            "version: 2\nmodels:\n"
            "  - name: plain\n"
            "    extraction:\n"
            "      backend: json\n"
            "    fields:\n"
            "      - name: title\n"
            "        type: string\n"
        ),
    )

    _, _, models = load_project(tmp_path)

    assert models[0].fields[0].values == []


# ─── the prompt fallback carries term descriptions ──────────────────────────


def _resolved_field(values: list[str], descriptions: dict[str, str]) -> FieldConfig:
    # Mirrors what loader resolution produces: model_copy bypasses
    # value_descriptions' authoring guard the same way the loader's own
    # resolution does.
    return FieldConfig(name="sector", type="enum", values=values).model_copy(
        update={"value_descriptions": descriptions}
    )


def test_prompt_fallback_renders_term_descriptions() -> None:
    from stel.backends.llm_backend import _apply_enum_portability
    from stel.llm_map import build_fields_spec

    field = _resolved_field(
        ["financials", "technology"], {"financials": "Banks and insurers"}
    )
    spec = build_fields_spec([field])

    class _NoSchemaEnumProvider:
        supports_schema_enum = False

    fields, system = _apply_enum_portability(_NoSchemaEnumProvider(), spec, "SYS")

    assert "enum" not in fields[0]
    assert "enum_descriptions" not in fields[0]
    assert "financials: Banks and insurers" in system
    assert "technology" in system


def test_prompt_fallback_without_descriptions_is_unchanged() -> None:
    from stel.backends.llm_backend import _apply_enum_portability
    from stel.llm_map import build_fields_spec

    spec = build_fields_spec([FieldConfig(name="sector", type="enum", values=["a", "b"])])

    class _NoSchemaEnumProvider:
        supports_schema_enum = False

    _, system = _apply_enum_portability(_NoSchemaEnumProvider(), spec, "SYS")

    assert "- sector: use exactly one of a, b" in system


def test_enum_descriptions_never_reach_the_provider_schema() -> None:
    from stel.backends.llm_backend import _input_schema
    from stel.llm_map import build_fields_spec

    field = _resolved_field(["financials"], {"financials": "Banks and insurers"})
    spec = build_fields_spec([field])

    schema = _input_schema(spec)

    assert schema["properties"]["sector"] == {"type": "string", "enum": ["financials"]}
