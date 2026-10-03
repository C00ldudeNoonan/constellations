"""Relation-type and class-pairing checks against a project declaration (issue #626).

`RelationRule.relation_type` and `ModelAssertionExtractorOptions.relation_types`
are free strings with nothing to check them against. A project's
`classes:`/`relations:` declaration — building on #625's vocabulary — gives
them something to check against; a project that declares neither is
unaffected, which these tests pin alongside the checks themselves.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from stel.compiler import validate_project_contract
from stel.config import load_project
from stel.config.loader import ConfigError
from stel.config.project import ProjectConfig
from stel.config.vocabulary import RelationTypeDef
from stel.relation_contracts import validate_relation_project_contracts

# ─── the project-level declaration ──────────────────────────────────────────


def test_classes_must_be_unique() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        ProjectConfig(name="p", classes=("org", "org"))


def test_class_names_follow_the_identifier_charset() -> None:
    with pytest.raises(ValueError, match="invalid"):
        ProjectConfig(name="p", classes=("not-an-identifier",))


def test_relation_subject_class_must_be_declared() -> None:
    with pytest.raises(ValueError, match=r"subject_class: org.*not declared"):
        ProjectConfig(
            name="p",
            classes=("gpe",),
            relations=(
                RelationTypeDef(name="located_in", subject_class="org", object_class="gpe"),
            ),
        )


def test_relation_object_class_must_be_declared() -> None:
    with pytest.raises(ValueError, match=r"object_class: gpe.*not declared"):
        ProjectConfig(
            name="p",
            classes=("org",),
            relations=(
                RelationTypeDef(name="located_in", subject_class="org", object_class="gpe"),
            ),
        )


def test_duplicate_relation_pairing_is_rejected() -> None:
    with pytest.raises(ValueError, match="declared twice"):
        ProjectConfig(
            name="p",
            classes=("org", "gpe"),
            relations=(
                RelationTypeDef(name="located_in", subject_class="org", object_class="gpe"),
                RelationTypeDef(name="located_in", subject_class="org", object_class="gpe"),
            ),
        )


def test_a_relation_may_repeat_with_a_different_pairing() -> None:
    # Polymorphic relation: `located_in` holds for two different subject classes.
    project = ProjectConfig(
        name="p",
        classes=("org", "person", "gpe"),
        relations=(
            RelationTypeDef(name="located_in", subject_class="org", object_class="gpe"),
            RelationTypeDef(name="located_in", subject_class="person", object_class="gpe"),
        ),
    )

    assert len(project.relations) == 2


# ─── compiling against the economic_nlp example ─────────────────────────────


def _example_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "economic_nlp"


def _copy_example(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    shutil.copytree(_example_dir(), project_dir, ignore=shutil.ignore_patterns("target"))
    return project_dir


def _append_declaration(project_dir: Path, declaration_yaml: str) -> None:
    project_file = project_dir / "stel_project.yml"
    existing = project_file.read_text(encoding="utf-8")
    project_file.write_text(existing + declaration_yaml, encoding="utf-8")


# `document_typed_relations` (the `rule` extractor) declares exactly these two
# rules against `examples/economic_nlp/models/document_typed_relations.yml`.
_MATCHING_DECLARATION = (
    "\nclasses: [ORG, GPE, MONEY]\n"
    "relations:\n"
    "  - name: references_geography\n"
    "    subject_class: ORG\n"
    "    object_class: GPE\n"
    "  - name: references_amount\n"
    "    subject_class: ORG\n"
    "    object_class: MONEY\n"
)


def test_a_project_with_no_declaration_is_unaffected(tmp_path: Path) -> None:
    # No `classes:`/`relations:` at all: identical to the undeclared example's
    # own compiler test — this is the "adds nothing" acceptance bullet.
    project_dir = _copy_example(tmp_path)

    project, sources, models = load_project(project_dir)
    validate_project_contract(project, sources, models, project_dir)


def test_matching_declaration_compiles(tmp_path: Path) -> None:
    project_dir = _copy_example(tmp_path)
    _append_declaration(project_dir, _MATCHING_DECLARATION)

    project, sources, models = load_project(project_dir)
    validate_project_contract(project, sources, models, project_dir)


def test_undeclared_relation_type_fails_at_compile(tmp_path: Path) -> None:
    project_dir = _copy_example(tmp_path)
    # Declares only one of the two rules' relation types.
    _append_declaration(
        project_dir,
        "\nclasses: [ORG, GPE]\n"
        "relations:\n"
        "  - name: references_geography\n"
        "    subject_class: ORG\n"
        "    object_class: GPE\n",
    )

    project, sources, models = load_project(project_dir)
    with pytest.raises(ConfigError) as excinfo:
        validate_project_contract(project, sources, models, project_dir)

    message = str(excinfo.value)
    assert "document_typed_relations" in message
    assert "references_amount" in message
    assert "`relations:` does not declare" in message


def test_disallowed_class_pair_fails_at_compile(tmp_path: Path) -> None:
    project_dir = _copy_example(tmp_path)
    # `references_geography` is declared, but backwards: GPE -> ORG, not the
    # rule's actual ORG -> GPE.
    _append_declaration(
        project_dir,
        "\nclasses: [ORG, GPE, MONEY]\n"
        "relations:\n"
        "  - name: references_geography\n"
        "    subject_class: GPE\n"
        "    object_class: ORG\n"
        "  - name: references_amount\n"
        "    subject_class: ORG\n"
        "    object_class: MONEY\n",
    )

    project, sources, models = load_project(project_dir)
    with pytest.raises(ConfigError) as excinfo:
        validate_project_contract(project, sources, models, project_dir)

    message = str(excinfo.value)
    assert "document_typed_relations" in message
    assert "ORG' to 'GPE'" in message
    assert "GPE -> ORG" in message


def test_co_occurrence_models_are_unaffected_by_a_declaration(tmp_path: Path) -> None:
    # `document_relations` uses the untyped co_occurrence extractor; a
    # classes:/relations: declaration constrains rule/model_assertion
    # extractors only.
    project_dir = _copy_example(tmp_path)
    _append_declaration(
        project_dir,
        "\nclasses: [ORG, GPE, MONEY]\n"
        "relations:\n"
        "  - name: references_geography\n"
        "    subject_class: ORG\n"
        "    object_class: GPE\n"
        "  - name: references_amount\n"
        "    subject_class: ORG\n"
        "    object_class: MONEY\n",
    )

    project, sources, models = load_project(project_dir)
    validate_project_contract(project, sources, models, project_dir)


# ─── model_assertion extractor ──────────────────────────────────────────────

_MA_MODEL_YAML = """\
version: 2

models:
  - name: document_relations_llm
    description: "Model-asserted relations for a declaration check"
    depends_on: [ref('document_entities')]
    transform:
      type: python
      module: stel.text.transforms.extract_relations
      uses_llm: true
      options:
        mentions: document_entities
        extractor: model_assertion
        relation_types: [acquired, invented_relation]
    materialization: full
"""


def test_model_assertion_relation_types_must_be_declared(tmp_path: Path) -> None:
    project_dir = _copy_example(tmp_path)
    (project_dir / "models" / "document_relations_llm.yml").write_text(
        _MA_MODEL_YAML, encoding="utf-8"
    )
    _append_declaration(
        project_dir,
        "\nclasses: [ORG, GPE]\n"
        "relations:\n"
        "  - name: acquired\n"
        "    subject_class: ORG\n"
        "    object_class: ORG\n",
    )

    project, sources, models = load_project(project_dir)
    with pytest.raises(ConfigError) as excinfo:
        validate_project_contract(project, sources, models, project_dir)

    message = str(excinfo.value)
    assert "document_relations_llm" in message
    assert "invented_relation" in message
    assert "`relations:` does not declare" in message



def test_a_project_local_relation_module_is_not_checked_against_the_builtin_parser(
    tmp_path: Path,
) -> None:
    # Same backwards declaration as test_disallowed_class_pair_fails_at_compile,
    # but a project file at the built-in module path owns the options shape, so
    # the built-in rule check must not run against it (Codex on #649).
    project_dir = _copy_example(tmp_path)
    _append_declaration(
        project_dir,
        "\nclasses: [ORG, GPE, MONEY]\n"
        "relations:\n"
        "  - name: references_geography\n"
        "    subject_class: GPE\n"
        "    object_class: ORG\n"
        "  - name: references_amount\n"
        "    subject_class: ORG\n"
        "    object_class: MONEY\n",
    )
    override = project_dir / "stel" / "text" / "transforms" / "extract_relations.py"
    override.parent.mkdir(parents=True)
    override.write_text("def validate_options(options):\n    pass\n", encoding="utf-8")

    project, _sources, models = load_project(project_dir)

    validate_relation_project_contracts(models, project, project_dir)
