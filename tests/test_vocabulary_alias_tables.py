"""`alias_table` resolving against a declared vocabulary (issue #627).

An alias table is already a preferred label plus alternative labels, kept by
hand. Once a vocabulary declaration (#625) holds the same information, the
`alias_table` resolver can read it directly — `aliases: vocab.<name>` instead
of an upstream model — so the two never drift apart. A hand-maintained alias
table keeps working unchanged: this is an additional source, not a
replacement, which these tests pin alongside the new one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from stel.adapters import parse_warehouse_config
from stel.compiler import validate_project_contract
from stel.config import load_project
from stel.config.loader import ConfigError
from stel.config.vocabulary import Vocabulary, VocabularyTerm
from stel.link_contracts import validate_link_project_contracts
from stel.text.linking import (
    ALIAS_RESOLVER_VERSION,
    parse_vocabulary_alias_source,
    vocabulary_alias_rows,
)
from stel.text.transforms import link_entities
from stel.transforms import TransformContext

_MENTIONS = pl.DataFrame(
    {
        "entity_id": ["m-fed", "m-nickname", "m-unknown"],
        "document_id": ["doc-1", "doc-1", "doc-1"],
        "entity_text": ["Federal Reserve", "the Fed", "Mercury"],
        "label": ["ORG", "ORG", "ORG"],
        "start": [0, 20, 40],
        "end": [15, 27, 47],
    }
)

_SECTOR_VOCABULARY = Vocabulary(
    terms=[
        VocabularyTerm(
            label="Federal Reserve", aliases=["the Fed", "FOMC"], description="US central bank"
        ),
        VocabularyTerm(label="European Central Bank", aliases=["the ECB"]),
    ]
)


def _ctx(
    *, options: dict[str, object] | None = None, vocabularies: dict[str, Vocabulary] | None = None
) -> TransformContext:
    merged: dict[str, object] = {
        "mentions": "mentions",
        "aliases": "vocab.sector",
    }
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
        vocabularies=vocabularies or {"sector": _SECTOR_VOCABULARY},
    )


# ─── linking.py helpers ──────────────────────────────────────────────────────


def test_parse_vocabulary_alias_source_recognizes_the_vocab_prefix() -> None:
    assert parse_vocabulary_alias_source("vocab.sector") == "sector"


def test_parse_vocabulary_alias_source_rejects_a_plain_model_name() -> None:
    assert parse_vocabulary_alias_source("entity_aliases") is None


def test_vocabulary_alias_rows_include_the_label_and_every_alias() -> None:
    rows = vocabulary_alias_rows("sector", _SECTOR_VOCABULARY)

    by_alias = {row["alias"]: row for row in rows}
    assert by_alias["Federal Reserve"]["canonical_id"] == "Federal Reserve"
    assert by_alias["the Fed"]["canonical_id"] == "Federal Reserve"
    assert by_alias["FOMC"]["canonical_id"] == "Federal Reserve"
    assert by_alias["the ECB"]["canonical_id"] == "European Central Bank"
    assert all(row["entity_namespace"] == "sector" for row in rows)


# ─── the driver resolving against a vocabulary ──────────────────────────────


def test_vocab_sourced_alias_table_matches_label_and_alias() -> None:
    result = link_entities.run({"mentions": _MENTIONS}, _ctx())

    by_mention = {row["mention_id"]: row for row in result.to_dicts()}
    assert by_mention["m-fed"]["status"] == "matched"
    assert by_mention["m-fed"]["canonical_id"] == "Federal Reserve"
    assert by_mention["m-fed"]["entity_namespace"] == "sector"
    assert by_mention["m-nickname"]["status"] == "matched"
    assert by_mention["m-nickname"]["canonical_id"] == "Federal Reserve"
    assert by_mention["m-unknown"]["status"] == "unmatched"


def test_vocab_sourced_alias_table_classifies_identically_to_a_hand_table() -> None:
    # Same alias content, two sources: status/method/canonical_id must agree.
    from_vocab = link_entities.run({"mentions": _MENTIONS}, _ctx())

    hand_aliases = pl.DataFrame(vocabulary_alias_rows("sector", _SECTOR_VOCABULARY))
    from_table = link_entities.run(
        {"mentions": _MENTIONS, "aliases": hand_aliases},
        _ctx(options={"aliases": "aliases"}),
    )

    key = ["mention_id", "status", "match_method", "canonical_id", "entity_namespace"]
    assert from_vocab.select(key).sort("mention_id").to_dicts() == from_table.select(
        key
    ).sort("mention_id").to_dicts()


def test_vocab_sourced_alias_table_honors_configured_column_names() -> None:
    # #643 review: the synthesized frame previously always used the three
    # default column names, so a configured override made `build_reference`
    # raise a missing-columns error even though compilation succeeded.
    result = link_entities.run(
        {"mentions": _MENTIONS},
        _ctx(
            options={
                "alias_text_field": "surface",
                "namespace_field": "ns",
                "canonical_id_field": "id",
            }
        ),
    )

    by_mention = {row["mention_id"]: row for row in result.to_dicts()}
    assert by_mention["m-fed"]["status"] == "matched"
    assert by_mention["m-fed"]["canonical_id"] == "Federal Reserve"


def test_resolver_version_is_reported_for_code_version_identity() -> None:
    assert link_entities.code_version_identity(
        {"mentions": "mentions", "aliases": "vocab.sector"}
    ) == {"resolver": "alias_table", "resolver_version": ALIAS_RESOLVER_VERSION}


def test_alias_resolver_version_bump_invalidates_incremental_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The regression #643 review asked for: a resolver-version bump must
    change `compute_model_code_version` for a `link_entities` model, or an
    already-materialized incremental model's rows silently stay at the old
    version forever (`transform_code_hash` is always "missing" for this
    built-in, installed module — it has no project-local file to hash)."""
    from stel import versioning
    from stel.config.model import ModelConfig, TransformConfig
    from stel.config.project import ExtractionDefaults, ProjectConfig
    from stel.text import linking

    project = ProjectConfig(
        name="p", extraction=ExtractionDefaults(default_backend="json")
    )
    model = ModelConfig(
        name="entity_links",
        depends_on=["ref('mentions')", "ref('aliases')"],
        transform=TransformConfig(
            type="python",
            module="stel.text.transforms.link_entities",
            options={"mentions": "mentions", "aliases": "aliases"},
        ),
        materialization="incremental",
    )

    before = versioning.compute_model_code_version(model, project, tmp_path)
    # Simulates a stel upgrade that bumps ALIAS_RESOLVER_VERSION: the
    # registry's already-constructed instance reads the version as a class
    # attribute, so patching the class is what an actual version bump does.
    monkeypatch.setattr(linking.AliasTableResolver, "version", "999")
    after = versioning.compute_model_code_version(model, project, tmp_path)

    assert before != after


def test_unknown_vocabulary_fails_at_run_time_with_a_clear_message() -> None:
    with pytest.raises(ValueError, match="not declared under"):
        link_entities.run(
            {"mentions": _MENTIONS},
            _ctx(options={"aliases": "vocab.nonexistent"}),
        )


def test_a_stray_aliases_dependency_is_rejected_when_vocab_sourced() -> None:
    with pytest.raises(ValueError, match="expects a dependency named"):
        link_entities.run(
            {"mentions": _MENTIONS, "aliases": pl.DataFrame()},
            _ctx(),
        )


def test_declared_dependencies_drop_the_alias_model_when_vocab_sourced() -> None:
    deps = link_entities.declared_dependencies({"mentions": "mentions", "aliases": "vocab.sector"})

    assert deps == ("mentions",)


def test_declared_dependencies_keep_the_alias_model_for_a_hand_table() -> None:
    deps = link_entities.declared_dependencies(
        {"mentions": "mentions", "aliases": "entity_aliases"}
    )

    assert deps == ("mentions", "entity_aliases")


def test_incremental_contract_has_no_reference_deps_when_vocab_sourced() -> None:
    contract = link_entities.declared_incremental_contract(
        {"mentions": "mentions", "aliases": "vocab.sector"}
    )

    assert contract.reference_deps == ()


def test_vocab_sourced_aliases_only_supported_for_alias_table() -> None:
    with pytest.raises(ValueError, match="only supported for `resolver: alias_table`"):
        link_entities.validate_options(
            {
                "mentions": "mentions",
                "aliases": "vocab.sector",
                "resolver": "fuzzy",
                "threshold": 0.8,
            }
        )


# ─── hand-maintained alias tables are unaffected ────────────────────────────


def test_a_hand_maintained_alias_table_still_requires_its_dependency() -> None:
    # An unrelated regression check: a plain model-name `aliases` is untouched
    # by any of the above — same two-dependency contract as before #627.
    deps = link_entities.declared_dependencies(
        {"mentions": "mentions", "aliases": "entity_aliases"}
    )
    contract = link_entities.declared_incremental_contract(
        {"mentions": "mentions", "aliases": "entity_aliases"}
    )

    assert deps == ("mentions", "entity_aliases")
    assert contract.reference_deps == ("entity_aliases",)


# ─── compiling against a project ────────────────────────────────────────────


def _example_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "economic_entity_links"


def _copy_example(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    shutil.copytree(_example_dir(), project_dir, ignore=shutil.ignore_patterns("target"))
    return project_dir


def _append_vocabulary(project_dir: Path) -> None:
    project_file = project_dir / "stel_project.yml"
    existing = project_file.read_text(encoding="utf-8")
    project_file.write_text(
        existing
        + "\nvocabularies:\n"
        "  sector:\n"
        "    terms:\n"
        "      - label: Federal Reserve\n"
        "        aliases: [the Fed]\n",
        encoding="utf-8",
    )


_VOCAB_MODEL_YAML = """\
version: 2

models:
  - name: entity_links_vocab
    description: "Vocabulary-sourced alias resolution, alongside the hand-maintained table"
    depends_on: [ref('entity_mentions')]
    transform:
      type: python
      module: stel.text.transforms.link_entities
      options:
        mentions: entity_mentions
        aliases: vocab.sector
        document_id_field: source_document_id
        start_field: null
        end_field: null
    materialization: incremental
"""


def test_vocab_sourced_model_compiles_alongside_the_hand_maintained_one(
    tmp_path: Path,
) -> None:
    project_dir = _copy_example(tmp_path)
    _append_vocabulary(project_dir)
    (project_dir / "models" / "entity_links_vocab.yml").write_text(
        _VOCAB_MODEL_YAML, encoding="utf-8"
    )

    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)

    assert "entity_links_vocab" in dag.execution_order()
    # The original, table-backed model is unaffected by the addition.
    assert "entity_links" in dag.execution_order()


def test_unknown_vocabulary_fails_to_compile(tmp_path: Path) -> None:
    project_dir = _copy_example(tmp_path)
    # No `vocabularies:` block at all.
    (project_dir / "models" / "entity_links_vocab.yml").write_text(
        _VOCAB_MODEL_YAML, encoding="utf-8"
    )

    project, sources, models = load_project(project_dir)
    with pytest.raises(ConfigError) as excinfo:
        validate_project_contract(project, sources, models, project_dir)

    message = str(excinfo.value)
    assert "entity_links_vocab" in message
    assert "vocab.sector" in message
    assert "not declared under" in message


def test_economic_entity_links_example_is_unaffected() -> None:
    # No vocabulary added: the shipped example compiles exactly as it did
    # before #627 exists.
    project, sources, models = load_project(_example_dir())
    validate_project_contract(project, sources, models, _example_dir())


def test_link_contract_does_not_swallow_options_that_fail_to_parse(
    tmp_path: Path,
) -> None:
    # Reached only if `validate_options` and this check disagree about the same
    # options. The check must raise rather than skip the model (#642 review).
    project_dir = _copy_example(tmp_path)
    _append_vocabulary(project_dir)
    malformed = _VOCAB_MODEL_YAML.replace(
        "        aliases: vocab.sector\n",
        "        aliases: vocab.sector\n        resolver: no_such_resolver\n",
    )
    (project_dir / "models" / "entity_links_vocab.yml").write_text(
        malformed, encoding="utf-8"
    )

    project, _sources, models = load_project(project_dir)

    with pytest.raises(ValueError, match=r"no_such_resolver|resolver"):
        validate_link_project_contracts(models, project, project_dir)


def test_a_project_local_link_module_is_not_checked_against_the_builtin_parser(
    tmp_path: Path,
) -> None:
    # `_load_transform_module` prefers a project file at the same dotted path, so
    # the built-in options shape does not apply to it (Codex on #649).
    project_dir = _copy_example(tmp_path)
    _append_vocabulary(project_dir)
    override = project_dir / "stel" / "text" / "transforms" / "link_entities.py"
    override.parent.mkdir(parents=True)
    override.write_text("def validate_options(options):\n    pass\n", encoding="utf-8")
    custom = _VOCAB_MODEL_YAML.replace(
        "        aliases: vocab.sector\n",
        "        aliases: vocab.sector\n        custom: true\n",
    )
    (project_dir / "models" / "entity_links_vocab.yml").write_text(custom, encoding="utf-8")

    project, _sources, models = load_project(project_dir)

    validate_link_project_contracts(models, project, project_dir)
