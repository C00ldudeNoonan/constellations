"""External vocabularies are bounded, validated snapshots (issue #718 / ALE-431)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest
import yaml

from stel.adapters import create_adapter
from stel.backends.llm_backend import _fields_spec, _input_schema
from stel.compiler import validate_project_contract
from stel.config.loader import ConfigError, load_project
from stel.config.model import LLMTransformConfig
from stel.config.vocabulary import Vocabulary, VocabularySource
from stel.dbt_embed.api import materialize
from stel.dbt_embed.codegen import generate_dbt_models
from stel.dbt_export import build_dbt_sources
from stel.llm_map import LLMMapError, build_fields_spec
from stel.manifest import build_manifest
from stel.profile import resolve_profile
from stel.versioning import compute_model_code_version
from stel.vocabulary_sources import resolve_warehouse_vocabularies


def _project(
    tmp_path: Path, source: dict[str, Any], *, backend: bool = False, **vocab: object
) -> None:
    declaration = {"from": source, **vocab}
    (tmp_path / "stel_project.yml").write_text(
        yaml.safe_dump(
            {
                "name": "vocab_sources",
                "duckdb": {"path": "warehouse.duckdb"},
                "vocabularies": {"sector": declaration},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "models").mkdir()
    field = {"name": "sector", "type": "enum", "values_from": "vocab.sector"}
    extraction: dict[str, Any] = {"backend": "llm" if backend else "json"}
    model: dict[str, Any] = {
        "name": "classified",
        "source": "ref('documents')",
        "extraction": extraction,
    }
    if backend:
        extraction["options"] = {"fields": [field]}
    else:
        model["fields"] = [field]
    (tmp_path / "models" / "m.yml").write_text(
        yaml.safe_dump({"version": 2, "models": [model]}),
        encoding="utf-8",
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "s.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "source",
    [
        {"label_column": "label"},
        {"path": "s.csv", "relation": "warehouse://s", "label_column": "label"},
        {"path": "s.json", "label_column": "label"},
        {"relation": "s", "label_column": "label"},
        {"relation": "warehouse://s;DROP TABLE t", "label_column": "label"},
        {"relation": "warehouse://s", "label_column": " "},
        {"path": "s.csv", "label_column": "label", "max_terms": 0},
        {"path": "s.csv", "label_column": "label", "max_terms": True},
        {"path": "s.csv", "label_column": "label", "sql": "SELECT 1"},
    ],
)
def test_source_declaration_rejects_ambiguous_or_unsafe_input(source: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        VocabularySource.model_validate(source)


def test_inline_and_external_terms_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="use exactly one"):
        Vocabulary.model_validate(
            {
                "terms": [{"label": "a"}],
                "from": {"path": "s.csv", "label_column": "label"},
            }
        )


@pytest.mark.parametrize("backend", [False, True])
def test_csv_values_constrain_both_enum_surfaces(tmp_path: Path, backend: bool) -> None:
    """#718: one seed reaches both native llm fields and backend: llm schemas."""
    _project(
        tmp_path,
        {"path": "sectors.csv", "label_column": "id", "description_column": "name"},
        backend=backend,
        extra_terms=[{"label": "none", "description": "No listed sector"}],
    )
    (tmp_path / "sectors.csv").write_text("id,name\nz,Zèbre\na,Alpha\n", encoding="utf-8")
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    model = models[0]
    if backend:
        assert model.extraction is not None
        schema = _input_schema(_fields_spec(model.extraction.options))
        assert schema["properties"]["sector"]["enum"] == ["a", "z", "none"]
    else:
        assert build_fields_spec(model.fields)[0]["enum"] == ["a", "z", "none"]
        assert model.fields[0].value_descriptions["z"] == "Zèbre"
    assert project.vocabularies["sector"].model_dump() == {
        "terms": [term.model_dump() for term in project.vocabularies["sector"].terms],
    }


@pytest.mark.parametrize(
    ("csv_text", "options", "message"),
    [
        ("id\na\n", {"label_column": "missing"}, "no configured column"),
        ("id\na\n", {"description_column": "missing"}, "no configured column"),
        ("id\n", {}, "non-empty"),
        ("id\nSENSITIVE\nSENSITIVE\n", {}, "unique"),
        ("id\n  \n", {}, "non-empty string"),
        ("id\na\nb\n", {"max_terms": 1}, "max_terms"),
        ("id,id\na,b\n", {}, "duplicate columns"),
        ("id\na,extra\n", {}, "does not match"),
        ('id\n"unterminated\n', {}, "cannot read UTF-8 CSV"),
    ],
)
def test_invalid_seed_errors_do_not_include_rows(
    tmp_path: Path,
    csv_text: str,
    options: dict[str, Any],
    message: str,
) -> None:
    _project(tmp_path, {"path": "s.csv", "label_column": "id", **options})
    (tmp_path / "s.csv").write_text(csv_text, encoding="utf-8")
    with pytest.raises(ConfigError, match=message) as caught:
        load_project(tmp_path)
    assert "SENSITIVE" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("backend", [False, True])
def test_seed_reorder_preserves_identity_but_label_changes_invalidate(
    tmp_path: Path,
    backend: bool,
) -> None:
    _project(tmp_path, {"path": "s.csv", "label_column": "id"}, backend=backend)
    seed = tmp_path / "s.csv"
    versions = []
    for text in ("id\na\nb\n", "id\nb\na\n", "id\na\nb\nc\n"):
        seed.write_text(text, encoding="utf-8")
        project, _, models = load_project(tmp_path)
        model = models[0]
        if not backend:
            model = model.model_copy(
                update={
                    "extraction": None,
                    "llm": LLMTransformConfig(
                        provider="deterministic",
                        input_field="text",
                        prompt="Choose a sector",
                    ),
                }
            )
        versions.append(compute_model_code_version(model, project, tmp_path))
    assert versions[0] == versions[1]
    assert versions[1] != versions[2]


def test_seed_cannot_escape_the_project(tmp_path: Path) -> None:
    _project(tmp_path, {"path": "../outside.csv", "label_column": "id"})
    with pytest.raises(ConfigError, match="outside the project"):
        load_project(tmp_path)


@pytest.mark.parametrize("directory", [False, True])
def test_seed_rejects_symlinks_even_inside_project(tmp_path: Path, directory: bool) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "s.csv").write_text("id\na\n", encoding="utf-8")
    link = tmp_path / ("linked" if directory else "linked.csv")
    try:
        link.symlink_to(real if directory else real / "s.csv", target_is_directory=directory)
    except OSError:
        pytest.skip("symlinks require privileges on Windows")
    _project(
        tmp_path, {"path": "linked/s.csv" if directory else "linked.csv", "label_column": "id"}
    )
    with pytest.raises(ConfigError, match="symlink"):
        load_project(tmp_path)


@pytest.mark.parametrize("backend", [False, True])
def test_warehouse_resolution_is_deferred_and_constrains_provider_schema(
    tmp_path: Path,
    backend: bool,
) -> None:
    _project(
        tmp_path, {"relation": "warehouse://main.sectors", "label_column": "id"}, backend=backend
    )
    project, sources, models = load_project(tmp_path)
    validate_project_contract(project, sources, models, tmp_path)
    # Static validation never replaces references with an executable fake enum.
    if backend:
        assert models[0].extraction is not None
        assert models[0].extraction.options["fields"][0]["values_from"] == "vocab.sector"
        with pytest.raises(ValueError, match="unresolved vocabulary"):
            _fields_spec(models[0].extraction.options)
    else:
        with pytest.raises(LLMMapError, match="unresolved"):
            build_fields_spec(models[0].fields)
    with pytest.raises(ConfigError, match="unresolved"):
        compute_model_code_version(models[0], project, tmp_path)
    assert not (tmp_path / "warehouse.duckdb").exists()
    with duckdb.connect(str(tmp_path / "warehouse.duckdb")) as con:
        con.execute("CREATE TABLE sectors AS SELECT 'z' AS id UNION ALL SELECT 'a'")
    resolved = resolve_profile(project, tmp_path)
    adapter = create_adapter(resolved.warehouse, project_dir=tmp_path)
    resolve_warehouse_vocabularies(project, sources, models, tmp_path, adapter)
    if backend:
        assert models[0].extraction is not None
        schema = _input_schema(_fields_spec(models[0].extraction.options))
        assert schema["properties"]["sector"]["enum"] == ["a", "z"]
    else:
        assert build_fields_spec(models[0].fields)[0]["enum"] == ["a", "z"]
    with duckdb.connect(str(tmp_path / "warehouse.duckdb")) as con:
        assert (
            con.execute(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name='stel'"
            ).fetchall()
            == []
        )


def test_manifest_contains_resolved_values_not_external_source_instructions(tmp_path: Path) -> None:
    _project(tmp_path, {"relation": "warehouse://main.sectors", "label_column": "id"})
    with duckdb.connect(str(tmp_path / "warehouse.duckdb")) as con:
        con.execute("CREATE TABLE sectors AS SELECT 'a' AS id")
    manifest = build_manifest(tmp_path)
    assert manifest["models"][0]["fields"][0]["values"] == ["a"]
    assert "warehouse://" not in json.dumps(manifest)


@pytest.mark.parametrize("operation", ["export", "codegen", "embed"])
def test_offline_consumers_reject_unresolved_warehouse_declarations(
    tmp_path: Path,
    operation: str,
) -> None:
    _project(tmp_path, {"relation": "warehouse://main.sectors", "label_column": "id"})
    with pytest.raises(ConfigError, match="CSV seed or inline terms"):
        if operation == "export":
            build_dbt_sources(tmp_path)
        elif operation == "codegen":
            generate_dbt_models(tmp_path, tmp_path / "generated")
        else:
            materialize("classified", project_dir=tmp_path)
    assert not (tmp_path / "warehouse.duckdb").exists()
    assert not (tmp_path / "generated").exists()


def test_duplicate_extra_terms_fail_before_warehouse_reads(tmp_path: Path) -> None:
    _project(
        tmp_path,
        {"relation": "warehouse://main.sectors", "label_column": "id"},
        extra_terms=[{"label": "a"}, {"label": "a"}],
    )
    with pytest.raises(ConfigError, match=r"extra_terms.*unique"):
        load_project(tmp_path)
    assert not (tmp_path / "warehouse.duckdb").exists()


def test_warehouse_vocabulary_uses_the_selected_profile_target(tmp_path: Path) -> None:
    _project(tmp_path, {"relation": "warehouse://main.sectors", "label_column": "id"})
    project_path = tmp_path / "stel_project.yml"
    project = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    project["profile"] = "vocab"
    project_path.write_text(yaml.safe_dump(project), encoding="utf-8")
    outputs = {}
    for name in ("dev", "other"):
        outputs[name] = {"warehouse": {"type": "duckdb", "path": f"{name}.duckdb"}}
        with duckdb.connect(str(tmp_path / f"{name}.duckdb")) as con:
            con.execute("CREATE TABLE sectors AS SELECT ? AS id", [name])
    (tmp_path / "profiles.yml").write_text(
        yaml.safe_dump({"vocab": {"target": "dev", "outputs": outputs}}),
        encoding="utf-8",
    )
    manifest = build_manifest(tmp_path, target="other")
    assert manifest["models"][0]["fields"][0]["values"] == ["other"]
