from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
from click.testing import CliRunner
from pydantic import BaseModel, ValidationError

from stel.cli import cli
from stel.config import ConfigError, load_project
from stel.config import model as model_config_module
from stel.config import profile as profile_config_module
from stel.config import project as project_config_module
from stel.config import source as source_config_module
from stel.config.model import ExtractionConfig, FieldConfig, ModelConfig, ModelFile
from stel.credentials import CredentialReference


def test_load_example_project(example_project_dir: Path) -> None:
    project, sources, models = load_project(example_project_dir)
    assert project.name == "invoice_pipeline"
    assert project.duckdb.schema_name == "stel"
    assert project.extraction.default_backend == "json"
    assert {s.name for s in sources} == {"vendor_invoices"}
    assert {m.name for m in models} == {"raw_invoices", "invoice_summary", "monthly_totals"}


def test_missing_project_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"stel_project\.yml"):
        load_project(tmp_path)


def test_missing_project_file_names_the_pre_rename_one_next_to_it(
    tmp_path: Path,
) -> None:
    # The first error an upgrading project hits (#324). Every other #313
    # hazard explains itself; this one used to just say the file was absent.
    (tmp_path / "dbt_ml_project.yml").write_text("name: x")

    with pytest.raises(ConfigError) as excinfo:
        load_project(tmp_path)

    message = str(excinfo.value)
    assert "dbt_ml_project.yml" in message
    assert "git mv dbt_ml_project.yml stel_project.yml" in message
    assert "stel migrate" in message


def test_legacy_project_file_is_reported_but_never_loaded(tmp_path: Path) -> None:
    # Detection only: two filenames that both work is how the old one never
    # dies, so the legacy file's contents must not reach the config.
    # A sentinel that cannot collide with the tmp_path directory name.
    (tmp_path / "dbt_ml_project.yml").write_text("name: n0tth3pr0j3ctnam3")

    with pytest.raises(ConfigError) as excinfo:
        load_project(tmp_path)

    assert "n0tth3pr0j3ctnam3" not in str(excinfo.value)


def test_missing_project_file_without_a_legacy_one_stays_plain(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_project(tmp_path)

    message = str(excinfo.value)
    assert "stel_project.yml" in message
    assert "dbt_ml_project.yml" not in message
    assert "git mv" not in message


def test_invalid_yaml_reports_path(tmp_path: Path) -> None:
    (tmp_path / "stel_project.yml").write_text("name: x\n")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "bad.yml").write_text(
        "version: 2\nsources:\n  - description: 'missing required name'\n"
    )
    with pytest.raises(ConfigError, match=r"bad\.yml"):
        load_project(tmp_path)


def test_raw_invoices_is_incremental(example_project_dir: Path) -> None:
    _, _, models = load_project(example_project_dir)
    raw = next(m for m in models if m.name == "raw_invoices")
    assert raw.materialization == "incremental"
    assert raw.extraction is not None
    assert raw.extraction.backend == "json"
    assert raw.extraction.options["fields"] == [
        "invoice_id",
        "vendor",
        "issue_date",
        "line_items",
        "total",
        "currency",
    ]


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("data_type", "integer", "integer"),
        ("data-type", "bool", "boolean"),
        ("type", "number", "float"),
        ("dtype", "varchar", "string"),
    ],
)
def test_field_data_type_aliases(
    key: str, value: str, expected: str
) -> None:
    field = FieldConfig.model_validate({"name": "value", key: value})
    assert field.data_type == expected
    assert field.model_dump()["data_type"] == expected


def test_field_rejects_unknown_data_type() -> None:
    with pytest.raises(ValueError, match="data_type"):
        FieldConfig(name="value", data_type="uuid")


def test_invoice_summary_depends_on_raw(example_project_dir: Path) -> None:
    _, _, models = load_project(example_project_dir)
    summary = next(m for m in models if m.name == "invoice_summary")
    assert summary.materialization == "full"
    assert summary.depends_on == ["ref('raw_invoices')"]
    assert summary.transform is not None
    assert summary.transform.module == "transforms.summarize"


def test_loads_classic_ml_model_config(tmp_path: Path) -> None:
    (tmp_path / "stel_project.yml").write_text(
        "\n".join(
            [
                "name: classic_ml_project",
                "version: '0.1.0'",
                "source-paths: ['sources']",
                "model-paths: ['models']",
            ]
        )
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "tickets.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "sources:",
                "  - name: tickets",
                "    path: data/tickets",
            ]
        )
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "ticket_features.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_tfidf",
                "    depends_on: [ref('raw_tickets')]",
                "    ml:",
                "      task: features",
                "      mode: fit_transform",
                "      provider: builtin.tfidf",
                "      text_field: body",
                "      artifact:",
                "        path: target/artifacts/ticket_tfidf",
                "      metrics: [vocabulary_size]",
                "      options:",
                "        ngram_range: [1, 2]",
                "        max_features: 50000",
            ]
        )
    )

    _, _, models = load_project(tmp_path)
    ml_model = models[0]
    assert ml_model.ml is not None
    assert ml_model.ml.task == "features"
    assert ml_model.ml.mode == "fit_transform"
    assert ml_model.ml.provider == "builtin.tfidf"
    assert ml_model.ml.text_field == "body"
    assert ml_model.ml.artifact.path == Path("target/artifacts/ticket_tfidf")
    assert ml_model.ml.metrics == ["vocabulary_size"]
    assert ml_model.ml.options["max_features"] == 50000


def test_multiple_kind_blocks_rejected() -> None:
    with pytest.raises(ValueError, match="multiple kind blocks"):
        ModelConfig(
            name="conflicted",
            extraction={"backend": "json"},
            transform={"type": "python", "module": "transforms.x"},
        )


def test_update_when_changed_requires_incremental() -> None:
    with pytest.raises(ValueError, match="requires `materialization: incremental`"):
        ModelConfig(
            name="doc_facts",
            extraction={"backend": "json"},
            materialization="full",
            update_when_changed=["content_hash"],
        )


def test_update_when_changed_rejects_invalid_identifier() -> None:
    with pytest.raises(ValueError, match="must be a valid identifier"):
        ModelConfig(
            name="doc_facts",
            extraction={"backend": "json"},
            materialization="incremental",
            update_when_changed=["not a column"],
        )


def test_update_when_changed_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate column 'content_hash'"):
        ModelConfig(
            name="doc_facts",
            extraction={"backend": "json"},
            materialization="incremental",
            update_when_changed=["content_hash", "content_hash"],
        )


def test_update_when_changed_accepts_valid_incremental() -> None:
    model = ModelConfig(
        name="doc_facts",
        extraction={"backend": "json"},
        materialization="incremental",
        update_when_changed=["content_hash", "code_version"],
    )
    assert model.update_when_changed == ["content_hash", "code_version"]


def test_model_file_requires_kind_block() -> None:
    # The list of options is derived from ModelKind rather than written out
    # here, which is how it stopped omitting `search` and `eval` (issue #494).
    with pytest.raises(
        ValueError,
        match=r"missing a kind block \(extraction/.*?/eval\)",
    ):
        ModelFile.model_validate(
            {"version": 2, "models": [{"name": "kindless"}]}
        )


def _llm_model(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "obligations",
        "depends_on": ['ref("chunks")'],
        "llm": {"input_field": "text", "id_field": "chunk_id", "prompt": "extract"},
        "fields": [{"name": "obligation", "type": "string"}],
    }
    base.update(overrides)
    return base


def test_llm_model_validates() -> None:
    model = ModelConfig.model_validate(_llm_model())
    assert model.llm is not None
    assert model.llm.mode == "map"
    assert model.llm.output_cardinality == "one"
    assert model.kind_block_count == 1


def test_llm_model_requires_fields() -> None:
    with pytest.raises(ValidationError, match="requires `fields:`"):
        ModelConfig.model_validate(_llm_model(fields=[]))


def test_llm_model_field_collides_with_metadata() -> None:
    with pytest.raises(ValidationError, match="collide with generated columns"):
        ModelConfig.model_validate(
            _llm_model(fields=[{"name": "llm_provider", "type": "string"}])
        )


def test_llm_model_field_collides_with_id_field() -> None:
    with pytest.raises(ValidationError, match="collide with generated columns"):
        ModelConfig.model_validate(
            _llm_model(fields=[{"name": "chunk_id", "type": "string"}])
        )


def test_llm_config_reserved_id_field() -> None:
    with pytest.raises(ValidationError, match="reserved for generation metadata"):
        ModelConfig.model_validate(
            _llm_model(
                llm={
                    "input_field": "text",
                    "id_field": "generated_at",
                    "prompt": "extract",
                }
            )
        )


def test_llm_model_conflicts_with_other_kind() -> None:
    with pytest.raises(ValidationError, match="multiple kind blocks"):
        ModelConfig.model_validate(
            _llm_model(transform={"type": "python", "module": "transforms.x"})
        )


def test_bare_model_config_allowed_programmatically() -> None:
    # DAG fixtures and docs tooling build ModelConfig directly without a
    # kind block; only the YAML load path requires one.
    assert ModelConfig(name="fixture_only").kind_block_count == 0


def test_kindless_model_fails_at_load(tmp_path: Path) -> None:
    (tmp_path / "stel_project.yml").write_text("name: p\n")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.yml").write_text(
        "version: 2\nmodels:\n  - name: no_kind\n"
    )
    with pytest.raises(ConfigError, match="no_kind"):
        load_project(tmp_path)


def _public_config_models(module: ModuleType) -> list[type[BaseModel]]:
    return [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, BaseModel)
        and value.__module__ == module.__name__
    ]


def test_all_public_config_models_forbid_unknown_fields() -> None:
    modules = (
        model_config_module,
        profile_config_module,
        project_config_module,
        source_config_module,
    )
    models = [model for module in modules for model in _public_config_models(module)]

    assert models
    assert all(model.model_config.get("extra") == "forbid" for model in models)


def _compile_fixture(tmp_path: Path, model_yaml: str) -> Path:
    (tmp_path / "stel_project.yml").write_text("name: strict_project\n")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "strict.yml").write_text(model_yaml)
    return tmp_path


def test_compile_unknown_model_key_is_precise_exit_2(tmp_path: Path) -> None:
    project = _compile_fixture(
        tmp_path,
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n"
        "    extraction:\n      backend: json\n"
        "    materializtion: incremental\n",
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "compile"])

    assert result.exit_code == 2, result.output
    assert "strict.yml" in result.output
    assert "strict.yml:7:5" in result.output
    assert "models.0.materializtion" in result.output
    assert "Extra inputs are not permitted" in result.output
    assert not (project / "target" / "manifest.json").exists()


def test_compile_unknown_source_key_is_precise_exit_2(tmp_path: Path) -> None:
    (tmp_path / "stel_project.yml").write_text("name: strict_project\n")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data\n"
        "    file_patern: '*.json'\n"
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(tmp_path), "compile"])

    assert result.exit_code == 2, result.output
    assert "docs.yml" in result.output
    assert "docs.yml:5:5" in result.output
    assert "sources.0.file_patern" in result.output


def test_nested_validation_error_reports_value_location(tmp_path: Path) -> None:
    project = _compile_fixture(
        tmp_path,
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    extraction:\n      backend: json\n      flush_every: 0\n",
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "compile"])

    assert result.exit_code == 2, result.output
    assert "strict.yml:6:20" in result.output
    assert "models.0.extraction.flush_every" in result.output
    assert "Input should be greater than 0" in result.output


def test_missing_field_reports_parent_location_and_full_path(tmp_path: Path) -> None:
    (tmp_path / "stel_project.yml").write_text("name: located_project\n")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "missing.yml").write_text(
        "version: 2\nsources:\n  - path: data\n"
    )

    with pytest.raises(ConfigError) as exc_info:
        load_project(tmp_path)

    message = str(exc_info.value)
    assert "missing.yml:3:5" in message
    assert "sources.0.name" in message
    assert "Field required" in message


def test_validation_diagnostic_does_not_echo_invalid_input(tmp_path: Path) -> None:
    secret = "distinctive-invalid-input-secret"
    project = _compile_fixture(
        tmp_path,
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    extraction:\n      backend: json\n"
        f"    materializtion: {secret}\n",
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "compile"])

    assert result.exit_code == 2, result.output
    assert "strict.yml:6:5" in result.output
    assert "models.0.materializtion" in result.output
    assert secret not in result.output


def test_malformed_yaml_reports_one_based_location_without_source_line(
    tmp_path: Path,
) -> None:
    secret = "distinctive-malformed-secret"
    (tmp_path / "stel_project.yml").write_text(
        f"name: project\ninvalid: [{secret}\n"
    )

    with pytest.raises(ConfigError) as exc_info:
        load_project(tmp_path)

    message = str(exc_info.value)
    assert "stel_project.yml:3:1" in message
    assert "expected ',' or ']'" in message
    assert secret not in message


@pytest.mark.parametrize(
    ("directory", "filename", "contents", "expected_location"),
    [
        (
            None,
            "stel_project.yml",
            "name: first\nname: distinctive-duplicate-secret\n",
            "stel_project.yml:2:1 [name]",
        ),
        (
            "sources",
            "duplicate.yml",
            "version: 2\nsources:\n  - name: first\n"
            "    name: distinctive-duplicate-secret\n    path: data\n",
            "duplicate.yml:4:5 [sources.0.name]",
        ),
        (
            "models",
            "duplicate.yml",
            "version: 2\nmodels:\n  - name: first\n"
            "    name: distinctive-duplicate-secret\n"
            "    extraction:\n      backend: json\n",
            "duplicate.yml:4:5 [models.0.name]",
        ),
    ],
)
def test_explicit_duplicate_mapping_keys_are_rejected_at_second_key(
    tmp_path: Path,
    directory: str | None,
    filename: str,
    contents: str,
    expected_location: str,
) -> None:
    if directory is None:
        path = tmp_path / filename
    else:
        (tmp_path / "stel_project.yml").write_text("name: duplicate_project\n")
        path = tmp_path / directory / filename
        path.parent.mkdir()
    path.write_text(contents)

    with pytest.raises(ConfigError) as exc_info:
        load_project(tmp_path)

    message = str(exc_info.value)
    assert expected_location in message
    assert "duplicate mapping key" in message
    assert "distinctive-duplicate-secret" not in message


def test_yaml_merge_key_can_be_overridden_explicitly(tmp_path: Path) -> None:
    (tmp_path / "stel_project.yml").write_text(
        "<<: &defaults\n  name: inherited\n  version: '0.1.0'\n"
        "name: explicit\n"
    )

    project, _, _ = load_project(tmp_path)

    assert project.name == "explicit"


def test_duplicate_merge_directive_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "stel_project.yml").write_text(
        "<<: &identity\n  name: inherited\n"
        "<<: &release\n  version: '0.1.0'\n"
    )

    with pytest.raises(ConfigError) as exc_info:
        load_project(tmp_path)

    message = str(exc_info.value)
    assert "stel_project.yml:3:1 [<<]" in message
    assert "duplicate mapping key" in message


@pytest.mark.parametrize("contents", ["[]\n", "0\n", "false\n"])
def test_falsy_non_mapping_project_document_is_not_treated_as_empty(
    tmp_path: Path,
    contents: str,
) -> None:
    (tmp_path / "stel_project.yml").write_text(contents)

    with pytest.raises(ConfigError) as exc_info:
        load_project(tmp_path)

    message = str(exc_info.value)
    assert "stel_project.yml:1:1 [<root>]" in message
    assert "Input should be a valid dictionary" in message


@pytest.mark.parametrize("directory", ["sources", "models"])
def test_falsy_yaml_file_document_reaches_its_file_schema(
    tmp_path: Path,
    directory: str,
) -> None:
    (tmp_path / "stel_project.yml").write_text("name: falsy_file_project\n")
    config_dir = tmp_path / directory
    config_dir.mkdir()
    config_path = config_dir / "falsy.yml"
    config_path.write_text("[]\n")

    with pytest.raises(ConfigError) as exc_info:
        load_project(tmp_path)

    message = str(exc_info.value)
    assert "falsy.yml:1:1 [<root>]" in message
    assert "Input should be a valid dictionary" in message


def test_loaded_project_exposes_excluded_yaml_provenance(tmp_path: Path) -> None:
    project_path = tmp_path / "stel_project.yml"
    project_path.write_text(
        "name: provenance_project\n"
        "extraction:\n  default_backend: json\n"
    )

    project, _, _ = load_project(tmp_path)

    provenance = project.yaml_provenance
    assert provenance is not None
    assert provenance.file_path == project_path.resolve()
    assert provenance.config_path == ()
    assert project.format_yaml_diagnostic(
        "backend is unavailable",
        relative_path=("extraction", "default_backend"),
    ) == (
        f"{project_path.resolve()}:3:20 "
        "[extraction.default_backend] backend is unavailable"
    )
    assert "yaml_provenance" not in project.model_dump()
    assert "yaml_provenance" not in project.model_dump_json()


def test_loaded_model_exposes_excluded_yaml_provenance(tmp_path: Path) -> None:
    (tmp_path / "stel_project.yml").write_text("name: provenance_project\n")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "located.yml"
    model_path.write_text(
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    extraction:\n      backend: json\n"
    )

    _, _, models = load_project(tmp_path)

    model = models[0]
    provenance = model.yaml_provenance
    assert provenance is not None
    assert provenance.file_path == model_path.resolve()
    assert provenance.config_path == ("models", 0)
    assert model.format_yaml_diagnostic(
        "backend is unavailable",
        relative_path=("extraction", "backend"),
    ) == (
        f"{model_path.resolve()}:5:16 "
        "[models.0.extraction.backend] backend is unavailable"
    )
    assert model.format_yaml_diagnostic(
        "source is required",
        relative_path=("source",),
    ) == (
        f"{model_path.resolve()}:3:5 [models.0.source] source is required"
    )
    assert "yaml_provenance" not in model.model_dump()
    assert "yaml_provenance" not in model.model_dump_json()


@pytest.mark.parametrize("directory, root_key", [("sources", "sources"), ("models", "models")])
def test_yaml_schema_version_other_than_2_is_rejected(
    tmp_path: Path, directory: str, root_key: str
) -> None:
    (tmp_path / "stel_project.yml").write_text("name: versioned\n")
    path = tmp_path / directory
    path.mkdir()
    if root_key == "sources":
        item = "  - name: docs\n    path: data\n"
    else:
        item = "  - name: raw_docs\n    extraction:\n      backend: json\n"
    (path / "bad_version.yml").write_text(
        f"version: 1\n{root_key}:\n{item}"
    )

    with pytest.raises(ConfigError, match=r"(?s)version.*Input should be 2"):
        load_project(tmp_path)


@pytest.mark.parametrize(
    "options",
    [
        {"fields": ["document_id"]},
        {"fields": [{"name": "source_uri", "type": "string"}]},
        {"frontmatter_fields": ["backend_name"]},
        {"text_field": "CONTENT_HASH"},
        {"body_field": "code_version"},
        {"selectors": {"extracted_at": "h1"}},
    ],
)
def test_extraction_options_reject_reserved_lineage_fields(
    options: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="reserved stel lineage columns"):
        ExtractionConfig(backend="llm", options=options)


def test_compile_reserved_extraction_field_is_exit_2(tmp_path: Path) -> None:
    project = _compile_fixture(
        tmp_path,
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n"
        "    extraction:\n      backend: llm\n      options:\n"
        "        fields:\n          - name: document_id\n            type: string\n",
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "compile"])

    assert result.exit_code == 2, result.output
    assert "strict.yml" in result.output
    assert "document_id" in result.output
    assert "reserved stel lineage columns" in result.output


def test_model_cannot_select_operator_credential_environment_variable(
    tmp_path: Path,
) -> None:
    project = _compile_fixture(
        tmp_path,
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n"
        "    extraction:\n      backend: llm\n      options:\n"
        "        api_key_env: GITHUB_TOKEN\n"
        "        fields:\n          - name: title\n            type: string\n",
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "compile"])

    assert result.exit_code == 2, result.output
    assert "operator-owned configuration" in result.output
    assert "profiles.yml" in result.output
    assert "GITHUB_TOKEN" not in result.output


def test_llm_model_credential_reference_is_opaque_before_compile() -> None:
    reference_name = "PRIVATE_MODEL_LLM_REFERENCE"
    config = ModelConfig(
        name="raw_docs",
        source="ref('docs')",
        extraction={
            "backend": "llm",
            "options": {
                "api_key_env": reference_name,
                "fields": [{"name": "title"}],
            },
        },
    )

    assert config.extraction is not None
    assert isinstance(
        config.extraction.options["api_key_env"], CredentialReference
    )
    rendered = repr(config) + repr(config.model_dump()) + config.model_dump_json()
    assert reference_name not in rendered


def test_invalid_model_credential_input_is_cleared_before_validation_error() -> None:
    sentinel = "literal-model-credential-secret"

    with pytest.raises(ValidationError) as exc_info:
        ModelConfig.model_validate(
            {
                "name": "raw_docs",
                "source": "ref('docs')",
                "extraction": {
                    "backend": "llm",
                    "options": {"api_key_env": sentinel},
                },
            }
        )

    error = exc_info.value
    rendered = "\n".join(
        (str(error), repr(error), repr(error.errors()), error.json())
    )
    assert sentinel not in rendered


def test_extraction_config_defers_backend_owned_option_names() -> None:
    options = {
        "api_key_env": "CUSTOM_CONFIG_VALUE",
        "temperature": -10.0,
        "max_tokens": 0,
    }

    config = ExtractionConfig(backend="custom", options=options)
    defaulted_config = ExtractionConfig(options=options)

    assert config.options == options
    assert defaulted_config.options == options


def test_publish_every_defaults_to_one_and_rejects_non_positive() -> None:
    """publish_every coalesces flushes into one upsert (issue #293); it defaults
    to per-flush publication and must be a positive flush count."""
    assert ExtractionConfig(backend="json").publish_every == 1
    assert ExtractionConfig(backend="json", publish_every=20).publish_every == 20
    with pytest.raises(ValidationError):
        ExtractionConfig(backend="json", publish_every=0)
    with pytest.raises(ValidationError):
        ExtractionConfig(backend="json", publish_every=-1)


def test_default_llm_model_is_protected_before_later_yaml_error(
    tmp_path: Path,
) -> None:
    reference_name = "EARLY_LOADED_DEFAULT_LLM_REFERENCE_SENTINEL_154"
    (tmp_path / "stel_project.yml").write_text(
        "name: p\nextraction:\n  default_backend: llm\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "a_valid.yml").write_text(
        "version: 2\nmodels:\n  - name: first\n"
        "    extraction:\n      options:\n"
        f"        api_key_env: {reference_name}\n"
    )
    (tmp_path / "models" / "b_invalid.yml").write_text(
        "version: 2\nmodels:\n  - name: invalid\n"
        "    extraction: {}\n    unexpected: true\n"
    )

    with pytest.raises(ConfigError) as exc_info:
        load_project(tmp_path)

    error = exc_info.value
    assert reference_name not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module, str) and module.startswith("stel"):
            assert reference_name not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def test_invalid_llm_option_fails_before_gcs_source_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "stel_project.yml").write_text(
        "name: llm_project\nextraction:\n  default_backend: llm\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: gs://bucket/docs\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "raw.yml").write_text(
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n"
        "    extraction:\n      options:\n        max_concurrent: 0\n"
    )
    discovered = False

    def _unexpected_client(*args: object, **kwargs: object) -> None:
        nonlocal discovered
        discovered = True
        raise AssertionError("GCS discovery must not run")

    monkeypatch.setattr(
        "stel.sources.gcs.GCSDocumentSource._make_client", _unexpected_client
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(tmp_path), "run"])

    assert result.exit_code == 2, result.output
    assert "max_concurrent" in result.output
    assert not discovered
