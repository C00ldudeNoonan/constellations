from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from pydantic import BaseModel, ConfigDict

from stel.backends import (
    BackendOptionsError,
    BaseBackend,
    ExtractionResult,
    get_backend,
    get_backend_option_contract,
    list_backend_option_contracts,
    list_backends,
    register,
    register_backend_option_contract,
    validate_backend_options,
)
from stel.cli import cli
from stel.credentials import CredentialReference


def test_option_contracts_cover_every_shipped_backend() -> None:
    assert list_backend_option_contracts() == list_backends()
    assert get_backend_option_contract("llm").native_batch
    assert get_backend_option_contract("llm").requires_credentials
    assert not get_backend_option_contract("json").native_batch


def test_shipped_backends_expose_distinct_implementation_identities() -> None:
    identities = {
        get_backend(name).implementation_identity() for name in list_backends()
    }

    assert len(identities) == len(list_backends())
    assert all(identity.startswith("stel/") for identity in identities)


@pytest.mark.parametrize(
    ("backend", "options"),
    [
        ("json", {"fields": ["id", "payload"]}),
        (
            "markdown",
            {
                "frontmatter_fields": ["title", "author"],
                "include_body": False,
                "compute_word_count": True,
            },
        ),
        (
            "pdf",
            {
                "text_field": "content",
                "include_text": False,
                "include_page_count": False,
                "include_metadata": True,
                "include_pages": True,
                "page_separator": "\n---\n",
            },
        ),
        (
            "html",
            {
                "text_field": "content",
                "include_text": False,
                "include_structure": True,
                "heading_selectors": [".title", ".subtitle"],
                "styled_headings": True,
                "selectors": {"author": ".byline"},
                "include_meta": True,
                "include_opengraph": True,
                "include_links": True,
                "parser": "html.parser",
            },
        ),
        (
            "email",
            {
                "include_body": False,
                "body_field": "content",
                "include_html": True,
                "include_headers": True,
            },
        ),
        (
            "llm",
            {
                "fields": [{"name": "title", "type": "string"}],
                "model": "claude-test",
                "system_prompt": "Extract the title.",
                "cache_path": "target/cache.duckdb",
                "temperature": 0.5,
                "max_tokens": 1024,
                "max_retries": 2,
                "max_concurrent": 8,
                "api_key_env": "TEST_ANTHROPIC_KEY",
                "batch": True,
                "batch_poll_seconds": 1.5,
            },
        ),
    ],
)
def test_documented_backend_option_shapes_are_valid(
    backend: str, options: dict[str, object]
) -> None:
    validated = validate_backend_options(backend, options)
    if backend != "llm":
        assert validated == options
        return

    reference = validated.pop("api_key_env")
    expected = dict(options)
    expected.pop("api_key_env")
    assert validated == expected
    assert isinstance(reference, CredentialReference)
    assert "TEST_ANTHROPIC_KEY" not in repr(reference)


def test_llm_credential_reference_is_opaque_but_survives_runtime_validation(
) -> None:
    raw_reference = "PRIVATE_LLM_CREDENTIAL"
    parsed = get_backend_option_contract("llm").options_model.model_validate(
        {
            "fields": [{"name": "title", "type": "string"}],
            "api_key_env": raw_reference,
        }
    )

    assert raw_reference not in repr(parsed)
    assert raw_reference not in parsed.model_dump()
    assert raw_reference not in parsed.model_dump_json()

    validated = validate_backend_options(
        "llm",
        {
            "fields": [{"name": "title", "type": "string"}],
            "api_key_env": raw_reference,
        },
    )
    assert isinstance(validated["api_key_env"], CredentialReference)
    assert raw_reference not in repr(validated)


def test_backend_validation_error_drops_credential_reference_input() -> None:
    raw_reference = "PRIVATE_INVALID_LLM_OPTIONS_REFERENCE"

    with pytest.raises(BackendOptionsError) as exc_info:
        validate_backend_options(
            "llm",
            {"fields": [], "api_key_env": raw_reference},
        )

    error = exc_info.value
    assert raw_reference not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module, str) and module.startswith("stel"):
            assert raw_reference not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


@pytest.mark.parametrize("alias", ["data_type", "data-type", "dtype"])
def test_llm_field_type_aliases_canonicalize(alias: str) -> None:
    options = validate_backend_options(
        "llm",
        {
            "fields": [
                {
                    "name": "tags",
                    alias: "array",
                    "items": {"type": "string", "minLength": 1},
                }
            ]
        },
    )

    assert options == {
        "fields": [
            {
                "name": "tags",
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            }
        ]
    }


@pytest.mark.parametrize(
    ("backend", "options", "location"),
    [
        ("json", {"fieldz": ["id"]}, "fieldz"),
        ("markdown", {"include_body": "false"}, "include_body"),
        ("pdf", {"include_pages": 1}, "include_pages"),
        ("html", {"parser": "html5lib"}, "parser"),
        ("email", {"body_field": ""}, "body_field"),
        ("llm", {"fields": []}, "fields"),
    ],
)
def test_invalid_backend_options_report_the_option_path_without_values(
    backend: str, options: dict[str, object], location: str
) -> None:
    with pytest.raises(BackendOptionsError) as exc_info:
        validate_backend_options(backend, options)

    message = str(exc_info.value)
    assert f"backend '{backend}'" in message
    assert location in message
    assert repr(options.get(location)) not in message


def test_unknown_option_error_does_not_echo_a_secret_value() -> None:
    secret = "sk-ant-must-not-appear"

    with pytest.raises(BackendOptionsError) as exc_info:
        validate_backend_options(
            "llm",
            {
                "fields": [{"name": "title", "type": "string"}],
                "api_key": secret,
            },
        )

    assert "api_key" in str(exc_info.value)
    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("temperature", -0.1),
        ("temperature", float("nan")),
        ("max_tokens", 0),
        ("max_tokens", 65_537),
        ("max_retries", 21),
        ("max_concurrent", 0),
        ("max_concurrent", True),
        ("batch_poll_seconds", 0),
        ("batch_poll_seconds", 3601),
    ],
)
def test_llm_numeric_options_are_bounded(name: str, value: object) -> None:
    with pytest.raises(BackendOptionsError, match=name):
        validate_backend_options(
            "llm",
            {
                "fields": [{"name": "title", "type": "string"}],
                name: value,
            },
        )


def test_llm_fields_must_be_unique_case_insensitively() -> None:
    with pytest.raises(BackendOptionsError, match="unique"):
        validate_backend_options(
            "llm",
            {
                "fields": [
                    {"name": "Title", "type": "string"},
                    {"name": "title", "type": "string"},
                ]
            },
        )


def test_llm_items_require_an_array_field() -> None:
    with pytest.raises(BackendOptionsError, match="items"):
        validate_backend_options(
            "llm",
            {
                "fields": [
                    {
                        "name": "title",
                        "type": "string",
                        "items": {"type": "string"},
                    }
                ]
            },
        )


@pytest.mark.parametrize("item_type", [["string"], {"type": "string"}, None, 1])
def test_llm_items_type_is_always_validated_as_a_string(item_type: object) -> None:
    with pytest.raises(BackendOptionsError, match=r"items\.type"):
        validate_backend_options(
            "llm",
            {
                "fields": [
                    {
                        "name": "tags",
                        "type": "array",
                        "items": {"type": item_type},
                    }
                ]
            },
        )


@pytest.mark.parametrize(
    "options",
    [
        {"heading_selectors": ["div["]},
        {"selectors": {"title": "h1>>p"}},
    ],
)
def test_html_css_selectors_are_validated_during_option_preflight(
    options: dict[str, object],
) -> None:
    with pytest.raises(BackendOptionsError, match="valid CSS selector") as exc_info:
        validate_backend_options("html", options)
    if "selectors" in options:
        assert exc_info.value.path == ("options", "selectors", "title")
    else:
        assert exc_info.value.path == ("options", "heading_selectors", 0)


def test_compile_rejects_invalid_html_selector(tmp_path: Path) -> None:
    (tmp_path / "stel_project.yml").write_text("name: selector_contract\n")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "raw.yml").write_text(
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n    extraction:\n      backend: html\n"
        "      options:\n        selectors:\n          title: 'h1>>p'\n"
    )

    result = CliRunner().invoke(
        cli, ["--project-dir", str(tmp_path), "compile"]
    )

    assert result.exit_code == 2, result.output
    assert "options.selectors.title" in result.output
    assert "valid CSS selector" in result.output


def test_html_parser_error_only_advertises_available_parsers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stel.backends.options._available_html_parsers",
        lambda: ("html.parser",),
    )

    with pytest.raises(BackendOptionsError) as exc_info:
        validate_backend_options("html", {"parser": "lxml"})

    assert "html.parser" in str(exc_info.value)
    assert "lxml" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("backend", "options"),
    [
        (
            "markdown",
            {"frontmatter_fields": ["Body"], "include_body": True},
        ),
        (
            "markdown",
            {"frontmatter_fields": ["Title", "title"], "include_body": False},
        ),
        ("pdf", {"text_field": "PAGES", "include_pages": True}),
        ("email", {"body_field": "Subject"}),
        (
            "html",
            {"include_structure": True, "text_field": "TABLES"},
        ),
        (
            "html",
            {"include_text": False, "selectors": {"META": "title"}, "include_meta": True},
        ),
        (
            "html",
            {"include_text": False, "selectors": {"Title": "h1", "title": "h2"}},
        ),
    ],
)
def test_backend_emitted_fields_must_not_collide_case_insensitively(
    backend: str, options: dict[str, object]
) -> None:
    with pytest.raises(BackendOptionsError, match=r"unique.*case-insensitive"):
        validate_backend_options(backend, options)


@pytest.mark.parametrize(
    ("backend", "options"),
    [
        (
            "markdown",
            {"frontmatter_fields": [], "include_body": True},
        ),
        (
            "pdf",
            {"include_text": False, "text_field": "pages", "include_pages": True},
        ),
        (
            "email",
            {"include_body": False, "body_field": "subject"},
        ),
        (
            "html",
            {"include_text": False, "text_field": "links", "include_links": True},
        ),
    ],
)
def test_disabled_or_dynamic_fields_do_not_create_false_collisions(
    backend: str, options: dict[str, object]
) -> None:
    assert validate_backend_options(backend, options) == options


def test_custom_register_decorator_installs_typed_option_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stel.backends.options as option_registry
    import stel.backends.registry as backend_registry

    monkeypatch.setattr(
        option_registry,
        "_OPTION_CONTRACTS",
        dict(option_registry._OPTION_CONTRACTS),
    )
    monkeypatch.setattr(
        backend_registry,
        "_REGISTRY",
        dict(backend_registry._REGISTRY),
    )

    class CustomOptions(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        api_key_env: str
        enabled: bool = False
        max_tokens: int
        temperature: float

    @register(options_model=CustomOptions, native_batch=True)
    class CustomBackend(BaseBackend):
        def name(self) -> str:
            return "custom_test"

        def supported_formats(self) -> list[str]:
            return [".custom"]

        def extract(
            self, path: Path, options: dict[str, Any]
        ) -> ExtractionResult:
            return ExtractionResult(fields={"path": str(path)})

    assert get_backend("custom_test").name() == "custom_test"
    custom_options = {
        "api_key_env": "PLUGIN_OWNED_VALUE",
        "enabled": True,
        "max_tokens": 0,
        "temperature": -10.0,
    }
    assert validate_backend_options("custom_test", custom_options) == custom_options
    assert get_backend_option_contract("custom_test").native_batch
    with pytest.raises(BackendOptionsError, match="unknown"):
        validate_backend_options("custom_test", {"unknown": True})

    @register
    class LegacyCustomBackend(BaseBackend):
        def name(self) -> str:
            return "legacy_custom_test"

        def supported_formats(self) -> list[str]:
            return [".legacy"]

        def extract(
            self, path: Path, options: dict[str, Any]
        ) -> ExtractionResult:
            return ExtractionResult(fields=dict(options))

    legacy_options = {
        "api_key_env": "PLUGIN_OWNED_VALUE",
        "max_tokens": 0,
        "plugin_owned": {"nested": True},
        "temperature": -10.0,
    }
    assert validate_backend_options("legacy_custom_test", legacy_options) == legacy_options

    class PreRegisteredOptions(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        limit: int

    register_backend_option_contract("pre_registered_test", PreRegisteredOptions)

    @register
    class PreRegisteredBackend(BaseBackend):
        def name(self) -> str:
            return "pre_registered_test"

        def supported_formats(self) -> list[str]:
            return [".pre"]

        def extract(
            self, path: Path, options: dict[str, Any]
        ) -> ExtractionResult:
            return ExtractionResult(fields=dict(options))

    assert validate_backend_options("pre_registered_test", {"limit": 3}) == {
        "limit": 3
    }


def test_backend_validates_options_before_reading_document(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(BackendOptionsError, match="fieldz"):
        get_backend("json").extract(missing, {"fieldz": ["id"]})
