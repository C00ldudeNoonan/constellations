from __future__ import annotations

from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from stel.adapters.duckdb import DuckDBWarehouseConfig
from stel.backends import (
    BaseBackend,
    ExtractionResult,
    register,
    validate_backend_options,
)
from stel.config.model import (
    ExtractionConfig,
    FieldConfig,
    ModelConfig,
    SearchConfig,
    SearchQueryConfig,
    TransformConfig,
)
from stel.config.profile import LLMConfig
from stel.config.project import ExtractionDefaults, ProjectConfig
from stel.hashing import HASH_DIGEST_SIZE
from stel.profile import ResolvedProfile
from stel.versioning import (
    compute_code_version,
    compute_content_hash,
    compute_document_id,
    compute_model_code_version,
    describe_model_inference,
    resolve_module_file,
)


class _FingerprintMode(Enum):
    FAST = "fast"


def _option_value_fingerprint(value: Any, project_dir: Path) -> str:
    return compute_code_version(
        extraction=ExtractionConfig(
            backend="fingerprint_test",
            options={"value": value},
        ),
        transform=None,
        project_dir=project_dir,
    )


def test_content_hash_changes_with_content(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    p.write_text("{}")
    h1 = compute_content_hash(p)
    p.write_text('{"x": 1}')
    h2 = compute_content_hash(p)
    assert h1 != h2
    assert len(h1) == HASH_DIGEST_SIZE * 2


def test_document_id_includes_scope() -> None:
    a = compute_document_id("source_a", "invoice.json")
    b = compute_document_id("source_b", "invoice.json")
    assert a != b
    assert len(a) == HASH_DIGEST_SIZE * 2


def test_code_version_stable(tmp_path: Path) -> None:
    cfg = ExtractionConfig(backend="json", options={"fields": ["a", "b"]})
    v1 = compute_code_version(extraction=cfg, transform=None, project_dir=tmp_path)
    v2 = compute_code_version(extraction=cfg, transform=None, project_dir=tmp_path)
    assert v1 == v2
    assert len(v1) == HASH_DIGEST_SIZE * 2


def test_fingerprint_set_cannot_collide_with_sentinel_shaped_mapping(
    tmp_path: Path,
) -> None:
    set_value = {"a", "b"}
    sentinel_mapping = {"__stel_set__": ["a", "b"]}

    assert _option_value_fingerprint(set_value, tmp_path) != (
        _option_value_fingerprint(sentinel_mapping, tmp_path)
    )


@pytest.mark.parametrize(
    ("typed_value", "string_value"),
    [
        (Path("data/output"), "data/output"),
        (date(2026, 7, 11), "2026-07-11"),
        (_FingerprintMode.FAST, "fast"),
    ],
)
def test_fingerprint_special_values_cannot_collide_with_strings(
    tmp_path: Path, typed_value: object, string_value: str
) -> None:
    assert _option_value_fingerprint(typed_value, tmp_path) != (
        _option_value_fingerprint(string_value, tmp_path)
    )


def test_fingerprint_preserves_list_and_tuple_identity(tmp_path: Path) -> None:
    assert _option_value_fingerprint(["a", "b"], tmp_path) != (
        _option_value_fingerprint(("a", "b"), tmp_path)
    )


def test_fingerprint_mapping_order_is_stable(tmp_path: Path) -> None:
    first = {"outer": {"b": 2, "a": 1}, "value": [1, 2]}
    reordered = {"value": [1, 2], "outer": {"a": 1, "b": 2}}

    assert _option_value_fingerprint(first, tmp_path) == (
        _option_value_fingerprint(reordered, tmp_path)
    )


def test_fingerprint_mapping_order_uses_full_key_value_pair(tmp_path: Path) -> None:
    first_nan = float("nan")
    second_nan = float("nan")
    first = {first_nan: "a", second_nan: "b"}
    reordered = {second_nan: "b", first_nan: "a"}
    assert len(first) == 2

    assert _option_value_fingerprint(first, tmp_path) == (
        _option_value_fingerprint(reordered, tmp_path)
    )


def test_code_version_changes_with_config(tmp_path: Path) -> None:
    a = compute_code_version(
        extraction=ExtractionConfig(backend="json", options={"fields": ["a"]}),
        transform=None,
        project_dir=tmp_path,
    )
    b = compute_code_version(
        extraction=ExtractionConfig(backend="json", options={"fields": ["b"]}),
        transform=None,
        project_dir=tmp_path,
    )
    assert a != b


def test_code_version_changes_with_field_data_type(tmp_path: Path) -> None:
    cfg = ExtractionConfig(backend="json", options={"fields": ["value"]})
    integer = compute_code_version(
        extraction=cfg,
        transform=None,
        fields=[FieldConfig(name="value", data_type="integer")],
        project_dir=tmp_path,
    )
    string = compute_code_version(
        extraction=cfg,
        transform=None,
        fields=[FieldConfig(name="value", data_type="string")],
        project_dir=tmp_path,
    )
    assert integer != string


def test_code_version_changes_with_module_contents(tmp_path: Path) -> None:
    mod_dir = tmp_path / "transforms"
    mod_dir.mkdir()
    mod = mod_dir / "summary.py"
    mod.write_text("def run(deps): return None\n")
    transform = TransformConfig(type="python", module="transforms.summary")

    v1 = compute_code_version(extraction=None, transform=transform, project_dir=tmp_path)
    mod.write_text("def run(deps): return 42\n")
    v2 = compute_code_version(extraction=None, transform=transform, project_dir=tmp_path)
    assert v1 != v2


def test_resolve_module_file_dotted(tmp_path: Path) -> None:
    assert resolve_module_file("transforms.summary", tmp_path) == (
        tmp_path / "transforms" / "summary.py"
    )


def test_code_version_ignores_flush_every(tmp_path: Path) -> None:
    """flush_every shapes execution, not output — changing it must not
    invalidate incremental state (issue #77)."""
    base = ExtractionConfig(backend="json", options={"fields": ["a"]})
    tuned = ExtractionConfig(
        backend="json", options={"fields": ["a"]}, flush_every=2
    )
    assert compute_code_version(
        extraction=base, transform=None, project_dir=tmp_path
    ) == compute_code_version(extraction=tuned, transform=None, project_dir=tmp_path)


def test_code_version_ignores_publish_every(tmp_path: Path) -> None:
    """publish_every shapes publication cadence, not output — changing it must
    not invalidate incremental state (issue #293)."""
    base = ExtractionConfig(backend="json", options={"fields": ["a"]})
    tuned = ExtractionConfig(
        backend="json", options={"fields": ["a"]}, publish_every=8
    )
    assert compute_code_version(
        extraction=base, transform=None, project_dir=tmp_path
    ) == compute_code_version(extraction=tuned, transform=None, project_dir=tmp_path)


def test_update_when_changed_does_not_change_model_code_version(
    tmp_path: Path,
) -> None:
    """update_when_changed shapes publication, not row content (issue #281):
    enabling it must not invalidate existing incremental state and force a
    reprocess."""
    project = ProjectConfig(
        name="p", extraction=ExtractionDefaults(default_backend="json")
    )
    base = ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction=ExtractionConfig(),
        materialization="incremental",
    )
    guarded = base.model_copy(update={"update_when_changed": ["content_hash"]})
    assert compute_model_code_version(base, project, tmp_path) == (
        compute_model_code_version(guarded, project, tmp_path)
    )


def test_effective_default_backend_changes_model_code_version(tmp_path: Path) -> None:
    model = ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction=ExtractionConfig(),
    )
    json_project = ProjectConfig(
        name="p", extraction=ExtractionDefaults(default_backend="json")
    )
    markdown_project = ProjectConfig(
        name="p", extraction=ExtractionDefaults(default_backend="markdown")
    )

    assert compute_model_code_version(model, json_project, tmp_path) != (
        compute_model_code_version(model, markdown_project, tmp_path)
    )


def _resolved_llm(tmp_path: Path, **overrides: Any) -> ResolvedProfile:
    return ResolvedProfile(
        profile_name="p",
        target_name="dev",
        warehouse=DuckDBWarehouseConfig(path=tmp_path / "db.duckdb"),
        llm=LLMConfig(**overrides),
        source_paths={},
        profiles_path=tmp_path / "profiles.yml",
    )


def test_profile_llm_semantics_change_model_code_version(tmp_path: Path) -> None:
    model = ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction=ExtractionConfig(
            backend="llm",
            options={"fields": [{"name": "title", "type": "string"}]},
        ),
    )
    project = ProjectConfig(name="p")
    baseline = compute_model_code_version(
        model,
        project,
        tmp_path,
        resolved=_resolved_llm(tmp_path, model="claude-a", system_prompt="prompt a"),
    )

    assert baseline != compute_model_code_version(
        model,
        project,
        tmp_path,
        resolved=_resolved_llm(tmp_path, model="claude-b", system_prompt="prompt a"),
    )
    assert baseline != compute_model_code_version(
        model,
        project,
        tmp_path,
        resolved=_resolved_llm(tmp_path, model="claude-a", system_prompt="prompt b"),
    )


def test_vllm_endpoint_changes_model_code_version_but_timeout_does_not(
    tmp_path: Path,
) -> None:
    model = ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction=ExtractionConfig(
            backend="llm",
            options={"fields": [{"name": "title", "type": "string"}]},
        ),
    )
    project = ProjectConfig(name="p")
    first = _resolved_llm(
        tmp_path,
        provider="vllm",
        model="invoice-extractor",
        base_url="https://first.example.test/v1",
        timeout_seconds=30,
    )
    same_endpoint = _resolved_llm(
        tmp_path,
        provider="vllm",
        model="invoice-extractor",
        base_url="HTTPS://FIRST.EXAMPLE.TEST:443/v1/",
        timeout_seconds=300,
    )
    second = _resolved_llm(
        tmp_path,
        provider="vllm",
        model="invoice-extractor",
        base_url="https://second.example.test/v1",
        timeout_seconds=30,
    )

    first_version = compute_model_code_version(
        model, project, tmp_path, resolved=first
    )
    assert first_version == compute_model_code_version(
        model, project, tmp_path, resolved=same_endpoint
    )
    assert first_version != compute_model_code_version(
        model, project, tmp_path, resolved=second
    )
    descriptor = describe_model_inference(model, project, resolved=first)
    assert descriptor is not None
    assert descriptor["provider"] == "vllm"
    assert descriptor["model"] == "invoice-extractor"
    assert len(descriptor["endpoint_identity"]) == HASH_DIGEST_SIZE * 2
    assert "first.example.test" not in repr(descriptor)


def test_llm_dependent_transform_versions_profile_inference(tmp_path: Path) -> None:
    model = ModelConfig(
        name="derived",
        depends_on=["ref('raw')"],
        transform=TransformConfig(type="python", uses_llm=True),
    )
    project = ProjectConfig(name="p")
    first = compute_model_code_version(
        model,
        project,
        tmp_path,
        resolved=_resolved_llm(
            tmp_path, model="claude-a", system_prompt="prompt a"
        ),
    )
    second = compute_model_code_version(
        model,
        project,
        tmp_path,
        resolved=_resolved_llm(
            tmp_path, model="claude-b", system_prompt="prompt a"
        ),
    )
    third = compute_model_code_version(
        model,
        project,
        tmp_path,
        resolved=_resolved_llm(
            tmp_path, model="claude-a", system_prompt="prompt b"
        ),
    )

    assert first != second
    assert first != third


def _llm_model(*, provider: str, model: str) -> ModelConfig:
    return ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction=ExtractionConfig(
            backend="llm",
            options={
                "provider": provider,
                "model": model,
                "fields": [{"name": "title", "type": "string"}],
            },
        ),
    )


def test_provider_and_model_change_model_code_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "stel.versioning.get_inference_provider",
        lambda name: SimpleNamespace(
            implementation_identity=lambda: f"implementation/{name}",
            resolve_model=lambda model: model,
            resolve_base_url=lambda base_url: base_url,
        ),
    )
    project = ProjectConfig(name="p")
    baseline = compute_model_code_version(
        _llm_model(provider="provider-a", model="model-a"), project, tmp_path
    )

    assert baseline != compute_model_code_version(
        _llm_model(provider="provider-b", model="model-a"), project, tmp_path
    )
    assert baseline != compute_model_code_version(
        _llm_model(provider="provider-a", model="model-b"), project, tmp_path
    )


def test_search_dependency_changes_model_code_version(tmp_path: Path) -> None:
    project = ProjectConfig(name="p")
    first = ModelConfig(
        name="search_index",
        depends_on=["ref('first_embeddings')"],
        materialization="incremental",
        search=SearchConfig(
            id_field="record_id",
            text_fields=("text",),
            query=SearchQueryConfig(modes=frozenset({"filter"})),
        ),
    )
    second = first.model_copy(
        update={"depends_on": ["ref('second_embeddings')"]},
    )

    assert compute_model_code_version(first, project, tmp_path) != (
        compute_model_code_version(second, project, tmp_path)
    )


def test_provider_implementation_changes_model_code_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _llm_model(provider="provider-a", model="model-a")
    project = ProjectConfig(name="p")
    monkeypatch.setattr(
        "stel.versioning.get_inference_provider",
        lambda name: SimpleNamespace(
            implementation_identity=lambda: "implementation/one",
            resolve_model=lambda selected: selected,
            resolve_base_url=lambda base_url: base_url,
        ),
    )
    first = compute_model_code_version(model, project, tmp_path)
    monkeypatch.setattr(
        "stel.versioning.get_inference_provider",
        lambda name: SimpleNamespace(
            implementation_identity=lambda: "implementation/two",
            resolve_model=lambda selected: selected,
            resolve_base_url=lambda base_url: base_url,
        ),
    )

    assert first != compute_model_code_version(model, project, tmp_path)


def test_llm_execution_only_options_do_not_change_model_code_version(
    tmp_path: Path,
) -> None:
    model = ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction=ExtractionConfig(
            backend="llm",
            options={"fields": [{"name": "title", "type": "string"}]},
        ),
    )
    project = ProjectConfig(name="p")
    first = _resolved_llm(
        tmp_path,
        api_key_env="FIRST_KEY",
        cache_path=tmp_path / "first.duckdb",
    )
    second = _resolved_llm(
        tmp_path,
        api_key_env="SECOND_KEY",
        cache_path=tmp_path / "second.duckdb",
    )

    assert compute_model_code_version(
        model, project, tmp_path, resolved=first
    ) == compute_model_code_version(model, project, tmp_path, resolved=second)


def test_backend_parser_identity_changes_model_code_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction=ExtractionConfig(backend="json"),
    )
    project = ProjectConfig(name="p")
    monkeypatch.setattr(
        "stel.versioning.get_backend",
        lambda name: SimpleNamespace(
            parser_identity=lambda: "parser/one",
            implementation_identity=lambda: "implementation/stable",
        ),
    )
    first = compute_model_code_version(model, project, tmp_path)
    monkeypatch.setattr(
        "stel.versioning.get_backend",
        lambda name: SimpleNamespace(
            parser_identity=lambda: "parser/two",
            implementation_identity=lambda: "implementation/stable",
        ),
    )

    assert first != compute_model_code_version(model, project, tmp_path)


def test_backend_implementation_changes_model_code_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction=ExtractionConfig(backend="json"),
    )
    project = ProjectConfig(name="p")
    monkeypatch.setattr(
        "stel.versioning.get_backend",
        lambda name: SimpleNamespace(
            parser_identity=lambda: None,
            implementation_identity=lambda: "implementation/one",
        ),
    )
    first = compute_model_code_version(model, project, tmp_path)
    monkeypatch.setattr(
        "stel.versioning.get_backend",
        lambda name: SimpleNamespace(
            parser_identity=lambda: None,
            implementation_identity=lambda: "implementation/two",
        ),
    )

    assert first != compute_model_code_version(model, project, tmp_path)


def test_release_version_does_not_change_model_code_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property issue #363 exists for: a release that touches no
    backend-reachable source leaves extraction code_version — and with it
    every document's incremental state — intact. The real json backend is
    used so the whole identity chain (implementation_identity, parser
    identity, version) is exercised, with only the reported release moved.
    """
    model = ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction=ExtractionConfig(backend="json"),
    )
    project = ProjectConfig(name="p")
    first = compute_model_code_version(model, project, tmp_path)
    monkeypatch.setattr(
        "stel.backends.base.distribution_version", lambda: "999.0.0"
    )

    assert compute_model_code_version(model, project, tmp_path) == first


def test_custom_backend_options_are_json_safe_and_runtime_values_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    class Mode(Enum):
        FAST = "fast"

    class NestedOptions(BaseModel):
        model_config = ConfigDict(strict=True)

        observed_at: datetime

    class TypedOptions(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)

        api_key_env: str
        max_tokens: int
        mode: Mode
        observed_on: date
        output_path: Path
        snapshot: NestedOptions
        tags: set[str]
        temperature: float

    @register(options_model=TypedOptions)
    class TypedFingerprintBackend(BaseBackend):
        def name(self) -> str:
            return "typed_fingerprint_test"

        def supported_formats(self) -> list[str]:
            return [".typed"]

        def extract(
            self, path: Path, options: dict[str, Any]
        ) -> ExtractionResult:
            return ExtractionResult(fields=dict(options))

    @register
    class PassthroughFingerprintBackend(BaseBackend):
        def name(self) -> str:
            return "passthrough_fingerprint_test"

        def supported_formats(self) -> list[str]:
            return [".passthrough"]

        def extract(
            self, path: Path, options: dict[str, Any]
        ) -> ExtractionResult:
            return ExtractionResult(fields=dict(options))

    snapshot = NestedOptions(observed_at=datetime(2026, 7, 11, 12, tzinfo=UTC))
    options: dict[str, Any] = {
        "api_key_env": "CUSTOM_VALUE",
        "max_tokens": 0,
        "mode": Mode.FAST,
        "observed_on": date(2026, 7, 11),
        "output_path": tmp_path / "output",
        "snapshot": snapshot,
        "tags": {"beta", "alpha"},
        "temperature": -10.0,
    }
    project = ProjectConfig(name="p")

    for backend_name in ("typed_fingerprint_test", "passthrough_fingerprint_test"):
        runtime_options = validate_backend_options(backend_name, options)
        assert isinstance(runtime_options["output_path"], Path)
        assert isinstance(runtime_options["observed_on"], date)
        assert runtime_options["mode"] is Mode.FAST
        assert isinstance(runtime_options["tags"], set)

        extraction = ExtractionConfig(backend=backend_name, options=dict(options))
        model = ModelConfig(
            name="raw",
            source="ref('docs')",
            extraction=extraction,
        )
        first = compute_model_code_version(model, project, tmp_path)
        reordered = ModelConfig(
            name="raw",
            source="ref('docs')",
            extraction=ExtractionConfig(
                backend=backend_name,
                options=dict(reversed(list(options.items()))),
            ),
        )

        assert first == compute_model_code_version(reordered, project, tmp_path)
        assert isinstance(extraction.options["output_path"], Path)
        assert isinstance(extraction.options["observed_on"], date)
        assert extraction.options["mode"] is Mode.FAST
        assert extraction.options["snapshot"] is snapshot

        changed_options = {**options, "max_tokens": 1}
        changed = ModelConfig(
            name="raw",
            source="ref('docs')",
            extraction=ExtractionConfig(
                backend=backend_name,
                options=changed_options,
            ),
        )
        assert first != compute_model_code_version(changed, project, tmp_path)
