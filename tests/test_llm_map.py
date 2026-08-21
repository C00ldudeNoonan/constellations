from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from stel.compiler import validate_project_contract
from stel.config import ConfigError, load_project
from stel.config.model import FieldConfig, LLMTransformConfig
from stel.llm_map import (
    LLMMapError,
    build_fields_spec,
    execute_map_item,
    resolve_llm_runtime,
)
from stel.profile import ResolvedProfile, resolve_profile
from stel.providers import InferenceResult, ProviderUsage
from stel.providers.base import InferenceProvider
from stel.providers.registry import _INFERENCE_PROVIDERS
from stel.versioning import compute_model_code_version


class _FakeMapProvider(InferenceProvider):
    provider_name = "llmmap_fake"
    implementation_version = "1"
    requires_credentials = False
    default_model = "fake-small"

    def complete(self, request: Any, *, credential: Any, runtime: Any) -> Any:
        properties = request.output_schema.get("properties", {})
        if "items" in properties:
            # output_cardinality: many
            output: dict[str, Any] = {
                "items": [
                    {"label": f"{request.content}-a", "score": 1},
                    {"label": f"{request.content}-b", "score": 2},
                ]
            }
        else:
            output = {"label": f"{request.content}-one", "score": 7}
        return InferenceResult(output, usage=ProviderUsage(input_tokens=3, output_tokens=5))


@pytest.fixture(autouse=True)
def _register_fake_provider() -> Any:
    _INFERENCE_PROVIDERS.setdefault("llmmap_fake", _FakeMapProvider)
    yield


_FIELDS = [
    FieldConfig(name="label", data_type="string"),
    FieldConfig(name="score", data_type="integer"),
]


def _runtime(**overrides: Any):
    params: dict[str, Any] = {
        "input_field": "text",
        "id_field": "chunk_id",
        "prompt": "classify",
        "provider": "llmmap_fake",
        "model": "fake-small",
    }
    params.update(overrides)
    config = LLMTransformConfig(**params)
    return resolve_llm_runtime(config, _FIELDS, resolved=None)


def test_build_fields_spec_maps_types() -> None:
    spec = build_fields_spec(_FIELDS)
    assert spec == [
        {"name": "label", "type": "string"},
        {"name": "score", "type": "integer"},
    ]


def test_execute_map_item_cardinality_one() -> None:
    runtime = _runtime(output_cardinality="one")
    rows, usage = execute_map_item("doc1", runtime)
    assert rows == [{"label": "doc1-one", "score": 7}]
    assert usage["api_calls"] == 1
    assert usage["input_tokens"] == 3


def test_execute_map_item_cardinality_many_fans_out() -> None:
    runtime = _runtime(output_cardinality="many")
    rows, _usage = execute_map_item("doc1", runtime)
    assert rows == [
        {"label": "doc1-a", "score": 1},
        {"label": "doc1-b", "score": 2},
    ]


def test_execute_map_item_projects_to_declared_fields() -> None:
    # Fields not returned by the provider default to None; provider extras drop.
    fields = [*_FIELDS, FieldConfig(name="missing", data_type="string")]
    config = LLMTransformConfig(
        input_field="text",
        prompt="classify",
        provider="llmmap_fake",
        model="fake-small",
    )
    runtime = resolve_llm_runtime(config, fields, resolved=None)
    rows, _usage = execute_map_item("doc1", runtime)
    assert rows == [{"label": "doc1-one", "score": 7, "missing": None}]


def test_config_hash_changes_with_prompt() -> None:
    a = _runtime(prompt="classify")
    b = _runtime(prompt="different")
    assert a.config_hash != b.config_hash


def test_config_hash_changes_with_cardinality() -> None:
    a = _runtime(output_cardinality="one")
    b = _runtime(output_cardinality="many")
    assert a.config_hash != b.config_hash


def _resolved_with_llm(**attrs: Any) -> ResolvedProfile:
    # Minimal duck-typed resolved profile: resolve_llm_runtime only reads
    # `resolved.llm`. Lets us vary endpoint/options without a full profile.
    llm = SimpleNamespace(
        provider="llmmap_fake",
        model="fake-small",
        base_url=None,
        api_key_env=None,
        provider_options={},
        cache_path=None,
        model_fields_set=set(),
    )
    for key, value in attrs.items():
        setattr(llm, key, value)
    return cast(ResolvedProfile, SimpleNamespace(llm=llm))


def test_config_hash_includes_base_url() -> None:
    # A profile endpoint change must invalidate incremental state (P2 review):
    # provider: default resolves base_url from the profile and sends it to the
    # provider + cache key, so it belongs in the output identity.
    config = LLMTransformConfig(
        input_field="text", prompt="p", provider="default", model="default"
    )
    a = resolve_llm_runtime(config, _FIELDS, _resolved_with_llm(base_url="http://a"))
    b = resolve_llm_runtime(config, _FIELDS, _resolved_with_llm(base_url="http://b"))
    assert a.config_hash != b.config_hash


def test_resolve_unknown_provider_raises() -> None:
    config = LLMTransformConfig(
        input_field="text", prompt="p", provider="does-not-exist-xyz"
    )
    with pytest.raises(LLMMapError):
        resolve_llm_runtime(config, _FIELDS, resolved=None)


def test_identity_is_artifact_safe() -> None:
    runtime = _runtime()
    identity = runtime.identity()
    assert set(identity) == {
        "provider",
        "model",
        "implementation",
        "output_cardinality",
        "config_hash",
        # Prompt *identity* is artifact-safe and deliberately included
        # (issue #303); the text below asserts the text itself is not.
        "prompt_name",
        "prompt_version",
    }
    assert "api_key" not in str(identity)
    assert runtime.system_prompt not in str(identity)


def _llm_project(tmp_path: Any, *, provider: str = "deterministic", extra: str = "") -> Any:
    project = Path(tmp_path) / "project"
    project.mkdir()
    (project / "stel_project.yml").write_text(
        "name: p\nversion: '0.1.0'\nprofile: p\n"
    )
    (project / "profiles.yml").write_text(
        "p:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n        schema: docs\n"
    )
    (project / "sources").mkdir()
    (project / "sources" / "s.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data\n"
        "    file_pattern: '*.json'\n"
    )
    (project / "models").mkdir()
    (project / "models" / "m.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: registry\n    source: ref('documents')\n"
        "    extraction:\n      backend: json\n      options:\n        fields: [body]\n"
        "    materialization: incremental\n"
        "  - name: facts\n    depends_on: [ref('registry')]\n"
        f"    llm:\n      input_field: body\n      prompt: p\n      provider: {provider}\n"
        f"      model: m\n{extra}"
        "    fields:\n      - {name: sentiment, type: string}\n"
        "    materialization: incremental\n"
    )
    return project


def test_compiler_rejects_unknown_llm_provider(tmp_path: Any) -> None:
    project = _llm_project(tmp_path, provider="no-such-provider-xyz")
    proj, sources, models = load_project(project)
    with pytest.raises(ConfigError, match=r"provider|registered"):
        validate_project_contract(proj, sources, models, project)


def test_compiler_requires_single_upstream(tmp_path: Any) -> None:
    project = _llm_project(tmp_path)
    (project / "models" / "m.yml").write_text(
        (project / "models" / "m.yml")
        .read_text()
        .replace("depends_on: [ref('registry')]", "depends_on: []")
    )
    proj, sources, models = load_project(project)
    with pytest.raises(ConfigError, match="exactly one"):
        validate_project_contract(proj, sources, models, project)


def test_backend_and_node_share_the_execution_core() -> None:
    # The migration invariant (issue #144): both `backend: llm` extraction and
    # native `llm:` models execute through one function, so provider/cache/retry
    # logic is never duplicated.
    import stel.backends.llm_backend as backend_module
    import stel.llm_map as llm_map_module

    assert (
        llm_map_module.extract_fields_with_usage
        is backend_module.extract_fields_with_usage
    )


def test_code_version_changes_with_prompt_and_provider(tmp_path: Any) -> None:
    project = _llm_project(tmp_path)
    proj, _sources, models = load_project(project)
    resolved = resolve_profile(proj, project)
    facts = next(m for m in models if m.name == "facts")
    base = compute_model_code_version(facts, proj, project, resolved=resolved)

    assert facts.llm is not None
    changed_prompt = facts.model_copy(
        update={"llm": facts.llm.model_copy(update={"prompt": "different"})}
    )
    assert (
        compute_model_code_version(changed_prompt, proj, project, resolved=resolved)
        != base
    )
