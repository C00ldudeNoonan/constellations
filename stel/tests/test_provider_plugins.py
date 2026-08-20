"""Entry-point provider discovery, provider-owned options, and billed
failures (issue #71)."""
from __future__ import annotations

import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from pydantic import BaseModel, ConfigDict

from stel.credentials import CredentialReference
from stel.providers import (
    PROVIDER_CONTRACT_VERSION,
    BatchInferenceItem,
    InferenceFailure,
    InferenceProvider,
    InferenceResult,
    ProviderConfigurationError,
    ProviderError,
    ProviderRegistrationError,
    ProviderUsage,
    artifact_safe_options,
    discover_providers,
    entry_point_group,
    get_inference_provider,
    list_inference_providers,
    parse_profile_options,
    profile_options_fingerprint,
    provider_inventory,
    provider_option,
    register_inference_provider,
)
from stel.providers import registry as registry_module
from stel.providers.base import validate_profile_options_model


class InferenceProviderStub(InferenceProvider):
    """Minimal concrete provider for registration tests."""

    provider_name = "stub"
    implementation_version = "1"
    requires_credentials = False
    default_model = "stub-small"

    def complete(self, request: Any, *, credential: Any, runtime: Any) -> Any:
        raise NotImplementedError


@pytest.fixture
def clean_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Snapshot registry state so discovered plugins never leak across tests."""
    monkeypatch.setattr(
        registry_module,
        "_INFERENCE_PROVIDERS",
        dict(registry_module._INFERENCE_PROVIDERS),
    )
    monkeypatch.setattr(
        registry_module,
        "_EMBEDDING_PROVIDERS",
        dict(registry_module._EMBEDDING_PROVIDERS),
    )
    monkeypatch.setattr(
        registry_module,
        "_INCOMPATIBLE_PLUGINS",
        {"inference": {}, "embedding": {}},
    )
    monkeypatch.setattr(registry_module, "_PLUGIN_DISTRIBUTIONS", {})
    monkeypatch.setattr(registry_module, "_DISCOVERY_COMPLETE", False)
    loaded_before = set(sys.modules)
    yield
    for module_name in set(sys.modules) - loaded_before:
        if module_name.startswith("plugin_"):
            del sys.modules[module_name]


_VALID_PROVIDER_SOURCE = """
from pydantic import BaseModel, ConfigDict

from stel.credentials import CredentialReference
from stel.providers import (
    InferenceProvider,
    InferenceResult,
    ProviderUsage,
    provider_option,
)


class {options_class}(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True
    )

    region: str = provider_option("semantic", default="us-east")
    pool_size: int = provider_option("execution", default=4)
    deployment: str = provider_option("artifact-safe", default="default")
    admin_key_env: CredentialReference | None = provider_option(
        "credential", default=None
    )


class {class_name}(InferenceProvider):
    provider_name = "{provider_name}"
    implementation_version = "1"
    requires_credentials = False
    default_model = "{provider_name}-small"

    @classmethod
    def profile_options_model(cls):
        return {options_class}

    def complete(self, request, *, credential, runtime):
        return InferenceResult(
            dict.fromkeys(request.output_schema.get("properties", {{}})),
            usage=ProviderUsage(input_tokens=1, output_tokens=1),
        )
"""


def _install_plugin(
    site_dir: Path,
    *,
    distribution: str,
    module: str,
    provider_name: str,
    group: str,
    class_name: str = "PluginProvider",
    source: str | None = None,
    entry_name: str | None = None,
) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    if source is None:
        source = _VALID_PROVIDER_SOURCE.format(
            class_name=class_name,
            options_class=f"{class_name}Options",
            provider_name=provider_name,
        )
    (site_dir / f"{module}.py").write_text(textwrap.dedent(source))
    dist_info = site_dir / f"{distribution}-0.1.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.1.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        f"[{group}]\n{entry_name or provider_name} = {module}:{class_name}\n"
    )


def _discover_from(site_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(site_dir))
    discover_providers(force=True)


# ─── discovery ──────────────────────────────────────────────────────────────


def test_entry_point_group_carries_contract_version() -> None:
    assert entry_point_group("inference") == (
        f"stel.inference_providers.v{PROVIDER_CONTRACT_VERSION}"
    )


def test_plugin_provider_loads_through_discovery(
    clean_registry: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_plugin(
        tmp_path / "site",
        distribution="plugin-good",
        module="plugin_good",
        provider_name="pluginone",
        group=entry_point_group("inference"),
    )
    _discover_from(tmp_path / "site", monkeypatch)

    assert "pluginone" in list_inference_providers()
    provider = get_inference_provider("pluginone")
    assert provider.name() == "pluginone"
    entries = {
        (entry.capability, entry.name): entry for entry in provider_inventory()
    }
    plugin_entry = entries[("inference", "pluginone")]
    assert plugin_entry.distribution == "plugin-good"
    assert plugin_entry.status == "available"
    builtin_entry = entries[("inference", "anthropic")]
    assert builtin_entry.distribution == "built-in"


def test_entry_point_name_must_match_provider_name(
    clean_registry: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_plugin(
        tmp_path / "site",
        distribution="plugin-misnamed",
        module="plugin_misnamed",
        provider_name="pluginone",
        entry_name="othername",
        group=entry_point_group("inference"),
    )
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    with pytest.raises(ProviderRegistrationError, match="plugin-misnamed"):
        discover_providers(force=True)


def test_duplicate_plugin_names_fail_with_both_distributions(
    clean_registry: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_plugin(
        tmp_path / "site-a",
        distribution="plugin-first",
        module="plugin_first",
        provider_name="pluginone",
        group=entry_point_group("inference"),
    )
    _install_plugin(
        tmp_path / "site-b",
        distribution="plugin-second",
        module="plugin_second",
        provider_name="pluginone",
        group=entry_point_group("inference"),
    )
    monkeypatch.syspath_prepend(str(tmp_path / "site-a"))
    monkeypatch.syspath_prepend(str(tmp_path / "site-b"))
    with pytest.raises(
        ProviderRegistrationError,
        match=r"plugin-first.*plugin-second|plugin-second.*plugin-first",
    ):
        discover_providers(force=True)


def test_plugin_cannot_shadow_builtin_provider(
    clean_registry: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_plugin(
        tmp_path / "site",
        distribution="plugin-shadow",
        module="plugin_shadow",
        provider_name="anthropic",
        group=entry_point_group("inference"),
    )
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    with pytest.raises(ProviderRegistrationError, match=r"built-in.*plugin-shadow"):
        discover_providers(force=True)


def test_plugin_load_failure_names_distribution_and_exception_type(
    clean_registry: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_plugin(
        tmp_path / "site",
        distribution="plugin-broken",
        module="plugin_broken",
        provider_name="pluginone",
        group=entry_point_group("inference"),
        source="raise RuntimeError('secret /etc/path must not leak')\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    with pytest.raises(ProviderRegistrationError) as excinfo:
        discover_providers(force=True)
    message = str(excinfo.value)
    assert "plugin-broken" in message
    assert "RuntimeError" in message
    assert "secret" not in message and "/etc/path" not in message


def test_wrong_contract_version_reports_mismatch_not_missing(
    clean_registry: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_plugin(
        tmp_path / "site",
        distribution="plugin-old",
        module="plugin_old",
        provider_name="pluginold",
        group="stel.inference_providers.v2",
    )
    _discover_from(tmp_path / "site", monkeypatch)

    assert "pluginold" not in list_inference_providers()
    with pytest.raises(ProviderConfigurationError, match="contract v2"):
        get_inference_provider("pluginold")
    incompatible = [
        entry for entry in provider_inventory() if entry.status == "incompatible"
    ]
    assert [(entry.name, entry.distribution) for entry in incompatible] == [
        ("pluginold", "plugin-old")
    ]


def test_invalid_plugin_metadata_fails_registration(
    clean_registry: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = """
    from stel.providers import InferenceProvider, InferenceResult


    class PluginProvider(InferenceProvider):
        provider_name = "Bad Name"
        implementation_version = "1"

        def complete(self, request, *, credential, runtime):
            raise NotImplementedError
    """
    _install_plugin(
        tmp_path / "site",
        distribution="plugin-badmeta",
        module="plugin_badmeta",
        provider_name="Bad Name",
        entry_name="badmeta",
        group=entry_point_group("inference"),
        source=source,
    )
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    with pytest.raises(ProviderRegistrationError, match="plugin-badmeta"):
        discover_providers(force=True)


def test_stock_cli_lists_discovered_plugin(
    clean_registry: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from stel.cli import cli

    _install_plugin(
        tmp_path / "site",
        distribution="plugin-good",
        module="plugin_cli_good",
        provider_name="plugincli",
        group=entry_point_group("inference"),
    )
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    result = CliRunner().invoke(cli, ["providers", "list"])
    assert result.exit_code == 0, result.output
    assert "plugincli" in result.output
    assert "plugin-good" in result.output
    assert "built-in" in result.output


# ─── provider-owned profile options ─────────────────────────────────────────


class _StrictOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    region: str = provider_option("semantic", default="us-east")
    pool_size: int = provider_option("execution", default=4)
    deployment: str = provider_option("artifact-safe", default="default")
    admin_key_env: CredentialReference | None = provider_option(
        "credential", default=None
    )


class _OptionedProvider:
    provider_name = "optioned"

    @classmethod
    def profile_options_model(cls) -> type[BaseModel] | None:
        return _StrictOptions


def test_classification_is_mandatory_and_single() -> None:
    class Unclassified(BaseModel):
        model_config = ConfigDict(
            extra="forbid", frozen=True, hide_input_in_errors=True
        )
        region: str = "us"

    with pytest.raises(ProviderRegistrationError, match="exactly one classification"):
        validate_profile_options_model("demo", Unclassified)
    with pytest.raises(ValueError, match="classification"):
        provider_option("umbrella")


def test_credential_options_must_be_credential_references() -> None:
    class WrongType(BaseModel):
        model_config = ConfigDict(
            extra="forbid", frozen=True, hide_input_in_errors=True
        )
        token: str = provider_option("credential", default="")

    with pytest.raises(ProviderRegistrationError, match="CredentialReference"):
        validate_profile_options_model("demo", WrongType)

    class MisclassifiedCredential(BaseModel):
        model_config = ConfigDict(
            extra="forbid", frozen=True, hide_input_in_errors=True
        )
        token: CredentialReference | None = provider_option("semantic", default=None)

    with pytest.raises(ProviderRegistrationError, match="classified credential"):
        validate_profile_options_model("demo", MisclassifiedCredential)


def test_loose_model_config_is_rejected() -> None:
    class NotFrozen(BaseModel):
        model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
        region: str = provider_option("semantic", default="us")

    with pytest.raises(ProviderRegistrationError, match="frozen"):
        validate_profile_options_model("demo", NotFrozen)


def test_parse_profile_options_rejects_unknown_keys_and_unpublished() -> None:
    parsed = parse_profile_options(cast(Any, _OptionedProvider), {"region": "eu-west"})
    assert isinstance(parsed, _StrictOptions)
    assert parsed.region == "eu-west"
    with pytest.raises(ProviderConfigurationError):
        parse_profile_options(cast(Any, _OptionedProvider), {"unknown": 1})

    class NoOptions:
        provider_name = "bare"

        @classmethod
        def profile_options_model(cls) -> type[BaseModel] | None:
            return None

    assert parse_profile_options(cast(Any, NoOptions), None) is None
    with pytest.raises(ProviderConfigurationError, match="does not accept"):
        parse_profile_options(cast(Any, NoOptions), {"anything": 1})


def test_fingerprint_tracks_semantic_fields_only() -> None:
    base = profile_options_fingerprint(_StrictOptions())
    semantic_change = profile_options_fingerprint(_StrictOptions(region="eu-west"))
    execution_change = profile_options_fingerprint(_StrictOptions(pool_size=32))
    credential_change = profile_options_fingerprint(
        _StrictOptions(admin_key_env=CredentialReference.from_env_name("ACME_ADMIN"))
    )
    assert base is not None
    assert semantic_change != base
    assert execution_change == base
    assert credential_change == base
    assert profile_options_fingerprint(None) is None


def test_artifact_safe_options_expose_only_artifact_safe_fields() -> None:
    options = _StrictOptions(region="eu-west", deployment="research")
    assert artifact_safe_options(options) == {"deployment": "research"}


def test_registered_provider_receives_frozen_options(clean_registry: None) -> None:
    @register_inference_provider
    class OptionedInference(InferenceProviderStub):
        provider_name = "optioned"

        @classmethod
        def profile_options_model(cls) -> type[BaseModel] | None:
            return _StrictOptions

    provider = get_inference_provider("optioned", profile_options={"region": "eu"})
    assert isinstance(provider.profile_options, _StrictOptions)
    assert provider.profile_options.region == "eu"
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        cast(Any, provider.profile_options).region = "us"


# ─── billed failures ────────────────────────────────────────────────────────


def test_inference_failure_validates_and_attaches() -> None:
    usage = ProviderUsage(input_tokens=10, output_tokens=2)
    failure = InferenceFailure(
        error_code="invalid_response",
        usage=usage,
        billed_requests=1,
        provider="anthropic",
        model="claude-test",
        implementation_identity="ident",
    )
    error = ProviderError("boom", safe_for_display=True)
    assert error.failure is None
    assert error.attach_failure(failure) is error
    assert error.failure is failure
    with pytest.raises(ValueError):
        InferenceFailure(
            error_code="",
            usage=usage,
            billed_requests=1,
            provider="p",
            model="m",
            implementation_identity="i",
        )


def test_batch_item_usage_only_accompanies_errors() -> None:
    usage = ProviderUsage(input_tokens=5, output_tokens=1)
    error_item = BatchInferenceItem(
        "req-0", error=ProviderError("bad", safe_for_display=True), usage=usage
    )
    assert error_item.usage == usage
    with pytest.raises(ValueError, match="accompanies a failed item"):
        BatchInferenceItem(
            "req-0",
            result=InferenceResult({"a": 1}, usage=usage),
            usage=usage,
        )


def test_anthropic_truncated_response_keeps_billed_usage() -> None:
    from stel.providers.anthropic import _parse_response_safely, _with_billed_failure
    from stel.providers.base import InferenceRequest, ProviderResponseError

    class _Usage:
        input_tokens = 17
        output_tokens = 9
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0

    class _Response:
        stop_reason = "max_tokens"
        usage = _Usage()
        content: ClassVar[list[Any]] = []

    request = InferenceRequest(
        model="claude-test",
        content="doc",
        system_prompt="sys",
        output_name="extract",
        output_description="extract fields",
        output_schema={"type": "object", "properties": {"a": {"type": "string"}}},
        temperature=0.0,
        max_tokens=16,
    )
    with pytest.raises(ProviderResponseError) as excinfo:
        try:
            _parse_response_safely(_Response(), request)
        except ProviderResponseError as error:
            raise _with_billed_failure(error, _Response(), request) from None
    failure = excinfo.value.failure
    assert failure is not None
    assert failure.usage.input_tokens == 17
    assert failure.billed_requests == 1
    assert failure.provider == "anthropic"


# ─── review follow-ups: redaction, call-path delivery, billed accounting ────


def test_provider_options_are_redacted_from_config_surfaces() -> None:
    from stel.backends.options import LLMBackendOptions, validate_backend_options
    from stel.config.profile import LLMConfig

    secret_name = "SECRET_ADMIN_ENV"
    llm = LLMConfig(provider="anthropic", provider_options={"admin_key_env": secret_name})
    assert secret_name not in repr(llm)
    assert "provider_options" not in llm.model_dump()

    raw = {
        "fields": [{"name": "a", "type": "string"}],
        "provider_options": {"admin_key_env": secret_name},
    }
    parsed = LLMBackendOptions.model_validate(raw)
    assert secret_name not in repr(parsed)
    assert "provider_options" not in parsed.model_dump()
    # The trusted in-process mapping still carries the options forward.
    validated = validate_backend_options("llm", raw)
    assert validated["provider_options"] == {"admin_key_env": secret_name}


def test_sync_call_path_delivers_profile_options_to_provider(
    clean_registry: None,
) -> None:
    from stel.backends.llm_backend import extract_fields_with_usage

    received: list[Any] = []

    @register_inference_provider
    class RecordingProvider(InferenceProviderStub):
        provider_name = "recorder"

        @classmethod
        def profile_options_model(cls) -> type[BaseModel] | None:
            return _StrictOptions

        def complete(self, request: Any, *, credential: Any, runtime: Any) -> Any:
            received.append(self.profile_options)
            return InferenceResult(
                {}, usage=ProviderUsage(input_tokens=1, output_tokens=1)
            )

    _fields, usage = extract_fields_with_usage(
        "document text",
        fields_spec=[{"name": "a", "type": "string"}],
        provider="recorder",
        provider_options={"region": "eu-west"},
    )
    assert usage["api_calls"] == 1
    assert received and isinstance(received[-1], _StrictOptions)
    assert received[-1].region == "eu-west"


def test_batch_execution_delivers_profile_options_to_provider(
    clean_registry: None,
) -> None:
    from stel.backends.llm_backend import _run_message_batch
    from stel.providers import (
        BatchInferenceRequest,
        BatchInferenceResult,
        BatchJobStatus,
        InferenceRequest,
    )

    received: list[Any] = []

    @register_inference_provider
    class BatchRecorder(InferenceProviderStub):
        provider_name = "batchrecorder"
        supports_native_batch = True

        @classmethod
        def profile_options_model(cls) -> type[BaseModel] | None:
            return _StrictOptions

        def submit_batch(
            self, requests: Any, *, credential: Any, runtime: Any
        ) -> str:
            received.append(self.profile_options)
            return "job-1"

        def poll_batch(self, batch_id: str, *, credential: Any, runtime: Any) -> Any:
            return BatchJobStatus(done=True, succeeded=1)

        def fetch_batch_results(
            self, batch_id: str, requests: Any, *, credential: Any, runtime: Any
        ) -> Any:
            return BatchInferenceResult(
                tuple(
                    BatchInferenceItem(
                        item.request_id,
                        result=InferenceResult(
                            {}, usage=ProviderUsage(input_tokens=1, output_tokens=1)
                        ),
                    )
                    for item in requests
                ),
                batch_submissions=1,
            )

        def cancel_batch(
            self, batch_id: str, *, credential: Any, runtime: Any
        ) -> None:
            return None

    request = BatchInferenceRequest(
        "req-0",
        InferenceRequest(
            model="stub-small",
            content="doc",
            system_prompt="sys",
            output_schema={"type": "object", "properties": {}},
        ),
    )
    result, resumed = _run_message_batch(
        [request],
        provider="batchrecorder",
        poll_seconds=0.1,
        provider_options={"region": "eu-west"},
    )
    assert not resumed
    assert len(result.items) == 1 and result.items[0].result is not None
    assert received and isinstance(received[-1], _StrictOptions)
    assert received[-1].region == "eu-west"


def test_sanitized_provider_error_preserves_billed_failure() -> None:
    from stel.providers import ProviderResponseError, sanitized_provider_error

    failure = InferenceFailure(
        error_code="invalid_response",
        usage=ProviderUsage(input_tokens=11, output_tokens=4),
        billed_requests=1,
        provider="anthropic",
        model="claude-test",
        implementation_identity="ident",
    )
    original = ProviderResponseError(
        "raw provider text with /internal/path", safe_for_display=False
    ).attach_failure(failure)
    sanitized = sanitized_provider_error("anthropic", "inference", original)
    assert sanitized is not original
    assert sanitized.failure is failure
    assert "raw provider text" not in str(sanitized)


def test_resolve_batch_item_synthesizes_failure_from_item_usage() -> None:
    from stel.backends.llm_backend import LLMBackend

    usage = ProviderUsage(input_tokens=7, output_tokens=3)
    item = BatchInferenceItem(
        "req-0",
        error=ProviderError("failed item", safe_for_display=True),
        usage=usage,
    )
    resolved = cast(Any, LLMBackend)._resolve_batch_item(
        item,
        cache_path=None,
        cache_key="key",
        model="stub-small",
        content_hash="content",
        schema_hash="schema",
        provider_name="acme",
        provider_identity="ident",
    )
    assert isinstance(resolved, ProviderError)
    assert resolved.failure is not None
    assert resolved.failure.usage == usage
    assert resolved.failure.provider == "acme"
    assert resolved.failure.billed_requests == 1
