from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

import dbt_ml.providers.base as provider_base
from dbt_ml.backends import get_backend
from dbt_ml.backends.llm_backend import extract_fields_from_text
from dbt_ml.config.profile import resolve_llm_credential
from dbt_ml.credentials import (
    CredentialReference,
    CredentialReferenceError,
    ProtectedCredential,
)
from dbt_ml.hashing import canonical_fingerprint
from dbt_ml.providers import (
    PROVIDER_CONTRACT_VERSION,
    BatchInferenceItem,
    BatchInferenceRequest,
    BatchInferenceResult,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResult,
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    ProviderBatchError,
    ProviderConfigurationError,
    ProviderCredential,
    ProviderNotFoundError,
    ProviderRegistrationError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderRuntimeOptions,
    ProviderUsage,
    get_embedding_provider,
    get_inference_provider,
    list_embedding_providers,
    list_inference_providers,
    register_embedding_provider,
    register_inference_provider,
    resolve_provider_model,
)


def _request(content: str = "hello") -> InferenceRequest:
    return InferenceRequest(
        model="test-model",
        content=content,
        system_prompt="Extract a value.",
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
    )


def _provider_traceback_locals(error: BaseException) -> str:
    rendered: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module, str) and module.startswith("dbt_ml"):
            rendered.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(rendered)


class _EchoInferenceProvider(InferenceProvider):
    provider_name = "contract-test"
    implementation_version = "1"
    requires_credentials = False

    def complete(
        self,
        request: InferenceRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> InferenceResult:
        assert credential is None
        assert runtime.max_retries >= 0
        if request.content == "provider-error":
            raise ProviderResponseError("safe provider response error")
        if request.content == "raw-error":
            raise RuntimeError("raw provider failure contained a-secret")
        return InferenceResult(
            {"value": request.content},
            usage=ProviderUsage(input_tokens=1, output_tokens=2),
        )


class _EchoEmbeddingProvider(EmbeddingProvider):
    provider_name = "contract-test"
    implementation_version = "1"
    requires_credentials = False

    def _embed(
        self,
        request: EmbeddingRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> EmbeddingResult:
        assert credential is None
        assert runtime.max_retries >= 0
        vectors = tuple((float(len(text)), 1.0) for text in request.texts)
        return EmbeddingResult(
            vectors=vectors,
            model=request.model,
            dimensions=2,
            usage=ProviderUsage(input_tokens=len(request.texts)),
        )


def test_registries_keep_inference_and_embedding_capabilities_separate() -> None:
    register_inference_provider(_EchoInferenceProvider)
    register_embedding_provider(_EchoEmbeddingProvider)

    assert isinstance(get_inference_provider("contract-test"), _EchoInferenceProvider)
    assert isinstance(get_embedding_provider("contract-test"), _EchoEmbeddingProvider)
    assert "contract-test" in list_inference_providers()
    assert "contract-test" in list_embedding_providers()


def test_llm_backend_runs_registered_provider_without_anthropic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    register_inference_provider(_EchoInferenceProvider)

    class ForbiddenAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            raise AssertionError("Anthropic must not be constructed")

    monkeypatch.setattr("anthropic.Anthropic", ForbiddenAnthropic)
    document = tmp_path / "document.txt"
    document.write_text("provider-neutral")

    result = get_backend("llm").extract(
        document,
        {
            "provider": "contract-test",
            "model": "test-model",
            "fields": [{"name": "value", "type": "string"}],
        },
    )

    assert result.fields == {"value": "provider-neutral"}
    assert result.metrics["input_tokens"] == 1
    assert result.metrics["output_tokens"] == 2


def test_public_helper_uses_selected_provider_credential_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, str] = {}

    class CredentialProvider(_EchoInferenceProvider):
        provider_name = "credential-helper-test"
        requires_credentials = True
        default_credential_env = "ACME_API_KEY"

        def complete(
            self,
            request: InferenceRequest,
            *,
            credential: ProviderCredential | None,
            runtime: ProviderRuntimeOptions,
        ) -> InferenceResult:
            del runtime
            assert credential is not None
            observed["value"] = credential.reveal()
            return InferenceResult({"value": request.content})

    register_inference_provider(CredentialProvider)
    register_inference_provider(_EchoInferenceProvider)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-cross-provider-boundary")
    monkeypatch.setenv("ACME_API_KEY", "selected-provider-key")

    with pytest.raises(ProviderConfigurationError, match="explicit model"):
        extract_fields_from_text(
            "provider-neutral",
            provider="credential-helper-test",
            fields_spec=[{"name": "value", "type": "string"}],
        )

    fields = extract_fields_from_text(
        "provider-neutral",
        provider="credential-helper-test",
        model="test-model",
        fields_spec=[{"name": "value", "type": "string"}],
    )

    assert fields == {"value": "provider-neutral"}
    assert observed == {"value": "selected-provider-key"}
    resolved_credential = resolve_llm_credential(
        {"provider": "credential-helper-test"}
    )
    assert isinstance(resolved_credential, ProtectedCredential)
    assert not isinstance(resolved_credential, tuple)
    assert resolved_credential.reveal() == "selected-provider-key"
    assert "ACME_API_KEY" not in repr(resolved_credential)
    assert resolve_llm_credential({"provider": "contract-test"}) is None


def test_registry_rejects_duplicate_and_invalid_providers() -> None:
    register_inference_provider(_EchoInferenceProvider)

    class DuplicateProvider(_EchoInferenceProvider):
        pass

    with pytest.raises(ProviderRegistrationError, match="already registered"):
        register_inference_provider(DuplicateProvider)

    class InvalidNameProvider(_EchoInferenceProvider):
        provider_name = "Invalid Provider"

    with pytest.raises(ProviderRegistrationError, match="provider name must match"):
        register_inference_provider(InvalidNameProvider)

    class InvalidBatchMetadataProvider(_EchoInferenceProvider):
        provider_name = "invalid-batch-metadata"
        max_batch_requests = 0

    with pytest.raises(ProviderRegistrationError, match="max_batch_requests"):
        register_inference_provider(InvalidBatchMetadataProvider)

    class InvalidDefaultModelProvider(_EchoInferenceProvider):
        provider_name = "invalid-default-model"
        default_model = ""

    with pytest.raises(ProviderRegistrationError, match="default_model"):
        register_inference_provider(InvalidDefaultModelProvider)

    class MissingNativeBatchImplementation(_EchoInferenceProvider):
        provider_name = "missing-native-batch"
        supports_native_batch = True

    with pytest.raises(ProviderRegistrationError, match="must override complete_batch"):
        register_inference_provider(MissingNativeBatchImplementation)

    class InvalidImplementationPackages(_EchoInferenceProvider):
        provider_name = "invalid-implementation-packages"
        implementation_packages = ("valid", "invalid package")

    with pytest.raises(ProviderRegistrationError, match="implementation_packages"):
        register_inference_provider(InvalidImplementationPackages)

    class InvalidImplementationVersion(_EchoInferenceProvider):
        provider_name = "invalid-implementation-version"
        implementation_version = "not valid!"

    with pytest.raises(ProviderRegistrationError, match="implementation_version"):
        register_inference_provider(InvalidImplementationVersion)


def test_registry_rejects_abstract_provider() -> None:
    class AbstractProvider(InferenceProvider):
        provider_name = "abstract-contract-test"

    with pytest.raises(ProviderRegistrationError, match="required methods"):
        register_inference_provider(AbstractProvider)  # type: ignore[arg-type]


def test_registry_requires_zero_argument_and_sanitized_initialization() -> None:
    class RequiredArgumentProvider(_EchoInferenceProvider):
        provider_name = "required-argument-contract-test"

        def __init__(self, required: str) -> None:
            self.required = required

    with pytest.raises(ProviderRegistrationError, match="must not require arguments"):
        register_inference_provider(RequiredArgumentProvider)

    class FailingInitializationProvider(_EchoInferenceProvider):
        provider_name = "failing-init-contract-test"

        def __init__(self) -> None:
            raise RuntimeError("initialization leaked a-secret")

    register_inference_provider(FailingInitializationProvider)
    with pytest.raises(ProviderConfigurationError) as exc_info:
        get_inference_provider("failing-init-contract-test")
    assert "a-secret" not in str(exc_info.value)
    assert "RuntimeError" in str(exc_info.value)
    assert "a-secret" not in _provider_traceback_locals(exc_info.value)
    assert exc_info.value.__context__ is None

    class ProviderErrorInitializationProvider(_EchoInferenceProvider):
        provider_name = "provider-error-init-contract-test"

        def __init__(self) -> None:
            raise ProviderConfigurationError("provider error leaked a-secret")

    register_inference_provider(ProviderErrorInitializationProvider)
    with pytest.raises(ProviderConfigurationError) as provider_exc_info:
        get_inference_provider("provider-error-init-contract-test")
    assert "a-secret" not in str(provider_exc_info.value)
    assert "ProviderConfigurationError" in str(provider_exc_info.value)
    assert "a-secret" not in _provider_traceback_locals(provider_exc_info.value)
    assert provider_exc_info.value.__context__ is None


def test_unknown_provider_error_lists_safe_available_names() -> None:
    with pytest.raises(ProviderNotFoundError) as exc_info:
        get_inference_provider("missing-contract-test")

    message = str(exc_info.value)
    assert "missing-contract-test" in message
    assert "anthropic" in message

    unsafe_name = "missing\nprovider-secret"
    with pytest.raises(ProviderNotFoundError) as unsafe_exc_info:
        get_inference_provider(unsafe_name)
    assert unsafe_name not in str(unsafe_exc_info.value)
    assert "<invalid>" in str(unsafe_exc_info.value)


def test_default_batch_preserves_order_and_sanitizes_raw_failures() -> None:
    provider = _EchoInferenceProvider()
    requests = tuple(
        BatchInferenceRequest(request_id, _request(content))
        for request_id, content in (
            ("a", "first"),
            ("b", "raw-error"),
            ("c", "provider-error"),
        )
    )

    result = provider.complete_batch(
        requests,
        credential=None,
        runtime=ProviderRuntimeOptions(max_retries=2),
        poll_seconds=0,
    )

    assert [item.request_id for item in result.items] == ["a", "b", "c"]
    assert result.items[0].result == InferenceResult(
        {"value": "first"},
        usage=ProviderUsage(input_tokens=1, output_tokens=2),
    )
    raw_error = result.items[1].error
    assert isinstance(raw_error, ProviderRequestError)
    assert raw_error.code == "RuntimeError"
    assert "a-secret" not in str(raw_error)
    assert isinstance(result.items[2].error, ProviderResponseError)
    assert result.batch_submissions == 0


def test_inference_provider_validates_result_type_and_explicit_model() -> None:
    provider = _EchoInferenceProvider()

    with pytest.raises(ProviderResponseError, match="invalid type"):
        provider.validate_result({})
    with pytest.raises(ProviderConfigurationError, match="non-empty model"):
        provider.resolve_model("")


def test_default_batch_rejects_duplicate_ids_and_invalid_poll_interval() -> None:
    provider = _EchoInferenceProvider()
    duplicate = (
        BatchInferenceRequest("same", _request()),
        BatchInferenceRequest("same", _request()),
    )

    with pytest.raises(ProviderBatchError, match="unique"):
        provider.complete_batch(
            duplicate,
            credential=None,
            runtime=ProviderRuntimeOptions(),
            poll_seconds=0,
        )
    with pytest.raises(ValueError, match="poll_seconds"):
        provider.complete_batch(
            (),
            credential=None,
            runtime=ProviderRuntimeOptions(),
            poll_seconds=float("nan"),
        )


def test_batch_result_must_align_one_to_one_with_requests() -> None:
    provider = _EchoInferenceProvider()
    requests = (
        BatchInferenceRequest("a", _request()),
        BatchInferenceRequest("b", _request()),
    )
    missing = BatchInferenceResult(
        (BatchInferenceItem("a", result=InferenceResult({"value": "a"})),)
    )

    with pytest.raises(ProviderBatchError, match="one-to-one"):
        provider.validate_batch_result(requests, missing)


def test_batch_result_runs_successful_items_through_provider_validation() -> None:
    secret = "unsafe-batch-validation-detail"

    class RejectingProvider(_EchoInferenceProvider):
        def validate_result(self, result: object) -> InferenceResult:
            del result
            raise ProviderResponseError(secret)

    request = BatchInferenceRequest("a", _request())
    raw = BatchInferenceResult(
        (BatchInferenceItem("a", result=InferenceResult({"value": "a"})),)
    )

    validated = RejectingProvider().validate_batch_result((request,), raw)

    assert validated.items[0].result is None
    assert isinstance(validated.items[0].error, ProviderResponseError)
    assert secret not in str(validated.items[0].error)


def test_batch_validation_crash_has_safe_opt_in_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "unsafe-validator-detail"

    class CrashingProvider(_EchoInferenceProvider):
        def validate_result(self, result: object) -> InferenceResult:
            del result
            raise RuntimeError(secret)

    monkeypatch.setenv("DBT_ML_DEBUG_PROVIDER_ERRORS", "1")
    caplog.set_level(logging.DEBUG)
    request = BatchInferenceRequest("a", _request())
    raw = BatchInferenceResult(
        (BatchInferenceItem("a", result=InferenceResult({"value": "a"})),)
    )

    validated = CrashingProvider().validate_batch_result((request,), raw)

    assert isinstance(validated.items[0].error, ProviderResponseError)
    assert secret not in caplog.text
    assert "builtins.RuntimeError" in caplog.text
    assert "batch item validation failed" in caplog.text


def test_embedding_contract_returns_one_vector_per_input() -> None:
    provider = _EchoEmbeddingProvider()
    request = EmbeddingRequest(model="embed-test", texts=("one", "three"))

    result = provider.embed(
        request,
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )

    assert result.vectors == ((3.0, 1.0), (5.0, 1.0))
    assert result.dimensions == 2
    assert result.usage.input_tokens == 2


def test_embedding_provider_rejects_misaligned_results() -> None:
    class MisalignedEmbeddingProvider(_EchoEmbeddingProvider):
        def _embed(
            self,
            request: EmbeddingRequest,
            *,
            credential: ProviderCredential | None,
            runtime: ProviderRuntimeOptions,
        ) -> EmbeddingResult:
            del request, credential, runtime
            return EmbeddingResult(
                vectors=((1.0, 2.0),),
                model="embed-test",
                dimensions=2,
            )

    with pytest.raises(ProviderResponseError, match="one embedding per input"):
        MisalignedEmbeddingProvider().embed(
            EmbeddingRequest(model="embed-test", texts=("one", "two")),
            credential=None,
            runtime=ProviderRuntimeOptions(),
        )

    class WrongModelEmbeddingProvider(_EchoEmbeddingProvider):
        def _embed(
            self,
            request: EmbeddingRequest,
            *,
            credential: ProviderCredential | None,
            runtime: ProviderRuntimeOptions,
        ) -> EmbeddingResult:
            del request, credential, runtime
            return EmbeddingResult(
                vectors=((1.0, 2.0),),
                model="different-model",
                dimensions=2,
            )

    with pytest.raises(ProviderResponseError, match="model does not match"):
        WrongModelEmbeddingProvider().embed(
            EmbeddingRequest(model="embed-test", texts=("one",)),
            credential=None,
            runtime=ProviderRuntimeOptions(),
        )


def test_embedding_error_drops_native_exception_and_credential_locals() -> None:
    sentinel = "distinctive-embedding-credential-secret"

    class CrashingEmbeddingProvider(_EchoEmbeddingProvider):
        def _embed(
            self,
            request: EmbeddingRequest,
            *,
            credential: ProviderCredential | None,
            runtime: ProviderRuntimeOptions,
        ) -> EmbeddingResult:
            del request, credential, runtime
            raise RuntimeError(sentinel)

    with pytest.raises(ProviderRequestError) as exc_info:
        CrashingEmbeddingProvider().embed(
            EmbeddingRequest(model="embed-test", texts=("one",)),
            credential=ProviderCredential(sentinel),
            runtime=ProviderRuntimeOptions(),
        )

    assert sentinel not in str(exc_info.value)
    assert sentinel not in _provider_traceback_locals(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"input_tokens": True}, "input_tokens"),
        ({"output_tokens": -1}, "output_tokens"),
        ({"reported_cost_usd": float("inf")}, "reported_cost_usd"),
        ({"unknown": 1}, "unknown provider usage fields"),
    ],
)
def test_usage_validation(values: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ProviderUsage.from_mapping(values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 1.5},
        {"temperature": True},
        {"output_schema": []},
    ],
)
def test_inference_request_rejects_runtime_type_mismatches(
    kwargs: dict[str, Any],
) -> None:
    values: dict[str, Any] = {
        "model": "test-model",
        "content": "hello",
        "system_prompt": "system",
        "output_schema": {"type": "object"},
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        InferenceRequest(**values)  # type: ignore[arg-type]


def test_embedding_result_rejects_bad_dimensions_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        EmbeddingResult(vectors=((1.0,),), model="embed", dimensions=2)
    with pytest.raises(ValueError, match="finite numbers"):
        EmbeddingResult(
            vectors=((float("nan"),),),
            model="embed",
            dimensions=1,
        )


def test_credentials_are_explicit_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "credential-that-must-not-leak"
    monkeypatch.setenv("CONTRACT_TEST_API_KEY", secret)

    class CredentialProvider(_EchoInferenceProvider):
        provider_name = "credential-contract-test"
        requires_credentials = True
        default_credential_env = "CONTRACT_TEST_API_KEY"

    credential = CredentialProvider().resolve_credential(None)

    assert credential is not None
    assert credential.reveal() == secret
    assert secret not in repr(credential)
    assert secret not in str(credential)
    assert "CONTRACT_TEST_API_KEY" not in repr(credential)

    other = ProviderCredential("other-private-value")
    assert credential == other
    assert hash(credential) == hash(other)
    assert canonical_fingerprint(
        credential, domain="provider-credential-test"
    ) == canonical_fingerprint(other, domain="provider-credential-test")

    monkeypatch.delenv("CONTRACT_TEST_API_KEY")
    with pytest.raises(ProviderConfigurationError) as exc_info:
        CredentialProvider().resolve_credential(None)
    assert secret not in str(exc_info.value)
    assert "CONTRACT_TEST_API_KEY" not in str(exc_info.value)
    assert "CONTRACT_TEST_API_KEY" not in _provider_traceback_locals(
        exc_info.value
    )

    unsafe_env_name = "BAD\nsecret-name"
    with pytest.raises(ProviderConfigurationError) as unsafe_exc_info:
        CredentialProvider().resolve_credential(unsafe_env_name)
    assert unsafe_env_name not in str(unsafe_exc_info.value)
    assert unsafe_env_name not in _provider_traceback_locals(
        unsafe_exc_info.value
    )

    with pytest.raises(CredentialReferenceError) as reference_exc_info:
        CredentialReference.from_env_name(unsafe_env_name)
    assert unsafe_env_name not in str(reference_exc_info.value)

    with pytest.raises(TypeError):
        ProviderCredential("LEGACY_ENV_NAME", secret)
    assert not hasattr(credential, "env_var")
    assert PROVIDER_CONTRACT_VERSION == 2


def test_llm_credential_helper_drops_reference_names_from_traceback_locals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_name = "PRIVATE_HELPER_TRACEBACK_REFERENCE"
    monkeypatch.delenv(reference_name, raising=False)

    class CredentialProvider(_EchoInferenceProvider):
        provider_name = "credential-traceback-test"
        requires_credentials = True
        default_credential_env = None

    register_inference_provider(CredentialProvider)

    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_llm_credential(
            {
                "provider": "credential-traceback-test",
                "api_key_env": reference_name,
            }
        )

    assert reference_name not in str(exc_info.value)
    assert reference_name not in _provider_traceback_locals(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None

    with pytest.raises(ProviderConfigurationError) as helper_exc_info:
        extract_fields_from_text(
            "credential-safe helper input",
            fields_spec=[{"name": "value", "type": "string"}],
            provider="credential-traceback-test",
            model="test-model",
            api_key_env=reference_name,
        )

    assert reference_name not in _provider_traceback_locals(
        helper_exc_info.value
    )
    assert helper_exc_info.value.__cause__ is None
    assert helper_exc_info.value.__context__ is None


def test_request_error_rejects_unsafe_provider_error_text() -> None:
    error = ProviderRequestError(
        "unsafe\nprovider-secret",
        "unsafe\noperation-secret",
        code="raw secret=do-not-expose",
    )

    assert error.code == "provider_error"
    assert "do-not-expose" not in str(error)
    assert "provider-secret" not in str(error)
    assert "operation-secret" not in str(error)
    assert str(error) == "provider request failed [provider_error]"

    with pytest.raises(ValueError, match="retryable must be boolean"):
        ProviderRequestError(
            "test",
            "inference",
            code="safe",
            retryable=1,  # type: ignore[arg-type]
        )


def test_provider_boundary_replaces_unsafe_provider_error_text() -> None:
    secret = "provider-authored-secret"

    class UnsafeErrorProvider(_EchoInferenceProvider):
        provider_name = "unsafe-error-contract-test"

        def complete(
            self,
            request: InferenceRequest,
            *,
            credential: ProviderCredential | None,
            runtime: ProviderRuntimeOptions,
        ) -> InferenceResult:
            del request, credential, runtime
            raise ProviderResponseError(f"unsafe upstream message {secret}")

    register_inference_provider(UnsafeErrorProvider)

    with pytest.raises(ProviderResponseError) as exc_info:
        extract_fields_from_text(
            "private document",
            provider="unsafe-error-contract-test",
            model="test-model",
            fields_spec=[{"name": "value"}],
        )

    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None

    with pytest.raises(ValueError, match="safe_for_display must be boolean"):
        ProviderResponseError(
            "must remain unsafe",
            safe_for_display="false",  # type: ignore[arg-type]
        )


def test_model_resolution_boundary_replaces_unsafe_provider_error() -> None:
    secret = "model-resolution-secret"

    class UnsafeModelProvider(_EchoInferenceProvider):
        provider_name = "unsafe-model-resolution-test"

        def resolve_model(self, model: str | None) -> str:
            del model
            raise ProviderConfigurationError(f"unsafe model error {secret}")

    with pytest.raises(ProviderConfigurationError) as exc_info:
        resolve_provider_model(UnsafeModelProvider(), "model")

    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in _provider_traceback_locals(exc_info.value)


def test_provider_implementation_identity_is_stable_and_class_specific() -> None:
    class AlternateProvider(_EchoInferenceProvider):
        provider_name = "alternate-contract-test"

        def complete(
            self,
            request: InferenceRequest,
            *,
            credential: ProviderCredential | None,
            runtime: ProviderRuntimeOptions,
        ) -> InferenceResult:
            del credential, runtime
            return InferenceResult({"alternate": request.content})

    first = _EchoInferenceProvider().implementation_identity()
    second = _EchoInferenceProvider().implementation_identity()
    alternate = AlternateProvider().implementation_identity()

    assert first == second
    assert first.startswith("provider-v")
    assert first != alternate


def test_provider_implementation_version_changes_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_base._implementation_identity.cache_clear()
    first = _EchoInferenceProvider().implementation_identity()
    with monkeypatch.context() as patch:
        patch.setattr(_EchoInferenceProvider, "implementation_version", "2")
        provider_base._implementation_identity.cache_clear()
        second = _EchoInferenceProvider().implementation_identity()
    provider_base._implementation_identity.cache_clear()

    assert first != second


def test_provider_debug_diagnostic_never_includes_exception_messages() -> None:
    secret = "private-one\nprivate-two"
    escaped = secret.replace("\n", r"\n")
    encoded = "private-one%0Aprivate-two"
    overlap = "private-one"
    source = compile(
        f'raise RuntimeError("{escaped} {encoded} {overlap}")',
        f"/tmp/{overlap}/sdk.py",
        "exec",
    )
    try:
        exec(source, {})
    except RuntimeError as error:
        diagnostic = provider_base.redacted_exception_text(
            error,
            sensitive=(secret, overlap),
        )

    assert "builtins.RuntimeError" in diagnostic
    assert "external frame" in diagnostic
    assert "private-one" not in diagnostic
    assert "private-two" not in diagnostic
    assert "%0A" not in diagnostic


def test_provider_dependency_version_changes_implementation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"dbt-ml": "1.0", "fake-provider-sdk": "1.0"}
    monkeypatch.setattr(
        provider_base,
        "package_version",
        lambda package: versions[package],
    )

    class DependencyProvider(_EchoInferenceProvider):
        provider_name = "dependency-version-contract-test"
        implementation_packages = ("fake-provider-sdk",)

    provider_base._implementation_identity.cache_clear()
    first = DependencyProvider().implementation_identity()
    versions["fake-provider-sdk"] = "2.0"
    provider_base._implementation_identity.cache_clear()
    second = DependencyProvider().implementation_identity()
    provider_base._implementation_identity.cache_clear()

    assert first != second
