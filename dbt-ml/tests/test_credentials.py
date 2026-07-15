from __future__ import annotations

import json
import pickle
from copy import copy, deepcopy
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from dbt_ml.backends.options import LLMBackendOptions
from dbt_ml.config.profile import LLMConfig, ProfileConfig, TargetConfig
from dbt_ml.credentials import (
    CredentialFreeUrl,
    CredentialReference,
    CredentialReferenceError,
    CredentialResolutionError,
    CredentialUrlError,
    ProtectedCredential,
)
from dbt_ml.hashing import canonical_fingerprint, canonical_json


class _CredentialConfig(BaseModel):
    credential: CredentialReference


class _UntypedProfileBlock(BaseModel):
    warehouse: dict[str, Any]


class _FutureRetrievalProfile(BaseModel):
    endpoint: str
    credential: CredentialReference = Field(repr=False, exclude=True)


def _dbt_ml_traceback_locals(error: BaseException) -> str:
    rendered: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        if "/src/dbt_ml/" in traceback.tb_frame.f_code.co_filename:
            rendered.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(rendered)


@pytest.mark.parametrize("name", ["API_KEY", "_PRIVATE_2", "lowercase_name"])
def test_literal_environment_name_reference(name: str) -> None:
    reference = CredentialReference.from_env_name(name)

    assert isinstance(reference, CredentialReference)


@pytest.mark.parametrize(
    "expression",
    [
        "{{ env_var('API_KEY') }}",
        '{{ env_var("API_KEY") }}',
        "{{env_var( 'API_KEY' )}}",
    ],
)
def test_exact_env_var_expression_reference(expression: str) -> None:
    reference = CredentialReference.from_env_var_expression(expression)

    assert isinstance(reference, CredentialReference)


@pytest.mark.parametrize(
    "expression",
    [
        "prefix {{ env_var('API_KEY') }}",
        "{{ env_var('API_KEY') }} suffix",
        "{{ env_var('API_KEY', 'default-secret') }}",
        "{{ env_var('API-KEY') }}",
        "${API_KEY}",
        "API_KEY",
    ],
)
def test_env_var_expression_rejects_defaults_and_mixed_values(
    expression: str,
) -> None:
    with pytest.raises(CredentialReferenceError) as exc_info:
        CredentialReference.from_env_var_expression(expression)

    assert expression not in str(exc_info.value)
    assert "default-secret" not in str(exc_info.value)


def test_literal_name_parser_does_not_accept_env_var_syntax() -> None:
    expression = "{{ env_var('DISTINCTIVE_CREDENTIAL_NAME') }}"

    with pytest.raises(CredentialReferenceError) as exc_info:
        CredentialReference.from_env_name(expression)

    assert "DISTINCTIVE_CREDENTIAL_NAME" not in str(exc_info.value)
    assert "DISTINCTIVE_CREDENTIAL_NAME" not in _dbt_ml_traceback_locals(
        exc_info.value
    )
    assert exc_info.value.__context__ is None


def test_credential_url_error_does_not_retain_user_information() -> None:
    sentinel = "distinctive-url-password"

    with pytest.raises(CredentialUrlError) as exc_info:
        CredentialFreeUrl(f"https://user:{sentinel}@example.test/token")

    assert sentinel not in str(exc_info.value)
    assert sentinel not in _dbt_ml_traceback_locals(exc_info.value)
    assert exc_info.value.__context__ is None


def test_missing_environment_variable_error_excludes_reference_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "DISTINCTIVE_MISSING_CREDENTIAL_NAME"
    monkeypatch.delenv(env_name, raising=False)
    reference = CredentialReference.from_env_name(env_name)

    with pytest.raises(CredentialResolutionError) as exc_info:
        reference.resolve()

    assert env_name not in str(exc_info.value)


def test_empty_environment_variable_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "EMPTY_CREDENTIAL_VALUE"
    monkeypatch.setenv(env_name, "")

    with pytest.raises(CredentialResolutionError):
        CredentialReference.from_env_name(env_name).resolve()


def test_reference_resolves_to_protected_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "TEST_PROTECTED_CREDENTIAL"
    secret = "distinctive-secret-value"
    monkeypatch.setenv(env_name, secret)

    credential = CredentialReference.from_env_name(env_name).resolve()

    assert isinstance(credential, ProtectedCredential)
    assert credential.reveal() == secret


@pytest.mark.parametrize(
    "factory,first,second",
    [
        (CredentialReference.from_env_name, "FIRST_ENV", "SECOND_ENV"),
        (ProtectedCredential, "first-secret", "second-secret"),
    ],
)
def test_opaque_representation_comparison_and_hashing(
    factory: Any,
    first: str,
    second: str,
) -> None:
    left = factory(first)
    right = factory(second)

    assert first not in repr(left)
    assert first not in str(left)
    assert first not in canonical_json(left)
    assert left == right
    assert hash(left) == hash(right)
    assert canonical_fingerprint(
        left, domain="credential-test"
    ) == canonical_fingerprint(right, domain="credential-test")


def test_pydantic_serialization_is_opaque_for_typed_and_untyped_fields() -> None:
    env_name = "DISTINCTIVE_SERIALIZED_CREDENTIAL_NAME"
    reference = CredentialReference.from_env_name(env_name)
    typed = _CredentialConfig(credential=reference)
    untyped = _UntypedProfileBlock(
        warehouse={"api_key_env": reference}
    )

    assert env_name not in repr(typed)
    assert env_name not in repr(typed.model_dump())
    assert env_name not in typed.model_dump_json()
    assert env_name not in repr(untyped.model_dump())
    assert env_name not in untyped.model_dump_json()
    assert typed.model_dump() == {"credential": "<redacted>"}
    assert json.loads(typed.model_dump_json()) == {"credential": "<redacted>"}
    assert json.loads(untyped.model_dump_json()) == {
        "warehouse": {"api_key_env": "<redacted>"}
    }


def test_future_retrieval_config_uses_the_same_excluded_reference_contract() -> None:
    env_name = "DISTINCTIVE_FUTURE_RETRIEVAL_CREDENTIAL"
    config = _FutureRetrievalProfile(
        endpoint="https://retrieval.example.test",
        credential=CredentialReference.from_env_name(env_name),
    )

    assert config.model_dump() == {"endpoint": "https://retrieval.example.test"}
    assert env_name not in repr(config)
    assert env_name not in config.model_dump_json()


def test_llm_configs_do_not_expose_or_fingerprint_reference_names() -> None:
    first_name = "FIRST_PRIVATE_LLM_REFERENCE"
    second_name = "SECOND_PRIVATE_LLM_REFERENCE"
    first_profile = LLMConfig(api_key_env=first_name)
    second_profile = LLMConfig(api_key_env=second_name)
    first_backend = LLMBackendOptions(
        fields=[{"name": "title", "type": "string"}],
        api_key_env=first_name,
    )
    second_backend = LLMBackendOptions(
        fields=[{"name": "title", "type": "string"}],
        api_key_env=second_name,
    )

    for config, name in (
        (first_profile, first_name),
        (second_profile, second_name),
        (first_backend, first_name),
        (second_backend, second_name),
    ):
        assert name not in repr(config)
        assert name not in repr(config.model_dump())
        assert name not in config.model_dump_json()
        assert config.api_key_env is not None
        assert name not in repr(config.api_key_env)

    assert first_profile == second_profile
    assert first_backend == second_backend
    assert hash(first_profile.api_key_env) == hash(second_profile.api_key_env)
    assert hash(first_backend.api_key_env) == hash(second_backend.api_key_env)
    assert canonical_fingerprint(
        first_profile, domain="llm-profile-config"
    ) == canonical_fingerprint(second_profile, domain="llm-profile-config")
    assert canonical_fingerprint(
        first_backend, domain="llm-backend-config"
    ) == canonical_fingerprint(second_backend, domain="llm-backend-config")


def test_pydantic_validation_error_does_not_echo_invalid_input() -> None:
    sentinel = "distinctive-secret-shaped-invalid-reference"

    with pytest.raises(ValidationError) as exc_info:
        _CredentialConfig(credential=sentinel)

    error = exc_info.value
    rendered = "\n".join(
        (str(error), repr(error), repr(error.errors()), error.json())
    )
    assert sentinel not in rendered
    assert sentinel.encode() not in pickle.dumps(error)


def _direct_profile_payload(
    model: type[BaseModel],
    credential: str,
) -> dict[str, Any]:
    warehouse = {
        "type": "bigquery",
        "project": "p",
        "token": credential,
    }
    if model is TargetConfig:
        return {"warehouse": warehouse}
    return {"outputs": {"dev": {"warehouse": warehouse}}}


@pytest.mark.parametrize("model", [TargetConfig, ProfileConfig])
@pytest.mark.parametrize(
    "unsafe_value",
    [
        "distinctive-direct-profile-literal",
        "prefix-{{ env_var('DISTINCTIVE_DIRECT_PROFILE_REFERENCE') }}",
    ],
)
def test_direct_profile_models_clear_rejected_credential_inputs(
    model: type[BaseModel],
    unsafe_value: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        model.model_validate(_direct_profile_payload(model, unsafe_value))

    error = exc_info.value
    rendered = "\n".join(
        (str(error), repr(error), repr(error.errors()), error.json())
    )
    assert "distinctive-direct-profile-literal" not in rendered
    assert "DISTINCTIVE_DIRECT_PROFILE_REFERENCE" not in rendered
    serialized = pickle.dumps(error)
    assert b"distinctive-direct-profile-literal" not in serialized
    assert b"DISTINCTIVE_DIRECT_PROFILE_REFERENCE" not in serialized
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize("model", [TargetConfig, ProfileConfig])
def test_direct_profile_models_keep_reference_names_opaque(
    model: type[BaseModel],
) -> None:
    reference_name = "DISTINCTIVE_DIRECT_PROFILE_REFERENCE"
    config = model.model_validate(
        _direct_profile_payload(
            model,
            f"{{{{ env_var('{reference_name}') }}}}",
        )
    )
    rendered = "\n".join(
        (
            str(config),
            repr(config),
            repr(config.model_dump()),
            config.model_dump_json(),
        )
    )

    assert reference_name not in rendered
    with pytest.raises(TypeError, match="cannot be serialized") as exc_info:
        pickle.dumps(config)
    assert reference_name not in str(exc_info.value)


def test_json_schema_contains_policy_not_values() -> None:
    env_name = "DISTINCTIVE_SCHEMA_CREDENTIAL_NAME"
    schema = _CredentialConfig.model_json_schema()
    rendered = json.dumps(schema)

    assert env_name not in rendered
    assert schema["properties"]["credential"]["format"] == "password"
    assert schema["properties"]["credential"]["writeOnly"] is True
    assert schema["properties"]["credential"]["pattern"] == (
        "^[A-Za-z_][A-Za-z0-9_]*$"
    )


@pytest.mark.parametrize(
    "value",
    [
        CredentialReference.from_env_name("PICKLE_REFERENCE"),
        ProtectedCredential("pickle-secret"),
    ],
)
def test_credential_objects_cannot_be_pickled(value: object) -> None:
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(value)


@pytest.mark.parametrize("copier", [copy, deepcopy])
def test_credential_objects_cannot_be_copied(copier: Any) -> None:
    reference = CredentialReference.from_env_name("PRIVATE_COPY_REFERENCE")

    with pytest.raises(TypeError, match="cannot be serialized"):
        copier(reference)
