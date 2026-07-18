from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable
from typing import overload

from ..endpoints import EndpointUrlError, OpenAICompatibleBaseUrl
from .base import (
    BaseProvider,
    EmbeddingProvider,
    InferenceProvider,
    ProviderConfigurationError,
    ProviderNotFoundError,
    ProviderRegistrationError,
)

_INFERENCE_PROVIDERS: dict[str, type[InferenceProvider]] = {}
_EMBEDDING_PROVIDERS: dict[str, type[EmbeddingProvider]] = {}
_PROVIDER_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DISTRIBUTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_IMPLEMENTATION_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@overload
def register_inference_provider(
    provider_cls: type[InferenceProvider], /
) -> type[InferenceProvider]: ...


@overload
def register_inference_provider(
    provider_cls: None = None, /
) -> Callable[[type[InferenceProvider]], type[InferenceProvider]]: ...


def register_inference_provider(
    provider_cls: type[InferenceProvider] | None = None, /
) -> (
    type[InferenceProvider]
    | Callable[[type[InferenceProvider]], type[InferenceProvider]]
):
    def decorator(cls: type[InferenceProvider]) -> type[InferenceProvider]:
        _validate_provider_class(cls, capability="inference")
        _validate_inference_metadata(cls)
        _register(_INFERENCE_PROVIDERS, cls, capability="inference")
        return cls

    return decorator if provider_cls is None else decorator(provider_cls)


@overload
def register_embedding_provider(
    provider_cls: type[EmbeddingProvider], /
) -> type[EmbeddingProvider]: ...


@overload
def register_embedding_provider(
    provider_cls: None = None, /
) -> Callable[[type[EmbeddingProvider]], type[EmbeddingProvider]]: ...


def register_embedding_provider(
    provider_cls: type[EmbeddingProvider] | None = None, /
) -> (
    type[EmbeddingProvider]
    | Callable[[type[EmbeddingProvider]], type[EmbeddingProvider]]
):
    def decorator(cls: type[EmbeddingProvider]) -> type[EmbeddingProvider]:
        _validate_provider_class(cls, capability="embedding")
        _register(_EMBEDDING_PROVIDERS, cls, capability="embedding")
        return cls

    return decorator if provider_cls is None else decorator(provider_cls)


def get_inference_provider(name: str) -> InferenceProvider:
    cls = _INFERENCE_PROVIDERS.get(name) if isinstance(name, str) else None
    if cls is None:
        display_name = name if _is_provider_name(name) else "<invalid>"
        raise ProviderNotFoundError(
            f"Inference provider '{display_name}' is not registered. "
            f"Available: {sorted(_INFERENCE_PROVIDERS)}"
        )
    return _instantiate(cls, name=name, capability="inference")


def get_embedding_provider(name: str) -> EmbeddingProvider:
    cls = _EMBEDDING_PROVIDERS.get(name) if isinstance(name, str) else None
    if cls is None:
        display_name = name if _is_provider_name(name) else "<invalid>"
        raise ProviderNotFoundError(
            f"Embedding provider '{display_name}' is not registered. "
            f"Available: {sorted(_EMBEDDING_PROVIDERS)}"
        )
    return _instantiate(cls, name=name, capability="embedding")


def list_inference_providers() -> list[str]:
    return sorted(_INFERENCE_PROVIDERS)


def list_embedding_providers() -> list[str]:
    return sorted(_EMBEDDING_PROVIDERS)


def _register[ProviderType: BaseProvider](
    registry: dict[str, type[ProviderType]],
    cls: type[ProviderType],
    *,
    capability: str,
) -> None:
    name = cls.name()
    existing = registry.get(name)
    if existing is not None and existing is not cls:
        raise ProviderRegistrationError(
            f"{capability.capitalize()} provider '{name}' is already registered "
            f"by {existing.__module__}.{existing.__qualname__}"
        )
    registry[name] = cls


def _validate_provider_class(
    cls: type[BaseProvider],
    *,
    capability: str,
) -> None:
    expected_type = (
        InferenceProvider if capability == "inference" else EmbeddingProvider
    )
    if not isinstance(cls, type) or not issubclass(cls, expected_type):
        raise ProviderRegistrationError(
            f"{capability} provider must subclass {expected_type.__name__}"
        )
    if inspect.isabstract(cls):
        raise ProviderRegistrationError(
            f"{capability} provider must implement its required methods"
        )
    try:
        inspect.signature(cls).bind()
    except (TypeError, ValueError):
        raise ProviderRegistrationError(
            f"{capability} provider constructor must not require arguments"
        ) from None
    name = getattr(cls, "provider_name", None)
    if not isinstance(name, str) or not _PROVIDER_NAME.fullmatch(name):
        raise ProviderRegistrationError(
            f"{capability} provider name must match {_PROVIDER_NAME.pattern}"
        )
    implementation_version = getattr(cls, "implementation_version", None)
    if not isinstance(implementation_version, str) or not (
        _IMPLEMENTATION_VERSION.fullmatch(implementation_version)
    ):
        raise ProviderRegistrationError(
            f"{capability} provider implementation_version must match "
            f"{_IMPLEMENTATION_VERSION.pattern}"
        )
    if not isinstance(cls.requires_credentials, bool):
        raise ProviderRegistrationError(
            f"{capability} provider requires_credentials must be boolean"
        )
    env_var = cls.default_credential_env
    if env_var is not None and (
        not isinstance(env_var, str) or not _ENV_VAR_NAME.fullmatch(env_var)
    ):
        raise ProviderRegistrationError(
            f"{capability} provider default_credential_env is invalid"
        )
    packages = cls.implementation_packages
    if (
        not isinstance(packages, tuple)
        or len(packages) != len(set(packages))
        or any(
            not isinstance(package, str)
            or not _DISTRIBUTION_NAME.fullmatch(package)
            for package in packages
        )
    ):
        raise ProviderRegistrationError(
            f"{capability} provider implementation_packages must be unique "
            "distribution names"
        )


def _validate_inference_metadata(cls: type[InferenceProvider]) -> None:
    if cls.default_model is not None and (
        not isinstance(cls.default_model, str) or not cls.default_model
    ):
        raise ProviderRegistrationError(
            "inference provider default_model must be a non-empty string"
        )
    if not isinstance(cls.supports_native_batch, bool):
        raise ProviderRegistrationError(
            "inference provider supports_native_batch must be boolean"
        )
    if not isinstance(cls.supports_custom_base_url, bool):
        raise ProviderRegistrationError(
            "inference provider supports_custom_base_url must be boolean"
        )
    if not isinstance(cls.requires_base_url, bool):
        raise ProviderRegistrationError(
            "inference provider requires_base_url must be boolean"
        )
    if cls.requires_base_url and not cls.supports_custom_base_url:
        raise ProviderRegistrationError(
            "inference provider requiring base_url must support custom base URLs"
        )
    if cls.default_base_url is not None:
        if not cls.supports_custom_base_url:
            raise ProviderRegistrationError(
                "inference provider with default_base_url must support custom base URLs"
            )
        try:
            OpenAICompatibleBaseUrl(cls.default_base_url)
        except EndpointUrlError as error:
            raise ProviderRegistrationError(
                "inference provider default_base_url is invalid"
            ) from error
    if cls.supports_native_batch:
        unimplemented = [
            name
            for name in (
                "submit_batch",
                "poll_batch",
                "fetch_batch_results",
                "cancel_batch",
            )
            if getattr(cls, name) is getattr(InferenceProvider, name)
        ]
        if unimplemented:
            raise ProviderRegistrationError(
                "inference provider advertising native batch support must "
                f"override {', '.join(unimplemented)}"
            )
    limit = cls.max_batch_requests
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ProviderRegistrationError(
            "inference provider max_batch_requests must be a positive integer"
        )
    multiplier = cls.batch_cost_multiplier
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, (int, float))
        or not math.isfinite(multiplier)
        or multiplier <= 0
    ):
        raise ProviderRegistrationError(
            "inference provider batch_cost_multiplier must be positive and finite"
        )


def _instantiate[ProviderType: BaseProvider](
    cls: type[ProviderType],
    *,
    name: str,
    capability: str,
) -> ProviderType:
    failure: ProviderConfigurationError | None = None
    try:
        return cls()
    except Exception as error:
        failure = ProviderConfigurationError(
            f"{capability} provider '{name}' could not be initialized "
            f"[{type(error).__name__}]"
        )
    raise failure


def _is_provider_name(value: object) -> bool:
    return isinstance(value, str) and bool(_PROVIDER_NAME.fullmatch(value))
