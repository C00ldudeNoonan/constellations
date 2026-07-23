from __future__ import annotations

from ..providers import (
    ProviderBatchError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRequestError,
    ProviderResponseError,
    sanitized_provider_error,
)


def provider_error_in_chain(error: BaseException) -> ProviderError | None:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, ProviderError):
            return current
        current = current.__cause__
    return None


def artifact_error_text(error: Exception) -> str:
    provider_error = provider_error_in_chain(error)
    if isinstance(provider_error, ProviderConfigurationError):
        return "ProviderConfigurationError: provider configuration is invalid"
    if isinstance(provider_error, ProviderRequestError):
        safe = sanitized_provider_error(
            provider_error.provider,
            provider_error.operation,
            provider_error,
        )
        return f"ProviderRequestError: {safe}"
    if isinstance(provider_error, ProviderResponseError):
        return "ProviderResponseError: provider response is invalid"
    if isinstance(provider_error, ProviderBatchError):
        return "ProviderBatchError: provider batch operation failed"
    if isinstance(provider_error, ProviderError):
        return "ProviderError: provider operation failed"
    return f"{type(error).__name__}: {error}"
