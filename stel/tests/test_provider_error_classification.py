from __future__ import annotations

from stel.providers.base import (
    ProviderBatchError,
    ProviderRequestError,
    http_status_of,
    provider_batch_error,
    provider_request_error,
    sanitized_provider_error,
)


class _GenAIError(Exception):
    """Shaped like google-genai ClientError/APIError (carries `.code`)."""

    def __init__(self, code: int) -> None:
        super().__init__("boom")
        self.code = code


class _AnthropicError(Exception):
    """Shaped like anthropic APIStatusError (carries `.status_code`)."""

    def __init__(self, status_code: int) -> None:
        super().__init__("boom")
        self.status_code = status_code


class _HttpxError(Exception):
    """Carries the status on a nested response, like httpx."""

    def __init__(self, status: int) -> None:
        super().__init__("boom")
        self.response = type("R", (), {"status_code": status})()


def test_http_status_of_reads_common_sdk_shapes() -> None:
    assert http_status_of(_GenAIError(429)) == 429
    assert http_status_of(_AnthropicError(403)) == 403
    assert http_status_of(_HttpxError(503)) == 503
    assert http_status_of(ValueError("no status")) is None
    # A non-HTTP int on `.code` (e.g. an errno) is not mistaken for a status.
    assert http_status_of(_GenAIError(2)) is None


def test_provider_request_error_preserves_status_and_retryability() -> None:
    # 429 -> retryable; the code carries the real status, not just the type name.
    e = provider_request_error("vertex", "inference", _GenAIError(429))
    assert isinstance(e, ProviderRequestError)
    assert e.code == "http_429"
    assert e.retryable is True

    # 403 -> permanent; distinguishable from the 429 above.
    e = provider_request_error("vertex", "inference", _AnthropicError(403))
    assert e.code == "http_403"
    assert e.retryable is False

    # 500 family -> retryable.
    assert provider_request_error("vertex", "inference", _HttpxError(503)).retryable is True

    # No status available -> falls back to the exception type name, non-retryable.
    e = provider_request_error("vertex", "inference", RuntimeError("x"))
    assert e.code == "RuntimeError"
    assert e.retryable is False


def test_provider_batch_error_classifies_status_retryability() -> None:
    retryable = provider_batch_error(
        "anthropic", "batch poll", _AnthropicError(429)
    )
    assert isinstance(retryable, ProviderBatchError)
    assert retryable.retryable is True

    permanent = provider_batch_error(
        "anthropic", "batch results", _AnthropicError(403)
    )
    assert permanent.retryable is False

    without_status = provider_batch_error(
        "anthropic", "batch cancel", RuntimeError("unsafe detail")
    )
    assert without_status.retryable is False
    assert "unsafe detail" not in str(without_status)


def test_batch_retryability_survives_provider_boundary_sanitization() -> None:
    error = provider_batch_error(
        "anthropic", "batch poll", _AnthropicError(429)
    )
    sanitized = sanitized_provider_error(
        "anthropic",
        "batch inference",
        error,
    )

    assert isinstance(sanitized, ProviderBatchError)
    assert sanitized.retryable is True
    assert "boom" not in str(sanitized)
