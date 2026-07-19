from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError

import pytest

from dbt_ml.config.profile import resolve_llm_credential
from dbt_ml.credentials import ProtectedCredential
from dbt_ml.providers import (
    InferenceRequest,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderRuntimeOptions,
    get_inference_provider,
    list_inference_providers,
)
from dbt_ml.providers import vllm as vllm_provider


def _request() -> InferenceRequest:
    return InferenceRequest(
        model="invoice-extractor",
        content="Invoice INV-42 totals USD 10.50.",
        system_prompt="Extract invoice fields.",
        output_schema={
            "type": "object",
            "properties": {
                "invoice_id": {"type": "string"},
                "total": {"type": "number"},
            },
        },
    )


def _response(*, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": '{"invoice_id":"INV-42","total":10.5}'
                },
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }


def test_vllm_provider_is_registered_and_requires_explicit_routing() -> None:
    provider = get_inference_provider("vllm")

    assert "vllm" in list_inference_providers()
    assert provider.requires_credentials is False
    with pytest.raises(ProviderConfigurationError, match="explicit model"):
        provider.resolve_model(None)
    with pytest.raises(ProviderConfigurationError, match="requires base_url"):
        provider.resolve_base_url(None)


def test_vllm_maps_openai_chat_request_and_normalizes_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        url: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        captured.update(
            url=url,
            payload=payload,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )
        return _response()

    monkeypatch.setattr(vllm_provider, "_post_json", fake_post)
    result = get_inference_provider("vllm").complete(
        _request(),
        credential=ProtectedCredential("private-token"),
        runtime=ProviderRuntimeOptions(
            base_url="HTTP://LOCALHOST:80/v1/",
            timeout_seconds=12.5,
        ),
    )

    assert result.output == {"invoice_id": "INV-42", "total": 10.5}
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 8
    assert result.provider_request_id == "chatcmpl-test"
    assert captured["url"] == "http://localhost/v1/chat/completions"
    assert captured["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer private-token",
    }
    assert captured["timeout_seconds"] == 12.5
    payload = captured["payload"]
    assert payload["model"] == "invoice-extractor"
    assert payload["messages"][1]["content"] == _request().content
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "extract",
            "description": "Return the extracted structured fields.",
            "schema": dict(_request().output_schema),
        },
    }


def test_vllm_allows_unauthenticated_local_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_headers: Mapping[str, str] = {}

    def fake_post(
        url: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        del url, payload, timeout_seconds
        nonlocal captured_headers
        captured_headers = headers
        return _response()

    monkeypatch.setattr(vllm_provider, "_post_json", fake_post)
    get_inference_provider("vllm").complete(
        _request(),
        credential=None,
        runtime=ProviderRuntimeOptions(base_url="http://127.0.0.1:8000/v1"),
    )

    assert captured_headers == {"Content-Type": "application/json"}


def test_vllm_optional_credential_is_resolved_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_TEST_KEY", "private-token")

    credential = resolve_llm_credential(
        {"provider": "vllm", "api_key_env": "VLLM_TEST_KEY"}
    )

    assert credential is not None
    assert credential.reveal() == "private-token"


def test_vllm_retries_only_retryable_request_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def flaky_post(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        del args, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderRequestError(
                "vllm", "inference", code="http_503", retryable=True
            )
        return _response()

    monkeypatch.setattr(vllm_provider, "_post_json", flaky_post)
    monkeypatch.setattr(vllm_provider.time, "sleep", sleeps.append)

    result = get_inference_provider("vllm").complete(
        _request(),
        credential=None,
        runtime=ProviderRuntimeOptions(
            base_url="https://inference.example.test/v1",
            max_retries=1,
        ),
    )

    assert result.output["invoice_id"] == "INV-42"
    assert calls == 2
    assert sleeps == [0.25]


def test_vllm_rejects_truncated_and_malformed_output() -> None:
    with pytest.raises(ProviderResponseError, match="truncated"):
        vllm_provider._parse_response_safely(
            _response(finish_reason="length"), _request()
        )

    malformed = _response()
    malformed["choices"][0]["message"]["content"] = "not-json"
    with pytest.raises(ProviderResponseError, match="malformed JSON"):
        vllm_provider._parse_response_safely(malformed, _request())


def test_vllm_http_failures_do_not_expose_endpoint_or_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint_marker = "private-routing-marker"

    def fail(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise HTTPError(
            f"https://example.test/{endpoint_marker}",
            503,
            "upstream body marker",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(vllm_provider, "_open_url", fail)
    with pytest.raises(ProviderRequestError) as exc_info:
        vllm_provider._post_json(
            f"https://example.test/{endpoint_marker}",
            {},
            headers={"Content-Type": "application/json"},
            timeout_seconds=1,
        )

    error = exc_info.value
    assert error.code == "http_503"
    assert error.retryable is True
    assert endpoint_marker not in str(error)
    assert "upstream body marker" not in str(error)


def test_vllm_does_not_follow_http_redirects() -> None:
    handler = vllm_provider._RejectRedirects()
    redirected = handler.redirect_request(
        object(),
        None,
        307,
        "redirect",
        {},
        "https://other.example.test/v1/chat/completions",
    )

    assert redirected is None


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://example.test/v1",
        "https://user:secret@example.test/v1",
        "https://example.test/v1?token=secret",
        "https://example.test/v1#fragment",
    ],
)
def test_vllm_base_url_rejects_unsafe_or_ambiguous_values(base_url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        ProviderRuntimeOptions(base_url=base_url)
