from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from dbt_ml.providers import (
    BatchInferenceRequest,
    InferenceRequest,
    ProviderBatchError,
    ProviderCredential,
    ProviderRequestError,
    ProviderResponseError,
    ProviderRuntimeOptions,
)
from dbt_ml.providers.anthropic import AnthropicInferenceProvider


def _provider_traceback_locals(error: BaseException) -> str:
    rendered: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module, str) and module.startswith("dbt_ml"):
            rendered.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(rendered)


def _request(content: str = "document") -> InferenceRequest:
    return InferenceRequest(
        model="claude-test",
        content=content,
        system_prompt="Extract the value.",
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        max_tokens=321,
    )


def _message(value: str, *, request_id: str = "msg_1") -> SimpleNamespace:
    return SimpleNamespace(
        id=request_id,
        stop_reason="tool_use",
        usage=SimpleNamespace(
            input_tokens=7,
            output_tokens=3,
            cache_read_input_tokens=2,
            cache_creation_input_tokens=1,
        ),
        content=[
            SimpleNamespace(type="tool_use", name="extract", input={"value": value})
        ],
    )


def test_anthropic_complete_maps_request_and_normalizes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Messages:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            captured["request"] = kwargs
            return _message("ok")

    class FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.messages = Messages()

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)
    provider = AnthropicInferenceProvider()

    result = provider.complete(
        _request(),
        credential=ProviderCredential("top-secret"),
        runtime=ProviderRuntimeOptions(max_retries=6),
    )

    assert captured["client"] == {"api_key": "top-secret", "max_retries": 6}
    request = captured["request"]
    assert request["model"] == "claude-test"
    assert request["max_tokens"] == 321
    assert request["tool_choice"] == {"type": "tool", "name": "extract"}
    assert request["tools"][0]["input_schema"]["required"] == ["value"]
    assert result.output == {"value": "ok"}
    assert result.usage.to_metrics() == {
        "input_tokens": 7,
        "output_tokens": 3,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 1,
    }
    assert result.provider_request_id == "msg_1"


def test_anthropic_request_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "top-secret-provider-detail"
    document_secret = "private-document-that-must-not-survive-traceback"

    class Messages:
        def create(self, **kwargs: Any) -> None:
            del kwargs
            raise RuntimeError(f"request failed: {secret}")

    class FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.messages = Messages()

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    with pytest.raises(ProviderRequestError) as exc_info:
        AnthropicInferenceProvider().complete(
            _request(document_secret),
            credential=ProviderCredential(secret),
            runtime=ProviderRuntimeOptions(),
        )

    assert exc_info.value.code == "RuntimeError"
    assert secret not in str(exc_info.value)
    assert secret not in _provider_traceback_locals(exc_info.value)
    assert document_secret not in _provider_traceback_locals(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_anthropic_rejects_truncated_and_malformed_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        SimpleNamespace(
            stop_reason="max_tokens",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            content=[],
        ),
        SimpleNamespace(
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            content=[SimpleNamespace(type="tool_use", name="extract", input=[])],
        ),
    ]

    class Messages:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            del kwargs
            return responses.pop(0)

    class FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.messages = Messages()

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)
    provider = AnthropicInferenceProvider()
    credential = ProviderCredential("api-key")

    with pytest.raises(ProviderResponseError, match="truncated"):
        provider.complete(
            _request(),
            credential=credential,
            runtime=ProviderRuntimeOptions(),
        )
    with pytest.raises(ProviderResponseError, match="must be a mapping"):
        provider.complete(
            _request(),
            credential=credential,
            runtime=ProviderRuntimeOptions(),
        )


def test_anthropic_batch_is_ordered_and_reports_item_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    raw_results = [
        SimpleNamespace(
            custom_id="b",
            result=SimpleNamespace(
                type="errored",
                error=SimpleNamespace(type="rate_limit_error"),
            ),
        ),
        SimpleNamespace(
            custom_id="a",
            result=SimpleNamespace(type="succeeded", message=_message("first")),
        ),
    ]

    class Batches:
        def create(self, *, requests: list[dict[str, Any]]) -> SimpleNamespace:
            captured["requests"] = requests
            return SimpleNamespace(id="batch_1")

        def retrieve(self, batch_id: str) -> SimpleNamespace:
            assert batch_id == "batch_1"
            return SimpleNamespace(id=batch_id, processing_status="ended")

        def results(self, batch_id: str) -> list[SimpleNamespace]:
            assert batch_id == "batch_1"
            return raw_results

    class FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.messages = SimpleNamespace(batches=Batches())

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)
    requests = tuple(
        BatchInferenceRequest(request_id, _request(request_id))
        for request_id in ("a", "b", "c")
    )

    result = AnthropicInferenceProvider().complete_batch(
        requests,
        credential=ProviderCredential("api-key"),
        runtime=ProviderRuntimeOptions(max_retries=8),
        poll_seconds=0,
    )

    assert captured["client"] == {"api_key": "api-key", "max_retries": 8}
    assert [item["custom_id"] for item in captured["requests"]] == ["a", "b", "c"]
    assert [item.request_id for item in result.items] == ["a", "b", "c"]
    assert result.items[0].result is not None
    assert result.items[0].result.output == {"value": "first"}
    assert isinstance(result.items[1].error, ProviderRequestError)
    assert result.items[1].error.code == "rate_limit_error"
    assert isinstance(result.items[2].error, ProviderRequestError)
    assert result.items[2].error.code == "missing_result"
    assert result.batch_submissions == 1


def test_anthropic_batch_detaches_malformed_item_error_from_raw_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "malformed-batch-message-that-must-not-survive"
    malformed_message = SimpleNamespace(
        provider_body=sentinel,
        stop_reason="tool_use",
        usage=None,
        content=[],
    )

    class Batches:
        def create(self, *, requests: list[dict[str, Any]]) -> SimpleNamespace:
            del requests
            return SimpleNamespace(id="batch_1")

        def retrieve(self, batch_id: str) -> SimpleNamespace:
            return SimpleNamespace(id=batch_id, processing_status="ended")

        def results(self, batch_id: str) -> list[SimpleNamespace]:
            del batch_id
            return [
                SimpleNamespace(
                    custom_id="a",
                    result=SimpleNamespace(
                        type="succeeded",
                        message=malformed_message,
                    ),
                )
            ]

    class FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.messages = SimpleNamespace(batches=Batches())

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    result = AnthropicInferenceProvider().complete_batch(
        (BatchInferenceRequest("a", _request()),),
        credential=ProviderCredential("api-key"),
        runtime=ProviderRuntimeOptions(),
        poll_seconds=0,
    )

    stored_error = result.items[0].error
    assert isinstance(stored_error, ProviderResponseError)
    assert sentinel not in str(stored_error)
    assert sentinel not in repr(stored_error)
    assert sentinel not in _provider_traceback_locals(stored_error)
    assert stored_error.__traceback__ is None
    assert stored_error.__cause__ is None
    assert stored_error.__context__ is None


def test_anthropic_batch_rejects_unknown_result_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_secret = "raw-batch-response-that-must-not-survive-traceback"
    document_secret = "private-batch-document-that-must-not-survive-traceback"

    class Batches:
        def create(self, *, requests: list[dict[str, Any]]) -> SimpleNamespace:
            del requests
            return SimpleNamespace(id="batch_1")

        def retrieve(self, batch_id: str) -> SimpleNamespace:
            return SimpleNamespace(id=batch_id, processing_status="ended")

        def results(self, batch_id: str) -> list[SimpleNamespace]:
            del batch_id
            return [
                SimpleNamespace(
                    custom_id="unknown",
                    provider_body=response_secret,
                    result=SimpleNamespace(
                        type="succeeded", message=_message("unknown")
                    ),
                )
            ]

    class FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.messages = SimpleNamespace(batches=Batches())

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    with pytest.raises(ProviderBatchError, match="unknown request ID") as exc_info:
        AnthropicInferenceProvider().complete_batch(
            (BatchInferenceRequest("a", _request(document_secret)),),
            credential=ProviderCredential("api-key"),
            runtime=ProviderRuntimeOptions(),
            poll_seconds=0,
        )

    traceback_locals = _provider_traceback_locals(exc_info.value)
    assert response_secret not in traceback_locals
    assert document_secret not in traceback_locals
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_anthropic_batch_sdk_failure_does_not_retain_credential_locals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "batch-credential-that-must-not-survive-traceback"

    class Batches:
        def create(self, *, requests: list[dict[str, Any]]) -> None:
            del requests
            raise RuntimeError(f"batch failed with {secret}")

    class FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.messages = SimpleNamespace(batches=Batches())

    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    with pytest.raises(ProviderBatchError) as exc_info:
        AnthropicInferenceProvider().complete_batch(
            (BatchInferenceRequest("a", _request()),),
            credential=ProviderCredential(secret),
            runtime=ProviderRuntimeOptions(),
            poll_seconds=0,
        )

    assert secret not in str(exc_info.value)
    assert secret not in _provider_traceback_locals(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
