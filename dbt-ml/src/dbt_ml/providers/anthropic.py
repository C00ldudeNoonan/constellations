from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .base import (
    BatchInferenceItem,
    BatchInferenceRequest,
    BatchInferenceResult,
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    ProviderBatchError,
    ProviderCredential,
    ProviderError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderRuntimeOptions,
    ProviderUsage,
)
from .registry import register_inference_provider

log = logging.getLogger(__name__)


@register_inference_provider
class AnthropicInferenceProvider(InferenceProvider):
    provider_name = "anthropic"
    implementation_packages = ("anthropic",)
    default_model = "claude-haiku-4-5"
    default_credential_env = "ANTHROPIC_API_KEY"
    supports_native_batch = True
    max_batch_requests = 100_000
    batch_cost_multiplier = 0.5

    def complete(
        self,
        request: InferenceRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> InferenceResult:
        api_key = _credential_value(credential)
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key, max_retries=runtime.max_retries)
            response = client.messages.create(  # type: ignore[call-overload]
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system_prompt,
                tools=[_structured_output_tool(request)],
                tool_choice={"type": "tool", "name": request.output_name},
                messages=[{"role": "user", "content": request.content}],
            )
        except Exception as error:
            raise ProviderRequestError(
                self.name(), "inference", code=type(error).__name__
            ) from None
        return _parse_response_safely(response, request)

    def complete_batch(
        self,
        requests: Sequence[BatchInferenceRequest],
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
        poll_seconds: float,
    ) -> BatchInferenceResult:
        self._validate_batch_inputs(requests, poll_seconds=poll_seconds)
        if not requests:
            return BatchInferenceResult(())
        api_key = _credential_value(credential)
        payload = [
            {
                "custom_id": item.request_id,
                "params": {
                    "model": item.request.model,
                    "max_tokens": item.request.max_tokens,
                    "temperature": item.request.temperature,
                    "system": item.request.system_prompt,
                    "tools": [_structured_output_tool(item.request)],
                    "tool_choice": {
                        "type": "tool",
                        "name": item.request.output_name,
                    },
                    "messages": [
                        {"role": "user", "content": item.request.content}
                    ],
                },
            }
            for item in requests
        ]
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key, max_retries=runtime.max_retries)
            batch = client.messages.batches.create(requests=payload)  # type: ignore[arg-type]
            log.info("submitted message batch %s (%d requests)", batch.id, len(payload))
            while True:
                batch = client.messages.batches.retrieve(batch.id)
                if batch.processing_status == "ended":
                    break
                counts = batch.request_counts
                log.info(
                    "batch %s: %s (processing=%d succeeded=%d errored=%d)",
                    batch.id,
                    batch.processing_status,
                    counts.processing,
                    counts.succeeded,
                    counts.errored,
                )
                time.sleep(poll_seconds)
            raw_items = list(client.messages.batches.results(batch.id))
        except Exception as error:
            raise ProviderBatchError(
                f"{self.name()} native batch failed [{type(error).__name__}]",
                safe_for_display=True,
            ) from None

        request_ids = {item.request_id for item in requests}
        raw_by_id: dict[str, Any] = {}
        for raw in raw_items:
            request_id = getattr(raw, "custom_id", None)
            if not isinstance(request_id, str) or not request_id:
                raise ProviderBatchError(
                    f"{self.name()} native batch returned an invalid request ID",
                    safe_for_display=True,
                )
            if request_id not in request_ids:
                raise ProviderBatchError(
                    f"{self.name()} native batch returned an unknown request ID",
                    safe_for_display=True,
                )
            if request_id in raw_by_id:
                raise ProviderBatchError(
                    f"{self.name()} native batch returned a duplicate request ID",
                    safe_for_display=True,
                )
            raw_by_id[request_id] = raw

        items: list[BatchInferenceItem] = []
        for item in requests:
            request_id = item.request_id
            raw_item = raw_by_id.get(request_id)
            if raw_item is None:
                items.append(
                    BatchInferenceItem(
                        request_id,
                        error=ProviderRequestError(
                            self.name(), "batch item", code="missing_result"
                        ),
                    )
                )
                continue
            raw_result = getattr(raw_item, "result", None)
            result_type = getattr(raw_result, "type", None)
            if result_type != "succeeded":
                raw_error = getattr(raw_result, "error", None)
                raw_code = getattr(raw_error, "type", None)
                error_code = raw_code if isinstance(raw_code, str) else "batch_error"
                items.append(
                    BatchInferenceItem(
                        request_id,
                        error=ProviderRequestError(
                            self.name(), "batch item", code=error_code
                        ),
                    )
                )
                continue
            try:
                result = _parse_response_safely(
                    getattr(raw_result, "message", None), item.request
                )
            except ProviderError as error:
                items.append(BatchInferenceItem(request_id, error=error))
            else:
                items.append(BatchInferenceItem(request_id, result=result))
        return BatchInferenceResult(tuple(items), batch_submissions=1)


def _credential_value(credential: ProviderCredential | None) -> str:
    if credential is None:
        raise ProviderRequestError(
            "anthropic", "credential resolution", code="missing_credential"
        )
    return credential.reveal()


def _structured_output_tool(request: InferenceRequest) -> dict[str, Any]:
    return {
        "name": request.output_name,
        "description": request.output_description,
        "input_schema": dict(request.output_schema),
    }


def _parse_response(response: Any, request: InferenceRequest) -> InferenceResult:
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise ProviderResponseError(
            f"LLM response truncated at max_tokens={request.max_tokens}; partial "
            "structured outputs are never used.",
            safe_for_display=True,
        )
    raw_usage = getattr(response, "usage", None)
    if raw_usage is None:
        raise ProviderResponseError(
            "provider response is missing usage metadata",
            safe_for_display=True,
        )
    usage = ProviderUsage(
        input_tokens=_usage_value(raw_usage, "input_tokens", required=True),
        output_tokens=_usage_value(raw_usage, "output_tokens", required=True),
        cache_read_input_tokens=_usage_value(
            raw_usage, "cache_read_input_tokens"
        ),
        cache_creation_input_tokens=_usage_value(
            raw_usage, "cache_creation_input_tokens"
        ),
    )
    content = getattr(response, "content", None)
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        raise ProviderResponseError(
            "provider response content is malformed",
            safe_for_display=True,
        )
    for block in content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == request.output_name
        ):
            output = getattr(block, "input", None)
            if not isinstance(output, Mapping):
                raise ProviderResponseError(
                    "provider structured output must be a mapping",
                    safe_for_display=True,
                )
            raw_request_id = getattr(response, "id", None)
            request_id = (
                raw_request_id
                if isinstance(raw_request_id, str) and raw_request_id
                else None
            )
            return InferenceResult(
                dict(output),
                usage=usage,
                provider_request_id=request_id,
            )
    raise ProviderResponseError(
        "anthropic did not return the requested structured output",
        safe_for_display=True,
    )


def _parse_response_safely(
    response: Any,
    request: InferenceRequest,
) -> InferenceResult:
    try:
        return _parse_response(response, request)
    except ProviderResponseError:
        raise
    except Exception:
        raise ProviderResponseError(
            "anthropic returned a malformed inference response",
            safe_for_display=True,
        ) from None


def _usage_value(usage: Any, name: str, *, required: bool = False) -> int:
    value = getattr(usage, name, None)
    if value is None and not required:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderResponseError(
            f"provider returned invalid usage field '{name}'",
            safe_for_display=True,
        )
    return value
