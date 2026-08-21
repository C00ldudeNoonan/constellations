from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anthropic.types import ToolParam

from .base import (
    BatchInferenceItem,
    BatchInferenceRequest,
    BatchInferenceResult,
    BatchJobStatus,
    InferenceFailure,
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
    provider_batch_error,
    provider_error_debug_enabled,
    provider_request_error,
    redacted_exception_text,
    sanitized_provider_error,
    validate_batch_job_id,
)
from .registry import register_inference_provider

log = logging.getLogger(__name__)


@register_inference_provider
class AnthropicInferenceProvider(InferenceProvider):
    provider_name = "anthropic"
    implementation_version = "1"
    implementation_packages = ("anthropic",)
    default_model = "claude-haiku-4-5"
    default_credential_env = "ANTHROPIC_API_KEY"
    supports_native_batch = True
    max_batch_requests = 100_000
    batch_cost_multiplier = 0.5
    # Tool `input_schema` is JSON Schema; `enum` rides through unchanged.
    supports_schema_enum = True

    def complete(
        self,
        request: InferenceRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> InferenceResult:
        failure: ProviderError | None = None
        try:
            return _complete_with_sdk(
                request,
                credential=credential,
                runtime=runtime,
            )
        except ProviderError as error:
            failure = sanitized_provider_error(self.name(), "inference", error)
        except Exception as error:
            if provider_error_debug_enabled() and log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "anthropic inference request failed:\n%s",
                    redacted_exception_text(error),
                )
            failure = provider_request_error(self.name(), "inference", error)
        if failure is not None:
            del request
            raise failure
        raise AssertionError("anthropic inference did not produce a result")

    def submit_batch(
        self,
        requests: Sequence[BatchInferenceRequest],
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> str:
        batch_id: str | None = None
        failure: Exception | None = None
        try:
            self._validate_batch_inputs(requests, poll_seconds=0.0)
            if not requests:
                raise ProviderBatchError(
                    "anthropic batch submission requires at least one request",
                    safe_for_display=True,
                )
            batch_id = _submit_batch_with_sdk(
                requests, credential=credential, runtime=runtime
            )
        except ValueError as error:
            failure = ValueError(str(error))
        except ProviderError as error:
            failure = sanitized_provider_error(
                self.name(), "batch submission", error
            )
        except Exception as error:
            failure = self._sanitized_sdk_batch_failure("batch submission", error)
        if failure is not None:
            requests = ()
            raise failure
        if batch_id is None:
            raise AssertionError("anthropic batch submission produced no job id")
        return batch_id

    def poll_batch(
        self,
        batch_id: str,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> BatchJobStatus:
        status: BatchJobStatus | None = None
        failure: Exception | None = None
        try:
            validate_batch_job_id(batch_id)
            status = _poll_batch_with_sdk(
                batch_id, credential=credential, runtime=runtime
            )
        except ProviderError as error:
            failure = sanitized_provider_error(self.name(), "batch poll", error)
        except Exception as error:
            failure = self._sanitized_sdk_batch_failure("batch poll", error)
        if failure is not None:
            raise failure
        if status is None:
            raise AssertionError("anthropic batch poll produced no status")
        return status

    def fetch_batch_results(
        self,
        batch_id: str,
        requests: Sequence[BatchInferenceRequest],
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> BatchInferenceResult:
        result: BatchInferenceResult | None = None
        failure: Exception | None = None
        try:
            validate_batch_job_id(batch_id)
            result = _fetch_batch_results_with_sdk(
                batch_id, requests, credential=credential, runtime=runtime
            )
        except ProviderError as error:
            failure = sanitized_provider_error(
                self.name(), "batch results", error
            )
        except Exception as error:
            failure = self._sanitized_sdk_batch_failure("batch results", error)
        if failure is not None:
            requests = ()
            raise failure
        if result is None:
            raise AssertionError("anthropic batch fetch produced no result")
        return result

    def cancel_batch(
        self,
        batch_id: str,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> None:
        failure: Exception | None = None
        try:
            validate_batch_job_id(batch_id)
            _cancel_batch_with_sdk(
                batch_id, credential=credential, runtime=runtime
            )
        except ProviderError as error:
            failure = sanitized_provider_error(self.name(), "batch cancel", error)
        except Exception as error:
            failure = self._sanitized_sdk_batch_failure("batch cancel", error)
        if failure is not None:
            raise failure

    def _sanitized_sdk_batch_failure(
        self, operation: str, error: Exception
    ) -> ProviderBatchError:
        if provider_error_debug_enabled() and log.isEnabledFor(logging.DEBUG):
            log.debug(
                "anthropic %s failed:\n%s",
                operation,
                redacted_exception_text(error),
            )
        return provider_batch_error(self.name(), operation, error)


def _complete_with_sdk(
    request: InferenceRequest,
    *,
    credential: ProviderCredential | None,
    runtime: ProviderRuntimeOptions,
) -> InferenceResult:
    from anthropic import Anthropic

    api_key = _credential_value(credential)
    client = Anthropic(
        api_key=api_key,
        max_retries=runtime.max_retries,
        timeout=runtime.timeout_seconds,
    )
    response = client.messages.create(
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        system=request.system_prompt,
        tools=[_structured_output_tool(request)],
        tool_choice={"type": "tool", "name": request.output_name},
        messages=[{"role": "user", "content": request.content}],
    )
    try:
        return _parse_response_safely(response, request)
    except ProviderResponseError as error:
        raise _with_billed_failure(error, response, request) from None


def _batch_client(
    credential: ProviderCredential | None,
    runtime: ProviderRuntimeOptions,
) -> Any:
    from anthropic import Anthropic

    api_key = _credential_value(credential)
    return Anthropic(
        api_key=api_key,
        max_retries=runtime.max_retries,
        timeout=runtime.timeout_seconds,
    )


def _submit_batch_with_sdk(
    requests: Sequence[BatchInferenceRequest],
    *,
    credential: ProviderCredential | None,
    runtime: ProviderRuntimeOptions,
) -> str:
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
                "messages": [{"role": "user", "content": item.request.content}],
            },
        }
        for item in requests
    ]
    client = _batch_client(credential, runtime)
    batch = client.messages.batches.create(requests=payload)
    batch_id = validate_batch_job_id(getattr(batch, "id", None))
    log.info("submitted message batch %s (%d requests)", batch_id, len(payload))
    return batch_id


def _poll_batch_with_sdk(
    batch_id: str,
    *,
    credential: ProviderCredential | None,
    runtime: ProviderRuntimeOptions,
) -> BatchJobStatus:
    client = _batch_client(credential, runtime)
    batch = client.messages.batches.retrieve(batch_id)
    counts = getattr(batch, "request_counts", None)

    def _count(name: str) -> int:
        value = getattr(counts, name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0
        return value

    status = BatchJobStatus(
        done=getattr(batch, "processing_status", None) == "ended",
        processing=_count("processing"),
        succeeded=_count("succeeded"),
        errored=_count("errored"),
    )
    log.info(
        "batch %s: %s (processing=%d succeeded=%d errored=%d)",
        batch_id,
        "ended" if status.done else "in_progress",
        status.processing,
        status.succeeded,
        status.errored,
    )
    return status


def _fetch_batch_results_with_sdk(
    batch_id: str,
    requests: Sequence[BatchInferenceRequest],
    *,
    credential: ProviderCredential | None,
    runtime: ProviderRuntimeOptions,
) -> BatchInferenceResult:
    client = _batch_client(credential, runtime)
    raw_items = list(client.messages.batches.results(batch_id))
    return _normalize_batch_results(requests, raw_items)


def _cancel_batch_with_sdk(
    batch_id: str,
    *,
    credential: ProviderCredential | None,
    runtime: ProviderRuntimeOptions,
) -> None:
    client = _batch_client(credential, runtime)
    client.messages.batches.cancel(batch_id)


def _normalize_batch_results(
    requests: Sequence[BatchInferenceRequest],
    raw_items: Sequence[Any],
) -> BatchInferenceResult:
    request_ids = {item.request_id for item in requests}
    raw_by_id: dict[str, Any] = {}
    for raw in raw_items:
        request_id = getattr(raw, "custom_id", None)
        if not isinstance(request_id, str) or not request_id:
            raise ProviderBatchError(
                "anthropic native batch returned an invalid request ID",
                safe_for_display=True,
            )
        if request_id not in request_ids:
            raise ProviderBatchError(
                "anthropic native batch returned an unknown request ID",
                safe_for_display=True,
            )
        if request_id in raw_by_id:
            raise ProviderBatchError(
                "anthropic native batch returned a duplicate request ID",
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
                        "anthropic", "batch item", code="missing_result"
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
                        "anthropic", "batch item", code=error_code
                    ),
                )
            )
            continue
        message = getattr(raw_result, "message", None)
        try:
            parsed = _parse_response_safely(message, item.request)
        except ProviderError as error:
            safe_error = sanitized_provider_error("anthropic", "batch item", error)
            # A rejected-but-billed response keeps its usage on the error
            # side so budgets and run results account for it (issue #71).
            usage = _best_effort_usage(message)
            if usage is not None:
                safe_error.attach_failure(
                    InferenceFailure(
                        error_code="invalid_response",
                        usage=usage,
                        billed_requests=1,
                        provider="anthropic",
                        model=item.request.model,
                        implementation_identity=(
                            AnthropicInferenceProvider().implementation_identity()
                        ),
                    )
                )
            items.append(
                BatchInferenceItem(request_id, error=safe_error, usage=usage)
            )
        else:
            items.append(BatchInferenceItem(request_id, result=parsed))
    return BatchInferenceResult(tuple(items), batch_submissions=1)


def _credential_value(credential: ProviderCredential | None) -> str:
    if credential is None:
        raise ProviderRequestError(
            "anthropic", "credential resolution", code="missing_credential"
        )
    return credential.reveal()


def _structured_output_tool(request: InferenceRequest) -> ToolParam:
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


def _best_effort_usage(response: Any) -> ProviderUsage | None:
    """Usage reported alongside a rejected response, for billed-failure
    accounting only; None when the response carries no valid usage."""
    raw_usage = getattr(response, "usage", None)
    if raw_usage is None:
        return None
    try:
        return ProviderUsage(
            input_tokens=_usage_value(raw_usage, "input_tokens", required=True),
            output_tokens=_usage_value(raw_usage, "output_tokens", required=True),
            cache_read_input_tokens=_usage_value(
                raw_usage, "cache_read_input_tokens"
            ),
            cache_creation_input_tokens=_usage_value(
                raw_usage, "cache_creation_input_tokens"
            ),
        )
    except (ProviderResponseError, ValueError):
        return None


def _with_billed_failure(
    error: ProviderResponseError,
    response: Any,
    request: InferenceRequest,
) -> ProviderResponseError:
    """Attach billed usage to a rejected-response error (issue #71)."""
    usage = _best_effort_usage(response)
    if usage is None:
        return error
    return error.attach_failure(
        InferenceFailure(
            error_code="invalid_response",
            usage=usage,
            billed_requests=1,
            provider="anthropic",
            model=request.model,
            implementation_identity=(
                AnthropicInferenceProvider().implementation_identity()
            ),
        )
    )


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
