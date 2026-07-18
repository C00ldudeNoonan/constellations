from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..credentials import (
    CredentialReference,
    CredentialReferenceError,
    CredentialResolutionError,
)
from .base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    ProviderConfigurationError,
    ProviderCredential,
    ProviderError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderRuntimeOptions,
    ProviderUsage,
    provider_error_debug_enabled,
    redacted_exception_text,
    sanitized_provider_error,
)
from .registry import register_inference_provider

log = logging.getLogger(__name__)

_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        del req, fp, code, msg, headers, newurl
        return None


_HTTP_OPENER = build_opener(_RejectRedirects())


@register_inference_provider
class VLLMInferenceProvider(InferenceProvider):
    provider_name = "vllm"
    implementation_version = "1"
    requires_credentials = False
    supports_custom_base_url = True
    requires_base_url = True

    def resolve_credential(
        self,
        env_var: str | CredentialReference | None,
    ) -> ProviderCredential | None:
        if env_var is None:
            return None
        try:
            reference = (
                env_var
                if isinstance(env_var, CredentialReference)
                else CredentialReference.from_env_name(env_var)
            )
        except (TypeError, CredentialReferenceError):
            raise ProviderConfigurationError(
                "vllm credential environment variable name is invalid",
                safe_for_display=True,
            ) from None
        try:
            return reference.resolve()
        except CredentialResolutionError:
            raise ProviderConfigurationError(
                "Inference provider 'vllm' credential environment variable is "
                "not set or is empty.",
                safe_for_display=True,
            ) from None
        except Exception as error:
            raise ProviderConfigurationError(
                "Inference provider 'vllm' credential resolution failed "
                f"[{type(error).__name__}].",
                safe_for_display=True,
            ) from None

    def complete(
        self,
        request: InferenceRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> InferenceResult:
        failure: ProviderError | None = None
        try:
            base_url = self.resolve_base_url(runtime.base_url)
            assert base_url is not None
            return _complete_with_http(
                request,
                credential=credential,
                runtime=runtime,
                base_url=base_url,
            )
        except ProviderError as error:
            failure = sanitized_provider_error(self.name(), "inference", error)
        except Exception as error:
            if provider_error_debug_enabled() and log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "vllm inference request failed:\n%s",
                    redacted_exception_text(error),
                )
            failure = ProviderRequestError(
                self.name(), "inference", code=type(error).__name__
            )
        if failure is not None:
            del request
            raise failure
        raise AssertionError("vllm inference did not produce a result")


def _complete_with_http(
    request: InferenceRequest,
    *,
    credential: ProviderCredential | None,
    runtime: ProviderRuntimeOptions,
    base_url: str,
) -> InferenceResult:
    headers = {"Content-Type": "application/json"}
    if credential is not None:
        headers["Authorization"] = f"Bearer {credential.reveal()}"
    payload = {
        "model": request.model,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.content},
        ],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": request.output_name,
                "description": request.output_description,
                "schema": dict(request.output_schema),
            },
        },
    }
    attempts = runtime.max_retries + 1
    for attempt in range(attempts):
        try:
            response = _post_json(
                f"{base_url}/chat/completions",
                payload,
                headers=headers,
                timeout_seconds=runtime.timeout_seconds,
            )
            return _parse_response_safely(response, request)
        except ProviderRequestError as error:
            if not error.retryable or attempt == attempts - 1:
                raise
            time.sleep(min(0.25 * (2**attempt), 4.0))
    raise AssertionError("vllm retry loop did not complete")


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers=dict(headers),
        method="POST",
    )
    try:
        response = _open_url(request, timeout_seconds)
        with response:
            status = getattr(response, "status", 200)
            if not isinstance(status, int) or not 200 <= status < 300:
                raise ProviderRequestError(
                    "vllm",
                    "inference",
                    code=_http_error_code(status),
                    retryable=status in _RETRYABLE_HTTP_STATUSES,
                )
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        status = error.code
        raise ProviderRequestError(
            "vllm",
            "inference",
            code=_http_error_code(status),
            retryable=status in _RETRYABLE_HTTP_STATUSES,
        ) from None
    except TimeoutError:
        raise ProviderRequestError(
            "vllm", "inference", code="timeout", retryable=True
        ) from None
    except URLError:
        raise ProviderRequestError(
            "vllm", "inference", code="network_error", retryable=True
        ) from None
    except ProviderError:
        raise
    except OSError:
        raise ProviderRequestError(
            "vllm", "inference", code="network_error", retryable=True
        ) from None

    if len(body) > _MAX_RESPONSE_BYTES:
        raise ProviderResponseError(
            "vllm response exceeded the maximum supported size",
            safe_for_display=True,
        )
    try:
        decoded = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ProviderResponseError(
            "vllm returned malformed JSON",
            safe_for_display=True,
        ) from None
    if not isinstance(decoded, Mapping):
        raise ProviderResponseError(
            "vllm response must be a JSON object",
            safe_for_display=True,
        )
    return decoded


def _open_url(request: Request, timeout_seconds: float) -> Any:
    return _HTTP_OPENER.open(request, timeout=timeout_seconds)


def _parse_response(
    response: Mapping[str, Any], request: InferenceRequest
) -> InferenceResult:
    choices = response.get("choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        raise ProviderResponseError(
            "vllm response choices are malformed",
            safe_for_display=True,
        )
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise ProviderResponseError(
            f"LLM response truncated at max_tokens={request.max_tokens}; partial "
            "structured outputs are never used.",
            safe_for_display=True,
        )
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ProviderResponseError(
            "vllm response message is malformed",
            safe_for_display=True,
        )
    content = message.get("content")
    if not isinstance(content, str):
        raise ProviderResponseError(
            "vllm response content is malformed",
            safe_for_display=True,
        )
    try:
        output = json.loads(content)
    except json.JSONDecodeError:
        raise ProviderResponseError(
            "vllm structured output is malformed JSON",
            safe_for_display=True,
        ) from None
    if not isinstance(output, Mapping):
        raise ProviderResponseError(
            "vllm structured output must be a mapping",
            safe_for_display=True,
        )

    raw_usage = response.get("usage")
    if not isinstance(raw_usage, Mapping):
        raise ProviderResponseError(
            "vllm response is missing usage metadata",
            safe_for_display=True,
        )
    usage = ProviderUsage(
        input_tokens=_usage_value(raw_usage, "prompt_tokens"),
        output_tokens=_usage_value(raw_usage, "completion_tokens"),
    )
    raw_request_id = response.get("id")
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


def _parse_response_safely(
    response: Mapping[str, Any], request: InferenceRequest
) -> InferenceResult:
    try:
        return _parse_response(response, request)
    except ProviderResponseError:
        raise
    except Exception:
        raise ProviderResponseError(
            "vllm returned a malformed inference response",
            safe_for_display=True,
        ) from None


def _usage_value(usage: Mapping[str, Any], name: str) -> int:
    value = usage.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderResponseError(
            f"provider returned invalid usage field '{name}'",
            safe_for_display=True,
        )
    return value


def _http_error_code(status: object) -> str:
    if isinstance(status, int) and 100 <= status <= 599:
        return f"http_{status}"
    return "http_error"
