from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..credentials import CredentialReference
from ..optional_dependencies import (
    OptionalDependencyError,
    import_optional_dependency,
)
from .base import (
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResult,
    InferenceFailure,
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
    provider_option,
    redacted_exception_text,
    sanitized_provider_error,
)
from .registry import register_embedding_provider, register_inference_provider

log = logging.getLogger(__name__)

_VERTEX_FEATURE = "Vertex AI embeddings"
_VERTEX_INFERENCE_FEATURE = "Vertex AI inference"
_RETRYABLE_STATUS_CODES = [408, 409, 425, 429, 500, 502, 503, 504]

VertexTaskType = Literal[
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
]


class VertexEmbeddingOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    project: str | None = provider_option(
        "execution",
        default=None,
        min_length=1,
        max_length=256,
    )
    location: str = provider_option(
        "execution",
        default="us-central1",
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    task_type: VertexTaskType = provider_option(
        "semantic",
        default="RETRIEVAL_DOCUMENT",
    )
    query_task_type: VertexTaskType = provider_option(
        "semantic",
        default="RETRIEVAL_QUERY",
    )
    auto_truncate: bool = provider_option("semantic", default=False)


@register_embedding_provider
class VertexEmbeddingProvider(EmbeddingProvider):
    provider_name = "vertex"
    implementation_version = "1"
    implementation_packages = ("google-genai",)
    requires_credentials = False
    accepts_api_key_env = False

    @classmethod
    def profile_options_model(cls) -> type[BaseModel] | None:
        return VertexEmbeddingOptions

    def validate_credential_reference(
        self,
        env_var: str | CredentialReference | None,
    ) -> None:
        if env_var is not None:
            raise ProviderConfigurationError(
                "Vertex AI embeddings use Application Default Credentials and "
                "do not accept api_key_env",
                safe_for_display=True,
            )

    def _embed(
        self,
        request: EmbeddingRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> EmbeddingResult:
        if credential is not None:
            raise ProviderConfigurationError(
                "Vertex AI embeddings use Application Default Credentials",
                safe_for_display=True,
            )
        options = self.profile_options
        if not isinstance(options, VertexEmbeddingOptions):
            raise ProviderConfigurationError(
                "Vertex AI embedding provider options are invalid",
                safe_for_display=True,
            )
        try:
            genai = _load_google_genai()
        except OptionalDependencyError as error:
            raise ProviderConfigurationError(
                str(error),
                safe_for_display=True,
            ) from None

        client_options: dict[str, Any] = {
            "vertexai": True,
            "location": options.location,
            "http_options": {
                "api_version": "v1",
                "timeout": round(runtime.timeout_seconds * 1000),
                "retry_options": {
                    "attempts": runtime.max_retries + 1,
                    "http_status_codes": _RETRYABLE_STATUS_CODES,
                },
            },
        }
        if options.project is not None:
            client_options["project"] = options.project
        client = genai.Client(**client_options)
        vectors: list[tuple[float, ...]] = []
        usage = ProviderUsage()
        provider_requests = _split_requests(request)
        try:
            for request_number, provider_request in enumerate(
                provider_requests, start=1
            ):
                response = client.models.embed_content(
                    model=provider_request.model,
                    contents=list(provider_request.texts),
                    config={
                        "task_type": (
                            options.query_task_type
                            if provider_request.input_type == "query"
                            else options.task_type
                        ),
                        "output_dimensionality": provider_request.dimensions,
                        "auto_truncate": options.auto_truncate,
                    },
                )
                try:
                    parsed = _parse_response(
                        response,
                        provider_request,
                        options=options,
                    )
                except ProviderResponseError as error:
                    raise _with_billed_failure(
                        error,
                        response,
                        request,
                        self,
                        prior_usage=usage,
                        billed_requests=request_number,
                    ) from None
                vectors.extend(parsed.vectors)
                usage = _add_usage(usage, parsed.usage)
        finally:
            client.close()
        try:
            return EmbeddingResult(
                vectors=tuple(vectors),
                model=request.model,
                dimensions=len(vectors[0]),
                input_ids=request.input_ids,
                usage=usage,
                provider_requests=len(provider_requests),
            )
        except (IndexError, ValueError):
            raise ProviderResponseError(
                "Vertex AI returned an invalid embedding response",
                safe_for_display=True,
            ) from None


class VertexInferenceOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    # Project and location are execution routing, not semantics: they select
    # where the request runs and never enter the transformation identity.
    project: str | None = provider_option(
        "execution",
        default=None,
        min_length=1,
        max_length=256,
    )
    location: str = provider_option(
        "execution",
        default="us-central1",
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )


@register_inference_provider
class VertexInferenceProvider(InferenceProvider):
    """Structured extraction on Vertex AI Gemini models via the optional
    ``google-genai`` SDK in explicit Vertex mode. Authentication is Application
    Default Credentials; ``api_key_env`` is rejected, matching the Vertex
    embedding provider. The provider-neutral ``output_schema`` is forwarded as
    the Gemini response schema with JSON output, so a model returns exactly the
    requested fields."""

    provider_name = "vertex"
    implementation_version = "1"
    implementation_packages = ("google-genai",)
    requires_credentials = False
    accepts_api_key_env = False

    @classmethod
    def profile_options_model(cls) -> type[BaseModel] | None:
        return VertexInferenceOptions

    def validate_credential_reference(
        self,
        env_var: str | CredentialReference | None,
    ) -> None:
        if env_var is not None:
            raise ProviderConfigurationError(
                "Vertex AI inference uses Application Default Credentials and "
                "does not accept api_key_env",
                safe_for_display=True,
            )

    def complete(
        self,
        request: InferenceRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> InferenceResult:
        failure: ProviderError | None = None
        try:
            return self._complete(request, credential=credential, runtime=runtime)
        except ProviderError as error:
            failure = sanitized_provider_error(self.name(), "inference", error)
        except Exception as error:
            if provider_error_debug_enabled() and log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "vertex inference request failed:\n%s",
                    redacted_exception_text(error),
                )
            failure = ProviderRequestError(
                self.name(), "inference", code=type(error).__name__
            )
        del request
        raise failure

    def _complete(
        self,
        request: InferenceRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> InferenceResult:
        if credential is not None:
            raise ProviderConfigurationError(
                "Vertex AI inference uses Application Default Credentials",
                safe_for_display=True,
            )
        options = self.profile_options
        if not isinstance(options, VertexInferenceOptions):
            raise ProviderConfigurationError(
                "Vertex AI inference provider options are invalid",
                safe_for_display=True,
            )
        try:
            genai = _load_google_genai_inference()
        except OptionalDependencyError as error:
            raise ProviderConfigurationError(
                str(error),
                safe_for_display=True,
            ) from None

        client_options: dict[str, Any] = {
            "vertexai": True,
            "location": options.location,
            "http_options": {
                "api_version": "v1",
                "timeout": round(runtime.timeout_seconds * 1000),
                "retry_options": {
                    "attempts": runtime.max_retries + 1,
                    "http_status_codes": _RETRYABLE_STATUS_CODES,
                },
            },
        }
        if options.project is not None:
            client_options["project"] = options.project
        client = genai.Client(**client_options)
        try:
            response = client.models.generate_content(
                model=request.model,
                contents=request.content,
                config={
                    "system_instruction": request.system_prompt,
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_tokens,
                    "response_mime_type": "application/json",
                    "response_schema": dict(request.output_schema),
                },
            )
            try:
                return _parse_inference_response(response, request)
            except ProviderResponseError as error:
                raise _with_billed_inference_failure(
                    error, response, request, self
                ) from None
        finally:
            client.close()


def _parse_inference_response(
    response: Any, request: InferenceRequest
) -> InferenceResult:
    candidates = getattr(response, "candidates", None)
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or not candidates
    ):
        raise ProviderResponseError(
            "Vertex AI returned no inference candidates",
            safe_for_display=True,
        )
    finish_reason = _finish_reason_name(getattr(candidates[0], "finish_reason", None))
    if finish_reason == "MAX_TOKENS":
        raise ProviderResponseError(
            f"LLM response truncated at max_tokens={request.max_tokens}; partial "
            "structured outputs are never used.",
            safe_for_display=True,
        )
    if finish_reason not in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
        # SAFETY, RECITATION, BLOCKLIST, etc. — the enum name is safe to surface
        # (it carries no prompt or response text).
        raise ProviderResponseError(
            f"Vertex AI stopped generation for reason '{finish_reason}'",
            safe_for_display=True,
        )
    text = _response_text(response, candidates[0])
    try:
        output = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        raise ProviderResponseError(
            "Vertex AI structured output is malformed JSON",
            safe_for_display=True,
        ) from None
    if not isinstance(output, Mapping):
        raise ProviderResponseError(
            "Vertex AI structured output must be a mapping",
            safe_for_display=True,
        )
    usage = _inference_usage(response, required=True)
    raw_request_id = getattr(response, "response_id", None)
    request_id = (
        raw_request_id if isinstance(raw_request_id, str) and raw_request_id else None
    )
    return InferenceResult(
        dict(output),
        usage=usage,
        provider_request_id=request_id,
    )


def _finish_reason_name(reason: Any) -> str | None:
    if reason is None:
        return None
    name = getattr(reason, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(reason, str) and reason:
        return reason
    return str(reason)


def _response_text(response: Any, candidate: Any) -> str:
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None)
    if isinstance(parts, Sequence) and not isinstance(parts, (str, bytes)):
        texts = [
            part.text
            for part in parts
            if isinstance(getattr(part, "text", None), str)
        ]
        if texts:
            return "".join(texts)
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    raise ProviderResponseError(
        "Vertex AI returned no text content",
        safe_for_display=True,
    )


def _inference_usage(response: Any, *, required: bool) -> ProviderUsage:
    raw_usage = getattr(response, "usage_metadata", None)
    if raw_usage is None:
        if required:
            raise ProviderResponseError(
                "Vertex AI response is missing usage metadata",
                safe_for_display=True,
            )
        return ProviderUsage()
    # Gemini thinking models bill reasoning tokens in a separate
    # `thoughts_token_count`; fold them into output usage so run metrics and
    # `max_tokens`/budget enforcement account for the full billable spend.
    output_tokens = _inference_usage_value(
        raw_usage, "candidates_token_count", required=required
    ) + _inference_usage_value(raw_usage, "thoughts_token_count")
    return ProviderUsage(
        input_tokens=_inference_usage_value(
            raw_usage, "prompt_token_count", required=required
        ),
        output_tokens=output_tokens,
        cache_read_input_tokens=_inference_usage_value(
            raw_usage, "cached_content_token_count"
        ),
    )


def _inference_usage_value(usage: Any, name: str, *, required: bool = False) -> int:
    value = getattr(usage, name, None)
    if value is None and not required:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderResponseError(
            f"Vertex AI returned invalid usage field '{name}'",
            safe_for_display=True,
        )
    return value


def _best_effort_inference_usage(response: Any) -> ProviderUsage | None:
    """Usage reported alongside a rejected response, for billed-failure
    accounting only; None when the response carries no valid usage."""
    try:
        return _inference_usage(response, required=False)
    except (ProviderResponseError, ValueError):
        return None


def _with_billed_inference_failure(
    error: ProviderResponseError,
    response: Any,
    request: InferenceRequest,
    provider: VertexInferenceProvider,
) -> ProviderResponseError:
    usage = _best_effort_inference_usage(response)
    if usage is None:
        return error
    return error.attach_failure(
        InferenceFailure(
            error_code="invalid_response",
            usage=usage,
            billed_requests=1,
            provider=provider.name(),
            model=request.model,
            implementation_identity=provider.implementation_identity(),
        )
    )


def _load_google_genai() -> Any:
    return _import_google_genai(_VERTEX_FEATURE)


def _load_google_genai_inference() -> Any:
    return _import_google_genai(_VERTEX_INFERENCE_FEATURE)


def _import_google_genai(feature: str) -> Any:
    return import_optional_dependency(
        "google.genai",
        distribution="google-genai",
        extra="vertex",
        feature=feature,
    )


def _split_requests(request: EmbeddingRequest) -> tuple[EmbeddingRequest, ...]:
    model_name = request.model.rsplit("/", maxsplit=1)[-1]
    request_limit = 1 if model_name.startswith("gemini-embedding") else 5
    return tuple(
        EmbeddingRequest(
            model=request.model,
            texts=request.texts[offset : offset + request_limit],
            dimensions=request.dimensions,
            input_ids=(
                request.input_ids[offset : offset + request_limit]
                if request.input_ids is not None
                else None
            ),
            input_type=request.input_type,
        )
        for offset in range(0, len(request.texts), request_limit)
    )


def _parse_response(
    response: Any,
    request: EmbeddingRequest,
    *,
    options: VertexEmbeddingOptions,
) -> EmbeddingResult:
    embeddings = getattr(response, "embeddings", None)
    if (
        not isinstance(embeddings, Sequence)
        or isinstance(embeddings, (str, bytes))
        or len(embeddings) != len(request.texts)
    ):
        raise ProviderResponseError(
            "Vertex AI returned embeddings that do not align with the inputs",
            safe_for_display=True,
        )
    vectors: list[tuple[float, ...]] = []
    for embedding in embeddings:
        values = getattr(embedding, "values", None)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                for value in values
            )
        ):
            raise ProviderResponseError(
                "Vertex AI returned a malformed embedding vector",
                safe_for_display=True,
            )
        statistics = getattr(embedding, "statistics", None)
        if (
            not options.auto_truncate
            and getattr(statistics, "truncated", False) is True
        ):
            raise ProviderResponseError(
                "Vertex AI truncated an embedding input while auto_truncate is disabled",
                safe_for_display=True,
            )
        vectors.append(tuple(float(value) for value in values))

    dimensions = len(vectors[0])
    try:
        return EmbeddingResult(
            vectors=tuple(vectors),
            model=request.model,
            dimensions=dimensions,
            input_ids=request.input_ids,
            usage=_response_usage(response),
        )
    except ValueError:
        raise ProviderResponseError(
            "Vertex AI returned an invalid embedding response",
            safe_for_display=True,
        ) from None


def _response_usage(response: Any) -> ProviderUsage:
    total = 0
    embeddings = getattr(response, "embeddings", None)
    if not isinstance(embeddings, Sequence) or isinstance(embeddings, (str, bytes)):
        return ProviderUsage()
    for embedding in embeddings:
        statistics = getattr(embedding, "statistics", None)
        token_count = getattr(statistics, "token_count", 0)
        if token_count is None:
            continue
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, (int, float))
            or token_count < 0
            or not float(token_count).is_integer()
        ):
            raise ProviderResponseError(
                "Vertex AI returned invalid embedding usage metadata",
                safe_for_display=True,
            )
        total += int(token_count)
    return ProviderUsage(input_tokens=total)


def _with_billed_failure(
    error: ProviderResponseError,
    response: Any,
    request: EmbeddingRequest,
    provider: VertexEmbeddingProvider,
    prior_usage: ProviderUsage | None = None,
    billed_requests: int = 1,
) -> ProviderResponseError:
    prior_usage = prior_usage or ProviderUsage()
    try:
        usage = _add_usage(prior_usage, _response_usage(response))
    except ProviderResponseError:
        usage = prior_usage
    return error.attach_failure(
        InferenceFailure(
            error_code="invalid_embedding_response",
            usage=usage,
            billed_requests=billed_requests,
            provider=provider.name(),
            model=request.model,
            implementation_identity=provider.implementation_identity(),
        )
    )


def _add_usage(first: ProviderUsage, second: ProviderUsage) -> ProviderUsage:
    reported_cost = (
        None
        if first.reported_cost_usd is None and second.reported_cost_usd is None
        else (first.reported_cost_usd or 0.0) + (second.reported_cost_usd or 0.0)
    )
    return ProviderUsage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        cache_read_input_tokens=(
            first.cache_read_input_tokens + second.cache_read_input_tokens
        ),
        cache_creation_input_tokens=(
            first.cache_creation_input_tokens + second.cache_creation_input_tokens
        ),
        reported_cost_usd=reported_cost,
    )
