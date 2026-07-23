from __future__ import annotations

from collections.abc import Sequence
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
    ProviderConfigurationError,
    ProviderCredential,
    ProviderResponseError,
    ProviderRuntimeOptions,
    ProviderUsage,
    provider_option,
)
from .registry import register_embedding_provider

_VERTEX_FEATURE = "Vertex AI embeddings"
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

    @classmethod
    def profile_options_model(cls) -> type[BaseModel] | None:
        return VertexEmbeddingOptions

    def resolve_credential(
        self,
        env_var: str | CredentialReference | None,
    ) -> ProviderCredential | None:
        if env_var is not None:
            raise ProviderConfigurationError(
                "Vertex AI embeddings use Application Default Credentials and "
                "do not accept api_key_env",
                safe_for_display=True,
            )
        return None

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
        try:
            response = client.models.embed_content(
                model=request.model,
                contents=list(request.texts),
                config={
                    "task_type": (
                        options.query_task_type
                        if request.input_type == "query"
                        else options.task_type
                    ),
                    "output_dimensionality": request.dimensions,
                    "auto_truncate": options.auto_truncate,
                },
            )
        finally:
            client.close()
        try:
            return _parse_response(response, request, options=options)
        except ProviderResponseError as error:
            raise _with_billed_failure(error, response, request, self) from None


def _load_google_genai() -> Any:
    return import_optional_dependency(
        "google.genai",
        distribution="google-genai",
        extra="vertex",
        feature=_VERTEX_FEATURE,
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
) -> ProviderResponseError:
    try:
        usage = _response_usage(response)
    except ProviderResponseError:
        usage = ProviderUsage()
    return error.attach_failure(
        InferenceFailure(
            error_code="invalid_embedding_response",
            usage=usage,
            billed_requests=1,
            provider=provider.name(),
            model=request.model,
            implementation_identity=provider.implementation_identity(),
        )
    )
