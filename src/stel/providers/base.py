from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from types import UnionType
from typing import Any, ClassVar, Literal, Self, Union, get_args, get_origin

from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import FieldInfo

from ..credentials import (
    CredentialReference,
    CredentialReferenceError,
    CredentialResolutionError,
    ProtectedCredential,
)
from ..endpoints import EndpointUrlError, OpenAICompatibleBaseUrl
from ..env import PROVIDER_DEBUG_ENV, read_env
from ..hashing import HASH_DIGEST_SIZE, canonical_fingerprint

log = logging.getLogger(__name__)

PROVIDER_CONTRACT_VERSION = 3


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str = "provider operation failed",
        *,
        safe_for_display: bool = False,
    ) -> None:
        if not isinstance(message, str):
            raise ValueError("provider error message must be a string")
        if not isinstance(safe_for_display, bool):
            raise ValueError("safe_for_display must be boolean")
        super().__init__(message)
        self.safe_for_display = safe_for_display
        # Billed-failure accounting (issue #71): a safe error may carry the
        # usage the provider charged for the failed work.
        self.failure: InferenceFailure | None = None

    def attach_failure(self, failure: InferenceFailure) -> Self:
        if not isinstance(failure, InferenceFailure):
            raise ValueError("attached failure must be InferenceFailure")
        self.failure = failure
        return self


class ProviderRegistrationError(ProviderError):
    pass


class ProviderNotFoundError(ProviderError):
    pass


class ProviderConfigurationError(ProviderError):
    pass


class ProviderRequestError(ProviderError):
    def __init__(
        self,
        provider: str,
        operation: str,
        *,
        code: str,
        retryable: bool = False,
    ) -> None:
        if not isinstance(retryable, bool):
            raise ValueError("retryable must be boolean")
        self.provider = _safe_error_label(provider, fallback="provider")
        self.operation = _safe_error_label(operation, fallback="request")
        self.code = _safe_error_code(code)
        self.retryable = retryable
        suffix = " (retryable)" if retryable else ""
        super().__init__(
            f"{self.provider} {self.operation} failed [{self.code}]{suffix}",
            safe_for_display=True,
        )


class ProviderResponseError(ProviderError):
    pass


class ProviderBatchError(ProviderError):
    def __init__(
        self,
        message: str = "provider batch operation failed",
        *,
        safe_for_display: bool = False,
        retryable: bool = False,
    ) -> None:
        if not isinstance(retryable, bool):
            raise ValueError("retryable must be boolean")
        self.retryable = retryable
        super().__init__(message, safe_for_display=safe_for_display)


_DEBUG_EXCEPTION_LABELS: dict[type[BaseException], str] = {
    ArithmeticError: "builtins.ArithmeticError",
    AssertionError: "builtins.AssertionError",
    AttributeError: "builtins.AttributeError",
    KeyError: "builtins.KeyError",
    LookupError: "builtins.LookupError",
    OSError: "builtins.OSError",
    RuntimeError: "builtins.RuntimeError",
    TypeError: "builtins.TypeError",
    ValueError: "builtins.ValueError",
    ProviderError: "stel.providers.ProviderError",
    ProviderRegistrationError: "stel.providers.ProviderRegistrationError",
    ProviderNotFoundError: "stel.providers.ProviderNotFoundError",
    ProviderConfigurationError: "stel.providers.ProviderConfigurationError",
    ProviderRequestError: "stel.providers.ProviderRequestError",
    ProviderResponseError: "stel.providers.ProviderResponseError",
    ProviderBatchError: "stel.providers.ProviderBatchError",
}



def provider_error_debug_enabled() -> bool:
    """Whether operators opted into allowlisted SDK diagnostics in debug logs.

    Off by default: an SDK error message can echo fragments of the request
    that no redaction can anticipate, and debug logs are often shipped to
    aggregators. Diagnostics contain only exception types and stack locations;
    the switch exists for local diagnosis.
    """
    value = read_env(PROVIDER_DEBUG_ENV, default="")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def redacted_exception_text(
    error: BaseException,
    *,
    sensitive: Sequence[str | None] = (),
) -> str:
    """Return allowlisted exception diagnostics without provider metadata.

    Exact-value replacement cannot safely redact repr-, JSON-, or URL-encoded
    request data. Keep the compatibility name and argument, but emit only
    recognized exception categories, stel module locations, and an external
    frame count. Raised errors and artifacts still use the sanitized
    `ProviderError` hierarchy.
    """
    del sensitive
    lines: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        lines.append(
            _DEBUG_EXCEPTION_LABELS.get(type(current), "external exception")
        )
        external_frames = 0
        for frame, line_number in traceback.walk_tb(current.__traceback__):
            module_name = frame.f_globals.get("__name__")
            module = sys.modules.get(module_name) if isinstance(module_name, str) else None
            if (
                isinstance(module_name, str)
                and (module_name == "stel" or module_name.startswith("stel."))
                and module is not None
                and vars(module) is frame.f_globals
            ):
                lines.append(f"  at {module_name}:{line_number}")
            else:
                external_frames += 1
        if external_frames:
            lines.append(f"  at {external_frames} external frame(s)")
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return "\n".join(lines)


def sanitized_provider_error(
    provider: str,
    operation: str,
    error: ProviderError,
) -> ProviderError:
    """Replace provider-authored text and exception chains at the boundary."""
    sanitized = _sanitized_provider_error(provider, operation, error)
    # Billed-failure accounting survives sanitization: the envelope holds
    # only a safe error code, normalized usage, and provenance (issue #71).
    if error.failure is not None:
        sanitized.attach_failure(error.failure)
    return sanitized


def _sanitized_provider_error(
    provider: str,
    operation: str,
    error: ProviderError,
) -> ProviderError:
    safe_provider = _safe_error_label(provider, fallback="provider")
    safe_operation = _safe_error_label(operation, fallback="request")
    if isinstance(error, ProviderRequestError):
        return ProviderRequestError(
            safe_provider,
            safe_operation,
            code=error.code,
            retryable=error.retryable,
        )
    if isinstance(error, ProviderConfigurationError):
        if error.safe_for_display:
            return ProviderConfigurationError(
                str(error), safe_for_display=True
            )
        return ProviderConfigurationError(
            f"Inference provider '{safe_provider}' configuration is invalid",
            safe_for_display=True,
        )
    if isinstance(error, ProviderResponseError):
        if error.safe_for_display:
            return ProviderResponseError(str(error), safe_for_display=True)
        return ProviderResponseError(
            f"{safe_provider} {safe_operation} returned an invalid response",
            safe_for_display=True,
        )
    if isinstance(error, ProviderBatchError):
        if error.safe_for_display:
            return ProviderBatchError(
                str(error),
                safe_for_display=True,
                retryable=error.retryable,
            )
        return ProviderBatchError(
            f"{safe_provider} {safe_operation} failed",
            safe_for_display=True,
            retryable=error.retryable,
        )
    return ProviderRequestError(
        safe_provider,
        safe_operation,
        code=type(error).__name__,
    )


# HTTP statuses worth retrying (transient): request timeout, conflict, too-early,
# rate limit, and the 5xx family. Mirrors the vLLM provider's own set.
RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def http_status_of(error: BaseException) -> int | None:
    """Best-effort HTTP status from a provider SDK exception, without importing
    any optional SDK. Duck-types the fields the google-genai, google-api-core,
    anthropic, and httpx exception hierarchies expose (`.status_code`, `.code`,
    `.response.status_code`); returns None when no HTTP status is present.

    The status is the one non-sensitive signal a caller needs to tell a
    transient failure (429/5xx) from a permanent one (400/403/404)."""
    candidates = (
        getattr(error, "status_code", None),
        getattr(error, "code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    )
    for value in candidates:
        if (
            not isinstance(value, bool)
            and isinstance(value, int)
            and 100 <= value <= 599
        ):
            return value
    return None


def provider_request_error(
    provider: str, operation: str, error: BaseException
) -> ProviderRequestError:
    """Wrap an unexpected SDK error as a `ProviderRequestError`, preserving its
    HTTP status (as `code="http_<status>"`) and classifying retryability so a
    caller can distinguish a retryable 429/5xx from a permanent 4xx. Falls back
    to the exception type name when the SDK exposes no status."""
    status = http_status_of(error)
    if status is not None:
        return ProviderRequestError(
            provider,
            operation,
            code=f"http_{status}",
            retryable=status in RETRYABLE_HTTP_STATUSES,
        )
    return ProviderRequestError(provider, operation, code=type(error).__name__)


def provider_batch_error(
    provider: str, operation: str, error: BaseException
) -> ProviderBatchError:
    """Wrap an unexpected batch SDK error without retaining provider text.

    When the SDK exposes an HTTP status, use the shared transient-status policy
    to let callers distinguish retryable batch interruptions from permanent
    failures. Errors without a status remain non-retryable.
    """
    status = http_status_of(error)
    safe_provider = _safe_error_label(provider, fallback="provider")
    safe_operation = _safe_error_label(operation, fallback="batch operation")
    code = _safe_error_code(type(error).__name__)
    return ProviderBatchError(
        f"{safe_provider} {safe_operation} failed [{code}]",
        safe_for_display=True,
        retryable=(
            status is not None and status in RETRYABLE_HTTP_STATUSES
        ),
    )


ProviderCredential = ProtectedCredential


@dataclass(frozen=True, slots=True)
class ProviderRuntimeOptions:
    max_retries: int = 4
    base_url: str | None = None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        if self.base_url is not None:
            try:
                normalized = OpenAICompatibleBaseUrl(self.base_url)
            except EndpointUrlError as error:
                raise ValueError(str(error)) from None
            object.__setattr__(self, "base_url", normalized)
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or not 0.1 <= self.timeout_seconds <= 3600.0
        ):
            raise ValueError("timeout_seconds must be between 0.1 and 3600")


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reported_cost_usd: float | None = None
    # Appended last so the v3 positional constructor signature keeps working
    # for separately installed provider plugins built against this contract.
    thinking_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "thinking_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.reported_cost_usd is not None:
            if (
                isinstance(self.reported_cost_usd, bool)
                or not isinstance(self.reported_cost_usd, (int, float))
                or not math.isfinite(self.reported_cost_usd)
                or self.reported_cost_usd < 0
            ):
                raise ValueError(
                    "reported_cost_usd must be a non-negative finite number"
                )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> ProviderUsage:
        values = values or {}
        allowed = {
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "thinking_tokens",
            "reported_cost_usd",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError("unknown provider usage fields")
        return cls(**dict(values))

    def to_metrics(self) -> dict[str, int | float]:
        metrics: dict[str, int | float] = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
        }
        # Reported only when a provider actually bills reasoning tokens, so
        # non-thinking providers and cache hits keep the metrics shape they
        # already had (mirrors reported_cost_usd).
        if self.thinking_tokens:
            metrics["thinking_tokens"] = self.thinking_tokens
        if self.reported_cost_usd is not None:
            metrics["reported_cost_usd"] = self.reported_cost_usd
        return metrics


@dataclass(frozen=True, slots=True)
class InferenceFailure:
    """Billing and provenance for provider work that produced no usable result.

    Truncated responses, schema-invalid tool output, and partially failed
    native batches still consume tokens. This envelope lets a sanitized error
    and normalized usage coexist so budgets and run results account for billed
    failures; it never carries prompts, response bodies, or credential data.
    """

    error_code: str
    usage: ProviderUsage
    billed_requests: int
    provider: str
    model: str
    implementation_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.error_code, str) or not self.error_code:
            raise ValueError("failure error_code must be a non-empty string")
        if not isinstance(self.usage, ProviderUsage):
            raise ValueError("failure usage must be ProviderUsage")
        if (
            isinstance(self.billed_requests, bool)
            or not isinstance(self.billed_requests, int)
            or self.billed_requests < 0
        ):
            raise ValueError("failure billed_requests must be a non-negative integer")
        for name in ("provider", "model", "implementation_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"failure {name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    model: str
    content: str = field(repr=False)
    system_prompt: str = field(repr=False)
    output_schema: Mapping[str, Any] = field(repr=False)
    output_name: str = "extract"
    output_description: str = "Return the extracted structured fields."
    temperature: float = 0.0
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("inference model must not be empty")
        if not isinstance(self.content, str):
            raise ValueError("inference content must be a string")
        if not isinstance(self.system_prompt, str):
            raise ValueError("inference system_prompt must be a string")
        if not isinstance(self.output_schema, Mapping):
            raise ValueError("inference output_schema must be a mapping")
        if not isinstance(self.output_name, str) or not self.output_name:
            raise ValueError("output_name must not be empty")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
        ):
            raise ValueError("temperature must be finite")
        if not isinstance(self.output_description, str):
            raise ValueError("output_description must be a string")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    output: Mapping[str, Any] = field(repr=False)
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    provider_request_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output, Mapping):
            raise ValueError("inference output must be a mapping")
        if not isinstance(self.usage, ProviderUsage):
            raise ValueError("inference usage must be ProviderUsage")
        if self.provider_request_id is not None and (
            not isinstance(self.provider_request_id, str)
            or not self.provider_request_id
        ):
            raise ValueError("provider_request_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class BatchInferenceRequest:
    request_id: str
    request: InferenceRequest

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("batch request_id must not be empty")
        if not isinstance(self.request, InferenceRequest):
            raise ValueError("batch request must be InferenceRequest")


@dataclass(frozen=True, slots=True)
class BatchInferenceItem:
    request_id: str
    result: InferenceResult | None = None
    error: ProviderError | None = None
    # Billed usage for a failed item (issue #71). Successful items carry
    # usage inside their result; the error side may carry what the provider
    # charged for the failure.
    usage: ProviderUsage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("batch item request_id must not be empty")
        if bool(self.result is not None) == bool(self.error is not None):
            raise ValueError("batch item must contain exactly one of result or error")
        if self.result is not None and not isinstance(self.result, InferenceResult):
            raise ValueError("batch item result must be InferenceResult")
        if self.error is not None and not isinstance(self.error, ProviderError):
            raise ValueError("batch item error must be ProviderError")
        if self.usage is not None:
            if self.error is None:
                raise ValueError(
                    "batch item usage accompanies a failed item; successful "
                    "items carry usage inside their result"
                )
            if not isinstance(self.usage, ProviderUsage):
                raise ValueError("batch item usage must be ProviderUsage")


@dataclass(frozen=True, slots=True)
class BatchInferenceResult:
    items: tuple[BatchInferenceItem, ...]
    batch_submissions: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or any(
            not isinstance(item, BatchInferenceItem) for item in self.items
        ):
            raise ValueError("batch result items must be BatchInferenceItem values")
        if (
            isinstance(self.batch_submissions, bool)
            or not isinstance(self.batch_submissions, int)
            or self.batch_submissions < 0
        ):
            raise ValueError("batch_submissions must be a non-negative integer")
        ids = [item.request_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("batch result request IDs must be unique")


@dataclass(frozen=True, slots=True)
class BatchJobStatus:
    """Progress of a submitted native batch job, artifact-safe."""

    done: bool
    processing: int = 0
    succeeded: int = 0
    errored: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.done, bool):
            raise ValueError("batch job done must be boolean")
        for name in ("processing", "succeeded", "errored"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"batch job {name} must be a non-negative integer")


_BATCH_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")


def validate_batch_job_id(value: object) -> str:
    """Provider job identifiers are persisted for resume; keep them boring."""
    if not isinstance(value, str) or not _BATCH_JOB_ID.fullmatch(value):
        raise ProviderBatchError(
            "provider returned an invalid batch job identifier",
            safe_for_display=True,
        )
    return value


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    model: str
    texts: tuple[str, ...] = field(repr=False)
    dimensions: int | None = None
    input_ids: tuple[str, ...] | None = field(default=None, repr=False)
    input_type: Literal["document", "query"] = "document"

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("embedding model must not be empty")
        if not isinstance(self.texts, tuple) or not self.texts:
            raise ValueError("embedding request must contain at least one text")
        if any(not isinstance(text, str) for text in self.texts):
            raise ValueError("embedding texts must be strings")
        if self.dimensions is not None and (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or self.dimensions < 1
        ):
            raise ValueError("embedding dimensions must be a positive integer")
        if self.input_ids is not None and (
            not isinstance(self.input_ids, tuple)
            or len(self.input_ids) != len(self.texts)
            or any(
                not isinstance(input_id, str) or not input_id
                for input_id in self.input_ids
            )
            or len(self.input_ids) != len(set(self.input_ids))
        ):
            raise ValueError(
                "embedding input_ids must align one-to-one with texts and be unique"
            )
        if self.input_type not in {"document", "query"}:
            raise ValueError("embedding input_type must be document or query")


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...] = field(repr=False)
    model: str
    dimensions: int
    input_ids: tuple[str, ...] | None = field(default=None, repr=False)
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    provider_requests: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("embedding result model must not be empty")
        if (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or self.dimensions < 1
        ):
            raise ValueError("embedding dimensions must be a positive integer")
        if not isinstance(self.vectors, tuple) or not self.vectors:
            raise ValueError("embedding result must contain at least one vector")
        if any(not isinstance(vector, tuple) for vector in self.vectors):
            raise ValueError("embedding vectors must be tuples")
        if any(len(vector) != self.dimensions for vector in self.vectors):
            raise ValueError("embedding vector dimensions do not match the result")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for vector in self.vectors
            for value in vector
        ):
            raise ValueError("embedding vectors must contain finite numbers")
        if self.input_ids is not None and (
            not isinstance(self.input_ids, tuple)
            or len(self.input_ids) != len(self.vectors)
            or any(
                not isinstance(input_id, str) or not input_id
                for input_id in self.input_ids
            )
            or len(self.input_ids) != len(set(self.input_ids))
        ):
            raise ValueError(
                "embedding result input_ids must align one-to-one with vectors "
                "and be unique"
            )
        if not isinstance(self.usage, ProviderUsage):
            raise ValueError("embedding usage must be ProviderUsage")
        if (
            isinstance(self.provider_requests, bool)
            or not isinstance(self.provider_requests, int)
            or self.provider_requests < 1
        ):
            raise ValueError("embedding provider_requests must be a positive integer")


OPTION_CLASSIFICATIONS = frozenset(
    {"credential", "semantic", "execution", "artifact-safe"}
)
_CLASSIFICATION_KEY = "stel_classification"


def provider_option(classification: str, **field_kwargs: Any) -> Any:
    """Declare a provider profile-option field with exactly one classification.

    `credential` fields must be typed `CredentialReference` and stay out of
    repr, serialization, artifacts, and fingerprints; `semantic` fields enter
    the response-cache key and model identity; `execution` fields never
    invalidate state; `artifact-safe` fields may additionally appear verbatim
    in manifest provider descriptors."""
    if classification not in OPTION_CLASSIFICATIONS:
        raise ValueError(
            f"provider option classification must be one of "
            f"{sorted(OPTION_CLASSIFICATIONS)}"
        )
    if "json_schema_extra" in field_kwargs:
        raise ValueError("provider_option owns json_schema_extra")
    if classification == "credential":
        field_kwargs.setdefault("repr", False)
        field_kwargs.setdefault("exclude", True)
    return Field(json_schema_extra={_CLASSIFICATION_KEY: classification}, **field_kwargs)


def option_classification(field: FieldInfo) -> str | None:
    extra = field.json_schema_extra
    if not isinstance(extra, dict):
        return None
    value = extra.get(_CLASSIFICATION_KEY)
    return value if isinstance(value, str) else None


def _is_credential_annotation(annotation: Any) -> bool:
    if annotation is CredentialReference:
        return True
    return get_origin(annotation) in {Union, UnionType} and all(
        arg is CredentialReference or arg is type(None)
        for arg in get_args(annotation)
    )


def validate_profile_options_model(
    provider_name: str, model: type[BaseModel]
) -> None:
    """Registration-time validation of a provider's published options model."""
    config = model.model_config
    if not (
        config.get("extra") == "forbid"
        and config.get("frozen") is True
        and config.get("hide_input_in_errors") is True
    ):
        raise ProviderRegistrationError(
            f"provider '{provider_name}' profile options model must set "
            "extra='forbid', frozen=True, and hide_input_in_errors=True"
        )
    for field_name, field_info in model.model_fields.items():
        classification = option_classification(field_info)
        if classification is None or classification not in OPTION_CLASSIFICATIONS:
            raise ProviderRegistrationError(
                f"provider '{provider_name}' profile option '{field_name}' must "
                "declare exactly one classification via provider_option(...)"
            )
        if classification == "credential":
            if not _is_credential_annotation(field_info.annotation):
                raise ProviderRegistrationError(
                    f"provider '{provider_name}' credential option '{field_name}' "
                    "must be typed CredentialReference"
                )
            if field_info.repr or not field_info.exclude:
                raise ProviderRegistrationError(
                    f"provider '{provider_name}' credential option '{field_name}' "
                    "must be excluded from repr and serialization"
                )
        elif _is_credential_annotation(field_info.annotation):
            raise ProviderRegistrationError(
                f"provider '{provider_name}' option '{field_name}' holds a "
                "credential reference and must be classified credential"
            )


def parse_profile_options(
    provider_cls: type[BaseProvider],
    options: Mapping[str, Any] | None,
) -> BaseModel | None:
    """Validate operator-supplied `provider_options:` for the selected provider."""
    model_factory = getattr(provider_cls, "profile_options_model", None)
    model_candidate = model_factory() if callable(model_factory) else None
    model: type[BaseModel] | None = (
        model_candidate
        if isinstance(model_candidate, type) and issubclass(model_candidate, BaseModel)
        else None
    )
    if model is None:
        if options:
            raise ProviderConfigurationError(
                f"provider '{provider_cls.provider_name}' does not accept "
                "provider_options",
                safe_for_display=True,
            )
        return None
    try:
        return model.model_validate(dict(options or {}))
    except ValidationError:
        raise ProviderConfigurationError(
            f"provider '{provider_cls.provider_name}' rejected provider_options; "
            "run with the provider's documented option schema",
            safe_for_display=True,
        ) from None


def _classified_option_values(
    instance: BaseModel, classifications: frozenset[str]
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field_name, field_info in type(instance).model_fields.items():
        if option_classification(field_info) in classifications:
            values[field_name] = getattr(instance, field_name)
    return values


def profile_options_fingerprint(instance: BaseModel | None) -> str | None:
    """Semantic-option fingerprint for cache keys and model identity.

    Credential and execution fields never enter it; tuning them cannot
    invalidate state or caches."""
    if not isinstance(instance, BaseModel):
        return None
    semantic = _classified_option_values(
        instance, frozenset({"semantic", "artifact-safe"})
    )
    if not semantic:
        return None
    return canonical_fingerprint(
        semantic, domain="dbt-ml-provider-profile-options", version=1
    )


def artifact_safe_options(instance: BaseModel | None) -> dict[str, Any]:
    if not isinstance(instance, BaseModel):
        return {}
    return _classified_option_values(instance, frozenset({"artifact-safe"}))


class BaseProvider(ABC):
    provider_name: ClassVar[str]
    implementation_version: ClassVar[str]
    requires_credentials: ClassVar[bool] = True
    accepts_api_key_env: ClassVar[bool] = True
    default_credential_env: ClassVar[str | None] = None
    implementation_packages: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, profile_options: BaseModel | None = None) -> None:
        # Immutable typed configuration delivered once at the provider
        # boundary (issue #71); None when the provider publishes no model.
        self.profile_options = profile_options

    @classmethod
    def profile_options_model(cls) -> type[BaseModel] | None:
        """Strict Pydantic model validating profile `provider_options:`,
        or None when the provider accepts none."""
        return None

    @classmethod
    def name(cls) -> str:
        return cls.provider_name

    def resolve_credential(
        self,
        env_var: str | CredentialReference | None,
    ) -> ProviderCredential | None:
        self.validate_credential_reference(env_var)
        if not self.requires_credentials:
            return None
        failure: ProviderConfigurationError | None = None
        reference: CredentialReference | None = None
        selected = env_var or self.default_credential_env
        if selected is None:
            failure = ProviderConfigurationError(
                f"{self.name()} requires a configured credential environment variable",
                safe_for_display=True,
            )
        else:
            try:
                reference = (
                    selected
                    if isinstance(selected, CredentialReference)
                    else CredentialReference.from_env_name(selected)
                )
            except (TypeError, CredentialReferenceError):
                failure = ProviderConfigurationError(
                    f"{self.name()} credential environment variable name is invalid",
                    safe_for_display=True,
                )
        if reference is not None:
            try:
                return reference.resolve()
            except CredentialResolutionError:
                failure = ProviderConfigurationError(
                    f"Inference provider '{self.name()}' credential environment "
                    "variable is not set or is empty.",
                    safe_for_display=True,
                )
            except Exception as error:
                failure = ProviderConfigurationError(
                    f"Inference provider '{self.name()}' credential resolution "
                    f"failed [{type(error).__name__}].",
                    safe_for_display=True,
                )
        env_var = None
        selected = None
        if failure is not None:
            raise failure
        raise AssertionError("provider credential resolution did not complete")

    def validate_credential_reference(
        self,
        env_var: str | CredentialReference | None,
    ) -> None:
        """Validate whether a profile may supply `api_key_env`, without I/O."""
        if env_var is not None and not self.accepts_api_key_env:
            raise ProviderConfigurationError(
                f"{self.name()} does not accept api_key_env",
                safe_for_display=True,
            )

    def implementation_identity(self) -> str:
        return _implementation_identity(type(self))


class InferenceProvider(BaseProvider):
    default_model: ClassVar[str | None] = None
    supports_custom_base_url: ClassVar[bool] = False
    requires_base_url: ClassVar[bool] = False
    default_base_url: ClassVar[OpenAICompatibleBaseUrl | None] = None
    supports_native_batch: ClassVar[bool] = False
    max_batch_requests: ClassVar[int | None] = None
    batch_cost_multiplier: ClassVar[float] = 1.0
    # Whether this provider's structured-output surface can carry an `enum`
    # constraint (issue #304). Opt-in, and deliberately so: a provider written
    # before this existed cannot have declared it, and sending a keyword its
    # API may reject is a hard failure, where the default path — the closed set
    # rendered into the prompt — still communicates the taxonomy. So an
    # undeclared provider degrades instead of breaking, and every shipped
    # provider opts in explicitly below.
    supports_schema_enum: ClassVar[bool] = False

    @abstractmethod
    def complete(
        self,
        request: InferenceRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> InferenceResult: ...

    def resolve_model(self, model: str | None) -> str:
        if model is not None and (not isinstance(model, str) or not model):
            raise ProviderConfigurationError(
                f"Inference provider '{self.name()}' requires a non-empty model",
                safe_for_display=True,
            )
        selected = self.default_model if model is None else model
        if not isinstance(selected, str) or not selected:
            raise ProviderConfigurationError(
                f"Inference provider '{self.name()}' requires an explicit model",
                safe_for_display=True,
            )
        return selected

    def resolve_base_url(
        self, base_url: str | OpenAICompatibleBaseUrl | None
    ) -> OpenAICompatibleBaseUrl | None:
        selected = self.default_base_url if base_url is None else base_url
        if selected is None:
            if self.requires_base_url:
                raise ProviderConfigurationError(
                    f"Inference provider '{self.name()}' requires base_url in "
                    "the active profile",
                    safe_for_display=True,
                )
            return None
        if not self.supports_custom_base_url:
            raise ProviderConfigurationError(
                f"Inference provider '{self.name()}' does not support base_url",
                safe_for_display=True,
            )
        try:
            return OpenAICompatibleBaseUrl(selected)
        except EndpointUrlError as error:
            raise ProviderConfigurationError(
                f"Inference provider '{self.name()}' {error}",
                safe_for_display=True,
            ) from None

    def validate_result(self, result: object) -> InferenceResult:
        if not isinstance(result, InferenceResult):
            raise ProviderResponseError(
                "provider inference result has an invalid type",
                safe_for_display=True,
            )
        return result

    def submit_batch(
        self,
        requests: Sequence[BatchInferenceRequest],
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> str:
        """Submit one native batch and return the provider's job identifier.

        Native-batch providers must override; the identifier is persisted so
        an interrupted run resumes the job instead of resubmitting it.
        """
        raise ProviderBatchError(
            f"{self.name()} does not support resumable native batches",
            safe_for_display=True,
        )

    def poll_batch(
        self,
        batch_id: str,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> BatchJobStatus:
        raise ProviderBatchError(
            f"{self.name()} does not support resumable native batches",
            safe_for_display=True,
        )

    def fetch_batch_results(
        self,
        batch_id: str,
        requests: Sequence[BatchInferenceRequest],
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> BatchInferenceResult:
        raise ProviderBatchError(
            f"{self.name()} does not support resumable native batches",
            safe_for_display=True,
        )

    def cancel_batch(
        self,
        batch_id: str,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> None:
        raise ProviderBatchError(
            f"{self.name()} does not support resumable native batches",
            safe_for_display=True,
        )

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
        if self.supports_native_batch:
            # Convenience one-shot driver over the resumable primitives.
            # Callers needing persistence, timeout, or cancellation drive
            # submit/poll/fetch themselves (the LLM backend does).
            batch_id = validate_batch_job_id(
                self.submit_batch(requests, credential=credential, runtime=runtime)
            )
            while not self.poll_batch(
                batch_id, credential=credential, runtime=runtime
            ).done:
                time.sleep(poll_seconds)
            fetched = self.fetch_batch_results(
                batch_id, requests, credential=credential, runtime=runtime
            )
            return BatchInferenceResult(fetched.items, batch_submissions=1)
        items: list[BatchInferenceItem] = []
        for item in requests:
            try:
                result = self.validate_result(
                    self.complete(
                        item.request,
                        credential=credential,
                        runtime=runtime,
                    )
                )
            except ProviderError as error:
                items.append(
                    BatchInferenceItem(
                        item.request_id,
                        error=sanitized_provider_error(
                            self.name(), "inference", error
                        ),
                    )
                )
            except Exception as error:
                if provider_error_debug_enabled() and log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        "%s sequential batch item failed:\n%s",
                        self.name(),
                        redacted_exception_text(error),
                    )
                items.append(
                    BatchInferenceItem(
                        item.request_id,
                        error=provider_request_error(
                            self.name(), "inference", error
                        ),
                    )
                )
            else:
                items.append(BatchInferenceItem(item.request_id, result=result))
        return BatchInferenceResult(tuple(items))

    def _validate_batch_inputs(
        self,
        requests: Sequence[BatchInferenceRequest],
        *,
        poll_seconds: float,
    ) -> None:
        _validate_batch_call(self, requests, poll_seconds=poll_seconds)

    def validate_batch_result(
        self,
        requests: Sequence[BatchInferenceRequest],
        result: BatchInferenceResult,
    ) -> BatchInferenceResult:
        if not isinstance(result, BatchInferenceResult):
            raise ProviderBatchError(
                "provider batch result has an invalid type",
                safe_for_display=True,
            )
        expected_ids = {request.request_id for request in requests}
        actual_ids = {item.request_id for item in result.items}
        if expected_ids != actual_ids or len(requests) != len(result.items):
            raise ProviderBatchError(
                f"{self.name()} batch result did not align one-to-one with requests",
                safe_for_display=True,
            )
        items: list[BatchInferenceItem] = []
        for item in result.items:
            if item.error is not None:
                items.append(
                    BatchInferenceItem(
                        item.request_id,
                        error=sanitized_provider_error(
                            self.name(), "batch item", item.error
                        ),
                    )
                )
                continue
            try:
                validated = self.validate_result(item.result)
            except ProviderError as error:
                items.append(
                    BatchInferenceItem(
                        item.request_id,
                        error=sanitized_provider_error(
                            self.name(), "batch item", error
                        ),
                    )
                )
            except Exception as error:
                if provider_error_debug_enabled() and log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        "%s batch item validation failed:\n%s",
                        self.name(),
                        redacted_exception_text(error),
                    )
                items.append(
                    BatchInferenceItem(
                        item.request_id,
                        error=ProviderResponseError(
                            f"{self.name()} batch item returned an invalid response",
                            safe_for_display=True,
                        ),
                    )
                )
            else:
                items.append(
                    BatchInferenceItem(item.request_id, result=validated)
                )
        return BatchInferenceResult(
            tuple(items),
            batch_submissions=result.batch_submissions,
        )


def resolve_provider_model(
    provider: InferenceProvider,
    model: str | None,
) -> str:
    failure: ProviderConfigurationError | None = None
    selected: Any = None
    try:
        selected = provider.resolve_model(model)
    except ProviderError as error:
        safe = sanitized_provider_error(
            provider.name(), "model resolution", error
        )
        if isinstance(safe, ProviderConfigurationError):
            failure = safe
        else:
            failure = ProviderConfigurationError(
                f"Inference provider '{provider.name()}' model resolution failed",
                safe_for_display=True,
            )
    except Exception:
        failure = ProviderConfigurationError(
            f"Inference provider '{provider.name()}' model resolution failed",
            safe_for_display=True,
        )
    if failure is not None:
        model = None
        raise failure
    if not isinstance(selected, str) or not selected:
        raise ProviderConfigurationError(
            f"Inference provider '{provider.name()}' returned an invalid model",
            safe_for_display=True,
        )
    return selected


class EmbeddingProvider(BaseProvider):
    def embed(
        self,
        request: EmbeddingRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> EmbeddingResult:
        failure: ProviderError | None = None
        result: Any = None
        try:
            result = self._embed(
                request,
                credential=credential,
                runtime=runtime,
            )
        except ProviderError as error:
            failure = sanitized_provider_error(
                self.name(), "embedding", error
            )
        except Exception as error:
            # Redacted diagnostics stay in local debug logs; raised errors
            # and artifacts carry only the sanitized form.
            if provider_error_debug_enabled() and log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "%s embedding failed:\n%s",
                    self.name(),
                    redacted_exception_text(error),
                )
            failure = provider_request_error(self.name(), "embedding", error)
        if failure is not None:
            raise failure
        if not isinstance(result, EmbeddingResult):
            raise ProviderResponseError(
                "provider embedding result has an invalid type",
                safe_for_display=True,
            )
        if len(result.vectors) != len(request.texts):
            raise ProviderResponseError(
                "provider must return exactly one embedding per input text",
                safe_for_display=True,
            )
        if result.model != request.model:
            raise ProviderResponseError(
                "provider embedding model does not match the request",
                safe_for_display=True,
            )
        if request.dimensions is not None and result.dimensions != request.dimensions:
            raise ProviderResponseError(
                "provider embedding dimensions do not match the request",
                safe_for_display=True,
            )
        if request.input_ids is not None and result.input_ids != request.input_ids:
            raise ProviderResponseError(
                "provider embedding result IDs do not align with the request",
                safe_for_display=True,
            )
        return result

    @abstractmethod
    def _embed(
        self,
        request: EmbeddingRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> EmbeddingResult: ...


def implementation_identity_for(provider_type: type[BaseProvider]) -> str:
    """Class-level implementation identity, without running the constructor."""
    return _implementation_identity(provider_type)


@cache
def _implementation_identity(provider_type: type[BaseProvider]) -> str:
    # Deliberately excludes the stel release and module source digests:
    # response caches survive unrelated upgrades. Provider behavior changes
    # bump implementation_version; contract-wide changes bump the contract.
    payload = {
        "contract_version": PROVIDER_CONTRACT_VERSION,
        "provider_class": _identity_qualname(provider_type),
        "provider_implementation_version": provider_type.implementation_version,
        "provider_dependency_versions": {
            package: _distribution_version(package)
            for package in provider_type.implementation_packages
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(
        canonical.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()
    return f"provider-v{PROVIDER_CONTRACT_VERSION}/{digest}"


# Frozen: the package this project used to be called. `provider_class` above is
# the one part of the provider identity that tracks our own module layout, and
# the identity is designed to stay stable across releases so cached provider
# responses survive an upgrade. Letting the #313 rename move it would have
# re-keyed every cache and every `llm:`/`embed:` model's state at once — a full
# reprocess at provider cost, which is exactly what the identity exists to
# avoid. In-tree providers keep reporting their original module path; a
# third-party provider's path is untouched either way.
_IDENTITY_PACKAGE = "dbt_ml"
_PACKAGE = __name__.split(".")[0]


def _identity_qualname(provider_type: type[BaseProvider]) -> str:
    module = provider_type.__module__
    head, separator, tail = module.partition(".")
    if head == _PACKAGE:
        module = _IDENTITY_PACKAGE + separator + tail
    return f"{module}.{provider_type.__qualname__}"


def _distribution_version(package: str) -> str:
    try:
        return package_version(package)
    except PackageNotFoundError:
        return "not-installed"


def _safe_error_code(code: str) -> str:
    if not isinstance(code, str) or not code:
        return "provider_error"
    if len(code) > 64 or any(
        not (
            character.isascii()
            and (character.isalnum() or character in "._:-")
        )
        for character in code
    ):
        return "provider_error"
    return code


def _safe_error_label(value: str, *, fallback: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 64:
        return fallback
    if any(
        not (
            character.isascii()
            and (character.isalnum() or character in " _.-")
        )
        for character in value
    ):
        return fallback
    return value


def _validate_batch_call(
    provider: InferenceProvider,
    requests: Sequence[BatchInferenceRequest],
    *,
    poll_seconds: float,
) -> None:
    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, (int, float))
        or not math.isfinite(poll_seconds)
        or poll_seconds < 0
    ):
        raise ValueError("poll_seconds must be a non-negative finite number")
    if any(not isinstance(request, BatchInferenceRequest) for request in requests):
        raise ValueError("batch requests must be BatchInferenceRequest values")
    request_ids = [request.request_id for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ProviderBatchError(
            "batch request IDs must be unique", safe_for_display=True
        )
    if (
        provider.max_batch_requests is not None
        and len(requests) > provider.max_batch_requests
    ):
        raise ProviderBatchError(
            f"{len(requests)} requests exceed the {provider.name()} batch limit "
            f"of {provider.max_batch_requests}",
            safe_for_display=True,
        )
