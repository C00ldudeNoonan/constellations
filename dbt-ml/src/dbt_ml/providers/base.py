from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import traceback
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, ClassVar

from ..credentials import (
    CredentialReference,
    CredentialReferenceError,
    CredentialResolutionError,
    ProtectedCredential,
)
from ..hashing import HASH_DIGEST_SIZE

log = logging.getLogger(__name__)

PROVIDER_CONTRACT_VERSION = 2


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
    pass


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
    ProviderError: "dbt_ml.providers.ProviderError",
    ProviderRegistrationError: "dbt_ml.providers.ProviderRegistrationError",
    ProviderNotFoundError: "dbt_ml.providers.ProviderNotFoundError",
    ProviderConfigurationError: "dbt_ml.providers.ProviderConfigurationError",
    ProviderRequestError: "dbt_ml.providers.ProviderRequestError",
    ProviderResponseError: "dbt_ml.providers.ProviderResponseError",
    ProviderBatchError: "dbt_ml.providers.ProviderBatchError",
}


PROVIDER_DEBUG_ENV = "DBT_ML_DEBUG_PROVIDER_ERRORS"


def provider_error_debug_enabled() -> bool:
    """Whether operators opted into allowlisted SDK diagnostics in debug logs.

    Off by default: an SDK error message can echo fragments of the request
    that no redaction can anticipate, and debug logs are often shipped to
    aggregators. Diagnostics contain only exception types and stack locations;
    the switch exists for local diagnosis.
    """
    value = os.environ.get(PROVIDER_DEBUG_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def redacted_exception_text(
    error: BaseException,
    *,
    sensitive: Sequence[str | None] = (),
) -> str:
    """Return allowlisted exception diagnostics without provider metadata.

    Exact-value replacement cannot safely redact repr-, JSON-, or URL-encoded
    request data. Keep the compatibility name and argument, but emit only
    recognized exception categories, dbt-ml module locations, and an external
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
                and (module_name == "dbt_ml" or module_name.startswith("dbt_ml."))
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
            return ProviderBatchError(str(error), safe_for_display=True)
        return ProviderBatchError(
            f"{safe_provider} {safe_operation} failed",
            safe_for_display=True,
        )
    return ProviderRequestError(
        safe_provider,
        safe_operation,
        code=type(error).__name__,
    )


ProviderCredential = ProtectedCredential


@dataclass(frozen=True, slots=True)
class ProviderRuntimeOptions:
    max_retries: int = 4

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reported_cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
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
        if self.reported_cost_usd is not None:
            metrics["reported_cost_usd"] = self.reported_cost_usd
        return metrics


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

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("batch item request_id must not be empty")
        if bool(self.result is not None) == bool(self.error is not None):
            raise ValueError("batch item must contain exactly one of result or error")
        if self.result is not None and not isinstance(self.result, InferenceResult):
            raise ValueError("batch item result must be InferenceResult")
        if self.error is not None and not isinstance(self.error, ProviderError):
            raise ValueError("batch item error must be ProviderError")


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
class EmbeddingRequest:
    model: str
    texts: tuple[str, ...] = field(repr=False)
    dimensions: int | None = None

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


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...] = field(repr=False)
    model: str
    dimensions: int
    usage: ProviderUsage = field(default_factory=ProviderUsage)

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
        if not isinstance(self.usage, ProviderUsage):
            raise ValueError("embedding usage must be ProviderUsage")


class BaseProvider(ABC):
    provider_name: ClassVar[str]
    implementation_version: ClassVar[str]
    requires_credentials: ClassVar[bool] = True
    default_credential_env: ClassVar[str | None] = None
    implementation_packages: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def name(cls) -> str:
        return cls.provider_name

    def resolve_credential(
        self,
        env_var: str | CredentialReference | None,
    ) -> ProviderCredential | None:
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

    def implementation_identity(self) -> str:
        return _implementation_identity(type(self))


class InferenceProvider(BaseProvider):
    default_model: ClassVar[str | None] = None
    supports_native_batch: ClassVar[bool] = False
    max_batch_requests: ClassVar[int | None] = None
    batch_cost_multiplier: ClassVar[float] = 1.0

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

    def validate_result(self, result: object) -> InferenceResult:
        if not isinstance(result, InferenceResult):
            raise ProviderResponseError(
                "provider inference result has an invalid type",
                safe_for_display=True,
            )
        return result

    def complete_batch(
        self,
        requests: Sequence[BatchInferenceRequest],
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
        poll_seconds: float,
    ) -> BatchInferenceResult:
        self._validate_batch_inputs(requests, poll_seconds=poll_seconds)
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
                        error=ProviderRequestError(
                            self.name(),
                            "inference",
                            code=type(error).__name__,
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
            failure = ProviderRequestError(
                self.name(), "embedding", code=type(error).__name__
            )
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
        return result

    @abstractmethod
    def _embed(
        self,
        request: EmbeddingRequest,
        *,
        credential: ProviderCredential | None,
        runtime: ProviderRuntimeOptions,
    ) -> EmbeddingResult: ...


@cache
def _implementation_identity(provider_type: type[BaseProvider]) -> str:
    # Deliberately excludes the dbt-ml release and module source digests:
    # response caches survive unrelated upgrades. Provider behavior changes
    # bump implementation_version; contract-wide changes bump the contract.
    payload = {
        "contract_version": PROVIDER_CONTRACT_VERSION,
        "provider_class": (
            f"{provider_type.__module__}.{provider_type.__qualname__}"
        ),
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
