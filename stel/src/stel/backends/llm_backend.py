from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import duckdb

from ..budget import BudgetExceededError, BudgetGuard
from ..config.profile import DEFAULT_LLM_PROVIDER
from ..credentials import CredentialReference, CredentialReferenceError
from ..hashing import HASH_DIGEST_SIZE
from ..providers import (
    PROVIDER_CONTRACT_VERSION,
    BatchInferenceItem,
    BatchInferenceRequest,
    BatchInferenceResult,
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
    get_inference_provider,
    profile_options_fingerprint,
    provider_error_debug_enabled,
    provider_request_error,
    redacted_exception_text,
    resolve_provider_model,
    sanitized_provider_error,
    validate_batch_job_id,
)
from .base import BaseBackend, BatchExtractionOutput, ExtractionResult
from .options import LLMBackendOptions, validate_llm_numeric_options
from .registry import register

log = logging.getLogger(__name__)

_DEFAULT_SYSTEM = (
    "You extract structured fields from documents. "
    "Return structured fields that match the requested output schema. "
    "If a field is genuinely missing from the document, use null."
)
# Extraction wants reproducibility, not creativity.
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_MAX_RETRIES = 4
_DEFAULT_MAX_CONCURRENT = 4
_DEFAULT_BATCH_POLL_SECONDS = 30.0
_DEFAULT_BATCH_POLL_MAX_SECONDS = 300.0
_DEFAULT_BATCH_TIMEOUT_SECONDS = 86_400.0
_DEFAULT_BATCH_SIZE = 1000
_POLL_BACKOFF_FACTOR = 1.5
# Submitted-job records older than this are unrecoverable provider-side and
# only accumulate; prune on the next job write.
_STALE_JOB_DAYS = 30


class BatchCancelledError(RuntimeError):
    """A native batch job hit batch_timeout_seconds and was cancelled.

    Message is artifact-safe: it names the timeout, never job contents.
    """


_DEFAULT_TIMEOUT_SECONDS = 60.0

# DuckDB cache writes can race when extraction is parallelized; serialize them.
_CACHE_WRITE_LOCK = threading.Lock()

# API-level concurrency caps are account-wide, so gates live at module scope
# and are shared across every model in the process. One gate per configured
# size: models that agree on max_concurrent share a limit; models that
# disagree get independent gates (combined ceiling = sum of distinct sizes).
_GATES: dict[int, threading.BoundedSemaphore] = {}
_GATES_LOCK = threading.Lock()

type CredentialSelector = str | CredentialReference | None


def _fields_spec(options: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_fields = options.get("fields")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise ValueError(
            "llm backend requires `options.fields: [{name, type, ...}]`"
        )
    fields: list[dict[str, Any]] = []
    for field in raw_fields:
        if not isinstance(field, dict) or any(
            not isinstance(key, str) for key in field
        ):
            raise ValueError(
                "llm backend requires `options.fields: [{name, type, ...}]`"
            )
        fields.append(
            {key: value for key, value in field.items() if isinstance(key, str)}
        )
    return fields


def _protect_credential_selector(
    selector: CredentialSelector,
) -> tuple[CredentialReference | None, bool]:
    if selector is None or isinstance(selector, CredentialReference):
        return selector, True
    try:
        return CredentialReference.from_env_name(selector), True
    except CredentialReferenceError:
        return None, False


def _gate(size: int) -> threading.BoundedSemaphore:
    with _GATES_LOCK:
        if size not in _GATES:
            _GATES[size] = threading.BoundedSemaphore(size)
        return _GATES[size]


@register(
    options_model=LLMBackendOptions,
    native_batch=True,
    requires_credentials=True,
)
class LLMBackend(BaseBackend):
    """LLM-based extraction backend.

    Configures a schema in YAML, delegates structured output to a registered
    inference provider, and caches responses in DuckDB so re-runs are free.

    Options:
        provider:       Registered inference provider (default: anthropic)
        model:          Provider model id (default is owned by the provider)
        system_prompt:  Override system prompt
        cache_path:     Path to cache file (recommended: ./target/llm_cache.duckdb)
        fields:         [{name, type, description?}] — structured output schema
        temperature:    Sampling temperature (default 0 — deterministic extraction;
                        part of the cache key)
        max_tokens:     Response budget (default 2048); a truncated response is
                        an error, never partial data
        max_retries:    SDK retry budget for rate limits / transient errors
                        (default 4, exponential backoff)
        max_concurrent: Max in-flight API calls process-wide (default 4)
        api_key_env:    Operator-owned credential environment variable
        base_url:       Operator-owned custom provider endpoint
        timeout_seconds: Per-request timeout (default 60 seconds)
        batch:          Use the provider's native batch API (default false)
        batch_poll_seconds: Poll interval while a batch runs (default 30)
    """

    def name(self) -> str:
        return "llm"

    def supported_formats(self) -> list[str]:
        return [".txt", ".md"]

    def extract(self, path: Path, options: dict[str, Any]) -> ExtractionResult:
        options = self.parse_options(options)
        validate_llm_numeric_options(options)
        provider_name = str(options.get("provider", DEFAULT_LLM_PROVIDER))
        provider_options = options.get("provider_options") or None
        provider = (
            get_inference_provider(provider_name, profile_options=provider_options)
            if provider_options
            else get_inference_provider(provider_name)
        )
        model = resolve_provider_model(provider, options.get("model"))
        base_url = provider.resolve_base_url(options.get("base_url"))
        api_key_env = _provider_api_key_env(options, provider)
        _resolve_provider_credential(provider, api_key_env)
        fields_spec = _fields_spec(options)

        call_options: dict[str, Any] = {
            "provider": provider_name,
            "api_key_env": api_key_env,
        }
        if provider_options:
            call_options["provider_options"] = provider_options
        if base_url is not None:
            call_options["base_url"] = base_url
        if "timeout_seconds" in options:
            call_options["timeout_seconds"] = options["timeout_seconds"]

        fields, usage = extract_fields_with_usage(
            path.read_text(),
            fields_spec=fields_spec,
            provider=provider_name,
            model=model,
            system=options.get("system_prompt", _DEFAULT_SYSTEM),
            cache_path=options.get("cache_path"),
            call_api=partial(self._call_api, **call_options),
            base_url=base_url,
            provider_options=provider_options,
            timeout_seconds=float(
                options.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
            ),
            temperature=float(options.get("temperature", _DEFAULT_TEMPERATURE)),
            max_tokens=int(options.get("max_tokens", _DEFAULT_MAX_TOKENS)),
            max_retries=int(options.get("max_retries", _DEFAULT_MAX_RETRIES)),
            max_concurrent=int(options.get("max_concurrent", _DEFAULT_MAX_CONCURRENT)),
        )
        return ExtractionResult(fields=fields, metrics=usage)

    def _call_api(
        self,
        content: str,
        model: str,
        system: str,
        fields_spec: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]] | dict[str, Any]:
        return _default_call_api(content, model, system, fields_spec, **kwargs)

    def extract_batch(
        self, paths: list[Path], options: dict[str, Any]
    ) -> list[ExtractionResult | Exception]:
        return self.extract_batch_with_metrics(paths, options).items

    def extract_batch_with_metrics(
        self,
        paths: list[Path],
        options: dict[str, Any],
        *,
        budget: BudgetGuard | None = None,
    ) -> BatchExtractionOutput:
        """Native batch extraction in deterministic bounded partitions.

        Cache hits resolve locally; uncached documents stream into
        partitions of min(batch_size, provider limit) and each partition is
        one resumable native batch submission (issue #149): the provider job
        id is persisted in the cache database before polling, so an
        interrupted run resumes the job instead of resubmitting it. Document
        text is held only for the partition in flight, bounding memory
        independently of corpus size. Budgets are checked before each
        submission and charged from each partition's results."""
        options = self.parse_options(options)
        validate_llm_numeric_options(options)
        provider_name = str(options.get("provider", DEFAULT_LLM_PROVIDER))
        provider_options = options.get("provider_options") or None
        provider = (
            get_inference_provider(provider_name, profile_options=provider_options)
            if provider_options
            else get_inference_provider(provider_name)
        )
        model = resolve_provider_model(provider, options.get("model"))
        base_url = provider.resolve_base_url(options.get("base_url"))
        api_key_env = _provider_api_key_env(options, provider)
        _resolve_provider_credential(provider, api_key_env)
        if not provider.supports_native_batch:
            raise RuntimeError(
                f"Inference provider '{provider_name}' does not support native "
                "batch execution; disable `batch:`."
            )
        fields_spec = _fields_spec(options)
        system = options.get("system_prompt", _DEFAULT_SYSTEM)
        temperature = float(options.get("temperature", _DEFAULT_TEMPERATURE))
        max_tokens = int(options.get("max_tokens", _DEFAULT_MAX_TOKENS))
        poll_seconds = float(
            options.get("batch_poll_seconds", _DEFAULT_BATCH_POLL_SECONDS)
        )
        poll_max_seconds = max(
            float(
                options.get(
                    "batch_poll_max_seconds", _DEFAULT_BATCH_POLL_MAX_SECONDS
                )
            ),
            poll_seconds,
        )
        timeout_seconds = float(
            options.get("batch_timeout_seconds", _DEFAULT_BATCH_TIMEOUT_SECONDS)
        )
        request_timeout_seconds = float(
            options.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
        )
        partition_size = _effective_batch_size(
            provider, int(options.get("batch_size", _DEFAULT_BATCH_SIZE))
        )
        cache_path = options.get("cache_path")
        cache_path_obj = Path(cache_path) if cache_path is not None else None
        schema_hash = _hash_schema(system, fields_spec, temperature)
        provider_identity = provider.implementation_identity()
        options_fingerprint = profile_options_fingerprint(
            getattr(provider, "profile_options", None)
        )
        max_retries = int(options.get("max_retries", _DEFAULT_MAX_RETRIES))

        by_index: dict[int, ExtractionResult | Exception] = {}
        batch_metrics: dict[str, Any] = {
            "batch_submissions": 0,
            "batches_resumed": 0,
            "batches_completed": 0,
        }
        failed_totals: dict[str, int | float] = {}
        pending: list[tuple[int, str, str, str]] = []

        def _flush_partition() -> None:
            nonlocal pending
            if not pending:
                return
            if budget is not None:
                budget.ensure_headroom(next_calls=len(pending))
            requests = [
                BatchInferenceRequest(
                    f"req-{j}",
                    _inference_request(
                        content=text,
                        model=model,
                        system=system,
                        fields_spec=fields_spec,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                )
                for j, (_, text, _, _) in enumerate(pending)
            ]
            job_key = _batch_job_key(
                provider=provider_name,
                provider_identity=provider_identity,
                model=model,
                schema_hash=schema_hash,
                max_tokens=max_tokens,
                cache_keys=[cache_key for *_rest, cache_key in pending],
            )
            batch_result, resumed = _run_message_batch(
                requests,
                provider=provider_name,
                poll_seconds=poll_seconds,
                api_key_env=api_key_env,
                max_retries=max_retries,
                poll_max_seconds=poll_max_seconds,
                batch_timeout_seconds=timeout_seconds,
                base_url=base_url,
                request_timeout_seconds=request_timeout_seconds,
                cache_path=cache_path_obj,
                job_key=job_key,
                provider_options=provider_options,
            )
            batch_metrics["batch_submissions"] += batch_result.batch_submissions
            batch_metrics["batches_resumed"] += 1 if resumed else 0
            batch_metrics["batches_completed"] += 1
            items = {item.request_id: item for item in batch_result.items}
            for j, (i, _, content_hash, cache_key) in enumerate(pending):
                resolved = self._resolve_batch_item(
                    items.get(f"req-{j}"),
                    cache_path=cache_path_obj,
                    cache_key=cache_key,
                    model=model,
                    content_hash=content_hash,
                    schema_hash=schema_hash,
                    provider_name=provider_name,
                    provider_identity=provider_identity,
                )
                if budget is not None and isinstance(resolved, ExtractionResult):
                    budget.charge_metrics(resolved.metrics)
                elif isinstance(resolved, ProviderError) and resolved.failure is not None:
                    # Billed failures consume budget and are reported like
                    # billed successes (issue #71).
                    failure = resolved.failure
                    billed = {
                        "api_calls": failure.billed_requests,
                        **failure.usage.to_metrics(),
                    }
                    for key, value in billed.items():
                        failed_totals[key] = failed_totals.get(key, 0) + value
                    if budget is not None:
                        budget.charge_metrics(billed)
                by_index[i] = resolved
            pending = []

        def _flush_and_record() -> None:
            _flush_partition()
            for key, value in failed_totals.items():
                batch_metrics[f"failed_{key}"] = value

        for i, path in enumerate(paths):
            try:
                if budget is not None:
                    budget.check_file_bytes(path.stat().st_size)
                text = path.read_text()
            except BudgetExceededError:
                raise
            except Exception as e:
                by_index[i] = e
                continue
            content_hash = hashlib.blake2b(
                text.encode(), digest_size=HASH_DIGEST_SIZE
            ).hexdigest()
            cache_key = _cache_key(
                provider=provider_name,
                provider_identity=provider_identity,
                model=model,
                content_hash=content_hash,
                schema_hash=schema_hash,
                max_tokens=max_tokens,
                base_url=base_url,
                options_fingerprint=options_fingerprint,
            )
            cached = (
                _cache_get(cache_path_obj, cache_key)
                if cache_path_obj is not None
                else None
            )
            if cached is not None:
                by_index[i] = ExtractionResult(
                    fields=cached,
                    metrics={"api_calls": 0, "cache_hits": 1, **_ZERO_USAGE},
                )
                continue
            if budget is not None:
                budget.charge_bytes(path.stat().st_size)
            pending.append((i, text, content_hash, cache_key))
            if len(pending) >= partition_size:
                _flush_partition()
        _flush_and_record()

        return BatchExtractionOutput(
            [by_index[i] for i in range(len(paths))],
            metrics=batch_metrics,
        )

    @staticmethod
    def _resolve_batch_item(
        item: BatchInferenceItem | None,
        *,
        cache_path: Path | None,
        cache_key: str,
        model: str,
        content_hash: str,
        schema_hash: str,
        provider_name: str,
        provider_identity: str,
    ) -> ExtractionResult | Exception:
        if item is None:
            return RuntimeError("Provider batch returned no result for document")
        if item.error is not None:
            # Providers may report billed usage on the item instead of
            # attaching an InferenceFailure themselves; synthesize the
            # envelope so budgets and failed_* metrics account for it.
            if item.usage is not None and item.error.failure is None:
                item.error.attach_failure(
                    InferenceFailure(
                        error_code="batch_item_failed",
                        usage=item.usage,
                        billed_requests=1,
                        provider=provider_name,
                        model=model,
                        implementation_identity=provider_identity,
                    )
                )
            return item.error
        if item.result is None:
            return RuntimeError("Provider batch returned an empty result")
        fields = dict(item.result.output)
        usage = item.result.usage.to_metrics()
        if cache_path is not None:
            _cache_put(
                cache_path,
                cache_key,
                model=model,
                content_hash=content_hash,
                schema_hash=schema_hash,
                fields=fields,
            )
        return ExtractionResult(
            fields=fields,
            metrics={"api_calls": 1, "cache_hits": 0, **_ZERO_USAGE, **usage},
        )


_ZERO_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}


def extract_fields_from_text(
    text: str,
    *,
    fields_spec: list[dict[str, Any]],
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str | None = None,
    system: str = _DEFAULT_SYSTEM,
    cache_path: str | Path | None = None,
    call_api: Any = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    api_key_env: CredentialSelector = None,
    base_url: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Extract structured fields from text with a registered provider.

    Reusable from transform models that need to LLM-process rows of text
    (e.g. text extracted from PDFs in an upstream model). Discards usage;
    call `extract_fields_with_usage` to get token accounting too.

    `call_api` remains injectable for compatibility and deterministic tests.
    """
    api_key_env, credential_reference_is_valid = _protect_credential_selector(
        api_key_env
    )
    if not credential_reference_is_valid:
        raise ValueError(
            "llm api_key_env must be a valid environment-variable name"
        )
    fields, _ = extract_fields_with_usage(
        text,
        fields_spec=fields_spec,
        provider=provider,
        model=model,
        system=system,
        cache_path=cache_path,
        call_api=call_api,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        max_concurrent=max_concurrent,
        api_key_env=api_key_env,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    return fields


def extract_fields_with_usage(
    text: str,
    *,
    fields_spec: list[dict[str, Any]],
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str | None = None,
    system: str = _DEFAULT_SYSTEM,
    cache_path: str | Path | None = None,
    call_api: Any = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    api_key_env: CredentialSelector = None,
    base_url: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    provider_options: Mapping[str, Any] | None = None,
    output_cardinality: str = "one",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Like `extract_fields_from_text`, but also returns usage accounting
    (issue #75): api_calls, cache_hits, and token counts for the call. A cache
    hit is zero tokens and zero API calls — the cache stores only fields, and
    cached responses cost nothing.

    This is the shared structured-completion core (issue #144): both the
    document-oriented ``backend: llm`` extraction path (via
    :meth:`LLMBackend.extract`) and native ``llm:`` map models (via
    :func:`stel.llm_map.execute_map_item`) route through it, so provider
    resolution, caching, retries, and usage accounting exist in one place.

    `output_cardinality` (issue #144) controls the requested output shape:
    ``one`` returns a single object; ``many`` returns ``{"items": [...]}`` for
    the caller to unwrap. The default preserves the single-object extraction
    contract and its cache keys unchanged."""
    api_key_env, credential_reference_is_valid = _protect_credential_selector(
        api_key_env
    )
    if not credential_reference_is_valid:
        raise ValueError(
            "llm api_key_env must be a valid environment-variable name"
        )
    validate_llm_numeric_options(
        {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_retries": max_retries,
            "max_concurrent": max_concurrent,
            "timeout_seconds": timeout_seconds,
        }
    )
    provider_instance = (
        get_inference_provider(provider, profile_options=provider_options)
        if provider_options
        else get_inference_provider(provider)
    )
    resolved_model = resolve_provider_model(provider_instance, model)
    resolved_base_url = provider_instance.resolve_base_url(base_url)
    content_hash = hashlib.blake2b(
        text.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()
    schema_hash = _hash_schema(system, fields_spec, temperature, output_cardinality)
    cache_key = _cache_key(
        provider=provider,
        provider_identity=provider_instance.implementation_identity(),
        model=resolved_model,
        content_hash=content_hash,
        schema_hash=schema_hash,
        max_tokens=max_tokens,
        base_url=resolved_base_url,
        options_fingerprint=profile_options_fingerprint(
            getattr(provider_instance, "profile_options", None)
        ),
    )

    cache_path_obj = Path(cache_path) if cache_path is not None else None
    if cache_path_obj is not None:
        cached = _cache_get(cache_path_obj, cache_key)
        if cached is not None:
            return cached, {"api_calls": 0, "cache_hits": 1, **_ZERO_USAGE}

    fn = call_api
    if fn is None:
        fn = partial(
            _default_call_api,
            provider=provider,
            api_key_env=api_key_env,
            base_url=resolved_base_url,
            timeout_seconds=timeout_seconds,
            provider_options=provider_options,
            output_cardinality=output_cardinality,
        )
    with _gate(max_concurrent):
        raw = fn(
            text,
            resolved_model,
            system,
            fields_spec,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    # Injected test fakes may still return bare fields (the pre-#75 contract);
    # the real call returns (fields, usage).
    if isinstance(raw, tuple):
        result_fields, call_usage = raw
    else:
        result_fields, call_usage = raw, {}
    if not isinstance(result_fields, dict):
        raise ProviderResponseError(
            "provider structured output must be a mapping",
            safe_for_display=True,
        )
    normalized_usage = ProviderUsage.from_mapping(call_usage).to_metrics()
    usage = {
        "api_calls": 1,
        "cache_hits": 0,
        **_ZERO_USAGE,
        **normalized_usage,
    }

    if cache_path_obj is not None:
        _cache_put(
            cache_path_obj,
            cache_key,
            model=resolved_model,
            content_hash=content_hash,
            schema_hash=schema_hash,
            fields=result_fields,
        )
    return result_fields, usage


def _default_call_api(
    content: str,
    model: str,
    system: str,
    fields_spec: list[dict[str, Any]],
    *,
    provider: str = DEFAULT_LLM_PROVIDER,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    api_key_env: CredentialSelector = None,
    base_url: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    provider_options: Mapping[str, Any] | None = None,
    output_cardinality: str = "one",
) -> tuple[dict[str, Any], dict[str, Any]]:
    provider_instance = (
        get_inference_provider(provider, profile_options=provider_options)
        if provider_options
        else get_inference_provider(provider)
    )
    api_key_env, credential_reference_is_valid = _protect_credential_selector(
        api_key_env
    )
    if not credential_reference_is_valid:
        raise ProviderRequestError(
            provider,
            "credential resolution",
            code="invalid_credential_reference",
        )
    credential = _resolve_provider_credential(provider_instance, api_key_env)
    request = _inference_request(
        content=content,
        model=model,
        system=system,
        fields_spec=fields_spec,
        temperature=temperature,
        max_tokens=max_tokens,
        output_cardinality=output_cardinality,
    )
    failure: ProviderError | None = None
    result: InferenceResult | None = None
    try:
        result = provider_instance.validate_result(
            provider_instance.complete(
                request,
                credential=credential,
                runtime=ProviderRuntimeOptions(
                    max_retries=max_retries,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                ),
            )
        )
    except ProviderError as error:
        failure = sanitized_provider_error(
            provider, "inference", error
        )
    except Exception as error:
        if provider_error_debug_enabled() and log.isEnabledFor(logging.DEBUG):
            log.debug(
                "provider '%s' inference failed:\n%s",
                provider,
                redacted_exception_text(error),
            )
        failure = provider_request_error(provider, "inference", error)
    if failure is not None:
        raise failure
    if result is None:
        raise AssertionError("provider inference did not produce a result")
    return dict(result.output), result.usage.to_metrics()


def _run_message_batch(
    requests: list[BatchInferenceRequest],
    *,
    provider: str = DEFAULT_LLM_PROVIDER,
    poll_seconds: float,
    api_key_env: CredentialSelector = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    poll_max_seconds: float = _DEFAULT_BATCH_POLL_MAX_SECONDS,
    batch_timeout_seconds: float = _DEFAULT_BATCH_TIMEOUT_SECONDS,
    base_url: str | None = None,
    request_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    cache_path: Path | None = None,
    job_key: str | None = None,
    provider_options: Mapping[str, Any] | None = None,
) -> tuple[BatchInferenceResult, bool]:
    """Run one native batch partition: submit (or resume), poll with bounded
    backoff, fetch, and clear the persisted job record.

    The provider job id is persisted before the first poll, so a crash or
    interrupt leaves a record that the next run resumes instead of
    resubmitting — the submitted work is billed exactly once. On timeout the
    job is cancelled and the record removed. Returns the result plus whether
    an existing job was resumed."""
    provider_instance = (
        get_inference_provider(provider, profile_options=provider_options)
        if provider_options
        else get_inference_provider(provider)
    )
    api_key_env, credential_reference_is_valid = _protect_credential_selector(
        api_key_env
    )
    if not credential_reference_is_valid:
        raise ProviderRequestError(
            provider,
            "credential resolution",
            code="invalid_credential_reference",
        )
    credential = _resolve_provider_credential(provider_instance, api_key_env)
    runtime = ProviderRuntimeOptions(
        max_retries=max_retries,
        base_url=base_url,
        timeout_seconds=request_timeout_seconds,
    )
    resumed = False
    failure: Exception | None = None
    result: BatchInferenceResult | None = None
    try:
        batch_id = (
            _job_get(cache_path, job_key)
            if cache_path is not None and job_key is not None
            else None
        )
        if batch_id is not None:
            resumed = True
            log.info("resuming provider batch %s", batch_id)
        else:
            batch_id = validate_batch_job_id(
                provider_instance.submit_batch(
                    requests, credential=credential, runtime=runtime
                )
            )
            if cache_path is not None and job_key is not None:
                _job_put(cache_path, job_key, provider=provider, batch_id=batch_id)
        deadline = time.monotonic() + batch_timeout_seconds
        interval = poll_seconds
        while True:
            status = provider_instance.poll_batch(
                batch_id, credential=credential, runtime=runtime
            )
            if status.done:
                break
            now = time.monotonic()
            if now >= deadline:
                _cancel_batch_job(
                    provider_instance,
                    batch_id,
                    credential=credential,
                    runtime=runtime,
                )
                if cache_path is not None and job_key is not None:
                    _job_delete(cache_path, job_key)
                raise BatchCancelledError(
                    f"provider batch exceeded batch_timeout_seconds="
                    f"{batch_timeout_seconds:g} and was cancelled"
                )
            time.sleep(min(interval, max(deadline - now, 0.0)))
            interval = min(interval * _POLL_BACKOFF_FACTOR, poll_max_seconds)
        fetched = provider_instance.fetch_batch_results(
            batch_id, requests, credential=credential, runtime=runtime
        )
        validated = provider_instance.validate_batch_result(requests, fetched)
        if cache_path is not None and job_key is not None:
            _job_delete(cache_path, job_key)
        result = BatchInferenceResult(
            validated.items, batch_submissions=0 if resumed else 1
        )
    except BatchCancelledError:
        raise
    except ProviderError as error:
        failure = sanitized_provider_error(
            provider, "batch inference", error
        )
    except Exception as error:
        if provider_error_debug_enabled() and log.isEnabledFor(logging.DEBUG):
            log.debug(
                "provider '%s' batch inference failed:\n%s",
                provider,
                redacted_exception_text(error),
            )
        failure = provider_request_error(provider, "batch inference", error)
    if failure is not None:
        raise failure
    if result is None:
        raise AssertionError("provider batch inference did not produce a result")
    return result, resumed


def _cancel_batch_job(
    provider_instance: InferenceProvider,
    batch_id: str,
    *,
    credential: Any,
    runtime: ProviderRuntimeOptions,
) -> None:
    """Best-effort cancel on timeout; the timeout error is raised regardless."""
    try:
        provider_instance.cancel_batch(
            batch_id, credential=credential, runtime=runtime
        )
    except Exception:
        log.debug("batch cancel failed for %s", batch_id)


def _effective_batch_size(provider: InferenceProvider, batch_size: int) -> int:
    limit = provider.max_batch_requests
    size = batch_size if limit is None else min(batch_size, limit)
    return max(size, 1)


def _batch_job_key(
    *,
    provider: str,
    provider_identity: str,
    model: str,
    schema_hash: str,
    max_tokens: int,
    cache_keys: list[str],
) -> str:
    """Deterministic identity of one submitted partition, built from hashes
    only — resumable without persisting any document content."""
    canonical = json.dumps(
        {
            "contract_version": PROVIDER_CONTRACT_VERSION,
            "provider": provider,
            "provider_identity": provider_identity,
            "model": model,
            "schema_hash": schema_hash,
            "max_tokens": max_tokens,
            "cache_keys": cache_keys,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(
        canonical.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()


def _inference_request(
    *,
    content: str,
    model: str,
    system: str,
    fields_spec: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    output_cardinality: str = "one",
) -> InferenceRequest:
    if output_cardinality == "many":
        output_description = (
            "Return a list of objects under `items`, one per distinct result "
            "extracted from the input."
        )
    else:
        output_description = (
            "Return the extracted structured fields from the document."
        )
    return InferenceRequest(
        model=model,
        content=content,
        system_prompt=system,
        output_schema=_input_schema(fields_spec, output_cardinality),
        output_name="extract",
        output_description=output_description,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _resolve_provider_credential(
    provider: InferenceProvider,
    api_key_env: CredentialSelector,
) -> ProviderCredential | None:
    api_key_env, credential_reference_is_valid = _protect_credential_selector(
        api_key_env
    )
    failure: ProviderError | None = None
    credential: ProviderCredential | None = None
    if not credential_reference_is_valid:
        failure = ProviderRequestError(
            provider.name(),
            "credential resolution",
            code="invalid_credential_reference",
        )
    try:
        if failure is None:
            credential = provider.resolve_credential(api_key_env)
    except ProviderError as error:
        failure = sanitized_provider_error(
            provider.name(), "credential resolution", error
        )
    except Exception as error:
        failure = ProviderRequestError(
            provider.name(),
            "credential resolution",
            code=type(error).__name__,
        )
    if failure is not None:
        raise failure
    return credential


def _api_key_env(
    options: dict[str, Any],
) -> tuple[CredentialReference | None, bool]:
    value = options.get("api_key_env")
    if not isinstance(value, str | CredentialReference | None):
        return None, False
    return _protect_credential_selector(value)


def _provider_api_key_env(
    options: dict[str, Any], provider: InferenceProvider
) -> CredentialReference | None:
    configured, credential_reference_is_valid = _api_key_env(options)
    if not credential_reference_is_valid:
        options = {}
        raise ValueError(
            "llm api_key_env must be a valid environment-variable name"
        )
    if configured is not None:
        return configured
    default = provider.default_credential_env
    if default is None:
        return None
    return CredentialReference.from_env_name(default)


def _input_schema(
    fields_spec: list[dict[str, Any]],
    output_cardinality: str = "one",
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for f in fields_spec:
        name = f["name"]
        ftype = f.get("type", "string")
        prop: dict[str, Any] = {"type": ftype}
        if "description" in f:
            prop["description"] = f["description"]
        if ftype == "array":
            prop["items"] = f.get("items", {"type": "string"})
        properties[name] = prop
    object_schema = {"type": "object", "properties": properties}
    if output_cardinality == "many":
        # Fan-out models ask the provider for a list of objects; the caller
        # unwraps `items`. `one` keeps the flat object schema unchanged.
        return {
            "type": "object",
            "properties": {"items": {"type": "array", "items": object_schema}},
            "required": ["items"],
        }
    return object_schema


def _hash_schema(
    system: str,
    fields_spec: list[dict[str, Any]],
    temperature: float,
    output_cardinality: str = "one",
) -> str:
    payload: dict[str, Any] = {
        "system": system,
        "fields": fields_spec,
        "temperature": temperature,
    }
    # Preserve existing cache keys for the default (single-object) path; only
    # fan-out requests fold the cardinality into the schema hash.
    if output_cardinality != "one":
        payload["output_cardinality"] = output_cardinality
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.blake2b(
        canonical.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()


def _cache_key(
    *,
    provider: str,
    provider_identity: str,
    model: str,
    content_hash: str,
    schema_hash: str,
    max_tokens: int,
    base_url: str | None = None,
    options_fingerprint: str | None = None,
) -> str:
    # Keyed on the semantic request plus the provider's contract identity —
    # not the backend implementation or stel release, so cached responses
    # survive routine upgrades. Row-shaping changes invalidate incremental
    # state through the model code version instead.
    payload = {
        "contract_version": PROVIDER_CONTRACT_VERSION,
        "provider": provider,
        "provider_identity": provider_identity,
        "model": model,
        "content_hash": content_hash,
        "schema_hash": schema_hash,
        "max_tokens": max_tokens,
    }
    if base_url is not None:
        payload["base_url"] = base_url
    if options_fingerprint is not None:
        # Semantic provider_options change what the provider returns, so
        # they isolate cache entries; execution/credential fields never
        # reach this fingerprint.
        payload["provider_options"] = options_fingerprint
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.blake2b(
        canonical.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()
    return f"provider-v{PROVIDER_CONTRACT_VERSION}|{provider}|{model}|{digest}"


_JOB_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS llm_batch_jobs (
    job_key VARCHAR PRIMARY KEY,
    provider VARCHAR NOT NULL,
    batch_id VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL
)
"""


def _job_get(path: Path, job_key: str) -> str | None:
    """Provider batch id persisted for this partition identity, if any.

    Rows hold only the job-key hash, provider name, provider job id, and a
    timestamp — no document content, prompt, credential, or response data.
    """
    if os.name == "nt":
        if not path.exists():
            return None
    elif not _harden_cache_files(path, require_main=True):
        return None
    con = duckdb.connect(str(path), read_only=True)
    try:
        row = con.execute(
            "SELECT batch_id FROM llm_batch_jobs WHERE job_key = ?", [job_key]
        ).fetchone()
    except duckdb.CatalogException:
        return None
    finally:
        con.close()
    if row is None:
        return None
    try:
        return validate_batch_job_id(row[0])
    except ProviderBatchError:
        return None


def _job_put(path: Path, job_key: str, *, provider: str, batch_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_WRITE_LOCK:
        if os.name != "nt":
            _harden_cache_files(path, require_main=False)
        with _private_cache_umask():
            con = duckdb.connect(str(path))
            try:
                con.execute(_JOB_TABLE_DDL)
                con.execute(
                    "DELETE FROM llm_batch_jobs WHERE created_at < "
                    f"current_timestamp - INTERVAL {_STALE_JOB_DAYS} DAY"
                )
                con.execute(
                    """
                    INSERT INTO llm_batch_jobs
                        (job_key, provider, batch_id, created_at)
                    VALUES (?, ?, ?, current_timestamp)
                    ON CONFLICT (job_key) DO UPDATE SET
                        provider   = excluded.provider,
                        batch_id   = excluded.batch_id,
                        created_at = excluded.created_at
                    """,
                    [job_key, provider, batch_id],
                )
            finally:
                con.close()
        if os.name != "nt":
            _harden_cache_files(path, require_main=True)


def _job_delete(path: Path, job_key: str) -> None:
    if not path.exists():
        return
    with _CACHE_WRITE_LOCK:
        con = duckdb.connect(str(path))
        try:
            con.execute(
                "DELETE FROM llm_batch_jobs WHERE job_key = ?", [job_key]
            )
        except duckdb.CatalogException:
            pass
        finally:
            con.close()


def _cache_get(path: Path, key: str) -> dict[str, Any] | None:
    if os.name == "nt":
        if not path.exists():
            return None
    elif not _harden_cache_files(path, require_main=True):
        return None
    con = duckdb.connect(str(path), read_only=True)
    try:
        row = con.execute(
            "SELECT response_json FROM llm_cache WHERE cache_key = ?", [key]
        ).fetchone()
    except duckdb.CatalogException:
        return None
    finally:
        con.close()
    return json.loads(row[0]) if row else None


def _cache_put(
    path: Path,
    key: str,
    *,
    model: str,
    content_hash: str,
    schema_hash: str,
    fields: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_WRITE_LOCK:
        _cache_put_locked(path, key, model, content_hash, schema_hash, fields)


def _cache_put_locked(
    path: Path,
    key: str,
    model: str,
    content_hash: str,
    schema_hash: str,
    fields: dict[str, Any],
) -> None:
    if os.name != "nt":
        _harden_cache_files(path, require_main=False)
    with _private_cache_umask():
        con = duckdb.connect(str(path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key VARCHAR PRIMARY KEY,
                    model VARCHAR NOT NULL,
                    content_hash VARCHAR NOT NULL,
                    schema_hash VARCHAR NOT NULL,
                    response_json VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
                """
            )
            con.execute(
                """
                INSERT INTO llm_cache
                    (cache_key, model, content_hash, schema_hash, response_json, created_at)
                VALUES (?, ?, ?, ?, ?, current_timestamp)
                ON CONFLICT (cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    created_at    = excluded.created_at
                """,
                [key, model, content_hash, schema_hash, json.dumps(fields)],
            )
        finally:
            con.close()
    if os.name != "nt":
        _harden_cache_files(path, require_main=True)


@contextmanager
def _private_cache_umask() -> Iterator[None]:
    if os.name == "nt":
        yield
        return
    # DuckDB creates its write-ahead log beside the database. The process-wide
    # mask is restored immediately, while the cache write lock serializes this
    # package's writers.
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


def _harden_cache_files(path: Path, *, require_main: bool) -> bool:
    main_exists = _harden_optional_cache_file(path)
    _harden_optional_cache_file(Path(f"{path}.wal"))
    if require_main and not main_exists:
        return False
    return main_exists


def _harden_optional_cache_file(path: Path) -> bool:
    try:
        _harden_existing_cache_file(path)
    except FileNotFoundError:
        return False
    return True


def _harden_existing_cache_file(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("LLM cache path must be a regular, non-symlink file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as error:
        if os.path.lexists(path):
            raise ValueError(
                "LLM cache path must be a regular, non-symlink file"
            ) from error
        raise
    except OSError as error:
        raise ValueError(
            "LLM cache path must be a regular, non-symlink file"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(
                "LLM cache path must be a regular, non-symlink file"
            )
        fchmod = getattr(os, "fchmod", None)
        if fchmod is not None:
            fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
    finally:
        os.close(descriptor)
