from __future__ import annotations

import hashlib
import json
import logging
import threading
from functools import partial
from pathlib import Path
from typing import Any

import duckdb

from ..config.profile import DEFAULT_LLM_PROVIDER
from ..hashing import HASH_DIGEST_SIZE
from ..providers import (
    PROVIDER_CONTRACT_VERSION,
    BatchInferenceItem,
    BatchInferenceRequest,
    BatchInferenceResult,
    InferenceProvider,
    InferenceRequest,
    ProviderCredential,
    ProviderError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderRuntimeOptions,
    ProviderUsage,
    get_inference_provider,
    provider_error_debug_enabled,
    redacted_exception_text,
    resolve_provider_model,
    sanitized_provider_error,
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

# DuckDB cache writes can race when extraction is parallelized; serialize them.
_CACHE_WRITE_LOCK = threading.Lock()

# API-level concurrency caps are account-wide, so gates live at module scope
# and are shared across every model in the process. One gate per configured
# size: models that agree on max_concurrent share a limit; models that
# disagree get independent gates (combined ceiling = sum of distinct sizes).
_GATES: dict[int, threading.BoundedSemaphore] = {}
_GATES_LOCK = threading.Lock()


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
        provider = get_inference_provider(provider_name)
        model = resolve_provider_model(provider, options.get("model"))
        api_key_env = _api_key_env(options) or provider.default_credential_env
        _resolve_provider_credential(provider, api_key_env)
        fields_spec = options.get("fields")
        if not fields_spec or not isinstance(fields_spec, list):
            raise ValueError(
                "llm backend requires `options.fields: [{name, type, ...}]`"
            )

        fields, usage = extract_fields_with_usage(
            path.read_text(),
            fields_spec=fields_spec,
            provider=provider_name,
            model=model,
            system=options.get("system_prompt", _DEFAULT_SYSTEM),
            cache_path=options.get("cache_path"),
            call_api=partial(
                self._call_api,
                provider=provider_name,
                api_key_env=api_key_env,
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
        self, paths: list[Path], options: dict[str, Any]
    ) -> BatchExtractionOutput:
        """One native batch submission for every uncached document (issue #75
        part 2): cache hits resolve locally, the rest go up as a single batch,
        results come back keyed by request ID, and every response is cached so
        re-runs are free. Per-document failures come back as Exception entries;
        only submission itself can fail the whole batch."""
        options = self.parse_options(options)
        validate_llm_numeric_options(options)
        provider_name = str(options.get("provider", DEFAULT_LLM_PROVIDER))
        provider = get_inference_provider(provider_name)
        model = resolve_provider_model(provider, options.get("model"))
        api_key_env = _api_key_env(options) or provider.default_credential_env
        _resolve_provider_credential(provider, api_key_env)
        if not provider.supports_native_batch:
            raise RuntimeError(
                f"Inference provider '{provider_name}' does not support native "
                "batch execution; disable `batch:`."
            )
        fields_spec = options.get("fields")
        if not fields_spec or not isinstance(fields_spec, list):
            raise ValueError(
                "llm backend requires `options.fields: [{name, type, ...}]`"
            )
        system = options.get("system_prompt", _DEFAULT_SYSTEM)
        temperature = float(options.get("temperature", _DEFAULT_TEMPERATURE))
        max_tokens = int(options.get("max_tokens", _DEFAULT_MAX_TOKENS))
        poll_seconds = float(
            options.get("batch_poll_seconds", _DEFAULT_BATCH_POLL_SECONDS)
        )
        cache_path = options.get("cache_path")
        cache_path_obj = Path(cache_path) if cache_path is not None else None
        schema_hash = _hash_schema(system, fields_spec, temperature)
        provider_identity = provider.implementation_identity()
        max_retries = int(options.get("max_retries", _DEFAULT_MAX_RETRIES))

        by_index: dict[int, ExtractionResult | Exception] = {}
        batch_metrics: dict[str, Any] = {}
        pending: list[tuple[int, str, str, str]] = []
        for i, path in enumerate(paths):
            try:
                text = path.read_text()
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
            pending.append((i, text, content_hash, cache_key))

        if pending:
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
            batch_result = _run_message_batch(
                requests,
                provider=provider_name,
                poll_seconds=poll_seconds,
                api_key_env=api_key_env,
                max_retries=max_retries,
            )
            batch_metrics["batch_submissions"] = batch_result.batch_submissions
            items = {item.request_id: item for item in batch_result.items}
            for j, (i, _, content_hash, cache_key) in enumerate(pending):
                by_index[i] = self._resolve_batch_item(
                    items.get(f"req-{j}"),
                    cache_path=cache_path_obj,
                    cache_key=cache_key,
                    model=model,
                    content_hash=content_hash,
                    schema_hash=schema_hash,
                )

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
    ) -> ExtractionResult | Exception:
        if item is None:
            return RuntimeError("Provider batch returned no result for document")
        if item.error is not None:
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
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Extract structured fields from text with a registered provider.

    Reusable from transform models that need to LLM-process rows of text
    (e.g. text extracted from PDFs in an upstream model). Discards usage;
    call `extract_fields_with_usage` to get token accounting too.

    `call_api` remains injectable for compatibility and deterministic tests.
    """
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
    api_key_env: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Like `extract_fields_from_text`, but also returns usage accounting
    (issue #75): api_calls, cache_hits, and token counts for the call. A cache
    hit is zero tokens and zero API calls — the cache stores only fields, and
    cached responses cost nothing."""
    validate_llm_numeric_options(
        {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_retries": max_retries,
            "max_concurrent": max_concurrent,
        }
    )
    provider_instance = get_inference_provider(provider)
    resolved_model = resolve_provider_model(provider_instance, model)
    content_hash = hashlib.blake2b(
        text.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()
    schema_hash = _hash_schema(system, fields_spec, temperature)
    cache_key = _cache_key(
        provider=provider,
        provider_identity=provider_instance.implementation_identity(),
        model=resolved_model,
        content_hash=content_hash,
        schema_hash=schema_hash,
        max_tokens=max_tokens,
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
    api_key_env: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provider_instance = get_inference_provider(provider)
    credential = _resolve_provider_credential(provider_instance, api_key_env)
    request = _inference_request(
        content=content,
        model=model,
        system=system,
        fields_spec=fields_spec,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        result = provider_instance.validate_result(
            provider_instance.complete(
                request,
                credential=credential,
                runtime=ProviderRuntimeOptions(max_retries=max_retries),
            )
        )
    except ProviderError as error:
        raise sanitized_provider_error(
            provider, "inference", error
        ) from None
    except Exception as error:
        if provider_error_debug_enabled() and log.isEnabledFor(logging.DEBUG):
            log.debug(
                "provider '%s' inference failed:\n%s",
                provider,
                redacted_exception_text(
                    error,
                    sensitive=(
                        credential.reveal() if credential else None,
                        request.content,
                        request.system_prompt,
                    ),
                ),
            )
        raise ProviderRequestError(
            provider, "inference", code=type(error).__name__
        ) from None
    return dict(result.output), result.usage.to_metrics()


def _run_message_batch(
    requests: list[BatchInferenceRequest],
    *,
    provider: str = DEFAULT_LLM_PROVIDER,
    poll_seconds: float,
    api_key_env: str | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> BatchInferenceResult:
    provider_instance = get_inference_provider(provider)
    credential = _resolve_provider_credential(provider_instance, api_key_env)
    try:
        result = provider_instance.complete_batch(
            requests,
            credential=credential,
            runtime=ProviderRuntimeOptions(max_retries=max_retries),
            poll_seconds=poll_seconds,
        )
        return provider_instance.validate_batch_result(requests, result)
    except ProviderError as error:
        raise sanitized_provider_error(
            provider, "batch inference", error
        ) from None
    except Exception as error:
        if provider_error_debug_enabled() and log.isEnabledFor(logging.DEBUG):
            sensitive: list[str | None] = [
                credential.reveal() if credential else None
            ]
            for item in requests:
                sensitive.append(item.request.content)
                sensitive.append(item.request.system_prompt)
            log.debug(
                "provider '%s' batch inference failed:\n%s",
                provider,
                redacted_exception_text(error, sensitive=tuple(sensitive)),
            )
        raise ProviderRequestError(
            provider, "batch inference", code=type(error).__name__
        ) from None


def _inference_request(
    *,
    content: str,
    model: str,
    system: str,
    fields_spec: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> InferenceRequest:
    return InferenceRequest(
        model=model,
        content=content,
        system_prompt=system,
        output_schema=_input_schema(fields_spec),
        output_name="extract",
        output_description=(
            "Return the extracted structured fields from the document."
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _resolve_provider_credential(
    provider: InferenceProvider,
    api_key_env: str | None,
) -> ProviderCredential | None:
    try:
        return provider.resolve_credential(api_key_env)
    except ProviderError as error:
        raise sanitized_provider_error(
            provider.name(), "credential resolution", error
        ) from None
    except Exception as error:
        raise ProviderRequestError(
            provider.name(),
            "credential resolution",
            code=type(error).__name__,
        ) from None


def _api_key_env(options: dict[str, Any]) -> str | None:
    value = options.get("api_key_env")
    return str(value) if value is not None else None


def _input_schema(fields_spec: list[dict[str, Any]]) -> dict[str, Any]:
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
    return {"type": "object", "properties": properties}


def _hash_schema(
    system: str, fields_spec: list[dict[str, Any]], temperature: float
) -> str:
    canonical = json.dumps(
        {"system": system, "fields": fields_spec, "temperature": temperature},
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
) -> str:
    # Keyed on the semantic request plus the provider's contract identity —
    # not the backend implementation or dbt-ml release, so cached responses
    # survive routine upgrades. Row-shaping changes invalidate incremental
    # state through the model code version instead.
    canonical = json.dumps(
        {
            "contract_version": PROVIDER_CONTRACT_VERSION,
            "provider": provider,
            "provider_identity": provider_identity,
            "model": model,
            "content_hash": content_hash,
            "schema_hash": schema_hash,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.blake2b(
        canonical.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()
    return f"provider-v{PROVIDER_CONTRACT_VERSION}|{provider}|{model}|{digest}"


def _cache_get(path: Path, key: str) -> dict[str, Any] | None:
    if not path.exists():
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
        _prune_legacy_entries(con, path)
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


# Paths already swept this process; guarded by _CACHE_WRITE_LOCK.
_PRUNED_CACHE_PATHS: set[str] = set()


def _prune_legacy_entries(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    """Drop pre-provider-contract rows (`{model}|{content}|{schema}` keys).

    They can never be read again — every current key carries the
    `provider-v…|` prefix — so they only grow the cache file. Versioned
    entries are kept even across contract bumps to keep downgrades cheap.
    """
    resolved = str(path.resolve())
    if resolved in _PRUNED_CACHE_PATHS:
        return
    con.execute("DELETE FROM llm_cache WHERE cache_key NOT LIKE 'provider-v%'")
    _PRUNED_CACHE_PATHS.add(resolved)
