from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import duckdb

from .base import BaseBackend, ExtractionResult
from .registry import register

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5"
_DEFAULT_SYSTEM = (
    "You extract structured fields from documents. "
    "Call the `extract` tool with the requested fields. "
    "If a field is genuinely missing from the document, use null."
)
# Extraction wants reproducibility, not creativity.
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_MAX_RETRIES = 4
_DEFAULT_MAX_CONCURRENT = 4
_DEFAULT_BATCH_POLL_SECONDS = 30.0
# Anthropic Message Batches API ceiling; multi-batch chunking is deferred.
_MAX_BATCH_REQUESTS = 100_000

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


@register
class LLMBackend(BaseBackend):
    """LLM-based extraction backend.

    Configures a schema in YAML; calls Claude with tool use to enforce structured
    output; caches responses in a DuckDB file keyed on (model, content_hash,
    schema_hash) so re-runs are free.

    Options:
        model:          Claude model id (default: claude-haiku-4-5)
        system_prompt:  Override system prompt
        cache_path:     Path to cache file (recommended: ./target/llm_cache.duckdb)
        fields:         [{name, type, description?}] — schema for tool input_schema
        temperature:    Sampling temperature (default 0 — deterministic extraction;
                        part of the cache key)
        max_tokens:     Response budget (default 2048); a truncated response is
                        an error, never partial data
        max_retries:    SDK retry budget for rate limits / transient errors
                        (default 4, exponential backoff)
        max_concurrent: Max in-flight API calls process-wide (default 4)
        batch:          Submit uncached documents through the Message Batches
                        API — 50% token cost, minutes-latency (default false;
                        keep off for dev loops)
        batch_poll_seconds: Poll interval while a batch runs (default 30)
    """

    def name(self) -> str:
        return "llm"

    def supported_formats(self) -> list[str]:
        return [".txt", ".md"]

    def extract(self, path: Path, options: dict[str, Any]) -> ExtractionResult:
        fields_spec = options.get("fields")
        if not fields_spec or not isinstance(fields_spec, list):
            raise ValueError(
                "llm backend requires `options.fields: [{name, type, ...}]`"
            )

        fields, usage = extract_fields_with_usage(
            path.read_text(),
            fields_spec=fields_spec,
            model=options.get("model", _DEFAULT_MODEL),
            system=options.get("system_prompt", _DEFAULT_SYSTEM),
            cache_path=options.get("cache_path"),
            call_api=self._call_api,
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
        """One Message Batches API submission for every uncached document
        (issue #75 part 2): cache hits resolve locally, the rest go up as a
        single batch, results come back keyed by custom_id, and every response
        is cached so re-runs are free. Per-document failures come back as
        Exception entries; only submission itself can fail the whole batch."""
        fields_spec = options.get("fields")
        if not fields_spec or not isinstance(fields_spec, list):
            raise ValueError(
                "llm backend requires `options.fields: [{name, type, ...}]`"
            )
        model = options.get("model", _DEFAULT_MODEL)
        system = options.get("system_prompt", _DEFAULT_SYSTEM)
        temperature = float(options.get("temperature", _DEFAULT_TEMPERATURE))
        max_tokens = int(options.get("max_tokens", _DEFAULT_MAX_TOKENS))
        poll_seconds = float(
            options.get("batch_poll_seconds", _DEFAULT_BATCH_POLL_SECONDS)
        )
        cache_path = options.get("cache_path")
        cache_path_obj = Path(cache_path) if cache_path is not None else None
        schema_hash = _hash_schema(system, fields_spec, temperature)

        by_index: dict[int, ExtractionResult | Exception] = {}
        pending: list[tuple[int, str, str]] = []
        for i, path in enumerate(paths):
            try:
                text = path.read_text()
            except Exception as e:
                by_index[i] = e
                continue
            content_hash = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
            cache_key = f"{model}|{content_hash}|{schema_hash}"
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
            pending.append((i, text, content_hash))

        if pending:
            if len(pending) > _MAX_BATCH_REQUESTS:
                raise RuntimeError(
                    f"{len(pending)} uncached documents exceed the Message "
                    f"Batches API limit of {_MAX_BATCH_REQUESTS} requests per "
                    "batch. Split the run with --select, or disable `batch:`."
                )
            tool = _extract_tool(fields_spec)
            requests = [
                {
                    "custom_id": f"req-{j}",
                    "params": {
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "system": system,
                        "tools": [tool],
                        "tool_choice": {"type": "tool", "name": "extract"},
                        "messages": [{"role": "user", "content": text}],
                    },
                }
                for j, (_, text, _) in enumerate(pending)
            ]
            items = _run_message_batch(requests, poll_seconds=poll_seconds)
            for j, (i, _, content_hash) in enumerate(pending):
                by_index[i] = self._resolve_batch_item(
                    items.get(f"req-{j}"),
                    max_tokens=max_tokens,
                    cache_path=cache_path_obj,
                    model=model,
                    content_hash=content_hash,
                    schema_hash=schema_hash,
                )

        return [by_index[i] for i in range(len(paths))]

    @staticmethod
    def _resolve_batch_item(
        item: Any,
        *,
        max_tokens: int,
        cache_path: Path | None,
        model: str,
        content_hash: str,
        schema_hash: str,
    ) -> ExtractionResult | Exception:
        if item is None:
            return RuntimeError("Message Batches API returned no result for document")
        result_type = item.result.type
        if result_type != "succeeded":
            detail = getattr(item.result, "error", None)
            suffix = f": {detail}" if detail is not None else ""
            return RuntimeError(f"batch request {result_type}{suffix}")
        try:
            fields, usage = _parse_extract_response(
                item.result.message, max_tokens=max_tokens
            )
        except RuntimeError as e:
            return e
        if cache_path is not None:
            _cache_put(
                cache_path,
                f"{model}|{content_hash}|{schema_hash}",
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
    model: str = _DEFAULT_MODEL,
    system: str = _DEFAULT_SYSTEM,
    cache_path: str | Path | None = None,
    call_api: Any = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
) -> dict[str, Any]:
    """Extract structured fields from a string of text by calling Claude.

    Reusable from transform models that need to LLM-process rows of text
    (e.g. text extracted from PDFs in an upstream model). Discards usage;
    call `extract_fields_with_usage` to get token accounting too.

    `call_api` is injectable for testing; defaults to the real Anthropic call.
    """
    fields, _ = extract_fields_with_usage(
        text,
        fields_spec=fields_spec,
        model=model,
        system=system,
        cache_path=cache_path,
        call_api=call_api,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        max_concurrent=max_concurrent,
    )
    return fields


def extract_fields_with_usage(
    text: str,
    *,
    fields_spec: list[dict[str, Any]],
    model: str = _DEFAULT_MODEL,
    system: str = _DEFAULT_SYSTEM,
    cache_path: str | Path | None = None,
    call_api: Any = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Like `extract_fields_from_text`, but also returns usage accounting
    (issue #75): api_calls, cache_hits, and token counts for the call. A cache
    hit is zero tokens and zero API calls — the cache stores only fields, and
    cached responses cost nothing."""
    content_hash = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
    schema_hash = _hash_schema(system, fields_spec, temperature)
    cache_key = f"{model}|{content_hash}|{schema_hash}"

    cache_path_obj = Path(cache_path) if cache_path is not None else None
    if cache_path_obj is not None:
        cached = _cache_get(cache_path_obj, cache_key)
        if cached is not None:
            return cached, {"api_calls": 0, "cache_hits": 1, **_ZERO_USAGE}

    fn = call_api or _default_call_api
    with _gate(max_concurrent):
        raw = fn(
            text,
            model,
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
    usage = {"api_calls": 1, "cache_hits": 0, **_ZERO_USAGE, **call_usage}

    if cache_path_obj is not None:
        _cache_put(
            cache_path_obj,
            cache_key,
            model=model,
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
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Either export it or seed the "
            "llm cache so re-runs hit cached responses."
        )
    from anthropic import Anthropic

    # The SDK retries 429s / 5xx / timeouts with exponential backoff.
    client = Anthropic(max_retries=max_retries)
    resp = client.messages.create(  # type: ignore[call-overload]
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        tools=[_extract_tool(fields_spec)],
        tool_choice={"type": "tool", "name": "extract"},
        messages=[{"role": "user", "content": content}],
    )
    return _parse_extract_response(resp, max_tokens=max_tokens)


def _extract_tool(fields_spec: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": "extract",
        "description": "Return the extracted structured fields from the document.",
        "input_schema": _input_schema(fields_spec),
    }


def _parse_extract_response(
    resp: Any, *, max_tokens: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fields + usage from a Messages API response — shared by the synchronous
    call and per-item batch results (identical message shape)."""
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(
            f"LLM response truncated at max_tokens={max_tokens}; partial "
            "extractions are never used. Raise `max_tokens` in the model's "
            "extraction options."
        )
    usage = {
        key: getattr(resp.usage, key, None) or 0
        for key in _ZERO_USAGE
    }
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "extract":
            return dict(block.input), usage
    raise RuntimeError("LLM did not return an `extract` tool call")


def _run_message_batch(
    requests: list[dict[str, Any]], *, poll_seconds: float
) -> dict[str, Any]:
    """Submit one Message Batch, poll to completion, return items keyed by
    custom_id (results stream back unordered). Injectable for testing, like
    `_default_call_api`."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Either export it or seed the "
            "llm cache so re-runs hit cached responses."
        )
    from anthropic import Anthropic

    client = Anthropic()
    batch = client.messages.batches.create(requests=requests)  # type: ignore[arg-type]
    log.info("submitted message batch %s (%d requests)", batch.id, len(requests))
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
    return {r.custom_id: r for r in client.messages.batches.results(batch.id)}


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
    return hashlib.blake2b(canonical.encode(), digest_size=8).hexdigest()


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
