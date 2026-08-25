"""Recognizing stel's own MCP calls inside a transcript (issue #380).

The reduction in #360 drops every tool result body, which is right for the
search corpus and wrong for exactly one case: a `search_context` call against
stel's own MCP server carries the retrieval judgment this project exists to
learn from — the query, the context ids that came back, and (with the prose
that follows) which of them the answer actually used.

This module is the narrow, structured exception. It keeps ids and a query
fingerprint, never result bodies and never — unless separately opted in — the
query text, mirroring `QueryLogConfig.capture_query_text`.

Recognition keys on the **response's own contract marker**
(`schema_version: mcp_context/v1`), not on the MCP server's name: the server
name is operator-chosen in client configuration, so matching it would
privilege one deployment's spelling and miss every other.
"""
from __future__ import annotations

import json
from typing import Any

from ..append_log import query_fingerprint
from ..mcp_server.contracts import MCP_CONTEXT_SCHEMA_VERSION
from .events import ContextCall

# The tool whose calls carry a (query, returned ids) judgment. `get_document`
# is a stel context call too, but it is a fetch of an already-chosen document
# rather than a retrieval to judge, so it is not captured here.
SEARCH_TOOL = "search_context"


def parse_context_call(
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    *,
    capture_query: bool,
) -> ContextCall | None:
    """The reduced record of one stel `search_context` call, or None when this
    is not one. `result` is the raw tool-result payload; anything that does not
    parse as an `mcp_context/v1` response is treated as an unrelated tool."""
    if not _is_search_tool(tool_name):
        return None
    payload = _as_context_payload(result)
    if payload is None:
        return None
    results = payload.get("results")
    rows = results if isinstance(results, list) else []
    context_ids: list[str] = []
    chunk_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        context_id = row.get("context_id")
        if isinstance(context_id, str) and context_id:
            context_ids.append(context_id)
        chunk_id = row.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            chunk_ids.append(chunk_id)
    query = args.get("query")
    query_text = query if isinstance(query, str) else None
    model = args.get("model")
    error_code = _error_code(payload)
    return ContextCall(
        model=model if isinstance(model, str) else None,
        # Fingerprinted with the same function and domain the MCP query log
        # uses, so a transcript row joins to a phase-1 log row directly.
        query_fingerprint=(
            query_fingerprint(query_text) if query_text is not None else None
        ),
        query_text=query_text if capture_query else None,
        returned_context_ids=tuple(context_ids),
        returned_chunk_ids=tuple(chunk_ids),
        cited_context_ids=(),
        # A failed call returned nothing because it failed, not because the
        # corpus had no match; only a successful empty result is a zero
        # result (Codex review).
        zero_results=error_code is None and not context_ids and not chunk_ids,
        error_code=error_code,
    )


def cited_ids(prose: str, candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Which returned ids the assistant's answer actually names.

    A mechanical substring test over opaque 32-hex ids, deliberately: a model
    judging its own citations is the self-reinforcement trap #329 rule 2
    names. Order follows `candidates` so the result is deterministic.
    """
    if not prose:
        return ()
    return tuple(
        candidate for candidate in candidates if candidate and candidate in prose
    )


def _error_code(payload: dict[str, Any]) -> str | None:
    """The MCP error code of a failed context response, else None.

    An error response still declares `mcp_context/v1` and carries empty
    `results`, so without this a denied or timed-out call would be recorded as
    a genuine zero-result retrieval.
    """
    error = payload.get("error")
    if error is None:
        return None
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str) and code:
            return code
    return "unknown"


def _is_search_tool(tool_name: str) -> bool:
    # Claude Code namespaces MCP tools as `mcp__<server>__<tool>`; other
    # harnesses pass the bare name. Both end in the tool's own name.
    return tool_name == SEARCH_TOOL or tool_name.endswith(f"__{SEARCH_TOOL}")


def _as_context_payload(result: Any) -> dict[str, Any] | None:
    """`result` parsed as an mcp_context/v1 response, or None.

    Handles the three shapes a tool result travels in: a JSON string, an
    already-decoded object, and the list of `{"type": "text"}` blocks MCP
    results usually arrive as.
    """
    if isinstance(result, list):
        for block in result:
            if not isinstance(block, dict):
                continue
            payload = _as_context_payload(block.get("text"))
            if payload is not None:
                return payload
        return None
    if isinstance(result, str):
        text = result.strip()
        if not text.startswith("{"):
            return None
        try:
            parsed = json.loads(text)
        except ValueError:
            return None
    else:
        parsed = result
    if not isinstance(parsed, dict):
        return None
    if parsed.get("schema_version") != MCP_CONTEXT_SCHEMA_VERSION:
        return None
    return parsed
