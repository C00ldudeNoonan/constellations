"""Parser for Claude Code session transcripts (issue #360).

Format: one JSON object per line under `~/.claude/projects/<slug>/<id>.jsonl`.
The format is unversioned and carries many auxiliary record types
(`attachment`, `system`, `queue-operation`, title records, …), so this parser
is tolerant by contract: unknown types, unparseable lines, and missing fields
skip rather than fail. Only `user` and `assistant` records become events, and
of those, sidechain (subagent) and meta records are dropped.

Reduction happens here because this is the last code that sees the raw file:
tool results contribute only an error flag and a byte count, tool arguments
only a fingerprint plus known file-bearing argument values, and thinking
blocks are dropped entirely.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..hashing import canonical_fingerprint
from .context_calls import parse_context_call
from .contract import Harness
from .events import (
    AssistantProse,
    ParsedSession,
    SessionEvent,
    ToolCall,
    UserTurn,
    file_arguments,
    parse_timestamp,
)

HARNESS: Harness = "claude-code"


def parse_claude_code(
    path: Path, *, capture_query: bool = False
) -> ParsedSession | None:
    """Parse one session file, or None when it holds no conversation."""
    entries = _entries(path)
    session_id: str | None = None
    project_path: str | None = None
    git_branch: str | None = None
    results = _tool_results(entries)
    events: list[SessionEvent] = []
    for entry in entries:
        kind = entry.get("type")
        if kind not in ("user", "assistant"):
            continue
        if entry.get("isSidechain") or entry.get("isMeta"):
            continue
        if session_id is None and isinstance(entry.get("sessionId"), str):
            session_id = entry["sessionId"]
        if project_path is None and isinstance(entry.get("cwd"), str):
            project_path = entry["cwd"]
        if git_branch is None and isinstance(entry.get("gitBranch"), str):
            git_branch = entry["gitBranch"]
        timestamp = parse_timestamp(entry.get("timestamp"))
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if kind == "user":
            text = _user_text(content)
            if text:
                events.append(UserTurn(text=text, timestamp=timestamp))
            continue
        events.extend(
            _assistant_events(
                content, timestamp, results, capture_query=capture_query
            )
        )
    if session_id is None or not any(isinstance(e, UserTurn) for e in events):
        return None
    return ParsedSession(
        harness=HARNESS,
        session_id=session_id,
        source_path=path,
        project_path=project_path,
        git_branch=git_branch,
        events=tuple(events),
    )


def _entries(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            # Tolerant by contract: a torn tail line on a live file, or a
            # record shape this parser has never seen, must not fail the run.
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    return entries


def _tool_results(
    entries: list[dict[str, Any]],
) -> dict[str, tuple[bool | None, int, Any]]:
    """tool_use id -> (ok, result byte count, raw payload). Results arrive as
    user-role `tool_result` blocks, possibly after intervening records, so
    they are collected up front and joined onto their calls by id.

    The raw payload is held only long enough for `parse_context_call` to
    recognize a stel context response; it never reaches a landing document.
    """
    results: dict[str, tuple[bool | None, int, Any]] = {}
    for entry in entries:
        if entry.get("type") != "user":
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            use_id = block.get("tool_use_id")
            if not isinstance(use_id, str):
                continue
            is_error = block.get("is_error")
            ok = None if not isinstance(is_error, bool) else not is_error
            payload = block.get("content")
            results[use_id] = (ok, _content_bytes(payload), payload)
    return results


def _content_bytes(content: Any) -> int:
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    if content is None:
        return 0
    return len(json.dumps(content, ensure_ascii=False, default=str).encode("utf-8"))


def _user_text(content: Any) -> str | None:
    """The prompt text of a real user turn; None for tool-result carriers."""
    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None
    if not isinstance(content, list):
        return None
    texts = [
        block["text"].strip()
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block["text"].strip()
    ]
    if any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    ):
        return None
    return "\n\n".join(texts) or None


def _assistant_events(
    content: Any,
    timestamp: Any,
    results: dict[str, tuple[bool | None, int, Any]],
    *,
    capture_query: bool,
) -> list[SessionEvent]:
    if not isinstance(content, list):
        return []
    events: list[SessionEvent] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                events.append(AssistantProse(text=text.strip(), timestamp=timestamp))
        elif kind == "tool_use":
            name = block.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            args = block.get("input")
            args_dict: dict[str, Any] = args if isinstance(args, dict) else {}
            use_id = block.get("id")
            ok: bool | None = None
            result_bytes: int | None = None
            payload: Any = None
            if isinstance(use_id, str) and use_id in results:
                ok, result_bytes, payload = results[use_id]
            events.append(
                ToolCall(
                    name=name,
                    args_fingerprint=canonical_fingerprint(
                        {"name": name, "args": args_dict},
                        domain="stel.transcript-tool-args",
                    ),
                    files=file_arguments(args_dict),
                    ok=ok,
                    result_bytes=result_bytes,
                    timestamp=timestamp,
                    context=parse_context_call(
                        name, args_dict, payload, capture_query=capture_query
                    ),
                )
            )
    return events
