"""Parser for Codex CLI session rollouts (issue #360).

Format: one JSON object per line under
`~/.codex/sessions/<yyyy>/<mm>/<dd>/rollout-*.jsonl`, each an envelope of
`{timestamp, type, payload}`. Conversation lives in `response_item` payloads
(`message`, `custom_tool_call`/`function_call` and their `_output` twins);
`session_meta` carries the session identity. Everything else — `event_msg`
progress records, `turn_context`, `world_state`, reasoning payloads — is
dropped, as are developer-role messages and the environment/instruction
pseudo-user messages Codex injects.

Tolerant by contract, and the reduction point for this harness: tool outputs
contribute a byte count only, tool arguments a fingerprint plus known
file-bearing argument values.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..hashing import canonical_fingerprint
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

HARNESS: Harness = "codex"

_TOOL_CALL_TYPES = ("custom_tool_call", "function_call")
_TOOL_OUTPUT_TYPES = ("custom_tool_call_output", "function_call_output")
# Pseudo-user messages Codex injects; real prompts never start with these.
_INJECTED_USER_PREFIXES = ("<environment_context>", "<user_instructions>")


def parse_codex(path: Path) -> ParsedSession | None:
    """Parse one rollout file, or None when it holds no conversation."""
    lines = _payload_lines(path)
    session_id: str | None = None
    project_path: str | None = None
    git_branch: str | None = None
    outputs = _tool_outputs(lines)
    events: list[SessionEvent] = []
    for kind, timestamp, payload in lines:
        if kind == "session_meta":
            raw_id = payload.get("session_id") or payload.get("id")
            if session_id is None and isinstance(raw_id, str):
                session_id = raw_id
            if project_path is None and isinstance(payload.get("cwd"), str):
                project_path = payload["cwd"]
            git = payload.get("git")
            if git_branch is None and isinstance(git, dict):
                branch = git.get("branch")
                if isinstance(branch, str):
                    git_branch = branch
            continue
        if kind != "response_item":
            continue
        payload_type = payload.get("type")
        if payload_type == "message":
            event = _message_event(payload, timestamp)
            if event is not None:
                events.append(event)
        elif payload_type in _TOOL_CALL_TYPES:
            call = _tool_call_event(payload, timestamp, outputs)
            if call is not None:
                events.append(call)
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


def _payload_lines(path: Path) -> list[tuple[str, Any, dict[str, Any]]]:
    lines: list[tuple[str, Any, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            # Tolerant by contract: torn tail lines and unknown shapes skip.
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            kind = obj.get("type")
            payload = obj.get("payload")
            if isinstance(kind, str) and isinstance(payload, dict):
                lines.append((kind, parse_timestamp(obj.get("timestamp")), payload))
    return lines


def _tool_outputs(
    lines: list[tuple[str, Any, dict[str, Any]]],
) -> dict[str, int]:
    """call_id -> output byte count. Codex outputs carry no error verdict, so
    outcome stays unknown; the byte count still records the exhaust dropped."""
    outputs: dict[str, int] = {}
    for kind, _timestamp, payload in lines:
        if kind != "response_item" or payload.get("type") not in _TOOL_OUTPUT_TYPES:
            continue
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            continue
        output = payload.get("output")
        if isinstance(output, str):
            outputs[call_id] = len(output.encode("utf-8"))
        elif output is not None:
            outputs[call_id] = len(
                json.dumps(output, ensure_ascii=False, default=str).encode("utf-8")
            )
    return outputs


def _message_event(
    payload: dict[str, Any], timestamp: Any
) -> UserTurn | AssistantProse | None:
    role = payload.get("role")
    if role not in ("user", "assistant"):
        return None
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    text_types = ("input_text",) if role == "user" else ("output_text",)
    texts = [
        block["text"].strip()
        for block in content
        if isinstance(block, dict)
        and block.get("type") in text_types
        and isinstance(block.get("text"), str)
        and block["text"].strip()
    ]
    if role == "user":
        texts = [
            text
            for text in texts
            if not text.startswith(_INJECTED_USER_PREFIXES)
        ]
    joined = "\n\n".join(texts)
    if not joined:
        return None
    if role == "user":
        return UserTurn(text=joined, timestamp=timestamp)
    return AssistantProse(text=joined, timestamp=timestamp)


def _tool_call_event(
    payload: dict[str, Any],
    timestamp: Any,
    outputs: dict[str, int],
) -> ToolCall | None:
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    raw_args = payload.get("input")
    if raw_args is None:
        raw_args = payload.get("arguments")
    args_dict: dict[str, Any] = {}
    if isinstance(raw_args, dict):
        args_dict = raw_args
    elif isinstance(raw_args, str) and raw_args.strip().startswith("{"):
        # Function arguments usually travel as a JSON string.
        try:
            parsed = json.loads(raw_args)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            args_dict = parsed
    status = payload.get("status")
    ok: bool | None = None
    if status == "completed":
        ok = True
    elif status == "failed":
        ok = False
    call_id = payload.get("call_id")
    result_bytes = (
        outputs.get(call_id) if isinstance(call_id, str) else None
    )
    return ToolCall(
        name=name,
        args_fingerprint=canonical_fingerprint(
            {"name": name, "args": args_dict if args_dict else raw_args},
            domain="stel.transcript-tool-args",
        ),
        files=file_arguments(args_dict),
        ok=ok,
        result_bytes=result_bytes,
        timestamp=timestamp,
    )
