"""Assembly, reduction accounting, rendering, and landing writes (issue #360).

The exchange is the unit: one user turn plus everything it caused, sealed by
the next user turn. Each exchange renders under a `## [<ordinal>] <prompt>`
markdown heading, so the existing chunk splitter's heading attribution maps
every chunk back to the request that produced it; the ordinal prefix keeps
headings unique when prompts repeat ("continue").

Landing writes are atomic (temp file + replace) and named
`{harness}-{session_id}.json`, so re-converting a grown live session rewrites
exactly one document and content-hash-based extraction reprocesses only it.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from .claude_code import parse_claude_code
from .codex import parse_codex
from .contract import TranscriptDocument, TranscriptExchange
from .events import AssistantProse, ParsedSession, SessionEvent, ToolCall, UserTurn

# Reduction caps. Prompt bodies carry pasted content — the main way secrets
# and bulk data would otherwise enter the corpus — so they are truncated with
# an explicit marker rather than kept whole (issue #360 §2).
_PROMPT_CHAR_CAP = 4000
_HEADING_CHAR_CAP = 100

# Default idle threshold for `sync`: a transcript modified more recently than
# this is treated as a live session and skipped (issue #360 §3).
DEFAULT_MIN_IDLE_SECONDS = 300


def detect_harness(path: Path) -> str | None:
    """Which parser understands this file, from its first parseable line."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                return None
            if not isinstance(obj, dict):
                return None
            if "payload" in obj and "type" in obj:
                return "codex"
            if "type" in obj and ("sessionId" in obj or "message" in obj):
                return "claude-code"
            return None
    return None


def parse_transcript(path: Path) -> ParsedSession | None:
    harness = detect_harness(path)
    if harness == "claude-code":
        return parse_claude_code(path)
    if harness == "codex":
        return parse_codex(path)
    return None


def build_document(session: ParsedSession) -> TranscriptDocument | None:
    """Assemble a transcript/v1 document, or None for an empty conversation.

    Events before the first user turn have no exchange to belong to and are
    dropped — for these harnesses that prefix is injected context, not
    conversation.
    """
    groups = _exchange_groups(session.events)
    if not groups:
        return None
    exchanges: list[TranscriptExchange] = []
    sections: list[str] = []
    for ordinal, group in enumerate(groups):
        exchange, rendered = _build_exchange(ordinal, group)
        exchanges.append(exchange)
        sections.append(rendered)
    tools_used = sorted({name for e in exchanges for name in e.tools_used})
    files_touched = sorted({name for e in exchanges for name in e.files_touched})
    timestamps = [e.started_at for e in exchanges if e.started_at is not None] + [
        e.ended_at for e in exchanges if e.ended_at is not None
    ]
    return TranscriptDocument(
        harness=session.harness,
        session_id=session.session_id,
        source_path=str(session.source_path),
        project_path=session.project_path,
        git_branch=session.git_branch,
        started_at=min(timestamps) if timestamps else None,
        ended_at=max(timestamps) if timestamps else None,
        exchange_count=len(exchanges),
        tools_used=tuple(tools_used),
        files_touched=tuple(files_touched),
        # The document opens directly with the first exchange heading: a
        # session-header line above it would leave every session's first
        # chunk attributed to no exchange (heading offsets are positional).
        text="\n\n".join(sections) + "\n",
        exchanges=tuple(exchanges),
    )


def convert_file(path: Path, out_dir: Path) -> Path | None:
    """Convert one transcript into the landing directory; None when the file
    is not a recognized transcript or holds no conversation."""
    session = parse_transcript(path)
    if session is None:
        return None
    document = build_document(session)
    if document is None:
        return None
    return write_landing_document(document, out_dir)


def write_landing_document(document: TranscriptDocument, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{document.harness}-{document.session_id}.json"
    payload = document.model_dump_json(indent=2)
    # Atomic publish: the landing directory is a live source for extraction,
    # which must never observe a half-written document.
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(payload + "\n", encoding="utf-8")
    temp.replace(target)
    return target


def sync_transcripts(
    *,
    out_dir: Path,
    claude_dir: Path | None,
    codex_dir: Path | None,
    min_idle_seconds: float,
) -> list[Path]:
    """Convert every settled transcript under the harness directories.

    A file modified more recently than `min_idle_seconds` is a live session
    still being appended; its sealed exchanges will land on a later sync.
    """
    candidates: list[Path] = []
    if claude_dir is not None and claude_dir.is_dir():
        candidates.extend(sorted(claude_dir.glob("*/*.jsonl")))
    if codex_dir is not None and codex_dir.is_dir():
        candidates.extend(sorted(codex_dir.rglob("*.jsonl")))
    cutoff = time.time() - min_idle_seconds
    written: list[Path] = []
    for path in candidates:
        if path.stat().st_mtime > cutoff:
            continue
        landed = convert_file(path, out_dir)
        if landed is not None:
            written.append(landed)
    return written


def default_claude_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".claude" / "projects"


def default_codex_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".codex" / "sessions"


def _exchange_groups(
    events: tuple[SessionEvent, ...],
) -> list[list[SessionEvent]]:
    groups: list[list[SessionEvent]] = []
    for event in events:
        if isinstance(event, UserTurn):
            groups.append([event])
        elif groups:
            groups[-1].append(event)
    return groups


def _build_exchange(
    ordinal: int, group: list[SessionEvent]
) -> tuple[TranscriptExchange, str]:
    prompt = group[0]
    assert isinstance(prompt, UserTurn)
    heading_text = _heading_text(prompt.text)
    heading = f"[{ordinal}] {heading_text}"
    body, truncated = _capped(prompt.text, _PROMPT_CHAR_CAP)
    lines = [f"## {heading}"]
    # A single-line prompt is already the heading; repeating it as the body
    # would only pad the index with duplicate text.
    if body != heading_text:
        lines.extend(["", body])

    prose_chars = 0
    tool_lines: list[str] = []
    tools_used: set[str] = set()
    files_touched: set[str] = set()
    tool_calls = 0
    tool_errors = 0
    timestamps: list[datetime] = []
    if prompt.timestamp is not None:
        timestamps.append(prompt.timestamp)
    for event in group[1:]:
        if event.timestamp is not None:
            timestamps.append(event.timestamp)
        if isinstance(event, AssistantProse):
            if tool_lines:
                lines.extend(["", *tool_lines])
                tool_lines = []
            lines.extend(["", event.text])
            prose_chars += len(event.text)
        elif isinstance(event, ToolCall):
            tool_calls += 1
            tools_used.add(event.name)
            files_touched.update(event.files)
            if event.ok is False:
                tool_errors += 1
            tool_lines.append(_tool_line(event))
    if tool_lines:
        lines.extend(["", *tool_lines])
    if files_touched:
        lines.extend(["", "Files: " + ", ".join(sorted(files_touched))])

    exchange = TranscriptExchange(
        ordinal=ordinal,
        heading=heading,
        started_at=min(timestamps) if timestamps else None,
        ended_at=max(timestamps) if timestamps else None,
        prompt_chars=len(prompt.text),
        prompt_truncated=truncated,
        assistant_chars=prose_chars,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        tools_used=tuple(sorted(tools_used)),
        files_touched=tuple(sorted(files_touched)),
    )
    return exchange, "\n".join(lines)


def _heading_text(prompt: str) -> str:
    first_line = next(
        (line.strip() for line in prompt.splitlines() if line.strip()), ""
    )
    # A leading '#' would change the markdown heading level; '[' would blur
    # the ordinal prefix that keeps headings unique.
    cleaned = " ".join(first_line.lstrip("#[ ").split())
    if len(cleaned) > _HEADING_CHAR_CAP:
        cleaned = cleaned[: _HEADING_CHAR_CAP - 1].rstrip() + "…"
    return cleaned or "(empty prompt)"


def _capped(text: str, cap: int) -> tuple[str, bool]:
    stripped = text.strip()
    if len(stripped) <= cap:
        return stripped, False
    dropped = len(stripped) - cap
    return stripped[:cap].rstrip() + f"\n[+{dropped} chars truncated]", True


def _tool_line(event: ToolCall) -> str:
    outcome = {True: "ok", False: "error"}.get(event.ok, "?")
    size = f", {event.result_bytes}B" if event.result_bytes is not None else ""
    return f"- tool {event.name} ({outcome}{size}) args:{event.args_fingerprint[:12]}"
