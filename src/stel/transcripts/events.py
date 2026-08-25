"""Neutral in-memory model of one agent session, shared by harness parsers.

Each parser reduces its harness's transcript format to this small vocabulary;
exchange assembly, reduction accounting, and markdown rendering happen once in
`convert` regardless of harness (issue #360).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .contract import Harness


@dataclass(frozen=True)
class UserTurn:
    """One real user prompt. Seals the previous exchange and opens the next."""

    text: str
    timestamp: datetime | None


@dataclass(frozen=True)
class AssistantProse:
    """Assistant text output. Thinking blocks and tool exhaust never get here."""

    text: str
    timestamp: datetime | None


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation, already reduced: never argument values or result
    bodies — only the fingerprint, the file paths named by known file-bearing
    argument keys, and the outcome accounting."""

    name: str
    args_fingerprint: str
    files: tuple[str, ...]
    # None when the transcript carries no verdict for this call.
    ok: bool | None
    result_bytes: int | None
    timestamp: datetime | None


SessionEvent = UserTurn | AssistantProse | ToolCall


@dataclass(frozen=True)
class ParsedSession:
    harness: Harness
    session_id: str
    source_path: Path
    project_path: str | None
    git_branch: str | None
    events: tuple[SessionEvent, ...]


# Tool-argument keys that name a file being read or written. Deterministic
# allow-list, never path guessing: a wrong guess would put noise in the best
# search filter this corpus has (issue #360 §2).
FILE_ARGUMENT_KEYS = ("file_path", "notebook_path")


def file_arguments(args: dict[str, object]) -> tuple[str, ...]:
    files = []
    for key in FILE_ARGUMENT_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            files.append(value)
    return tuple(files)


def parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    # fromisoformat has no non-raising variant; harness timestamps are
    # untrusted input, and an unparseable one degrades to None by design.
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
