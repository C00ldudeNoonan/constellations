"""The transcript/v1 landing contract (issue #360).

One JSON document per agent session, written into a landing directory that an
ordinary local source then consumes through the json backend. The rendered
`text` carries the whole reduced session with one `## [<ordinal>] <prompt>`
markdown heading per exchange, so the existing chunk splitter attributes every
chunk to the exchange that produced it; `exchanges` carries the structured
metadata a chunk-grain wrapper joins back by that ordinal.

This shape is an explicit versioned contract: additions bump the minor
version, incompatible changes bump `transcript/v<N>`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

TRANSCRIPT_SCHEMA_VERSION = "transcript/v1"

Harness = Literal["claude-code", "codex"]


class TranscriptExchange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ordinal: int
    # The `## [<ordinal>] <first prompt line>` heading exactly as rendered in
    # `text` (without the leading `## `), unique within the session by the
    # ordinal prefix — identical prompts must not collide the join key.
    heading: str
    started_at: datetime | None
    ended_at: datetime | None
    prompt_chars: int
    prompt_truncated: bool
    assistant_chars: int
    tool_calls: int
    tool_errors: int
    tools_used: tuple[str, ...]
    files_touched: tuple[str, ...]


class TranscriptDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["transcript/v1"] = TRANSCRIPT_SCHEMA_VERSION
    harness: Harness
    session_id: str
    source_path: str
    project_path: str | None
    git_branch: str | None
    started_at: datetime | None
    ended_at: datetime | None
    exchange_count: int
    tools_used: tuple[str, ...]
    files_touched: tuple[str, ...]
    text: str
    exchanges: tuple[TranscriptExchange, ...]
