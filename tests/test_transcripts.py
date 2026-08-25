"""Agent transcript conversion (issue #360): harness parsing, exchange
assembly, reduction, rendering, and landing writes."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from stel.cli import cli
from stel.transcripts import (
    build_document,
    convert_file,
    detect_harness,
    parse_transcript,
    sync_transcripts,
)

_SID_CLAUDE = "0f5a2c1e-1111-4aaa-8bbb-000000000001"
_SID_CODEX = "0199aaaa-2222-7bbb-8ccc-000000000002"


def _claude_lines(prompts: list[str] | None = None) -> list[dict[str, Any]]:
    base = {"sessionId": _SID_CLAUDE, "cwd": "/work/app", "gitBranch": "main"}
    lines: list[dict[str, Any]] = [
        {"type": "queue-operation", "operation": "enqueue", "sessionId": _SID_CLAUDE},
        {
            **base,
            "type": "user",
            "timestamp": "2026-08-20T14:00:00Z",
            "message": {"role": "user", "content": "Fix the rounding test"},
        },
        {
            **base,
            "type": "assistant",
            "timestamp": "2026-08-20T14:00:10Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "thinking must never land"},
                    {"type": "text", "text": "Reading the formatter first."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "Read",
                        "input": {"file_path": "src/app/formatting.py"},
                    },
                ],
            },
        },
        {
            **base,
            "type": "user",
            "timestamp": "2026-08-20T14:00:12Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "is_error": False,
                        "content": "result body must never land",
                    }
                ],
            },
        },
        {
            **base,
            "type": "assistant",
            "timestamp": "2026-08-20T14:00:40Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_2",
                        "name": "Bash",
                        "input": {"command": "pytest -q  # arg must never land"},
                    }
                ],
            },
        },
        {
            **base,
            "type": "user",
            "timestamp": "2026-08-20T14:00:55Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_2",
                        "is_error": True,
                        "content": "failure output must never land",
                    }
                ],
            },
        },
        {
            **base,
            "type": "assistant",
            "isSidechain": True,
            "timestamp": "2026-08-20T14:01:00Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "sidechain must never land"}],
            },
        },
        {
            **base,
            "type": "user",
            "isMeta": True,
            "timestamp": "2026-08-20T14:01:01Z",
            "message": {"role": "user", "content": "meta must never land"},
        },
    ]
    for index, prompt in enumerate(prompts or []):
        lines.append(
            {
                **base,
                "type": "user",
                "timestamp": f"2026-08-20T14:1{index}:00Z",
                "message": {"role": "user", "content": prompt},
            }
        )
    return lines


def _codex_lines() -> list[dict[str, Any]]:
    return [
        {
            "timestamp": "2026-08-21T09:00:00Z",
            "type": "session_meta",
            "payload": {
                "session_id": _SID_CODEX,
                "cwd": "/work/app",
                "git": {"branch": "feature/etl"},
            },
        },
        {
            "timestamp": "2026-08-21T09:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": "instructions must never land"}
                ],
            },
        },
        {
            "timestamp": "2026-08-21T09:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "<environment_context>env must never land</environment_context>",
                    }
                ],
            },
        },
        {
            "timestamp": "2026-08-21T09:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Profile the ETL job"}],
            },
        },
        {
            "timestamp": "2026-08-21T09:00:20Z",
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "call_id": "c1",
                "name": "shell",
                "status": "completed",
                "input": json.dumps({"command": "python -m cProfile etl.py"}),
            },
        },
        {
            "timestamp": "2026-08-21T09:00:50Z",
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "call_id": "c1",
                "output": "profile exhaust must never land " * 100,
            },
        },
        {
            "timestamp": "2026-08-21T09:01:10Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "The insert loop dominates."}
                ],
            },
        },
    ]


def _write_jsonl(path: Path, lines: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\nnot json\n",
        encoding="utf-8",
    )
    return path


def test_claude_transcript_reduces_to_exchanges(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "session.jsonl", _claude_lines())
    session = parse_transcript(path)
    assert session is not None
    assert session.harness == "claude-code"
    assert session.session_id == _SID_CLAUDE
    assert session.project_path == "/work/app"
    assert session.git_branch == "main"

    document = build_document(session)
    assert document is not None
    assert document.exchange_count == 1
    exchange = document.exchanges[0]
    assert exchange.heading == "[0] Fix the rounding test"
    assert exchange.tool_calls == 2
    assert exchange.tool_errors == 1
    assert exchange.tools_used == ("Bash", "Read")
    assert exchange.files_touched == ("src/app/formatting.py",)
    # The rendered document opens with the first exchange heading, so the
    # first chunk of every session attributes to exchange 0.
    assert document.text.startswith("## [0] ")
    # Reduction is the contract: no thinking, tool bodies, tool argument
    # values, sidechain, or meta content may reach the landing document.
    assert "must never land" not in document.model_dump_json()


def test_repeated_prompts_get_unique_headings(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "session.jsonl", _claude_lines(prompts=["continue", "continue"])
    )
    session = parse_transcript(path)
    assert session is not None
    document = build_document(session)
    assert document is not None
    headings = [exchange.heading for exchange in document.exchanges]
    assert headings[1:] == ["[1] continue", "[2] continue"]
    assert len(set(headings)) == len(headings)


def test_long_prompt_is_truncated_with_marker(tmp_path: Path) -> None:
    prompt = "paste " * 2000
    path = _write_jsonl(tmp_path / "session.jsonl", _claude_lines(prompts=[prompt]))
    session = parse_transcript(path)
    assert session is not None
    document = build_document(session)
    assert document is not None
    exchange = document.exchanges[1]
    assert exchange.prompt_truncated
    assert exchange.prompt_chars == len(prompt.strip())
    assert "chars truncated]" in document.text


def test_codex_transcript_reduces_to_exchanges(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "rollout-x.jsonl", _codex_lines())
    session = parse_transcript(path)
    assert session is not None
    assert session.harness == "codex"
    assert session.session_id == _SID_CODEX
    assert session.git_branch == "feature/etl"

    document = build_document(session)
    assert document is not None
    assert document.exchange_count == 1
    exchange = document.exchanges[0]
    assert exchange.heading == "[0] Profile the ETL job"
    assert exchange.tool_calls == 1
    assert exchange.tools_used == ("shell",)
    assert "must never land" not in document.model_dump_json()
    # The dropped exhaust is still accounted for by byte count.
    assert "3200B" in document.text


def test_detect_harness_and_unrecognized_files(tmp_path: Path) -> None:
    claude = _write_jsonl(tmp_path / "a.jsonl", _claude_lines())
    codex = _write_jsonl(tmp_path / "b.jsonl", _codex_lines())
    other = tmp_path / "c.jsonl"
    other.write_text('{"rows": [1, 2]}\n', encoding="utf-8")
    assert detect_harness(claude) == "claude-code"
    assert detect_harness(codex) == "codex"
    assert detect_harness(other) is None
    assert convert_file(other, tmp_path / "out") is None


def test_landing_write_is_stable_and_atomic(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "session.jsonl", _claude_lines())
    out = tmp_path / "landing"
    first = convert_file(path, out)
    second = convert_file(path, out)
    assert first is not None
    assert first == second
    assert first.name == f"claude-code-{_SID_CLAUDE}.json"
    assert not list(out.glob("*.tmp"))
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "transcript/v1"


def test_sync_skips_the_live_session(tmp_path: Path) -> None:
    claude_dir = tmp_path / "projects"
    settled = _write_jsonl(claude_dir / "proj-a" / "old.jsonl", _claude_lines())
    _write_jsonl(claude_dir / "proj-a" / "live.jsonl", _claude_lines())
    old = time.time() - 3600
    os.utime(settled, (old, old))

    out = tmp_path / "landing"
    written = sync_transcripts(
        out_dir=out,
        claude_dir=claude_dir,
        codex_dir=tmp_path / "no-such-dir",
        min_idle_seconds=600,
    )
    # Both files hold the same session; only the settled one was converted.
    assert len(written) == 1
    assert len(list(out.glob("*.json"))) == 1


def test_cli_convert_writes_and_rejects(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "session.jsonl", _claude_lines())
    out = tmp_path / "landing"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["transcripts", "convert", str(path), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert f"claude-code-{_SID_CLAUDE}.json" in result.output

    empty = tmp_path / "empty.jsonl"
    empty.write_text('{"rows": 1}\n', encoding="utf-8")
    result = runner.invoke(
        cli, ["transcripts", "convert", str(empty), "--out", str(out)]
    )
    assert result.exit_code != 0
    assert "Not a recognized agent transcript" in result.output


def test_conversationless_transcript_is_rejected(tmp_path: Path) -> None:
    # Auxiliary records only — parseable as claude-code, but no user turn.
    lines = [
        {"type": "queue-operation", "sessionId": _SID_CLAUDE},
        {
            "type": "assistant",
            "sessionId": _SID_CLAUDE,
            "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        },
    ]
    path = _write_jsonl(tmp_path / "session.jsonl", lines)
    assert parse_transcript(path) is None


@pytest.mark.parametrize("harness_lines", [_claude_lines, _codex_lines])
def test_documents_validate_against_the_contract(
    tmp_path: Path, harness_lines: Any
) -> None:
    from stel.transcripts.contract import TranscriptDocument

    path = _write_jsonl(tmp_path / "t.jsonl", harness_lines())
    landed = convert_file(path, tmp_path / "out")
    assert landed is not None
    payload = json.loads(landed.read_text(encoding="utf-8"))
    TranscriptDocument.model_validate(payload)
