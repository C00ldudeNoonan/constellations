"""Capturing stel's own MCP calls inside a transcript (issue #380).

The reduction in #360 drops every tool result body. This is the one narrow
exception: a `search_context` call against stel's own server carries the
retrieval judgment the feedback loop exists to learn from, so its ids and
query fingerprint survive — and nothing else does.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stel.append_log import query_fingerprint
from stel.transcripts import build_document, parse_transcript

_SID = "0f5a2c1e-1111-4aaa-8bbb-000000000001"
_CTX_A = "a" * 32
_CTX_B = "b" * 32
_CHUNK_A = "c" * 32
_QUERY = "how does incremental state advance"

# The tool name a Claude Code client gives stel's server; the `stel-context`
# half is operator-chosen in client config, which is exactly why recognition
# must not key on it.
_TOOL = "mcp__stel-context__search_context"


def _context_response(*context_ids: str) -> str:
    return json.dumps(
        {
            "schema_version": "mcp_context/v1",
            "results": [
                {
                    "rank": rank,
                    "score": 0.9 - rank / 10,
                    "document_id": "d" * 32,
                    "document_version_id": "e" * 32,
                    "context_id": context_id,
                    "chunk_id": _CHUNK_A,
                    "snippet": "chunk text that must never land",
                }
                for rank, context_id in enumerate(context_ids)
            ],
        }
    )


def _session(
    *,
    result: Any,
    tool_name: str = _TOOL,
    prose_before: str | None = None,
    prose_after: str | None = None,
) -> list[dict[str, Any]]:
    base = {"sessionId": _SID, "cwd": "/work/app", "gitBranch": "main"}
    assistant_blocks: list[dict[str, Any]] = []
    if prose_before is not None:
        assistant_blocks.append({"type": "text", "text": prose_before})
    assistant_blocks.append(
        {
            "type": "tool_use",
            "id": "tu_1",
            "name": tool_name,
            "input": {"model": "ctx", "query": _QUERY},
        }
    )
    lines: list[dict[str, Any]] = [
        {
            **base,
            "type": "user",
            "timestamp": "2026-08-20T14:00:00Z",
            "message": {"role": "user", "content": "How does incremental state work?"},
        },
        {
            **base,
            "type": "assistant",
            "timestamp": "2026-08-20T14:00:05Z",
            "message": {"role": "assistant", "content": assistant_blocks},
        },
        {
            **base,
            "type": "user",
            "timestamp": "2026-08-20T14:00:06Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "is_error": False,
                        "content": result,
                    }
                ],
            },
        },
    ]
    if prose_after is not None:
        lines.append(
            {
                **base,
                "type": "assistant",
                "timestamp": "2026-08-20T14:00:20Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": prose_after}],
                },
            }
        )
    return lines


def _write(tmp_path: Path, lines: list[dict[str, Any]]) -> Path:
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    return path


def _only_call(tmp_path: Path, lines: list[dict[str, Any]], **kwargs: Any) -> Any:
    session = parse_transcript(_write(tmp_path, lines), **kwargs)
    assert session is not None
    document = build_document(session)
    assert document is not None
    calls = document.exchanges[0].context_calls
    assert len(calls) == 1
    return calls[0], document


def test_search_context_call_is_captured_without_its_body(tmp_path: Path) -> None:
    call, document = _only_call(
        tmp_path,
        _session(
            result=_context_response(_CTX_A, _CTX_B),
            prose_after=f"The answer is in {_CTX_A}.",
        ),
    )

    assert call.model == "ctx"
    assert call.returned_context_ids == (_CTX_A, _CTX_B)
    assert call.returned_chunk_ids == (_CHUNK_A, _CHUNK_A)
    assert call.zero_results is False
    assert call.error_code is None
    # The snippet bodies that carried those ids are still dropped entirely.
    assert "must never land" not in document.model_dump_json()


def test_query_fingerprint_joins_to_the_mcp_query_log(tmp_path: Path) -> None:
    """The whole point of the fingerprint: a transcript row and a served-side
    query-log row for the same question must carry the same value."""
    call, _ = _only_call(tmp_path, _session(result=_context_response(_CTX_A)))

    assert call.query_fingerprint == query_fingerprint(_QUERY)
    # Same question, different typography — the log normalizes, so this must
    # too, or the join silently misses.
    assert call.query_fingerprint == query_fingerprint(
        "  How Does Incremental\nState Advance "
    )


def test_query_text_is_withheld_unless_opted_in(tmp_path: Path) -> None:
    lines = _session(result=_context_response(_CTX_A))
    call, document = _only_call(tmp_path, lines)
    assert call.query_text is None
    assert _QUERY not in document.model_dump_json()

    call, document = _only_call(tmp_path, lines, capture_query=True)
    assert call.query_text == _QUERY
    assert call.query_fingerprint == query_fingerprint(_QUERY)


def test_citation_requires_prose_after_the_call(tmp_path: Path) -> None:
    # An id named only *before* the call cannot have been cited from it.
    call, _ = _only_call(
        tmp_path,
        _session(
            result=_context_response(_CTX_A, _CTX_B),
            prose_before=f"I will look for {_CTX_A}.",
        ),
    )
    assert call.cited_context_ids == ()

    call, _ = _only_call(
        tmp_path,
        _session(
            result=_context_response(_CTX_A, _CTX_B),
            prose_before=f"I will look for {_CTX_A}.",
            prose_after=f"Per {_CTX_B}, state advances after publication.",
        ),
    )
    assert call.cited_context_ids == (_CTX_B,)


def test_zero_result_call_is_recorded(tmp_path: Path) -> None:
    call, _ = _only_call(tmp_path, _session(result=_context_response()))
    assert call.zero_results is True
    assert call.error_code is None
    assert call.returned_context_ids == ()
    assert call.cited_context_ids == ()


def test_failed_call_is_not_a_zero_result(tmp_path: Path) -> None:
    """A denied or timed-out search returns no rows because it failed, not
    because the corpus had no match. Recording it as a zero result would
    poison the retrieval-quality signal this feature exists to produce
    (Codex review).
    """
    for code in ("not_found_or_denied", "timeout", "busy"):
        error_response = json.dumps(
            {
                "schema_version": "mcp_context/v1",
                "results": [],
                "error": {"code": code, "message": "m", "retryable": True},
            }
        )
        call, _ = _only_call(tmp_path, _session(result=error_response))
        assert call.error_code == code
        assert call.zero_results is False
        assert call.returned_context_ids == ()


def test_malformed_error_still_marks_the_call_failed(tmp_path: Path) -> None:
    malformed = json.dumps(
        {"schema_version": "mcp_context/v1", "results": [], "error": "boom"}
    )
    call, _ = _only_call(tmp_path, _session(result=malformed))
    assert call.error_code == "unknown"
    assert call.zero_results is False


def test_mcp_text_block_result_shape_is_recognized(tmp_path: Path) -> None:
    call, _ = _only_call(
        tmp_path,
        _session(result=[{"type": "text", "text": _context_response(_CTX_A)}]),
    )
    assert call.returned_context_ids == (_CTX_A,)


def test_unrelated_tools_are_not_context_calls(tmp_path: Path) -> None:
    """Recognition keys on the response contract, not the tool name: a tool
    named like ours but answering something else is not a stel call, and a
    stel-named tool whose payload is unparseable is not one either."""
    not_ours = json.dumps({"results": []})
    other_contract = json.dumps({"schema_version": "something/v9", "results": []})
    for lines in (
        _session(tool_name="mcp__other__search_context", result=not_ours),
        _session(result="plain text output"),
        _session(result=other_contract),
    ):
        session = parse_transcript(_write(tmp_path, lines))
        assert session is not None
        document = build_document(session)
        assert document is not None
        assert document.exchanges[0].context_calls == ()


def test_codex_search_context_call_is_captured(tmp_path: Path) -> None:
    sid = "0199aaaa-2222-7bbb-8ccc-000000000002"
    lines = [
        {
            "timestamp": "2026-08-21T09:00:00Z",
            "type": "session_meta",
            "payload": {"session_id": sid, "cwd": "/work/app"},
        },
        {
            "timestamp": "2026-08-21T09:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "How does state work?"}],
            },
        },
        {
            "timestamp": "2026-08-21T09:00:10Z",
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "call_id": "c1",
                "name": "search_context",
                "status": "completed",
                "input": json.dumps({"model": "ctx", "query": _QUERY}),
            },
        },
        {
            "timestamp": "2026-08-21T09:00:20Z",
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "call_id": "c1",
                "output": _context_response(_CTX_A),
            },
        },
        {
            "timestamp": "2026-08-21T09:00:30Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"See {_CTX_A}."}],
            },
        },
    ]
    path = tmp_path / "rollout-x.jsonl"
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    session = parse_transcript(path)
    assert session is not None
    document = build_document(session)
    assert document is not None
    call = document.exchanges[0].context_calls[0]
    assert call.query_fingerprint == query_fingerprint(_QUERY)
    assert call.returned_context_ids == (_CTX_A,)
    assert call.cited_context_ids == (_CTX_A,)
    assert "must never land" not in document.model_dump_json()


def test_v1_documents_without_context_calls_still_validate() -> None:
    """The field is additive: a v1 landing document stays readable."""
    from stel.transcripts.contract import TranscriptDocument

    payload = {
        "schema_version": "transcript/v1",
        "harness": "claude-code",
        "session_id": _SID,
        "source_path": "s.jsonl",
        "project_path": None,
        "git_branch": None,
        "started_at": None,
        "ended_at": None,
        "exchange_count": 1,
        "tools_used": [],
        "files_touched": [],
        "text": "## [0] hi\n",
        "exchanges": [
            {
                "ordinal": 0,
                "heading": "[0] hi",
                "started_at": None,
                "ended_at": None,
                "prompt_chars": 2,
                "prompt_truncated": False,
                "assistant_chars": 0,
                "tool_calls": 0,
                "tool_errors": 0,
                "tools_used": [],
                "files_touched": [],
            }
        ],
    }
    document = TranscriptDocument.model_validate(payload)
    assert document.exchanges[0].context_calls == ()
