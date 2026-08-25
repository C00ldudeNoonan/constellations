"""End-to-end proof that examples/agent_transcripts (issue #360) turns
transcript/v1 landing documents into an exchange-attributed, governed search
index using only shipped primitives."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import duckdb

from stel.checks import run_project_tests
from stel.runner import run_project


def _project(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "examples" / "agent_transcripts"
    dst = tmp_path / "agent_transcripts"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("target", "__pycache__"))
    return dst


def _rows(project: Path, sql: str) -> list[tuple[Any, ...]]:
    con = duckdb.connect(
        str(project / "target" / "transcripts.duckdb"), read_only=True
    )
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_transcripts_example_all_checks_pass(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)
    results = run_project_tests(project)
    failed = [r for r in results if not r.passed]
    assert failed == [], f"unexpected failures: {[(r.test_name, r.message) for r in failed]}"


def test_transcripts_example_chunks_attribute_to_exchanges(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)

    rows = _rows(
        project,
        'SELECT harness, exchange_heading, tools_used, files_touched '
        'FROM "transcripts".transcripts.document_chunks',
    )
    assert {harness for harness, *_ in rows} == {"claude-code", "codex"}
    # Every chunk in this corpus attributes to an exchange, and the heading
    # carries the ordinal prefix that keyed the join.
    for _harness, heading, tools_used, files_touched in rows:
        assert heading is not None and heading.startswith("[")
        assert tools_used is not None
        assert files_touched is not None
    # The exchange attributes carry real reduction output, not placeholders.
    assert any("Bash" in tools for _h, _s, tools, _f in rows)
    assert any("formatting.py" in files for _h, _s, _t, files in rows)


def test_transcripts_example_reruns_incrementally(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)
    results = run_project(project)

    chunks = next(r for r in results if r.model_name == "document_chunks")
    # Unchanged landing corpus: the keyed-reference incremental contract
    # (issue #364) skips every session on the second run.
    assert chunks.documents_processed == 0
    assert chunks.documents_skipped == 2


def test_transcripts_example_derives_candidate_judgments(tmp_path: Path) -> None:
    """The corpus's own MCP calls become #329 phase 3 candidates (issue #380):
    a cited id, the id that came back beside it, and a query that matched
    nothing — the three things a reviewer needs to tell apart."""
    project = _project(tmp_path)
    run_project(project)

    rows = _rows(
        project,
        "SELECT harness, judgment, context_id, id_space, query_fingerprint "
        'FROM "transcripts".transcripts.retrieval_judgment_candidates',
    )
    judgments = sorted(judgment for _h, judgment, *_rest in rows)
    assert judgments == ["cited", "returned_not_cited", "zero_result"]

    for _harness, judgment, context_id, id_space, fingerprint in rows:
        assert fingerprint, "a candidate with no fingerprint cannot be promoted"
        # Constraint 3 of #380: promotion must reconcile this against the
        # target index's id_field rather than assume, so it is recorded.
        assert id_space == "context_id"
        assert (context_id is None) == (judgment == "zero_result")
