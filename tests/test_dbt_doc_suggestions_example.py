"""examples/dbt_doc_suggestions: the analysis half of `stel suggest` (#361).

#381 shipped the patching half and nothing produced a candidate row, so the
loop did not close. These tests run it end to end — transcript corpus, gap
detection, drafted description, reviewable diff — because the claim being
made is that the whole chain works, and every intermediate assertion in the
world does not establish that.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import duckdb
import pytest

from stel.cli_services.suggest import suggest_dbt
from stel.runner import run_project

CANDIDATES = "suggestions.dbt_doc_candidates"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build the corpus and the analysis once; both are real project runs."""
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path_factory.mktemp("suggest")
    ignore = shutil.ignore_patterns("target", "__pycache__")
    transcripts = root / "agent_transcripts"
    analysis = root / "dbt_doc_suggestions"
    shutil.copytree(repo / "examples" / "agent_transcripts", transcripts, ignore=ignore)
    shutil.copytree(
        repo / "examples" / "dbt_doc_suggestions", analysis, ignore=ignore
    )
    # The demo sessions ship with the analysis example and join the corpus,
    # exactly as the README instructs a reader to do.
    for fixture in (analysis / "fixtures" / "landing").glob("*.json"):
        shutil.copy(fixture, transcripts / "fixtures" / "landing" / fixture.name)
    run_project(transcripts)
    run_project(analysis)
    return {"transcripts": transcripts, "analysis": analysis}


def _rows(built: dict[str, Path], sql: str) -> list[tuple[Any, ...]]:
    database = built["transcripts"] / "target" / "transcripts.duckdb"
    connection = duckdb.connect(str(database), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


# ─── the loop closes ────────────────────────────────────────────────────────


def test_the_corpus_yields_a_candidate_with_its_provenance(
    built: dict[str, Path],
) -> None:
    rows = _rows(
        built,
        "SELECT dbt_model, dbt_column, evidence_count, evidence_sessions "
        f"FROM {CANDIDATES}",
    )

    assert len(rows) == 1
    model, column, count, sessions = rows[0]
    assert model == "fct_orders"
    # files_touched names files, not columns, so model-level only.
    assert column is None
    assert count == 3
    # Provenance is not a count alone: a reviewer must be able to go and read
    # the sessions the suggestion came from.
    assert len(str(sessions).split(",")) == 3


def test_a_real_diff_reaches_the_dbt_project(built: dict[str, Path]) -> None:
    """The end the issue actually cares about: a patch a human can read."""
    analysis = built["analysis"]

    diff, outcomes = suggest_dbt(
        analysis,
        profiles_dir=None,
        target=None,
        relation=CANDIDATES,
        dbt_project_dir=analysis / "dbt_project",
        min_evidence=3,
        write=False,
    )

    assert outcomes
    assert "--- a/models/marts/schema.yml" in diff
    assert "+    description:" in diff
    assert "fct_orders" in diff


def test_nothing_is_written_without_write(built: dict[str, Path]) -> None:
    """`stel suggest` proposes; it does not commit, and without --write it
    does not even touch the file."""
    analysis = built["analysis"]
    schema = analysis / "dbt_project" / "models" / "marts" / "schema.yml"
    before = schema.read_text(encoding="utf-8")

    suggest_dbt(
        analysis,
        profiles_dir=None,
        target=None,
        relation=CANDIDATES,
        dbt_project_dir=analysis / "dbt_project",
        min_evidence=3,
        write=False,
    )

    assert schema.read_text(encoding="utf-8") == before


def test_an_existing_description_is_never_overwritten(
    built: dict[str, Path],
) -> None:
    """`dim_customers` is documented; a suggestion must not touch it. This is
    the patching half's refusal, asserted from the analysis side because that
    is where a bad candidate would come from."""
    analysis = built["analysis"]

    diff, _ = suggest_dbt(
        analysis,
        profiles_dir=None,
        target=None,
        relation=CANDIDATES,
        dbt_project_dir=analysis / "dbt_project",
        min_evidence=3,
        write=False,
    )

    assert "dim_customers" not in diff


# ─── the rules the design binds every suggestion to ─────────────────────────


def test_the_threshold_is_applied_in_the_analysis_not_the_cli(
    built: dict[str, Path],
) -> None:
    """`--min-evidence` is a second gate. If the first gate were the CLI's,
    the analysis would draft — and pay for — a description for every
    one-session file it ever saw."""
    gaps = _rows(
        built, "SELECT dbt_model, evidence_count FROM suggestions.documentation_gaps"
    )

    assert gaps == [("fct_orders", 3)]
    # Files touched in only one session exist in the corpus and are absent
    # here: the analysis dropped them before any provider call.
    touched = _rows(
        built,
        "SELECT count(*) FROM suggestions.exchange_rows "
        "WHERE files_touched LIKE '%formatting.py%'",
    )
    assert touched[0][0] >= 1


def test_transcript_body_text_never_reaches_the_analysis(
    built: dict[str, Path],
) -> None:
    """Sensitivity travels (#361 rule 4): a proposed description must not be
    able to quote free text a session opted out of capturing. The exchange
    body is not extracted at all, so it cannot reach a prompt by accident."""
    columns = {
        str(row[0])
        for row in _rows(
            built,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'exchange_rows'",
        )
    }

    assert "text" not in columns
    assert "exchange_heading" in columns


def test_evidence_prompts_are_headings_not_bodies(built: dict[str, Path]) -> None:
    prompts = _rows(
        built, "SELECT evidence_prompts FROM suggestions.documentation_gaps"
    )

    assert len(prompts) == 1
    text = str(prompts[0][0])
    # Every line is a heading the human wrote, not assistant prose.
    assert all(line.startswith("[") for line in text.splitlines() if line.strip())
    assert "fct_orders" in text
