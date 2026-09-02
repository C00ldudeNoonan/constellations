"""End-to-end proof that examples/agent_transcripts (issue #360) turns
transcript/v1 landing documents into an exchange-attributed, governed search
index using only shipped primitives."""
from __future__ import annotations

import json
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


# ─── the eval: half of #329 phase 3 (issue #456) ────────────────────────────


def test_correction_inputs_carry_prose_and_the_ids_it_could_correct(
    tmp_path: Path,
) -> None:
    """The chain's first step, and the one that decides what is even eligible.

    A correction is only ground truth if it attaches to a record, and the
    context ids an exchange retrieved are the only record identity a
    transcript names — so an exchange that retrieved nothing is not a
    candidate however clearly it corrects something.
    """
    project = _project(tmp_path)
    run_project(project)

    rows = _rows(
        project,
        "SELECT exchange_text, candidate_context_ids, id_space "
        'FROM "transcripts".transcripts.correction_inputs',
    )

    assert rows, "no exchange in the corpus retrieved context with prose"
    for text, ids, id_space in rows:
        assert str(text).strip(), "an input row with no prose has nothing to classify"
        assert json.loads(str(ids)), "an input row with no id has nothing to label"
        assert id_space == "context_id"


def test_the_label_chain_reaches_the_eval_expected_shape(tmp_path: Path) -> None:
    """The acceptance criterion: candidates land in the shape `eval.expected`
    reads — a key column and a label column — carrying the provenance a
    reviewer asks for first.

    The label itself is the offline provider's stub, so this proves the chain
    and the contract, not the classifier's judgement. What counts as a
    correction is prompt-shaped and only a real provider exercises it.
    """
    project = _project(tmp_path)
    run_project(project)

    rows = _rows(
        project,
        "SELECT context_id, expected_label, session_id, exchange_ordinal, id_space "
        'FROM "transcripts".transcripts.classification_label_candidates',
    )

    assert rows
    for context_id, label, session_id, ordinal, id_space in rows:
        assert context_id, "a candidate with no key cannot join to predictions"
        assert str(label).strip(), "an empty label is not a candidate"
        # Provenance survives to the row a human would promote from.
        assert session_id
        assert isinstance(ordinal, int)
        assert id_space == "context_id"


def test_no_eval_reads_the_candidates(tmp_path: Path) -> None:
    """#329 rule 2, asserted rather than trusted to the naming: the candidate
    relation is not wired into any `eval:` or `retrieval_tests:` block, so
    nothing promotes itself."""
    import yaml

    project = _project(tmp_path)
    declared: list[str] = []
    consumers: list[str] = []
    for path in (project / "models").glob("*.yml"):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for model in document.get("models", []):
            declared.append(str(model.get("name")))
            # Block keys, not the words: the example discusses `eval:` in its
            # own comments, and a substring search fails on those.
            consumers.extend(
                key for key in ("eval", "retrieval_tests") if key in model
            )

    assert "classification_label_candidates" in declared
    assert consumers == [], (
        f"{consumers} declared in the example; candidates must never be read "
        "by an eval"
    )
