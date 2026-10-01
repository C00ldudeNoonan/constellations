"""The unchanged-scan skip, end to end (issue #573/#611): a model whose
immediate parent has not changed by either sync-watermark signal, and whose
own code is unchanged, skips its full parent scan.

Also pins the two properties a review of #612 found missing from the first
cut: the watermark it writes always reflects content read *before* this
model's own run, never re-read after (a parent mutated mid-run must never be
folded into a watermark claiming this model consumed it); and any failure
reading or writing a watermark falls back to the real scan rather than
failing an otherwise-successful run.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import duckdb
import pytest

from stel.adapters.duckdb import DuckDBAdapter
from stel.runner import run_project

# Runs a whole project, so it belongs to the `e2e` tier (issue #518).
pytestmark = pytest.mark.e2e

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def rag_project(tmp_path: Path) -> Path:
    pytest.importorskip("lancedb")
    dst = tmp_path / "rag_chunks_pipeline"
    shutil.copytree(
        EXAMPLES / "rag_chunks_pipeline",
        dst,
        ignore=shutil.ignore_patterns("target", "__pycache__"),
    )
    return dst


def test_a_second_untouched_run_skips_every_downstream_scan(rag_project: Path) -> None:
    run_project(rag_project)  # first run: real work, nothing to skip yet

    results = {r.model_name: r for r in run_project(rag_project)}

    # document_registry has no stel-model parent -- its own scan (new source
    # documents) is exactly what can never be skipped this way.
    assert results["document_registry"].status != "unchanged"
    # document_chunks and chunk_embeddings each have exactly one immediate
    # parent, and that parent published nothing: both take the fast path.
    assert results["document_chunks"].status == "unchanged"
    assert results["document_chunks"].documents_skipped > 0
    assert results["chunk_embeddings"].status == "unchanged"


def test_a_new_document_still_reaches_every_child(rag_project: Path) -> None:
    run_project(rag_project)
    (rag_project / "documents" / "new.html").write_text(
        "<html><body><p>A brand new document.</p></body></html>", encoding="utf-8"
    )

    results = {r.model_name: r for r in run_project(rag_project)}

    assert results["document_chunks"].status != "unchanged"
    assert results["chunk_embeddings"].status != "unchanged"


# ─── the watermark never reflects content read after the run (#612 review) ──


def test_a_code_change_run_establishes_a_watermark_for_next_time(
    rag_project: Path,
) -> None:
    """A real-work run (here, document_chunks's own `chunk_size` changing)
    still reads the parent's content once, before its own dispatch, and
    writes a fresh watermark from it -- otherwise a model would never
    acquire its first watermark, since the next run would find nothing to
    compare against either and the skip could never engage for it at all."""
    run_project(rag_project)
    model_yml = rag_project / "models" / "document_chunks.yml"
    model_yml.write_text(
        model_yml.read_text(encoding="utf-8").replace("chunk_size: 800", "chunk_size: 400"),
        encoding="utf-8",
    )
    second = {r.model_name: r for r in run_project(rag_project, accept_reprocess=True)}
    assert second["document_chunks"].status != "unchanged"

    third = {r.model_name: r for r in run_project(rag_project)}
    assert third["document_chunks"].status == "unchanged"


def test_a_parent_mutated_during_the_scan_is_not_folded_into_the_watermark(
    rag_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The race a review of #612 found: an earlier cut read the parent's
    content *after* this model's own dispatch to build the watermark it
    wrote, so a parent mutated during that window would be folded into a
    watermark claiming this model had consumed content it never actually
    saw -- and every later run would then wrongly skip forever, since both
    signals would agree with that never-consumed content from then on.

    Simulated by mutating `document_registry` from inside the monkeypatched
    chunk dispatch itself: `parent_content` is captured before this call
    runs at all, so the watermark it records must reflect the *pre*-mutation
    content, not the mutated content this run's own dispatch goes on to see.
    A follow-up run against the now-stable mutated content must still notice
    it does not match that pre-mutation watermark -- an extra, redundant
    scan is an acceptable cost; a silent, permanent skip is not."""
    run_project(rag_project)
    model_yml = rag_project / "models" / "document_chunks.yml"
    model_yml.write_text(
        model_yml.read_text(encoding="utf-8").replace("chunk_size: 800", "chunk_size: 400"),
        encoding="utf-8",
    )

    import stel.runner as runner_module

    original_run_chunk_model = runner_module._run_chunk_model

    def mutate_then_run(*args: Any, **kwargs: Any) -> Any:
        db_path = rag_project / "target" / "stel.duckdb"
        con = duckdb.connect(str(db_path))
        try:
            con.execute("UPDATE rag.document_registry SET text = text || ' mutated'")
        finally:
            con.close()
        return original_run_chunk_model(*args, **kwargs)

    monkeypatch.setattr(runner_module, "_run_chunk_model", mutate_then_run)

    run_project(rag_project, accept_reprocess=True)
    monkeypatch.undo()

    # Nothing mutates further from here. If the watermark had captured the
    # *mutated* content (the bug), this run would wrongly skip; it must
    # instead still notice the mismatch against the pre-mutation watermark.
    third = {r.model_name: r for r in run_project(rag_project)}
    assert third["document_chunks"].status != "unchanged"


# ─── a watermark failure never fails the run (#612 review) ──────────────────


def test_a_watermark_write_failure_does_not_fail_the_run(
    rag_project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Reached here via the direct-mutation case: a raw `UPDATE` on
    `document_registry` that moves content without moving its `stel_state`
    row count or `last_run_at`, so the cheap signal alone matches and the
    content check -- which reads fresh, pre-dispatch -- is what actually
    disqualifies the skip and triggers both the real scan and a fresh
    watermark-write attempt afterward."""
    run_project(rag_project)
    run_project(rag_project)  # establishes a watermark for document_chunks

    db_path = rag_project / "target" / "stel.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute("UPDATE rag.document_registry SET text = text || ' mutated'")
    finally:
        con.close()

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("warehouse said no")

    monkeypatch.setattr(DuckDBAdapter, "write_sync_watermark", fail)

    results = {r.model_name: r for r in run_project(rag_project)}

    assert results["document_chunks"].status != "unchanged"
    assert not results["document_chunks"].errors
    assert "could not record its sync watermark" in caplog.text
    assert "warehouse said no" not in caplog.text


def test_a_watermark_read_failure_falls_back_to_the_real_scan(
    rag_project: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    run_project(rag_project)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("warehouse said no")

    monkeypatch.setattr(DuckDBAdapter, "state_generation", fail)

    results = {r.model_name: r for r in run_project(rag_project)}

    assert results["document_chunks"].status != "unchanged"
    assert not results["document_chunks"].errors
    assert "could not read the parent's generation" in caplog.text
    assert "warehouse said no" not in caplog.text
