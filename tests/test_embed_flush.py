"""Embed models publish at flush boundaries (issue #401).

Embeds were the last all-or-nothing stage, and the worst one to leave that
way: their re-run cost is metered provider spend, not CPU. A 3.6M-chunk corpus
ran 28 hours of paid Vertex calls and lost every one of them to a MemoryError
at the final assembly, because nothing published until the end.

These tests are about *when* rows land, not what they contain. The content
tests live in test_embedding.py and must keep passing unchanged — flush
cadence changes execution, never output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from stel.execution.contracts import RunError
from stel.providers.deterministic import DeterministicEmbeddingProvider
from stel.runner import run_project

DOCUMENTS = 7


def _project(tmp_path: Path, *, flush_every: int) -> Path:
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "stel_project.yml").write_text(
        "name: embeddings\nversion: '0.1.0'\nprofile: embeddings\n",
        encoding="utf-8",
    )
    (project / "profiles.yml").write_text(
        "embeddings:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: docs\n",
        encoding="utf-8",
    )
    (project / "sources").mkdir()
    (project / "sources" / "documents.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data\n"
        "    file_pattern: '*.json'\n",
        encoding="utf-8",
    )
    (project / "models").mkdir()
    (project / "models" / "documents.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: document_registry\n"
        "    source: ref('documents')\n"
        "    extraction:\n      backend: json\n"
        "      options:\n        fields: [title, body]\n"
        "    materialization: incremental\n"
        "  - name: document_chunks\n"
        "    depends_on: [ref('document_registry')]\n"
        "    chunk:\n      text_field: body\n      chunk_size: 1000\n"
        "      chunk_overlap: 0\n"
        "    materialization: incremental\n"
        "  - name: document_embeddings\n"
        "    depends_on: [ref('document_chunks')]\n"
        "    embed:\n      provider: deterministic\n"
        "      model: contract-v1\n      text_field: text\n"
        "      id_field: chunk_id\n      vector_field: embedding\n"
        "      dimensions: 4\n      batch_size: 1\n"
        f"      flush_every: {flush_every}\n"
        "    materialization: incremental\n",
        encoding="utf-8",
    )
    data = project / "data"
    data.mkdir()
    for index in range(DOCUMENTS):
        (data / f"doc{index}.json").write_text(
            json.dumps({"title": f"Release {index}", "body": f"body text {index}"}),
            encoding="utf-8",
        )
    return project


def _query(project: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def _embedded_count(project: Path) -> int:
    return int(
        _query(project, 'SELECT COUNT(*) FROM "db".docs.document_embeddings')[0][0]
    )


def _fail_after(calls: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the provider serve `calls` embeddings, then fail like a real one."""
    original = DeterministicEmbeddingProvider._embed
    served = 0

    def limited(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal served
        if served >= calls:
            raise RuntimeError("provider exhausted")
        served += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DeterministicEmbeddingProvider, "_embed", limited)


# ─── the point of the issue: paid work survives a later failure ─────────────


def test_a_failure_mid_run_keeps_earlier_flushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The incident in #401: 28 hours of paid provider calls thrown away by a
    failure at the end. Rows embedded before the failure must already be in
    the warehouse."""
    project = _project(tmp_path, flush_every=2)
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")
    _fail_after(5, monkeypatch)

    with pytest.raises(RunError):
        run_project(project, select="document_embeddings")

    # Five calls served: flushes of two published twice; the third flush died
    # mid-way and is unpublished.
    assert _embedded_count(project) == 4


def test_the_rerun_only_pays_for_what_was_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """State advances per flush, so a resumed run must not re-embed rows it
    already paid for — that is the whole economic argument of the issue."""
    project = _project(tmp_path, flush_every=2)
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")
    _fail_after(5, monkeypatch)
    with pytest.raises(RunError):
        run_project(project, select="document_embeddings")
    monkeypatch.undo()

    [resumed] = run_project(project, select="document_embeddings")

    # Four already published and state-advanced; three remain.
    assert resumed.metrics["provider_calls"] == DOCUMENTS - 4
    assert _embedded_count(project) == DOCUMENTS


def test_nothing_is_lost_across_the_whole_corpus(tmp_path: Path) -> None:
    project = _project(tmp_path, flush_every=2)

    run_project(project)

    assert _embedded_count(project) == DOCUMENTS
    vectors = _query(
        project, 'SELECT embedding FROM "db".docs.document_embeddings'
    )
    assert all(vector[0] is not None for vector in vectors)


# ─── cadence must not change content ────────────────────────────────────────


def test_flush_size_does_not_change_the_output(tmp_path: Path) -> None:
    """`flush_every` is an execution knob. A corpus embedded in one flush and
    the same corpus embedded in seven must come out identical, or the knob is
    silently a correctness setting."""
    columns = (
        "SELECT chunk_id, embedding, embedding_input_hash, embedding_config_hash "
        'FROM "db".docs.document_embeddings ORDER BY chunk_id'
    )

    def _run(flush_every: int) -> list[tuple[Any, ...]]:
        project = _project(tmp_path / f"f{flush_every}", flush_every=flush_every)
        run_project(project)
        return _query(project, columns)

    one_at_a_time = _run(1)
    two_at_a_time = _run(2)
    all_at_once = _run(1000)

    assert len(one_at_a_time) == DOCUMENTS
    assert one_at_a_time == two_at_a_time == all_at_once


def test_flush_every_does_not_move_code_version(tmp_path: Path) -> None:
    """A changed code_version re-embeds every existing corpus at provider
    prices, silently, because the run still succeeds. Adding this option must
    not do that, and neither must changing it."""
    from stel.config.model import EmbedConfig
    from stel.versioning import compute_code_version

    def _version(flush_every: int) -> str:
        return compute_code_version(
            extraction=None,
            transform=None,
            embed=EmbedConfig(
                provider="deterministic",
                model="contract-v1",
                dimensions=4,
                flush_every=flush_every,
            ),
            project_dir=tmp_path,
        )

    assert _version(1) == _version(5000) == _version(100_000)


# ─── memory is the other half of the incident ───────────────────────────────


def test_publications_are_bounded_by_flush_every(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Peak memory is one flush, not the corpus. Asserted where it is
    observable: no single publication may carry more rows than `flush_every`.

    The old shape held every row, every vector, and a DataFrame over both at
    once; a test that only checked the final row count passed happily while
    doing that.
    """
    from stel.adapters.duckdb import DuckDBAdapter

    heights: list[int] = []
    # Both entry points: on a first run the target does not exist yet, so the
    # opening flush replaces the table and every later one merges into it.
    # Watching only the incremental path would miss the first flush entirely.
    originals = {
        "materialize_full": DuckDBAdapter.materialize_full,
        "materialize_incremental": DuckDBAdapter.materialize_incremental,
    }

    def _spy(name: str) -> Any:
        original = originals[name]

        def spy(self: Any, table: str, frame: Any, **kwargs: Any) -> Any:
            if table == "document_embeddings":
                heights.append(frame.height)
            return original(self, table, frame, **kwargs)

        return spy

    for name in originals:
        monkeypatch.setattr(DuckDBAdapter, name, _spy(name))
    project = _project(tmp_path, flush_every=2)

    run_project(project)

    assert heights, "the embed model never published"
    assert max(heights) <= 2
    assert sum(heights) == DOCUMENTS
