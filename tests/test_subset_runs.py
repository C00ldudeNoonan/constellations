"""Subset runs and --read-filter past extraction (issue #417).

`--source-filter` promises extraction that a deliberately narrowed run is
additive: absence from the slice is not removal. That promise stopped one
model kind into the pipeline — chunk, transform, embed, and search_index all
computed "removed = state - what this run saw" ungated, so a partitioned or
sliced invocation would have silently deleted every other partition's rows.

These tests run the real pipeline against DuckDB. The pattern throughout:
narrow a run, prove the rows outside the slice survive with state intact,
then prove the next unfiltered run still reconciles — gating deletion must
defer it, never lose it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from stel.execution.contracts import RunError
from stel.runner import run_project

DOCUMENTS = 5


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "stel_project.yml").write_text(
        "name: subset\nversion: '0.1.0'\nprofile: subset\n", encoding="utf-8"
    )
    (project / "profiles.yml").write_text(
        "subset:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
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


def _mutate(project: Path, sql: str) -> None:
    connection = duckdb.connect(str(project / "target" / "db.duckdb"))
    try:
        connection.execute(sql)
    finally:
        connection.close()


def _embedded_titles(project: Path) -> set[str]:
    return {
        str(row[0])
        for row in _query(
            project, 'SELECT DISTINCT title FROM "db".docs.document_embeddings'
        )
    }


# ─── --read-filter narrows the embed read, additively ───────────────────────


def test_read_filter_embeds_only_the_slice_and_deletes_nothing(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    run_project(project)
    assert len(_embedded_titles(project)) == DOCUMENTS
    # Change one document inside the slice and one outside it.
    _mutate(
        project,
        "UPDATE \"db\".docs.document_chunks SET text = 'revised-' || chunk_id "
        "WHERE title IN ('Release 1', 'Release 3')",
    )

    [result] = run_project(
        project,
        select="document_embeddings",
        read_filter=[("title", "eq", "Release 1")],
    )

    # Only the in-slice change was paid for; the out-of-slice change waits.
    assert result.metrics["provider_calls"] == 1
    assert result.documents_deleted == 0
    # Every other partition's rows survive untouched.
    assert len(_embedded_titles(project)) == DOCUMENTS
    revised = _query(
        project,
        "SELECT title FROM \"db\".docs.document_embeddings "
        "WHERE embedding_input_hash IN (SELECT embedding_input_hash FROM "
        '"db".docs.document_embeddings) AND title = \'Release 3\'',
    )
    assert revised  # still present, still the old vector's row


def test_the_next_unfiltered_run_settles_what_the_slice_deferred(
    tmp_path: Path,
) -> None:
    """Gating deletion defers reconciliation; it must never lose it."""
    project = _project(tmp_path)
    run_project(project)
    _mutate(
        project,
        "DELETE FROM \"db\".docs.document_chunks WHERE title = 'Release 4'",
    )

    [filtered] = run_project(
        project,
        select="document_embeddings",
        read_filter=[("title", "eq", "Release 1")],
    )
    assert filtered.documents_deleted == 0
    assert "Release 4" in _embedded_titles(project)

    [unfiltered] = run_project(project, select="document_embeddings")
    assert unfiltered.documents_deleted == 1
    assert "Release 4" not in _embedded_titles(project)


def test_in_operator_takes_a_json_array(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)
    _mutate(
        project,
        "UPDATE \"db\".docs.document_chunks SET text = 'revised-' || chunk_id",
    )

    [result] = run_project(
        project,
        select="document_embeddings",
        read_filter=[("title", "in", '["Release 0", "Release 2"]')],
    )

    assert result.metrics["provider_calls"] == 2
    assert result.documents_deleted == 0


# ─── chunk models under a subset invocation ─────────────────────────────────


def test_chunk_deletion_is_gated_by_the_subset_invocation(tmp_path: Path) -> None:
    """The chunk model's read is not narrowed by --read-filter, but the
    invocation is a subset run, so its reconciliation must defer exactly as
    extraction's does under --source-filter."""
    project = _project(tmp_path)
    run_project(project)
    _mutate(
        project,
        "DELETE FROM \"db\".docs.document_registry WHERE title = 'Release 2'",
    )

    [filtered] = run_project(
        project,
        select="document_chunks",
        read_filter=[("title", "eq", "Release 1")],
    )
    assert filtered.documents_deleted == 0
    chunk_titles = {
        str(row[0])
        for row in _query(
            project, 'SELECT DISTINCT title FROM "db".docs.document_chunks'
        )
    }
    assert "Release 2" in chunk_titles

    [unfiltered] = run_project(project, select="document_chunks")
    assert unfiltered.documents_deleted == 1


# ─── validation: the flag fails closed ──────────────────────────────────────


def test_read_filter_refuses_full_refresh(tmp_path: Path) -> None:
    """A full refresh rebuilds from what it reads; rebuilding from a slice
    silently truncates the model to the slice."""
    project = _project(tmp_path)

    with pytest.raises(RunError, match="full-refresh"):
        run_project(
            project,
            full_refresh=True,
            read_filter=[("title", "eq", "Release 1")],
        )


def test_read_filter_refuses_a_full_materialization_embed(tmp_path: Path) -> None:
    project = _project(tmp_path)
    model_path = project / "models" / "documents.yml"
    text = model_path.read_text(encoding="utf-8")
    model_path.write_text(
        text[: text.rindex("    materialization: incremental\n")]
        + "    materialization: full\n",
        encoding="utf-8",
    )

    with pytest.raises(RunError, match="incremental"):
        run_project(
            project,
            select="document_embeddings",
            read_filter=[("title", "eq", "Release 1")],
        )


def test_an_unknown_operator_is_a_run_error(tmp_path: Path) -> None:
    project = _project(tmp_path)

    with pytest.raises(RunError, match="operator"):
        run_project(
            project,
            select="document_embeddings",
            read_filter=[("title", "matches", "x")],
        )


def test_a_missing_filter_column_names_the_available_ones(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")

    with pytest.raises(RunError, match="no_such_column"):
        run_project(
            project,
            select="document_embeddings",
            read_filter=[("no_such_column", "eq", "x")],
        )
