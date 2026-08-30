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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
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


# ─── review follow-ups (PR #402) ────────────────────────────────────────────


def test_a_column_null_in_the_first_flush_does_not_break_a_later_one(
    tmp_path: Path,
) -> None:
    """Every flush frame is built with a schema fixed for the whole run.

    Inferring each frame from its own rows lets a passthrough column that
    happens to be all-NULL in the opening flush create the target column from
    `Null`. The later flush carrying real values then fails on conversion —
    after its provider calls have been paid for, which is precisely the loss
    this issue exists to stop.
    """
    project = _project(tmp_path, flush_every=2)
    # `note` is absent from the first four documents and present afterwards,
    # so with flush_every=2 the first two flushes see nothing but NULL.
    (project / "models" / "documents.yml").write_text(
        (project / "models" / "documents.yml")
        .read_text(encoding="utf-8")
        .replace("fields: [title, body]", "fields: [title, body, note]"),
        encoding="utf-8",
    )
    for index in range(DOCUMENTS):
        payload: dict[str, Any] = {"title": f"Release {index}", "body": f"body {index}"}
        if index >= 4:
            payload["note"] = f"note {index}"
        (project / "data" / f"doc{index}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    run_project(project)

    notes = _query(
        project,
        'SELECT note FROM "db".docs.document_embeddings ORDER BY chunk_id',
    )
    assert len(notes) == DOCUMENTS
    assert sum(1 for note in notes if note[0] is not None) == DOCUMENTS - 4


def test_a_warehouse_failure_is_not_reported_as_a_provider_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication errors and provider errors take different sanitizers.
    Routing a warehouse error through the provider path hands its raw text to
    the fallback, and that text can quote the offending row and the SQL."""
    from stel.adapters.duckdb import DuckDBAdapter

    secret = "row value 123-45-6789 in INSERT INTO"

    def exploding(self: Any, *args: Any, **kwargs: Any) -> Any:
        del self, args, kwargs
        raise ValueError(secret)

    project = _project(tmp_path, flush_every=2)
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")
    monkeypatch.setattr(DuckDBAdapter, "materialize_full", exploding)
    monkeypatch.setattr(DuckDBAdapter, "materialize_incremental", exploding)

    with pytest.raises(RunError) as caught:
        run_project(project, select="document_embeddings")

    assert secret not in str(caught.value)
    assert "provider execution failed" not in str(caught.value)


# ─── the resume path is bounded (issue #401 follow-up) ──────────────────────


def test_resume_never_reads_the_whole_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix for the accumulate-everything run made a second wall reachable:
    on resume, the old code read the entire existing target -- vectors
    included -- into Python dicts before the first provider call. ~25KB per
    768-dim row is ~90GB at 3.6M chunks, on a path only reachable once the
    corpus has already proven itself that large.

    Asserted where it is observable: a resume may read the target only
    through zero-row schema probes and projected snapshot reads, never as a
    full read_table materialization.
    """
    from stel.adapters.duckdb import DuckDBAdapter

    project = _project(tmp_path, flush_every=2)
    run_project(project)  # first run publishes the target

    full_reads: list[str] = []
    original = DuckDBAdapter.read_table

    def spy(self: Any, table: str, *, limit: int | None = None) -> Any:
        if table == "document_embeddings" and limit != 0:
            full_reads.append(table)
        return original(self, table, limit=limit)

    monkeypatch.setattr(DuckDBAdapter, "read_table", spy)
    # Touch one chunk's text so the resume has real work to do.
    connection = duckdb.connect(str(project / "target" / "db.duckdb"))
    try:
        connection.execute(
            "UPDATE \"db\".docs.document_chunks SET text = 'revised' "
            "WHERE chunk_id = (SELECT min(chunk_id) FROM \"db\".docs.document_chunks)"
        )
    finally:
        connection.close()

    run_project(project, select="document_embeddings")

    assert full_reads == []


def test_a_fresh_run_never_reads_the_whole_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The input half of the same wall (issue #410). #401 streamed the output
    and #407 bounded the resume lookup, but a fresh run still did
    `read_table(upstream)` -- one `SELECT *` into one frame -- before anything
    else happened. On the 3.6M-chunk corpus memory climbed ~1.2GiB/min for six
    minutes with zero flushes committed and zero provider calls made: a 10GiB
    container would have been OOM-killed at read time, before any of the flush
    machinery engaged.

    Asserted the same way the resume guard is: the upstream may be touched
    only by a zero-row schema probe and by streamed snapshot reads, never as a
    full read_table materialization.
    """
    from stel.adapters.duckdb import DuckDBAdapter

    project = _project(tmp_path, flush_every=2)
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")

    full_reads: list[str] = []
    original = DuckDBAdapter.read_table

    def spy(self: Any, table: str, *, limit: int | None = None) -> Any:
        if table == "document_chunks" and limit != 0:
            full_reads.append(table)
        return original(self, table, limit=limit)

    monkeypatch.setattr(DuckDBAdapter, "read_table", spy)
    run_project(project, select="document_embeddings")  # fresh: no prior target

    assert full_reads == []
    assert _embedded_count(project) == DOCUMENTS


def test_streaming_the_input_preserves_input_fingerprints(tmp_path: Path) -> None:
    """The upgrade hazard #410 has to clear before it is worth shipping.

    `input_fingerprint` is a hash of the *whole* upstream record, and
    incremental state compares against it. If reading the upstream in Arrow
    batches produced even subtly different Python values than reading it whole
    -- a dtype widened, a datetime unit shifted -- every existing embed model
    would silently re-embed its entire corpus on upgrade. That failure is
    metered provider spend and it raises nothing, so it is pinned here rather
    than left to the end-to-end tests, which only ever exercise one path.
    """
    from stel.adapters.duckdb import DuckDBAdapter, DuckDBWarehouseConfig
    from stel.hashing import canonical_fingerprint

    project = _project(tmp_path, flush_every=2)
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")

    config = DuckDBWarehouseConfig(
        path=project / "target" / "db.duckdb", schema_name="docs"
    )
    with DuckDBAdapter(config) as adapter:
        whole = adapter.read_table("document_chunks")
        # The dtypes the output schema is now built from must survive a
        # zero-row read, or a passthrough column lands in the target as the
        # wrong type.
        assert dict(adapter.read_table("document_chunks", limit=0).schema) == dict(
            whole.schema
        )

        def _fingerprint(record: dict[str, Any]) -> str:
            return canonical_fingerprint(
                record, domain="embedding-input-row", version=1
            )

        read_whole = {
            str(row["chunk_id"]): _fingerprint(row)
            for row in whole.iter_rows(named=True)
        }
        streamed: dict[str, str] = {}
        with adapter.table_snapshot("document_chunks", batch_size=2) as snapshot:
            for batch in snapshot:
                frame = pl.from_arrow(batch)
                assert isinstance(frame, pl.DataFrame)
                for row in frame.iter_rows(named=True):
                    streamed[str(row["chunk_id"])] = _fingerprint(row)

    assert read_whole  # the corpus is not empty, or this proves nothing
    assert streamed == read_whole


def test_a_duplicate_upstream_id_still_fails_before_any_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Id validation moved into a streamed pass; it must stay *ahead* of the
    embedding loop. Folding it into the loop would turn a contract violation
    in the last batch into a failure the operator pays for the whole corpus to
    discover."""
    project = _project(tmp_path, flush_every=2)
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")

    connection = duckdb.connect(str(project / "target" / "db.duckdb"))
    try:
        connection.execute(
            'INSERT INTO "db".docs.document_chunks '
            'SELECT * FROM "db".docs.document_chunks LIMIT 1'
        )
    finally:
        connection.close()

    calls = 0
    original = DeterministicEmbeddingProvider._embed

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DeterministicEmbeddingProvider, "_embed", counted)

    with pytest.raises(RunError, match="duplicate"):
        run_project(project, select="document_embeddings")
    assert calls == 0


def test_resume_still_reuses_vectors_for_metadata_only_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole-target load existed to serve vector reuse. The bounded
    per-window lookup must keep serving it, or every metadata-only change
    becomes a paid re-embedding -- correct output, silent cost."""
    project = _project(tmp_path, flush_every=2)
    run_project(project)

    # A run over an unchanged corpus: everything skips via state, no provider
    # calls, no reuse needed. Then force fingerprints to move without moving
    # text, which is exactly the reuse case: state misses, text hash matches.
    calls = {"n": 0}
    original = DeterministicEmbeddingProvider._embed

    def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DeterministicEmbeddingProvider, "_embed", counting)
    connection = duckdb.connect(str(project / "target" / "db.duckdb"))
    try:
        # Mutate an existing passthrough column so every fingerprint moves
        # while every text hash stays put -- the exact reuse case.
        connection.execute(
            "UPDATE \"db\".docs.document_chunks SET title = title || ' (reclassified)'"
        )
    finally:
        connection.close()

    [result] = run_project(project, select="document_embeddings")

    # Every row was re-fingerprinted, none re-embedded.
    assert result.documents_processed == DOCUMENTS
    assert calls["n"] == 0
    assert result.metrics["cache_hits"] == DOCUMENTS


# ─── the run budget can finally see embed spend ─────────────────────────────


def test_a_budget_stop_is_graceful_and_resumable(tmp_path: Path) -> None:
    """Embeds were the only provider-spending stage the run budget could not
    gate: --max-cost stopped extraction and llm calls while a Vertex embed
    run spent freely. And a cap is only worth having if hitting it behaves
    like a crash at the same point -- published windows stay, state covers
    exactly them, and raising the cap resumes for the remainder.
    """
    project = _project(tmp_path, flush_every=2)
    profiles = project / "profiles.yml"
    profiles.write_text(
        profiles.read_text(encoding="utf-8")
        + "      llm:\n"
        "        provider: deterministic\n"
        "        model: deterministic-v1\n"
        "        budget:\n"
        "          max_api_calls: 4\n",
        encoding="utf-8",
    )
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")

    [capped] = run_project(project, select="document_embeddings")

    assert capped.status == "budget_exceeded"
    assert any("BudgetExceededError" in error for error in capped.errors)
    # Two windows of two landed before the cap; nothing after it mutated.
    assert _embedded_count(project) == 4

    profiles.write_text(
        profiles.read_text(encoding="utf-8").replace(
            "max_api_calls: 4", "max_api_calls: 100"
        ),
        encoding="utf-8",
    )
    [resumed] = run_project(project, select="document_embeddings")

    assert resumed.status is None
    assert resumed.metrics["provider_calls"] == DOCUMENTS - 4
    assert _embedded_count(project) == DOCUMENTS


# ─── review follow-ups (PR #407) ────────────────────────────────────────────


def test_headroom_reserves_every_split_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider may fan one logical batch into many billed requests --
    Vertex issues one per text for gemini-embedding models. Reserving one
    slot per batch would let max_api_calls: 1 admit all of them before the
    charge lands; the reservation must match what the provider says it will
    bill."""
    monkeypatch.setattr(
        DeterministicEmbeddingProvider,
        "estimate_provider_requests",
        lambda self, request: len(request.texts),
    )
    project = _project(tmp_path, flush_every=7)
    # batch_size 7 in one window; a fan-out-aware reservation of 7 must trip
    # a cap of 4 BEFORE the first call, so nothing is billed at all.
    model_path = project / "models" / "documents.yml"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace(
            "batch_size: 1", "batch_size: 7"
        ),
        encoding="utf-8",
    )
    profiles = project / "profiles.yml"
    profiles.write_text(
        profiles.read_text(encoding="utf-8")
        + "      llm:\n"
        "        provider: deterministic\n"
        "        model: deterministic-v1\n"
        "        budget:\n"
        "          max_api_calls: 4\n",
        encoding="utf-8",
    )
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")

    [capped] = run_project(project, select="document_embeddings")

    assert capped.status == "budget_exceeded"
    assert capped.metrics["provider_calls"] == 0
    # Nothing was billed, so nothing was published -- not even the table.
    tables = _query(
        project,
        "SELECT table_name FROM duckdb_tables() "
        "WHERE table_name = 'document_embeddings'",
    )
    assert tables == []


def test_a_budget_stop_does_not_claim_deletions_it_skipped(
    tmp_path: Path,
) -> None:
    """The budget stop skips the removed-row deletion pass; the result must
    say zero, not the plan. A manifest claiming mutations that did not occur
    is the kind of lie an operator acts on."""
    project = _project(tmp_path, flush_every=2)
    run_project(project)
    connection = duckdb.connect(str(project / "target" / "db.duckdb"))
    try:
        # Remove one upstream chunk and change two rows' text, so the run has
        # a deletion to plan and enough provider work for a cap of 1 (the
        # ledger's floor) to trip mid-window, before the deletion pass.
        connection.execute(
            'DELETE FROM "db".docs.document_chunks WHERE chunk_id = '
            '(SELECT min(chunk_id) FROM "db".docs.document_chunks)'
        )
        connection.execute(
            "UPDATE \"db\".docs.document_chunks SET text = 'revised-' || chunk_id "
            "WHERE chunk_id IN (SELECT chunk_id FROM \"db\".docs.document_chunks "
            "ORDER BY chunk_id DESC LIMIT 2)"
        )
    finally:
        connection.close()
    profiles = project / "profiles.yml"
    profiles.write_text(
        profiles.read_text(encoding="utf-8")
        + "      llm:\n"
        "        provider: deterministic\n"
        "        model: deterministic-v1\n"
        "        budget:\n"
        "          max_api_calls: 1\n",
        encoding="utf-8",
    )

    [capped] = run_project(project, select="document_embeddings")

    assert capped.status == "budget_exceeded"
    assert capped.documents_deleted == 0
    # The removed row is still in the target, consistent with the report.
    assert _embedded_count(project) == DOCUMENTS

    profiles.write_text(
        profiles.read_text(encoding="utf-8").replace(
            "max_api_calls: 1", "max_api_calls: 100"
        ),
        encoding="utf-8",
    )
    [resumed] = run_project(project, select="document_embeddings")

    assert resumed.documents_deleted == 1
    assert _embedded_count(project) == DOCUMENTS - 1


def test_a_decimal_id_degrades_to_no_reuse_instead_of_failing(
    tmp_path: Path,
) -> None:
    """The read-predicate contract carries strings, numbers, bools, and
    temporals -- not decimal.Decimal, which a DuckDB DECIMAL or BigQuery
    NUMERIC id column produces. The old whole-target dict handled those ids,
    so the bounded path must not turn them into a failed resume: it skips
    reuse for such rows (a paid re-embed, same price as changed text), never
    an error."""
    from stel.adapters import create_adapter, parse_warehouse_config
    from stel.config.model import EmbedConfig
    from stel.execution.embed import _EmbeddingReuseReader

    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "w.duckdb"), "schema": "docs"}
    )
    with create_adapter(config) as adapter:
        adapter.execute(
            'CREATE TABLE "w".docs.emb ('
            "chunk_id DECIMAL(10, 2), embedding_input_hash VARCHAR, "
            "embedding_config_hash VARCHAR, embedding DOUBLE[], "
            "embedded_at VARCHAR)"
        )
        adapter.execute(
            "INSERT INTO \"w\".docs.emb VALUES (1.50, 'h', 'g', [0.1], 't')"
        )
        reader = _EmbeddingReuseReader(
            adapter,
            "emb",
            config=EmbedConfig(
                provider="deterministic", model="m", dimensions=1,
                id_field="chunk_id", vector_field="embedding",
            ),
        )

        assert reader.target_key("1.50") is not None
        assert reader.rows_for(["1.50"]) == {}


def test_reuse_reader_retries_complete_mutable_target_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stel.adapters import (
        TableSnapshotGenerationChangedError,
        create_adapter,
        parse_warehouse_config,
    )
    from stel.config.model import EmbedConfig
    from stel.execution.embed import _EmbeddingReuseReader

    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "w.duckdb"), "schema": "docs"}
    )
    with create_adapter(config) as adapter:
        adapter.materialize_full(
            "emb",
            pl.DataFrame(
                {
                    "chunk_id": ["a"],
                    "embedding_input_hash": ["input-hash"],
                    "embedding_config_hash": ["config-hash"],
                    "embedding": [[0.1]],
                    "embedded_at": ["2026-08-30T00:00:00+00:00"],
                }
            ),
        )
        original_snapshot = adapter.table_snapshot
        snapshot_calls = 0
        changing_calls = {1, 3}

        @contextmanager
        def changing_snapshot(*args: Any, **kwargs: Any) -> Iterator[Any]:
            nonlocal snapshot_calls
            snapshot_calls += 1
            this_call = snapshot_calls
            with original_snapshot(*args, **kwargs) as snapshot:
                yield snapshot
            if this_call in changing_calls:
                raise TableSnapshotGenerationChangedError(
                    "simulated target generation change"
                )

        monkeypatch.setattr(adapter, "table_snapshot", changing_snapshot)
        reader = _EmbeddingReuseReader(
            adapter,
            "emb",
            config=EmbedConfig(
                provider="deterministic",
                model="m",
                dimensions=1,
                id_field="chunk_id",
                vector_field="embedding",
            ),
        )

        assert snapshot_calls == 2
        assert reader.rows_for(["a"])["a"]["embedding_input_hash"] == "input-hash"
        assert snapshot_calls == 4

        changing_calls.update({5, 6, 7})
        with pytest.raises(
            TableSnapshotGenerationChangedError,
            match="simulated target generation change",
        ):
            reader.rows_for(["a"])
        assert snapshot_calls == 7
