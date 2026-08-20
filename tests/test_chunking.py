"""Chunk model kind (issue #86): splitter behaviour, deterministic IDs, and
the registry → chunks contract end to end through the runner."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

from stel.chunking import chunk_id, split_text
from stel.config.model import ChunkConfig
from stel.hashing import HASH_DIGEST_SIZE
from stel.runner import (
    _CHUNK_INPUT_EXCLUDED_FIELDS,
    RunError,
    _chunk_document_ids,
    _chunk_input_hash,
)

# ─── splitter ───────────────────────────────────────────────────────────────


def test_short_text_is_one_chunk() -> None:
    chunks = split_text("a short sentence.", ChunkConfig(chunk_size=100, chunk_overlap=10))
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == "a short sentence."


def test_empty_and_whitespace_yield_no_chunks() -> None:
    assert split_text("", ChunkConfig()) == []
    assert split_text("   \n  ", ChunkConfig()) == []


def test_recursive_respects_size_and_indexes() -> None:
    text = ". ".join(f"sentence number {i}" for i in range(40))
    chunks = split_text(text, ChunkConfig(chunk_size=80, chunk_overlap=10))
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # only a hard word-boundary cut may exceed size; recursive splits shouldn't
    assert all(len(c.text) <= 80 + 20 for c in chunks)


def test_recursive_overlap_carries_context() -> None:
    text = " ".join(f"word{i}" for i in range(100))
    chunks = split_text(text, ChunkConfig(chunk_size=60, chunk_overlap=20))
    # adjacent chunks should share some trailing/leading content
    overlaps = 0
    for a, b in pairwise(chunks):
        tail = a.text[-15:]
        if any(tok in b.text for tok in tail.split()):
            overlaps += 1
    assert overlaps >= 1


def test_long_unbroken_token_is_hard_cut() -> None:
    chunks = split_text("x" * 250, ChunkConfig(chunk_size=100, chunk_overlap=0))
    assert len(chunks) == 3
    assert "".join(c.text for c in chunks) == "x" * 250


def test_tokens_strategy_splits_by_tokens() -> None:
    text = " ".join(f"token{i}" for i in range(200))
    chunks = split_text(text, ChunkConfig(strategy="tokens", chunk_size=50, chunk_overlap=10))
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_splitting_is_deterministic() -> None:
    text = ". ".join(f"clause {i}" for i in range(50))
    cfg = ChunkConfig(chunk_size=90, chunk_overlap=15)
    assert split_text(text, cfg) == split_text(text, cfg)


def test_splitter_does_not_inject_trailing_separator() -> None:
    """The source ends in `sentence 19` (no trailing `. `); the final chunk
    must not gain an injected period — chunk text is what gets embedded."""
    text = ". ".join(f"sentence {i}" for i in range(20))
    assert text.endswith("sentence 19")
    chunks = split_text(text, ChunkConfig(chunk_size=80, chunk_overlap=10))
    assert chunks[-1].text.endswith("sentence 19")
    assert not chunks[-1].text.endswith(".")


def test_chunks_contain_only_source_characters() -> None:
    """Concatenating chunk text (minus overlap) introduces no characters that
    weren't in the source."""
    text = "\n\n".join(f"Paragraph {i} has several words in it." for i in range(15))
    chunks = split_text(text, ChunkConfig(chunk_size=100, chunk_overlap=0))
    source_chars = set(text)
    for chunk in chunks:
        assert set(chunk.text) <= source_chars


# ─── chunk ids ──────────────────────────────────────────────────────────────


def test_chunk_id_is_content_addressed() -> None:
    document_id = "0" * 32
    other_document_id = "1" * 32
    assert chunk_id(document_id, 0, "hello") == chunk_id(document_id, 0, "hello")
    assert chunk_id(document_id, 0, "hello") != chunk_id(document_id, 1, "hello")
    assert chunk_id(document_id, 0, "hello") != chunk_id(document_id, 0, "world")
    assert chunk_id(document_id, 0, "hello") != chunk_id(other_document_id, 0, "hello")


def test_chunk_id_rejects_non_contract_document_id() -> None:
    with pytest.raises(ValueError, match="document_id"):
        chunk_id("doc", 0, "hello")


def test_chunk_config_validation() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ChunkConfig(chunk_size=0)
    with pytest.raises(ValueError, match="overlap must be smaller"):
        ChunkConfig(chunk_size=100, chunk_overlap=100)


def test_chunk_input_fingerprint_is_typed_and_canonical() -> None:
    first = {
        "document_id": "doc-1",
        "body": "unchanged text",
        "metadata": {"tenant": "econ", "groups": ["reader", "admin"]},
        "effective_at": datetime(2026, 7, 15, tzinfo=UTC),
        "weight": Decimal("1.00"),
        "payload": b"\x00\xff",
    }
    reordered = {
        "payload": b"\x00\xff",
        "weight": Decimal("1.00"),
        "effective_at": datetime(2026, 7, 15, tzinfo=UTC),
        "metadata": {"groups": ["reader", "admin"], "tenant": "econ"},
        "body": "unchanged text",
        "document_id": "doc-1",
    }

    assert _chunk_input_hash(first, text_field="body") == _chunk_input_hash(
        reordered, text_field="body"
    )

    changed = dict(first)
    changed["metadata"] = {"tenant": "econ", "groups": ["admin", "reader"]}
    assert _chunk_input_hash(first, text_field="body") != _chunk_input_hash(
        changed, text_field="body"
    )


@pytest.mark.parametrize(
    "field",
    ["title", "source_uri", "tenant", "access_groups", "effective_date"],
)
def test_chunk_input_fingerprint_includes_filter_metadata(field: str) -> None:
    record = {
        "document_id": "doc-1",
        "body": "unchanged text",
        "title": "old",
        "source_uri": "gs://bucket/old",
        "tenant": "tenant-a",
        "access_groups": ["analyst"],
        "effective_date": "2026-07-15",
    }
    changed = dict(record)
    changed[field] = "different"

    assert _chunk_input_hash(record, text_field="body") != _chunk_input_hash(
        changed, text_field="body"
    )


def test_chunk_input_fingerprint_excludes_only_generated_values() -> None:
    assert _CHUNK_INPUT_EXCLUDED_FIELDS == {
        "chunk_id",
        "document_id",
        "chunk_index",
        "chunk_count",
        "text",
        "chunk_strategy",
        "code_version",
        "chunked_at",
    }
    record = {
        "document_id": "doc-1",
        "body": "unchanged text",
        "chunk_id": "old-chunk",
        "chunk_index": 99,
        "chunk_count": 100,
        "text": "overwritten",
        "chunk_strategy": "old",
        "code_version": "upstream-version",
        "chunked_at": "yesterday",
    }
    changed = {
        **record,
        "chunk_id": "other",
        "chunk_index": 0,
        "chunk_count": 1,
        "text": "other",
        "chunk_strategy": "other",
        "code_version": "other",
        "chunked_at": "today",
    }

    assert _chunk_input_hash(record, text_field="body") == _chunk_input_hash(
        changed, text_field="body"
    )
    changed["document_id"] = "doc-2"
    assert _chunk_input_hash(record, text_field="body") != _chunk_input_hash(
        changed, text_field="body"
    )


def test_chunk_input_text_coercion_keeps_zero_false_and_null_distinct() -> None:
    fingerprints = {
        _chunk_input_hash({"document_id": "doc", "body": value}, text_field="body")
        for value in (0, False, None)
    }

    assert len(fingerprints) == 3


@pytest.mark.parametrize(
    ("values", "message"),
    [(["doc", None], "NULL"), (["doc", ""], "empty"), (["doc", "doc"], "duplicate")],
)
def test_chunk_document_identity_is_validated_before_mutation(
    values: list[str | None], message: str
) -> None:
    with pytest.raises(RunError, match=message):
        _chunk_document_ids(pl.DataFrame({"document_id": values}), "chunks")


# ─── end to end: registry → chunks ──────────────────────────────────────────


def _chunk_project(tmp_path: Path, *, chunk_size: int = 120) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "stel_project.yml").write_text("name: docs\nversion: '0.1.0'\nprofile: docs\n")
    (project / "profiles.yml").write_text(
        "docs:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n        schema: docs\n"
    )
    (project / "sources").mkdir()
    (project / "sources" / "src.yml").write_text(
        "version: 2\nsources:\n  - name: raw_docs\n    path: data/docs\n"
        "    file_pattern: '*.json'\n"
    )
    (project / "models").mkdir()
    (project / "models" / "registry.yml").write_text(
        "version: 2\nmodels:\n  - name: document_registry\n"
        "    source: ref('raw_docs')\n    extraction:\n      backend: json\n"
        "      options:\n"
        "        fields: [title, body, tenant, access_groups, effective_date]\n"
        "    materialization: incremental\n"
    )
    (project / "models" / "chunks.yml").write_text(
        "version: 2\nmodels:\n  - name: document_chunks\n"
        "    depends_on: [ref('document_registry')]\n    chunk:\n"
        "      strategy: recursive\n      text_field: body\n"
        f"      chunk_size: {chunk_size}\n      chunk_overlap: 20\n"
        "    materialization: incremental\n"
    )
    docs = project / "data" / "docs"
    docs.mkdir(parents=True)
    return project


def _write_doc(
    project: Path,
    name: str,
    title: str,
    body: str,
    **metadata: object,
) -> None:
    import json

    (project / "data" / "docs" / name).write_text(
        json.dumps({"title": title, "body": body, **metadata})
    )


def _chunks(project: Path) -> list[tuple[Any, ...]]:
    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        return con.execute(
            "SELECT chunk_id, document_id, chunk_index, chunk_count, text, "
            'title, source_uri, chunk_strategy FROM "db".docs.document_chunks '
            "ORDER BY document_id, chunk_index"
        ).fetchall()
    finally:
        con.close()


def test_chunk_model_produces_contract_rows(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _chunk_project(tmp_path)
    _write_doc(project, "a.json", "Doc A", ". ".join(f"sentence {i}" for i in range(30)))
    _write_doc(project, "b.json", "Doc B", "short body")

    results = run_project(project)
    by_name = {r.model_name: r for r in results}
    assert by_name["document_chunks"].kind == "chunk"
    assert by_name["document_chunks"].documents_processed == 2

    rows = _chunks(project)
    # doc A splits into several chunks; doc B is one
    a_chunks = [r for r in rows if r[5] == "Doc A"]
    b_chunks = [r for r in rows if r[5] == "Doc B"]
    assert len(a_chunks) > 1
    assert len(b_chunks) == 1
    # contract: stable ids, lineage carried, chunk metadata present
    assert len({r[0] for r in rows}) == len(rows)  # unique chunk_ids
    for cid, doc_id, _idx, _count, text, _title, source_uri, strategy in rows:
        assert cid and doc_id and text
        assert source_uri.startswith("file://")
        assert strategy == "recursive"
    # chunk_index is contiguous per document and chunk_count matches
    assert [r[2] for r in a_chunks] == list(range(len(a_chunks)))
    assert all(r[3] == len(a_chunks) for r in a_chunks)


def test_chunk_ids_stable_across_reruns(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _chunk_project(tmp_path)
    _write_doc(project, "a.json", "Doc A", ". ".join(f"sentence {i}" for i in range(30)))
    run_project(project)
    first = {r[0] for r in _chunks(project)}

    results = run_project(project)
    # unchanged upstream → chunk model skips the document
    assert next(r for r in results if r.model_name == "document_chunks").documents_skipped == 1
    assert {r[0] for r in _chunks(project)} == first


def test_legacy_chunk_state_migrates_then_rewrites_once(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _chunk_project(tmp_path)
    body = ". ".join(f"sentence {i}" for i in range(30))
    _write_doc(project, "a.json", "Doc A", body)
    run_project(project)
    original_ids = {row[0] for row in _chunks(project)}

    old_chunk_hash = "text:" + hashlib.blake2b(
        body.encode(), digest_size=HASH_DIGEST_SIZE
    ).hexdigest()
    db_path = project / "target" / "db.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE "db".docs.stel_state_v1 (
                model_name VARCHAR NOT NULL,
                document_id VARCHAR NOT NULL,
                content_hash VARCHAR NOT NULL,
                code_version VARCHAR NOT NULL,
                last_run_at TIMESTAMP NOT NULL,
                PRIMARY KEY (model_name, document_id)
            )
            """
        )
        con.execute(
            """
            INSERT INTO "db".docs.stel_state_v1
            SELECT model_name, record_key,
                   CASE WHEN model_name = 'document_chunks' THEN ?
                        ELSE input_fingerprint END,
                   code_version, last_run_at
            FROM "db".docs.stel_state
            """,
            [old_chunk_hash],
        )
        con.execute('DROP TABLE "db".docs.stel_state')
        con.execute(
            'ALTER TABLE "db".docs.stel_state_v1 RENAME TO stel_state'
        )
    finally:
        con.close()

    migrated = run_project(project)
    chunk_result = next(r for r in migrated if r.model_name == "document_chunks")
    assert chunk_result.documents_processed == 1
    assert chunk_result.documents_skipped == 0
    assert {row[0] for row in _chunks(project)} == original_ids

    unchanged = run_project(project)
    unchanged_chunk = next(
        r for r in unchanged if r.model_name == "document_chunks"
    )
    assert unchanged_chunk.documents_processed == 0
    assert unchanged_chunk.documents_skipped == 1


@pytest.mark.parametrize(
    ("column", "updated"),
    [
        ("title", "Reclassified filing"),
        ("source_uri", "gs://governed-bucket/reclassified.json"),
        ("tenant", "tenant-b"),
        ("access_groups", '["admin","auditor"]'),
        ("effective_date", "2026-08-01"),
    ],
)
def test_metadata_only_update_rewrites_chunks_with_stable_ids(
    tmp_path: Path, column: str, updated: str
) -> None:
    from stel.runner import run_project

    project = _chunk_project(tmp_path)
    _write_doc(
        project,
        "a.json",
        "Economic release",
        ". ".join(f"sentence {i}" for i in range(30)),
        tenant="tenant-a",
        access_groups=["analyst"],
        effective_date="2026-07-15",
    )
    run_project(project)
    before_ids = {row[0] for row in _chunks(project)}

    db_path = project / "target" / "db.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            f'UPDATE "db".docs.document_registry SET "{column}" = ?',
            [updated],
        )
    finally:
        con.close()

    results = run_project(project)
    chunk_result = next(r for r in results if r.model_name == "document_chunks")
    assert chunk_result.documents_processed == 1
    assert chunk_result.documents_skipped == 0

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(f'SELECT chunk_id, "{column}" FROM "db".docs.document_chunks').fetchall()
    finally:
        con.close()
    assert {row[0] for row in rows} == before_ids
    assert {row[1] for row in rows} == {updated}

    unchanged = run_project(project)
    unchanged_chunk = next(r for r in unchanged if r.model_name == "document_chunks")
    assert unchanged_chunk.documents_processed == 0
    assert unchanged_chunk.documents_skipped == 1


def test_failed_chunk_replacement_cannot_leave_old_state_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """replace_children is atomic: a failure rolls back the whole transaction,
    so chunks and state are never left inconsistent (issue #229)."""
    from stel.adapters.base import AdapterError
    from stel.adapters.duckdb import DuckDBAdapter
    from stel.runner import RunError, run_project

    project = _chunk_project(tmp_path)
    _write_doc(project, "a.json", "Original title", "unchanged body")
    run_project(project)
    original_ids = {row[0] for row in _chunks(project)}

    db_path = project / "target" / "db.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            'UPDATE "db".docs.document_registry SET title = ?',
            ["Restricted title"],
        )
    finally:
        con.close()

    original_replace = DuckDBAdapter.replace_children
    failed = False

    def fail_replace_once(
        self: DuckDBAdapter, table: str, *args: Any, **kwargs: Any
    ) -> int:
        nonlocal failed
        if table == "document_chunks" and not failed:
            failed = True
            raise AdapterError("simulated chunk publication failure")
        return original_replace(self, table, *args, **kwargs)

    monkeypatch.setattr(DuckDBAdapter, "replace_children", fail_replace_once)
    with pytest.raises(RunError, match="simulated chunk publication failure"):
        run_project(project)

    # Atomic rollback: old chunks and state must both still be present.
    con = duckdb.connect(str(db_path))
    try:
        assert con.execute('SELECT COUNT(*) FROM "db".docs.document_chunks').fetchone() == (
            len(original_ids),
        )
        assert con.execute(
            "SELECT COUNT(*) FROM \"db\".docs.stel_state WHERE model_name = 'document_chunks'"
        ).fetchone() == (1,)
        # Restore the title so the document hash matches the pre-failure state.
        con.execute(
            'UPDATE "db".docs.document_registry SET title = ?',
            ["Original title"],
        )
    finally:
        con.close()

    # Document hash now matches stored state → skipped, not reprocessed.
    results = run_project(project)
    chunk_result = next(r for r in results if r.model_name == "document_chunks")
    assert chunk_result.documents_processed == 0
    assert chunk_result.documents_skipped == 1
    assert {row[0] for row in _chunks(project)} == original_ids


def test_changed_document_rechunks_without_orphans(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _chunk_project(tmp_path)
    _write_doc(project, "a.json", "Doc A", ". ".join(f"sentence {i}" for i in range(30)))
    run_project(project)
    before = _chunks(project)
    assert len(before) > 1

    # shrink the document — must not leave orphan chunks from the longer version
    _write_doc(project, "a.json", "Doc A", "now just one short body")
    results = run_project(project)
    chunk_res = next(r for r in results if r.model_name == "document_chunks")
    assert chunk_res.documents_processed == 1

    after = _chunks(project)
    assert len(after) == 1
    assert after[0][4] == "now just one short body"
    # none of the old chunk_ids survive
    assert not ({r[0] for r in after} & {r[0] for r in before})


def test_changed_chunk_upstream_rechunks_same_source_hash(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _chunk_project(tmp_path, chunk_size=500)
    _write_doc(project, "a.json", "Doc A", "base text for the filing")
    (project / "transforms").mkdir()
    (project / "transforms" / "noisy_docs.py").write_text(
        "import polars as pl\n\n"
        "def run(deps):\n"
        "    df = deps['document_registry']\n"
        "    return df.with_columns(\n"
        "        (pl.lit('NOISY ') + pl.col('body')).alias('body')\n"
        "    )\n"
    )
    (project / "transforms" / "clean_docs.py").write_text(
        "import polars as pl\n\n"
        "def run(deps):\n"
        "    df = deps['document_registry']\n"
        "    return df.with_columns(\n"
        "        (pl.lit('CLEAN ') + pl.col('body')).alias('body')\n"
        "    )\n"
    )
    (project / "models" / "variants.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: noisy_docs\n"
        "    depends_on: [ref('document_registry')]\n"
        "    transform:\n"
        "      type: python\n"
        "      module: transforms.noisy_docs\n"
        "  - name: clean_docs\n"
        "    depends_on: [ref('document_registry')]\n"
        "    transform:\n"
        "      type: python\n"
        "      module: transforms.clean_docs\n"
    )
    chunks_yml = project / "models" / "chunks.yml"
    chunks_yml.write_text(
        chunks_yml.read_text().replace(
            "depends_on: [ref('document_registry')]",
            "depends_on: [ref('noisy_docs')]",
        )
    )

    run_project(project)
    assert _chunks(project)[0][4] == "NOISY base text for the filing"

    chunks_yml.write_text(
        chunks_yml.read_text().replace(
            "depends_on: [ref('noisy_docs')]",
            "depends_on: [ref('clean_docs')]",
        )
    )
    results = run_project(project)
    chunk_res = next(r for r in results if r.model_name == "document_chunks")

    assert chunk_res.documents_processed == 1
    assert chunk_res.documents_skipped == 0
    rows = _chunks(project)
    assert len(rows) == 1
    assert rows[0][4] == "CLEAN base text for the filing"


def test_deleted_document_prunes_its_chunks(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _chunk_project(tmp_path)
    _write_doc(project, "a.json", "Doc A", ". ".join(f"sentence {i}" for i in range(30)))
    _write_doc(project, "b.json", "Doc B", "keep me")
    run_project(project)
    assert {r[5] for r in _chunks(project)} == {"Doc A", "Doc B"}

    (project / "data" / "docs" / "a.json").unlink()
    results = run_project(project)
    chunk_res = next(r for r in results if r.model_name == "document_chunks")
    assert chunk_res.documents_deleted == 1

    remaining = _chunks(project)
    assert {r[5] for r in remaining} == {"Doc B"}


def test_full_materialization_chunks(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _chunk_project(tmp_path)
    # switch the chunk model to full materialization
    chunks_yml = project / "models" / "chunks.yml"
    chunks_yml.write_text(
        chunks_yml.read_text().replace("materialization: incremental", "materialization: full")
    )
    _write_doc(project, "a.json", "Doc A", ". ".join(f"sentence {i}" for i in range(20)))
    run_project(project)
    assert len(_chunks(project)) > 1


def test_chunk_model_requires_text_field(tmp_path: Path) -> None:
    from stel.runner import RunError, run_project

    project = _chunk_project(tmp_path)
    chunks_yml = project / "models" / "chunks.yml"
    chunks_yml.write_text(chunks_yml.read_text().replace("text_field: body", "text_field: nope"))
    _write_doc(project, "a.json", "Doc A", "some body text here")
    with pytest.raises(RunError, match="no column 'nope'"):
        run_project(project)


def test_chunk_dag_orders_after_extraction(tmp_path: Path) -> None:
    from stel.config import load_project
    from stel.dag import ProjectDAG

    project = _chunk_project(tmp_path)
    _, sources, models = load_project(project)
    dag = ProjectDAG(sources, models)
    order = dag.execution_order()
    assert order.index("document_registry") < order.index("document_chunks")


def test_ls_reports_chunk_kind() -> None:
    """`stel ls` must show chunk models as `chunk`, not `unknown`."""
    from stel.cli import _model_kind
    from stel.config.model import ChunkConfig, ModelConfig

    model = ModelConfig(
        name="document_chunks",
        depends_on=["ref('document_registry')"],
        chunk=ChunkConfig(),
    )
    assert _model_kind(model) == "chunk"
