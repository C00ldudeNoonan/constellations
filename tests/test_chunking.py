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

from stel.chunking import (
    ChunkingError,
    chunk_id,
    measure,
    render_metadata_block,
    split_text,
)
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


def _chunk_project(
    tmp_path: Path, *, chunk_size: int = 120, in_text_metadata: str | None = None
) -> Path:
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
    metadata_line = (
        f"      in_text_metadata: [{in_text_metadata}]\n"
        if in_text_metadata is not None
        else ""
    )
    (project / "models" / "chunks.yml").write_text(
        "version: 2\nmodels:\n  - name: document_chunks\n"
        "    depends_on: [ref('document_registry')]\n    chunk:\n"
        "      strategy: recursive\n      text_field: body\n"
        f"      chunk_size: {chunk_size}\n      chunk_overlap: 20\n"
        f"{metadata_line}"
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


# ─── in-text metadata (issue #308) ──────────────────────────────────────────


def test_metadata_block_renders_in_declared_order() -> None:
    # Declared order, not sorted: it is the author's, and a stable rendering is
    # what keeps chunk ids stable.
    record = {"title": "Q3 Report", "published_date": "2026-03-14", "body": "..."}

    block = render_metadata_block(record, ["published_date", "title"])

    assert block == "published_date: 2026-03-14\ntitle: Q3 Report\n---\n"


def test_metadata_block_skips_null_values() -> None:
    record = {"title": "Q3 Report", "published_date": None}

    block = render_metadata_block(record, ["title", "published_date"])

    assert block == "title: Q3 Report\n---\n"
    assert "None" not in block


def test_metadata_block_is_empty_when_every_value_is_null() -> None:
    # No lines means no separator either — an empty block must not prepend a
    # bare rule to the text.
    assert render_metadata_block({"title": None}, ["title"]) == ""


def test_reserved_budget_shrinks_chunks_so_the_block_still_fits() -> None:
    config = ChunkConfig(strategy="recursive", chunk_size=100, chunk_overlap=0)
    text = "x" * 500
    block = "title: T\n---\n"

    pieces = split_text(text, config, reserved=len(block))

    assert pieces
    for piece in pieces:
        assert len(block + piece.text) <= config.chunk_size


def test_reserved_budget_leaving_no_room_for_text_raises() -> None:
    config = ChunkConfig(strategy="recursive", chunk_size=20, chunk_overlap=0)

    with pytest.raises(ChunkingError, match="leaving no room for text"):
        split_text("some text", config, reserved=20)


def test_overlap_at_or_above_the_remaining_budget_raises() -> None:
    # chunk_overlap validated against chunk_size at config load, but the block
    # moves the ceiling it has to stay under.
    config = ChunkConfig(strategy="recursive", chunk_size=100, chunk_overlap=60)

    with pytest.raises(ChunkingError, match="chunk_overlap"):
        split_text("some text", config, reserved=50)


def test_in_text_metadata_config_rejects_duplicates_and_the_text_field() -> None:
    with pytest.raises(ValueError, match="twice"):
        ChunkConfig(in_text_metadata=["title", "title"])
    with pytest.raises(ValueError, match="must not name the text field"):
        ChunkConfig(text_field="body", in_text_metadata=["body"])


def test_in_text_metadata_is_additive(tmp_path: Path) -> None:
    """The block goes into the text *and* the columns stay.

    The design rule (#308): SQL reads columns, the embedding model reads only
    the text, and a rendering aimed at one reader must never remove the copy
    the other depends on.
    """
    from stel.runner import run_project

    project = _chunk_project(tmp_path, in_text_metadata="title")
    _write_doc(project, "a.json", "Doc A", ". ".join(f"sentence {i}" for i in range(30)))

    run_project(project)

    rows = _chunks(project)
    assert rows
    for _cid, _doc_id, _idx, _count, text, title, _source_uri, _strategy in rows:
        # in the text, for the embedder...
        assert text.startswith("title: Doc A\n---\n")
        # ...and still in its own column, for SQL.
        assert title == "Doc A"


def test_in_text_metadata_chunks_stay_within_chunk_size(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _chunk_project(tmp_path, chunk_size=120, in_text_metadata="title")
    _write_doc(project, "a.json", "Doc A", ". ".join(f"sentence {i}" for i in range(40)))

    run_project(project)

    rows = _chunks(project)
    assert len(rows) > 1
    for row in rows:
        assert len(row[4]) <= 120


def test_chunk_id_tracks_the_text_that_is_stored(tmp_path: Path) -> None:
    """chunk_id must derive from the text the row actually carries.

    The agent_context document_chunks contract recomputes it from the stored
    `text` and rejects a mismatch, so deriving the id from the pre-block text
    while storing the post-block text would fail validation downstream.
    """
    from stel.runner import run_project

    project = _chunk_project(tmp_path, in_text_metadata="title")
    _write_doc(project, "a.json", "Doc A", ". ".join(f"sentence {i}" for i in range(30)))

    run_project(project)

    for cid, doc_id, idx, _count, text, *_rest in _chunks(project):
        assert cid == chunk_id(doc_id, idx, text)


def test_in_text_metadata_naming_a_missing_column_fails_fast(tmp_path: Path) -> None:
    from stel.runner import run_project

    project = _chunk_project(tmp_path, in_text_metadata="nonexistent_column")
    _write_doc(project, "a.json", "Doc A", "short body")

    with pytest.raises(RunError, match="nonexistent_column"):
        run_project(project)


def test_tokens_strategy_charges_the_block_against_chunk_size() -> None:
    """The reservation is exact in the unit the strategy splits by.

    Tokenizers can merge across a concatenation boundary, so measuring the
    block alone and adding it to a piece is not obviously safe — this asserts
    the combined text really does fit, rather than assuming it.
    """
    tiktoken = pytest.importorskip("tiktoken")
    config = ChunkConfig(
        strategy="tokens",
        chunk_size=64,
        chunk_overlap=8,
        in_text_metadata=["title", "published_date"],
    )
    record = {"title": "Q3 Monetary Policy Report", "published_date": "2026-03-14"}
    block = render_metadata_block(record, config.in_text_metadata)
    text = ". ".join(f"sentence number {i} about rates" for i in range(120))

    pieces = split_text(text, config, reserved=measure(block, config))

    encoding = tiktoken.get_encoding(config.encoding)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(encoding.encode(block + piece.text)) <= config.chunk_size


# ─── boundary-aware overlap (issue #331) ────────────────────────────────────


def _structured_document() -> str:
    """Paragraphs of sentences, the shape of a filing or report."""
    sentences = [
        f"Sentence number {index} discusses revenue and margin at length."
        for index in range(400)
    ]
    return "\n\n".join(
        " ".join(sentences[start : start + 5]) for start in range(0, 400, 5)
    )


def _starts_mid_word(chunk: str, document: str) -> bool:
    """True when the chunk begins partway through a word of the source."""
    position = document.find(chunk[:40])
    if position <= 0:
        return False
    return not document[position - 1].isspace()


def test_overlap_no_longer_starts_chunks_mid_word() -> None:
    """The measured defect: a fixed slice lands wherever the count says.

    On a real 10-K that put 81.5% of chunk starts mid-word, which defeated the
    separator hierarchy entirely — upstream work to emit real paragraph
    structure showed almost no improvement because the overlap was
    reintroducing arbitrary offsets on its own.
    """
    document = _structured_document()

    chunks = split_text(
        document, ChunkConfig(chunk_size=800, chunk_overlap=100)
    )

    assert len(chunks) > 5
    broken = [c.text for c in chunks if _starts_mid_word(c.text, document)]
    assert broken == []


def test_overlap_still_carries_context_across_the_boundary() -> None:
    """Snapping must not quietly become `overlap: 0`.

    Disabling overlap also gives clean boundaries — the point is keeping the
    redundancy that lets a query matching the far half of a straddling concept
    still find the chunk.
    """
    document = _structured_document()

    overlapped = split_text(
        document, ChunkConfig(chunk_size=800, chunk_overlap=100)
    )
    without = split_text(document, ChunkConfig(chunk_size=800, chunk_overlap=0))

    # Each chunk after the first repeats a real tail of its predecessor.
    for previous, following in pairwise(overlapped):
        shared = following.text.split(".")[0]
        assert shared and shared in previous.text
    # Redundancy is the point of overlap, so the chunks together cover more
    # than the document once. (Chunk *count* is not a reliable proxy: whether
    # the carried tail pushes a chunk over the size limit depends on where the
    # paragraphs fall.)
    assert sum(len(c.text) for c in overlapped) > sum(
        len(c.text) for c in without
    )


def test_the_carried_overlap_stays_near_the_requested_size() -> None:
    """`approximately N`, not `whatever the nearest paragraph break was`."""
    document = _structured_document()

    chunks = split_text(
        document, ChunkConfig(chunk_size=800, chunk_overlap=100)
    )

    for previous, following in pairwise(chunks):
        carried = 0
        for size in range(min(len(previous.text), len(following.text)), 0, -1):
            if previous.text.endswith(following.text[:size]):
                carried = size
                break
        # The snap band is [overlap/2, 2*overlap]; strip() can trim a little
        # more off the front, so the floor is generous.
        assert carried <= 200, f"carried {carried} characters, far beyond 100"


def test_a_long_unbroken_token_falls_back_to_a_plain_slice() -> None:
    # No separator exists anywhere in the band, so there is nothing to snap
    # to and the old behavior is the honest answer.
    document = "x" * 5000

    chunks = split_text(document, ChunkConfig(chunk_size=500, chunk_overlap=50))

    assert len(chunks) > 1
    assert all(set(c.text) == {"x"} for c in chunks)


def test_zero_overlap_is_untouched() -> None:
    document = _structured_document()

    chunks = split_text(document, ChunkConfig(chunk_size=800, chunk_overlap=0))

    for previous, following in pairwise(chunks):
        assert not previous.text.endswith(following.text[:40])


def test_snapping_is_deterministic() -> None:
    document = _structured_document()
    config = ChunkConfig(chunk_size=800, chunk_overlap=100)

    assert split_text(document, config) == split_text(document, config)


def test_overlap_prefers_the_strongest_available_boundary() -> None:
    from stel.chunking import _overlap_tail

    # A paragraph break and a sentence break both sit inside the band; the
    # hierarchy the splitter walks prefers the paragraph.
    text = "alpha beta gamma.\n\ndelta epsilon zeta. eta theta iota kappa"

    tail = _overlap_tail(text, 45)

    assert tail.startswith("delta")
    # An overlap of exactly 40 would land on that boundary by arithmetic
    # coincidence, which would let this pass with the old fixed slice too;
    # 45 lands mid-word unless the snap actually happens.
    assert not text[-45:].startswith("delta")


# ─── heading attribution (issue #332) ───────────────────────────────────────

_FILING = (
    "Item 1. Business\n\n"
    + "We operate in several segments. " * 6
    + "\n\nItem 1A. Risk Factors\n\n"
    + "Tariffs could affect margins. " * 6
    + "\n\nItem 7. Discussion\n\n"
    + "Revenue grew modestly. " * 6
)
_ITEM_PATTERN = r"^(Item\s+\d{1,2}[A-C]?)[.:]"


def _sectioned(**overrides: Any) -> list[Any]:
    config = ChunkConfig(
        chunk_size=overrides.pop("chunk_size", 150),
        chunk_overlap=overrides.pop("chunk_overlap", 20),
        headings={"pattern": _ITEM_PATTERN, **overrides},
    )
    return split_text(_FILING, config)


def test_every_chunk_is_attributed_to_its_heading() -> None:
    """The splitter knows the full text and every boundary; use that.

    A downstream transform re-derives section membership from chunk fragments
    and misses the cases offsets settle outright.
    """
    chunks = _sectioned()

    assert len(chunks) > 3
    for chunk in chunks:
        expected = None
        for name in ("Item 7", "Item 1A", "Item 1"):
            if _FILING.index(f"{name}.") <= _FILING.index(chunk.text[:30]):
                expected = name
                break
        assert chunk.section == expected


def test_a_chunk_opening_a_section_belongs_to_it() -> None:
    # The heading sits at the chunk's very first character; it covers that
    # chunk rather than the previous section.
    chunks = _sectioned()

    opening = next(c for c in chunks if c.text.startswith("Item 1A."))
    assert opening.section == "Item 1A"


def test_text_before_the_first_heading_has_no_section() -> None:
    preamble = "Cover page text here.\n\nItem 1. Business\n\nBody follows here."

    chunks = split_text(
        preamble,
        ChunkConfig(
            chunk_size=40, chunk_overlap=0, headings={"pattern": _ITEM_PATTERN}
        ),
    )

    assert chunks[0].section is None
    assert chunks[-1].section == "Item 1"


def test_a_capture_group_names_the_section() -> None:
    # Without a group the whole match is the name, so the author chooses
    # whether trailing punctuation is part of it rather than stel guessing.
    with_group = _sectioned()[0].section
    without_group = split_text(
        _FILING,
        ChunkConfig(
            chunk_size=150,
            chunk_overlap=20,
            headings={"pattern": r"^Item\s+\d{1,2}[A-C]?[.:]"},
        ),
    )[0].section

    assert with_group == "Item 1"
    assert without_group == "Item 1."


def test_no_headings_configured_leaves_the_section_unset() -> None:
    chunks = split_text(_FILING, ChunkConfig(chunk_size=150, chunk_overlap=20))

    assert all(chunk.section is None for chunk in chunks)


def test_a_pattern_matching_nothing_attributes_nothing() -> None:
    chunks = split_text(
        _FILING,
        ChunkConfig(
            chunk_size=150, chunk_overlap=20, headings={"pattern": r"^Section\s+\d+"}
        ),
    )

    assert all(chunk.section is None for chunk in chunks)


def test_attribution_survives_overlap() -> None:
    """The carried tail moves a chunk's start earlier, and the offset with it.

    A chunk whose overlap reaches back into the previous section still belongs
    to the section its own start falls in.
    """
    without = _sectioned(chunk_overlap=0)
    with_overlap = _sectioned(chunk_overlap=40)

    assert [c.section for c in without].count("Item 1A") > 0
    assert [c.section for c in with_overlap].count("Item 1A") > 0
    for chunk in with_overlap:
        assert chunk.section in {"Item 1", "Item 1A", "Item 7"}


def test_headings_require_the_recursive_strategy() -> None:
    # Token splitting has no source offsets, so attribution would be a guess.
    with pytest.raises(ValueError, match="requires `strategy: recursive`"):
        ChunkConfig(strategy="tokens", headings={"pattern": _ITEM_PATTERN})


def test_an_invalid_heading_pattern_is_rejected_at_config_load() -> None:
    with pytest.raises(ValueError, match="not a valid regex"):
        ChunkConfig(headings={"pattern": "["})
    with pytest.raises(ValueError, match="at most one capture group"):
        ChunkConfig(headings={"pattern": "^(a)(b)"})


def test_a_chunk_straddling_a_boundary_belongs_where_it_starts() -> None:
    """The rule is "the last heading at or before the chunk's start".

    A chunk can contain the *next* section's heading in its tail while still
    being mostly the previous section's content, and it belongs to the
    previous one. Pinned because it is a semantics choice a reader has to
    know, and the case a downstream transform re-deriving from fragments gets
    wrong — it sees the heading text and claims the whole chunk.
    """
    document = (
        "Item 1. Business\n\n"
        + "Body sentence here. " * 4
        + "\n\nItem 1A. Risk Factors\n\n"
        + "Risk sentence here. " * 4
    )

    chunks = split_text(
        document,
        ChunkConfig(
            chunk_size=170, chunk_overlap=0, headings={"pattern": _ITEM_PATTERN}
        ),
    )

    straddling = [c for c in chunks if "Item 1A." in c.text and c.section == "Item 1"]
    assert straddling, "expected a chunk carrying the next heading in its tail"
    # And the heading text itself is never dropped from the corpus.
    assert any("Item 1A." in c.text for c in chunks)


# ─── review follow-ups (PR #343) ────────────────────────────────────────────


@pytest.mark.parametrize("column", ["chunk_id", "text", "chunk_index"])
def test_a_generated_column_is_never_a_heading_column(column: str) -> None:
    """The upstream-column guard cannot catch these.

    An extraction model has no `chunk_id`, so `column: chunk_id` passed that
    check and then overwrote every generated chunk id with a section name —
    duplicate identifiers on a full materialization, failed key validation on
    an incremental one.
    """
    # `text` is additionally the default `text_field`, so it trips that guard
    # first — either refusal is correct, the point is that none of these can
    # silently overwrite a generated value.
    with pytest.raises(
        ValueError, match=r"chunk model generates|must not be the text field"
    ):
        ChunkConfig(headings={"pattern": "^x", "column": column})


def test_a_heading_above_the_bmp_attributes_correctly() -> None:
    """Attribution bisects offsets, not `(offset, name)` tuples.

    Comparing tuples needs a sentinel name sorting after every possible
    heading, and none exists: "\uffff" sorts before an astral character, so a
    chunk opening "🚀 Overview" landed in the previous section.
    """
    document = (
        "\U0001F680 Overview\n\n"
        + "Body text here. " * 6
        + "\n\n\U0001F30D Global\n\n"
        + "More body text. " * 6
    )

    chunks = split_text(
        document,
        ChunkConfig(
            chunk_size=110, chunk_overlap=0, headings={"pattern": r"^(\S+ \w+)$"}
        ),
    )

    opening = next(c for c in chunks if c.text.startswith("\U0001F30D"))
    assert opening.section == "\U0001F30D Global"
    assert chunks[0].section == "\U0001F680 Overview"


def test_a_heading_less_first_batch_does_not_fix_the_column_type(
    tmp_path: Path,
) -> None:
    """Explicit dtype, not one inferred from the first batch.

    A first document whose pattern matches nothing supplies only nulls, which
    polars infers as `Null` and DuckDB materializes as an integer column — so
    the next document that *does* find a heading fails converting a string
    into it. Same failure mode as the append-only logs in #333.
    """
    from stel.runner import run_project

    project = _heading_project(tmp_path)
    _write_document(project, "a", "Plain prose with no heading at all. " * 6)
    run_project(project)

    _write_document(
        project, "b", "Item 1. Business\n\n" + "Real content here. " * 6
    )
    run_project(project)

    con = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        dtype = con.execute(
            "SELECT section FROM main.filing_chunks LIMIT 0"
        ).description[0][1]
        sections = {
            row[0]
            for row in con.execute(
                "SELECT DISTINCT section FROM main.filing_chunks"
            ).fetchall()
        }
    finally:
        con.close()
    assert dtype == "VARCHAR"
    assert sections == {None, "Item 1"}


def _heading_project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    (project / "models").mkdir(parents=True)
    (project / "sources").mkdir()
    (project / "data" / "docs").mkdir(parents=True)
    (project / "stel_project.yml").write_text(
        "name: ns\nversion: '0.1.0'\nprofile: ns\n"
    )
    (project / "profiles.yml").write_text(
        "ns:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: main\n"
    )
    (project / "sources" / "s.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data/docs\n"
        "    file_pattern: '*.json'\n"
    )
    (project / "models" / "m.yml").write_text(
        "version: 2\nmodels:\n  - name: filings\n    source: ref('docs')\n"
        "    extraction:\n      backend: json\n      options:\n"
        "        fields: [doc_id, body]\n    materialization: incremental\n"
        "  - name: filing_chunks\n    depends_on: [ref('filings')]\n"
        "    chunk:\n      strategy: recursive\n      text_field: body\n"
        "      chunk_size: 120\n      chunk_overlap: 0\n"
        "      headings:\n        pattern: '^(Item\s+\d+)[.:]'\n"
        "    materialization: incremental\n"
    )
    return project


def _write_document(project: Path, name: str, body: str) -> None:
    import json

    (project / "data" / "docs" / f"{name}.json").write_text(
        json.dumps({"doc_id": name, "body": body})
    )


# ─── bounded input reads (issue #423) ───────────────────────────────────────


def test_streaming_the_input_preserves_chunk_input_hashes(tmp_path: Path) -> None:
    """The upgrade hazard #423 has to clear before it is worth shipping.

    `chunk_input_hash` is computed from the upstream record and incremental
    state compares against it. If reading the registry in Arrow batches
    produced even subtly different Python values than reading it whole — a
    widened dtype, a shifted datetime unit — every existing corpus would
    silently re-chunk, and everything downstream of it would re-embed at full
    provider cost. That failure raises nothing, so it is pinned directly rather
    than left to the end-to-end tests, which only ever exercise one read path.
    """
    from stel.adapters.duckdb import DuckDBAdapter, DuckDBWarehouseConfig
    from stel.execution.chunk import chunk_input_hash

    config = DuckDBWarehouseConfig(path=tmp_path / "w.duckdb", schema_name="main")
    with DuckDBAdapter(config) as adapter:
        adapter.materialize_full(
            "registry",
            pl.DataFrame(
                {
                    "document_id": ["a", "b", "c"],
                    "text": ["one", "two", None],
                    "count": [1, 2, 3],
                    "ratio": [1.5, 2.5, None],
                    "flag": [True, False, True],
                    "seen_at": [datetime(2024, 1, 1, 12, 30, tzinfo=UTC)] * 3,
                    "tags": [["x", "y"], ["z"], []],
                }
            ),
        )
        whole = adapter.read_table("registry")
        # The dtypes the chunk contract now reads from a zero-row probe must
        # survive it, or a carried column lands in the target as the wrong type.
        assert dict(adapter.read_table("registry", limit=0).schema) == dict(
            whole.schema
        )

        read_whole = {
            str(row["document_id"]): chunk_input_hash(row, text_field="text")
            for row in whole.iter_rows(named=True)
        }
        streamed: dict[str, str] = {}
        with adapter.table_snapshot("registry", batch_size=2) as snapshot:
            for batch in snapshot:
                frame = pl.from_arrow(batch)
                assert isinstance(frame, pl.DataFrame)
                for row in frame.iter_rows(named=True):
                    streamed[str(row["document_id"])] = chunk_input_hash(
                        row, text_field="text"
                    )

    assert read_whole  # not vacuous
    assert streamed == read_whole


def test_chunk_never_reads_the_whole_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asserted where it is observable: the upstream may be touched only by a
    zero-row schema probe and by streamed snapshot reads."""
    from stel.adapters.duckdb import DuckDBAdapter
    from stel.runner import run_project

    project = _chunk_project(tmp_path)
    _write_doc(project, "a.json", "Doc A", ". ".join(f"s {i}" for i in range(30)))
    _write_doc(project, "b.json", "Doc B", "short body")
    run_project(project, select="document_registry")

    full_reads: list[str] = []
    original = DuckDBAdapter.read_table

    def spy(self: Any, table: str, *, limit: int | None = None) -> Any:
        if table == "document_registry" and limit != 0:
            full_reads.append(table)
        return original(self, table, limit=limit)

    monkeypatch.setattr(DuckDBAdapter, "read_table", spy)
    run_project(project, select="document_chunks")
    assert full_reads == []


def _set_chunk_flush_every(project: Path, value: int) -> None:
    path = project / "models" / "chunks.yml"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("    chunk:\n", f"    chunk:\n      flush_every: {value}\n"),
        encoding="utf-8",
    )


def test_chunk_publishes_at_window_boundaries(tmp_path: Path) -> None:
    """Chunk amplifies — one registry row becomes many chunk rows — so holding
    the output until the end is the larger of the two O(corpus) problems. A
    failure partway through must leave the earlier windows published."""
    from stel.runner import run_project

    project = _chunk_project(tmp_path, chunk_size=40)
    _set_chunk_flush_every(project, 1)
    _write_doc(project, "a.json", "Doc A", ". ".join(f"s {i}" for i in range(30)))
    _write_doc(project, "b.json", "Doc B", "another body here")
    run_project(project, select="document_registry")

    published: list[int] = []
    from stel.adapters.duckdb import DuckDBAdapter

    original = DuckDBAdapter.replace_children

    def spy(self: Any, table: str, **kwargs: Any) -> Any:
        written = original(self, table, **kwargs)
        published.append(written)
        return written

    DuckDBAdapter.replace_children = spy  # type: ignore[method-assign]
    try:
        run_project(project, select="document_chunks")
    finally:
        DuckDBAdapter.replace_children = original  # type: ignore[method-assign]

    # One publication per document, not one for the whole run.
    assert len(published) > 1


def test_chunk_flush_every_does_not_move_code_version(tmp_path: Path) -> None:
    """A publication cadence must not invalidate state: changing it would
    re-chunk and re-embed a corpus for an execution-only setting."""
    from stel.config.model import ChunkConfig
    from stel.versioning import compute_code_version

    def version(flush_every: int) -> str:
        return compute_code_version(
            extraction=None,
            transform=None,
            chunk=ChunkConfig(flush_every=flush_every),
            depends_on=["registry"],
            project_dir=tmp_path,
        )

    assert version(5000) == version(7)
