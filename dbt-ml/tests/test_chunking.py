"""Chunk model kind (issue #86): splitter behaviour, deterministic IDs, and
the registry → chunks contract end to end through the runner."""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import duckdb
import pytest

from dbt_ml.chunking import chunk_id, split_text
from dbt_ml.config.model import ChunkConfig

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
    chunks = split_text(
        text, ChunkConfig(strategy="tokens", chunk_size=50, chunk_overlap=10)
    )
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
    text = "\n\n".join(
        f"Paragraph {i} has several words in it." for i in range(15)
    )
    chunks = split_text(text, ChunkConfig(chunk_size=100, chunk_overlap=0))
    source_chars = set(text)
    for chunk in chunks:
        assert set(chunk.text) <= source_chars


# ─── chunk ids ──────────────────────────────────────────────────────────────


def test_chunk_id_is_content_addressed() -> None:
    assert chunk_id("doc", 0, "hello") == chunk_id("doc", 0, "hello")
    assert chunk_id("doc", 0, "hello") != chunk_id("doc", 1, "hello")
    assert chunk_id("doc", 0, "hello") != chunk_id("doc", 0, "world")
    assert chunk_id("doc", 0, "hello") != chunk_id("other", 0, "hello")


def test_chunk_config_validation() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ChunkConfig(chunk_size=0)
    with pytest.raises(ValueError, match="overlap must be smaller"):
        ChunkConfig(chunk_size=100, chunk_overlap=100)


# ─── end to end: registry → chunks ──────────────────────────────────────────


def _chunk_project(tmp_path: Path, *, chunk_size: int = 120) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "dbt_ml_project.yml").write_text(
        "name: docs\nversion: '0.1.0'\nprofile: docs\n"
    )
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
        "      options:\n        fields: [title, body]\n"
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


def _write_doc(project: Path, name: str, title: str, body: str) -> None:
    import json

    (project / "data" / "docs" / name).write_text(
        json.dumps({"title": title, "body": body})
    )


def _chunks(project: Path) -> list[tuple]:
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
    from dbt_ml.runner import run_project

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
    from dbt_ml.runner import run_project

    project = _chunk_project(tmp_path)
    _write_doc(project, "a.json", "Doc A", ". ".join(f"sentence {i}" for i in range(30)))
    run_project(project)
    first = {r[0] for r in _chunks(project)}

    results = run_project(project)
    # unchanged upstream → chunk model skips the document
    assert next(r for r in results if r.model_name == "document_chunks").documents_skipped == 1
    assert {r[0] for r in _chunks(project)} == first


def test_changed_document_rechunks_without_orphans(tmp_path: Path) -> None:
    from dbt_ml.runner import run_project

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


def test_deleted_document_prunes_its_chunks(tmp_path: Path) -> None:
    from dbt_ml.runner import run_project

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
    from dbt_ml.runner import run_project

    project = _chunk_project(tmp_path)
    # switch the chunk model to full materialization
    chunks_yml = project / "models" / "chunks.yml"
    chunks_yml.write_text(
        chunks_yml.read_text().replace(
            "materialization: incremental", "materialization: full"
        )
    )
    _write_doc(project, "a.json", "Doc A", ". ".join(f"sentence {i}" for i in range(20)))
    run_project(project)
    assert len(_chunks(project)) > 1


def test_chunk_model_requires_text_field(tmp_path: Path) -> None:
    from dbt_ml.runner import RunError, run_project

    project = _chunk_project(tmp_path)
    chunks_yml = project / "models" / "chunks.yml"
    chunks_yml.write_text(chunks_yml.read_text().replace("text_field: body", "text_field: nope"))
    _write_doc(project, "a.json", "Doc A", "some body text here")
    with pytest.raises(RunError, match="no column 'nope'"):
        run_project(project)


def test_chunk_dag_orders_after_extraction(tmp_path: Path) -> None:
    from dbt_ml.config import load_project
    from dbt_ml.dag import ProjectDAG

    project = _chunk_project(tmp_path)
    _, sources, models = load_project(project)
    dag = ProjectDAG(sources, models)
    order = dag.execution_order()
    assert order.index("document_registry") < order.index("document_chunks")


def test_ls_reports_chunk_kind() -> None:
    """`dbt-ml ls` must show chunk models as `chunk`, not `unknown`."""
    from dbt_ml.cli import _model_kind
    from dbt_ml.config.model import ChunkConfig, ModelConfig

    model = ModelConfig(
        name="document_chunks",
        depends_on=["ref('document_registry')"],
        chunk=ChunkConfig(),
    )
    assert _model_kind(model) == "chunk"
