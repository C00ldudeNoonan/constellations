from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest
from click.testing import CliRunner

from stel.cli import cli
from stel.compiler import validate_project_contract
from stel.config import ConfigError, load_project
from stel.config.model import EmbedConfig
from stel.embedding import EmbeddingIdentity, embed_query
from stel.manifest import build_manifest
from stel.providers.deterministic import DeterministicEmbeddingProvider
from stel.runner import RunError, run_project
from stel.versioning import compute_model_code_version


def _embedding_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "stel_project.yml").write_text(
        "name: embeddings\nversion: '0.1.0'\nprofile: embeddings\n"
    )
    (project / "profiles.yml").write_text(
        "embeddings:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: docs\n"
    )
    (project / "sources").mkdir()
    (project / "sources" / "documents.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data\n    file_pattern: '*.json'\n"
    )
    (project / "models").mkdir()
    (project / "models" / "documents.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: document_registry\n"
        "    source: ref('documents')\n"
        "    extraction:\n      backend: json\n"
        "      options:\n        fields: [title, body, tenant]\n"
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
        "    materialization: incremental\n"
    )
    data = project / "data"
    data.mkdir()
    for name, title, body in (
        ("a.json", "Release A", "employment increased"),
        ("b.json", "Release B", "inflation moderated"),
    ):
        (data / name).write_text(
            json.dumps(
                {
                    "title": title,
                    "body": body,
                    "tenant": "economic-data-project",
                }
            )
        )
    return project


def _query(project: Path, sql: str, params: list[object] | None = None) -> list[tuple[Any, ...]]:
    connection = duckdb.connect(
        str(project / "target" / "db.duckdb"),
        read_only=params is None,
    )
    try:
        return connection.execute(sql, params or []).fetchall()
    finally:
        connection.close()


def test_embed_config_validates_execution_and_canonical_fields() -> None:
    config = EmbedConfig(
        provider="deterministic",
        model="contract-v1",
        dimensions=4,
    )
    assert config.batch_size == 128
    assert config.vector_field == "embedding"

    with pytest.raises(ValueError, match="must be distinct"):
        EmbedConfig(
            provider="deterministic",
            model="contract-v1",
            dimensions=4,
            text_field="chunk_id",
        )
    with pytest.raises(ValueError, match="less than or equal to 10000"):
        EmbedConfig(
            provider="deterministic",
            model="contract-v1",
            dimensions=4,
            batch_size=10_001,
        )


def test_deterministic_provider_and_query_helper_are_reproducible() -> None:
    config = EmbedConfig(
        provider="deterministic",
        model="contract-v1",
        dimensions=4,
    )
    identity = EmbeddingIdentity.from_config(config)

    first = embed_query("economic release", identity)
    second = embed_query("economic release", identity.to_dict())

    assert first == second
    assert len(first) == 4
    assert sum(value * value for value in first) == pytest.approx(1.0)

    tampered = identity.to_dict()
    tampered["model"] = "different-model"
    with pytest.raises(ValueError, match="config_hash"):
        embed_query("economic release", tampered)


def test_embed_code_version_excludes_execution_tuning(tmp_path: Path) -> None:
    project_dir = _embedding_project(tmp_path)
    project, _, models = load_project(project_dir)
    model = next(item for item in models if item.name == "document_embeddings")
    assert model.embed is not None
    baseline = compute_model_code_version(model, project, project_dir)
    tuned = model.model_copy(
        update={
            "embed": model.embed.model_copy(
                update={"batch_size": 32, "max_retries": 9}
            )
        }
    )
    changed = model.model_copy(
        update={"embed": model.embed.model_copy(update={"model": "contract-v2"})}
    )

    assert compute_model_code_version(tuned, project, project_dir) == baseline
    assert compute_model_code_version(changed, project, project_dir) != baseline


def test_compiler_rejects_an_unregistered_embedding_provider(tmp_path: Path) -> None:
    project_dir = _embedding_project(tmp_path)
    model_path = project_dir / "models" / "documents.yml"
    model_path.write_text(
        model_path.read_text().replace(
            "provider: deterministic",
            "provider: unavailable",
        )
    )
    project, sources, models = load_project(project_dir)

    with pytest.raises(ConfigError, match=r"unavailable.*not registered"):
        validate_project_contract(project, sources, models, project_dir)


def test_embed_model_materializes_canonical_rows_and_artifacts(tmp_path: Path) -> None:
    project = _embedding_project(tmp_path)

    results = run_project(project)
    result = next(item for item in results if item.model_name == "document_embeddings")

    assert result.kind == "embed"
    assert result.provider == "deterministic"
    assert result.provider_model == "contract-v1"
    assert result.documents_processed == 2
    assert result.rows_written == 2
    assert result.metrics["provider_calls"] == 2
    assert result.metrics["cache_misses"] == 2
    rows = _query(
        project,
        "SELECT chunk_id, document_id, title, embedding, embedding_provider, "
        "embedding_model, embedding_dimensions, embedding_input_hash, "
        "embedding_config_hash, embedded_at "
        'FROM "db".docs.document_embeddings ORDER BY title',
    )
    assert len(rows) == 2
    assert all(len(row[3]) == 4 for row in rows)
    assert {row[4] for row in rows} == {"deterministic"}
    assert {row[5] for row in rows} == {"contract-v1"}
    assert {row[6] for row in rows} == {4}
    assert all(row[7] and row[8] and row[9] for row in rows)

    model = next(
        item for item in build_manifest(project)["models"] if item["name"] == "document_embeddings"
    )
    assert model["kind"] == "embed"
    assert result.artifact_metadata is not None
    assert model["embedding"] == result.artifact_metadata["embedding"]
    assert set(model["embedding"]) == {
        "provider",
        "model",
        "dimensions",
        "implementation",
        "config_hash",
    }
    listed = CliRunner().invoke(
        cli,
        ["--project-dir", str(project), "ls", "--output", "json"],
    )
    assert listed.exit_code == 0, listed.output
    listed_models = json.loads(listed.output)
    assert next(
        item for item in listed_models if item["name"] == "document_embeddings"
    )["kind"] == "embed"
    shown = CliRunner().invoke(
        cli,
        ["--project-dir", str(project), "show", "document_embeddings", "--limit", "1"],
    )
    assert shown.exit_code == 0, shown.output
    assert "shape: (1, 25)" in shown.output


def test_incremental_embed_reuses_vectors_for_metadata_only_updates(
    tmp_path: Path,
) -> None:
    project = _embedding_project(tmp_path)
    run_project(project)
    before = _query(
        project,
        'SELECT chunk_id, embedding, embedded_at FROM "db".docs.document_embeddings '
        "ORDER BY chunk_id",
    )

    _query(
        project,
        'UPDATE "db".docs.document_chunks SET title = ? WHERE title = ?',
        ["Reclassified release", "Release A"],
    )
    [result] = run_project(project, select="document_embeddings")

    assert result.documents_processed == 1
    assert result.documents_skipped == 1
    assert result.metrics["provider_calls"] == 0
    assert result.metrics["cache_hits"] == 1
    assert result.metrics["metadata_updates"] == 1
    after = _query(
        project,
        'SELECT chunk_id, embedding, embedded_at FROM "db".docs.document_embeddings '
        "ORDER BY chunk_id",
    )
    assert after == before
    assert _query(
        project,
        'SELECT title FROM "db".docs.document_embeddings WHERE title = ?',
        ["Reclassified release"],
    ) == [("Reclassified release",)]

    [unchanged] = run_project(project, select="document_embeddings")
    assert unchanged.documents_processed == 0
    assert unchanged.documents_skipped == 2


def test_incremental_embed_recomputes_text_and_removes_deleted_rows(
    tmp_path: Path,
) -> None:
    project = _embedding_project(tmp_path)
    run_project(project)
    chunk_id, old_vector = _query(
        project,
        "SELECT chunk_id, embedding FROM \"db\".docs.document_embeddings WHERE title = 'Release A'",
    )[0]

    _query(
        project,
        'UPDATE "db".docs.document_chunks SET text = ? WHERE chunk_id = ?',
        ["employment declined", chunk_id],
    )
    [updated] = run_project(project, select="document_embeddings")
    assert updated.documents_processed == 1
    assert updated.metrics["provider_calls"] == 1
    assert updated.metrics["cache_hits"] == 0
    new_vector = _query(
        project,
        'SELECT embedding FROM "db".docs.document_embeddings WHERE chunk_id = ?',
        [chunk_id],
    )[0][0]
    assert new_vector != old_vector

    _query(
        project,
        'DELETE FROM "db".docs.document_chunks WHERE chunk_id = ?',
        [chunk_id],
    )
    [deleted] = run_project(project, select="document_embeddings")
    assert deleted.documents_deleted == 1
    assert deleted.documents_processed == 0
    assert _query(
        project,
        'SELECT COUNT(*) FROM "db".docs.document_embeddings WHERE chunk_id = ?',
        [chunk_id],
    ) == [(0,)]


def test_embed_rejects_generated_columns_case_insensitively(tmp_path: Path) -> None:
    project = _embedding_project(tmp_path)
    run_project(project)
    _query(
        project,
        'ALTER TABLE "db".docs.document_chunks ADD COLUMN "Embedding_Model" VARCHAR',
        [],
    )

    with pytest.raises(RunError, match="Embedding_Model"):
        run_project(project, select="document_embeddings")


def test_incremental_embed_deletes_rows_with_typed_keys(tmp_path: Path) -> None:
    project = _embedding_project(tmp_path)
    run_project(project)
    _query(
        project,
        'ALTER TABLE "db".docs.document_chunks ADD COLUMN numeric_id BIGINT',
        [],
    )
    _query(
        project,
        'UPDATE "db".docs.document_chunks SET numeric_id = '
        "CASE WHEN title = 'Release A' THEN 1 ELSE 2 END",
        [],
    )
    model_path = project / "models" / "documents.yml"
    model_path.write_text(
        model_path.read_text().replace(
            "      id_field: chunk_id\n",
            "      id_field: numeric_id\n",
        )
    )
    run_project(project, select="document_embeddings", full_refresh=True)
    _query(
        project,
        'DELETE FROM "db".docs.document_chunks WHERE numeric_id = ?',
        [1],
    )

    [result] = run_project(project, select="document_embeddings")

    assert result.documents_deleted == 1
    assert _query(
        project,
        'SELECT numeric_id FROM "db".docs.document_embeddings ORDER BY numeric_id',
    ) == [(2,)]


def test_incremental_embed_handles_an_empty_upstream_relation(tmp_path: Path) -> None:
    project = _embedding_project(tmp_path)
    run_project(project)
    _query(project, 'DELETE FROM "db".docs.document_chunks', [])

    [result] = run_project(project, select="document_embeddings")

    assert result.documents_deleted == 2
    assert result.documents_processed == 0
    assert result.metrics["provider_calls"] == 0
    columns = _query(
        project,
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'docs' AND table_name = 'document_embeddings'",
    )
    assert ("embedding",) in columns
    assert _query(
        project,
        'SELECT COUNT(*) FROM "db".docs.document_embeddings',
    ) == [(0,)]


def test_provider_failure_does_not_publish_rows_or_advance_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _embedding_project(tmp_path)
    run_project(project)
    chunk_id, old_vector = _query(
        project,
        "SELECT chunk_id, embedding FROM \"db\".docs.document_embeddings WHERE title = 'Release A'",
    )[0]
    _query(
        project,
        'UPDATE "db".docs.document_chunks SET text = ? WHERE chunk_id = ?',
        ["revised release", chunk_id],
    )

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("unsafe provider payload")

    monkeypatch.setattr(DeterministicEmbeddingProvider, "_embed", fail)
    with pytest.raises(RunError, match="ProviderRequestError"):
        run_project(project, select="document_embeddings")

    assert _query(
        project,
        'SELECT embedding FROM "db".docs.document_embeddings WHERE chunk_id = ?',
        [chunk_id],
    ) == [(old_vector,)]
    monkeypatch.undo()

    [recovered] = run_project(project, select="document_embeddings")
    assert recovered.documents_processed == 1
    assert recovered.metrics["provider_calls"] == 1
    assert (
        _query(
            project,
            'SELECT embedding FROM "db".docs.document_embeddings WHERE chunk_id = ?',
            [chunk_id],
        )[0][0]
        != old_vector
    )


# ─── removal detection is an anti-join, not a set difference (issue #428) ───


class _CountingPageReader:
    """Delegates to a real state page reader, recording page sizes.

    A wrapper rather than a patched method: the reader's own `fetch_page`
    validates ordering and page bounds, and this has to sit outside that, not
    replace it.
    """

    def __init__(self, reader: Any, surfaced: list[int]) -> None:
        self._reader = reader
        self._surfaced = surfaced

    def fetch_page(self, cursor: str | None = None) -> Any:
        page = self._reader.fetch_page(cursor)
        self._surfaced.append(len(page.records))
        return page

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reader, name)


def _count_anti_join_keys(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record how many state keys the anti-join actually surfaces to Python.

    Only the probing reader is counted: embed opens others for ordinary state
    work, and counting those would measure the wrong thing.
    """
    from contextlib import contextmanager

    from stel.adapters.duckdb import DuckDBAdapter

    surfaced: list[int] = []
    original = DuckDBAdapter.state_page_reader

    def counting_reader(self: Any, scope: Any, **kwargs: Any) -> Any:
        manager = original(self, scope, **kwargs)
        if kwargs.get("absent_from") is None:
            return manager

        @contextmanager
        def wrapper() -> Any:
            with manager as reader:
                yield _CountingPageReader(reader, surfaced)

        return wrapper()

    monkeypatch.setattr(DuckDBAdapter, "state_page_reader", counting_reader)
    return surfaced


def test_removal_detection_surfaces_only_the_removed_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measurement #428 asked for, and the one `test_bounded_memory.py`
    structurally cannot make: it records the largest single frame, so it
    cannot see a container that accumulates across batches.

    This measures the cumulative thing directly. Removal detection used to
    build a `set[str]` of every upstream id and subtract it from every state
    key -- both key domains in Python, however little had changed. The
    warehouse evaluates the anti-join now, so what reaches Python is only the
    keys to delete. Asserting *that* is what catches a regression back to a
    set: a reintroduced set difference still deletes the right row, so a
    correctness assertion alone would pass.
    """
    project = _embedding_project(tmp_path)
    run_project(project)
    surfaced = _count_anti_join_keys(monkeypatch)
    # Two documents embedded; remove exactly one of them.
    _query(
        project, 'DELETE FROM "db".docs.document_chunks WHERE title = ?', ["Release A"]
    )

    [result] = run_project(project, select="document_embeddings")

    assert result.documents_deleted == 1
    # The engine filtered: one key crossed into Python, not the whole domain.
    assert surfaced, "removal detection did not use the paged anti-join"
    assert sum(surfaced) == 1


def test_removal_detection_pages_nothing_when_nothing_was_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common case, and the one a set difference paid full price for: an
    unchanged corpus should surface no keys at all."""
    project = _embedding_project(tmp_path)
    run_project(project)
    surfaced = _count_anti_join_keys(monkeypatch)

    [result] = run_project(project, select="document_embeddings")

    assert result.documents_deleted == 0
    # One page was fetched and it was empty -- the anti-join ran and found
    # nothing, rather than the anti-join not running at all.
    assert surfaced == [0]


def test_a_boolean_id_reconciles_in_python_not_in_the_warehouse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure mode is deletion, not a slow query (PR #457 review).

    State keys are written as `str(value)`, so a boolean id is stored as
    `True`; DuckDB and BigQuery both cast booleans to `true`. Pushed into the
    warehouse, the anti-join matches nothing, calls every unchanged row
    absent, and the delete pass removes the whole target. So a column whose
    cast cannot be proven identical reconciles in Python instead, where both
    sides go through the same `str()` that wrote the key.
    """
    project = _embedding_project(tmp_path)
    run_project(project)
    _query(project, 'ALTER TABLE "db".docs.document_chunks ADD COLUMN flag BOOLEAN', [])
    _query(
        project,
        'UPDATE "db".docs.document_chunks SET flag = (title = ?)',
        ["Release A"],
    )
    model_path = project / "models" / "documents.yml"
    model_path.write_text(
        model_path.read_text().replace(
            "      id_field: chunk_id\n", "      id_field: flag\n"
        )
    )
    run_project(project, select="document_embeddings", full_refresh=True)
    surfaced = _count_anti_join_keys(monkeypatch)

    # Nothing removed upstream, so nothing may be deleted.
    [result] = run_project(project, select="document_embeddings")

    assert result.documents_deleted == 0
    # And it did not ask the warehouse: the cast would not have matched.
    assert surfaced == []
    assert (
        _query(project, 'SELECT count(*) FROM "db".docs.document_embeddings')[0][0] == 2
    )


def test_removing_the_whole_corpus_deletes_page_by_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Residency has to follow the page, not the removal count (PR #457
    review). Emptying the upstream removes every row, and accumulating those
    keys would restore the O(corpus) term this change exists to remove -- so
    each page is deleted as it arrives.

    The page size is shrunk to 1 rather than building a corpus larger than
    the real one: what is under test is that deletion is driven per page, and
    a run that deletes N rows in one call looks identical at any page size
    unless the pages are small enough to count.
    """
    from stel.adapters.duckdb import DuckDBAdapter
    from stel.execution import embed as embed_module

    project = _embedding_project(tmp_path)
    run_project(project)
    monkeypatch.setattr(embed_module, "_REMOVAL_PAGE_ROWS", 1)
    delete_batches: list[int] = []
    original_delete = DuckDBAdapter.delete_rows_and_state

    def counting_delete(
        self: Any, table: str, *args: Any, **kwargs: Any
    ) -> Any:
        keys = kwargs.get("state_record_keys") or kwargs.get("keys") or []
        delete_batches.append(len(keys))
        return original_delete(self, table, *args, **kwargs)

    monkeypatch.setattr(DuckDBAdapter, "delete_rows_and_state", counting_delete)
    _query(project, 'DELETE FROM "db".docs.document_chunks', [])

    [result] = run_project(project, select="document_embeddings")

    assert result.documents_deleted == 2
    # Two single-key deletes, not one delete carrying both keys.
    assert delete_batches == [1, 1]
    assert (
        _query(project, 'SELECT count(*) FROM "db".docs.document_embeddings')[0][0] == 0
    )


# ─── wall-clock attribution (issue #432 item 1) ─────────────────────────────


def test_an_embed_run_attributes_its_phases(tmp_path: Path) -> None:
    """#432 opens by asking where an embed run's wall clock goes — provider
    wait, warehouse write, the read — and says everything it proposes after
    that is a guess about which term dominates. Nothing in a run said so."""
    project = _embedding_project(tmp_path)

    results = run_project(project)

    [embed] = [r for r in results if r.kind == "embed"]
    for phase in ("seconds_provider", "seconds_publish", "seconds_read"):
        assert phase in embed.metrics, embed.metrics
        assert embed.metrics[phase] >= 0.0


def test_phase_totals_reach_run_results_json(tmp_path: Path) -> None:
    """The deliverable is a number an operator reads after a production run,
    not one only a test can see."""
    from stel.manifest import write_run_results

    project = _embedding_project(tmp_path)
    results = run_project(project)

    path = write_run_results(project, results)
    payload = json.loads(path.read_text(encoding="utf-8"))

    row = next(
        r for r in payload["results"] if r["model_name"] == "document_embeddings"
    )
    assert "seconds_provider" in row["metrics"]
    # Beside the wall clock: summed phase time can exceed it once provider
    # batches overlap, and that ratio is the concurrency actually achieved.
    assert "duration_seconds" in row
