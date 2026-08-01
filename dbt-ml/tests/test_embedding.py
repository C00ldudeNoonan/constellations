from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest
from click.testing import CliRunner

from dbt_ml.cli import cli
from dbt_ml.compiler import validate_project_contract
from dbt_ml.config import ConfigError, load_project
from dbt_ml.config.model import EmbedConfig
from dbt_ml.embedding import EmbeddingIdentity, embed_query
from dbt_ml.manifest import build_manifest
from dbt_ml.providers.deterministic import DeterministicEmbeddingProvider
from dbt_ml.runner import RunError, run_project
from dbt_ml.versioning import compute_model_code_version


def _embedding_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "dbt_ml_project.yml").write_text(
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
