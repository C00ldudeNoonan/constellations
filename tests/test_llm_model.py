from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest

from stel.manifest import build_manifest
from stel.runner import RunError, run_project

_PROJECT_YML = "name: llmmodels\nversion: '0.1.0'\nprofile: llmmodels\n"
_PROFILES_YML = (
    "llmmodels:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
    "        type: duckdb\n        path: ./target/db.duckdb\n        schema: docs\n"
)
_FACTS = '"db".docs.document_facts'


def _write_models(project: Path, *, cardinality: str = "one") -> None:
    (project / "models").mkdir(exist_ok=True)
    fan = "      output_cardinality: many\n" if cardinality == "many" else ""
    (project / "models" / "models.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: document_registry\n"
        "    source: ref('documents')\n"
        "    extraction:\n      backend: json\n"
        "      options:\n        fields: [title, body]\n"
        "    materialization: incremental\n"
        "  - name: document_facts\n"
        "    depends_on: [ref('document_registry')]\n"
        "    llm:\n      input_field: body\n      id_field: document_id\n"
        "      prompt: 'Extract the key fact.'\n"
        "      provider: deterministic\n      model: deterministic-v1\n"
        f"{fan}"
        "    fields:\n"
        "      - {name: sentiment, type: string}\n"
        "      - {name: score, type: integer}\n"
        "    materialization: incremental\n"
    )


def _seed(project: Path, docs: dict[str, str]) -> None:
    data = project / "data"
    data.mkdir(exist_ok=True)
    for name, body in docs.items():
        (data / f"{name}.json").write_text(json.dumps({"title": name, "body": body}))


def _project(
    tmp_path: Path,
    *,
    cardinality: str = "one",
    docs: dict[str, str] | None = None,
) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "stel_project.yml").write_text(_PROJECT_YML)
    (project / "profiles.yml").write_text(_PROFILES_YML)
    (project / "sources").mkdir()
    (project / "sources" / "documents.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data\n"
        "    file_pattern: '*.json'\n"
    )
    _write_models(project, cardinality=cardinality)
    _seed(project, docs or {"a": "employment rose", "b": "inflation cooled"})
    return project


def _rows(project: Path, sql: str) -> list[tuple[Any, ...]]:
    connection = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.close()


def _columns(project: Path) -> dict[str, str]:
    rows = _rows(
        project,
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'docs' AND table_name = 'document_facts'",
    )
    return {name: dtype for name, dtype in rows}


def test_llm_model_materializes_typed_rows_and_metadata(tmp_path: Path) -> None:
    project = _project(tmp_path)
    results = run_project(project)
    result = next(r for r in results if r.model_name == "document_facts")
    assert result.kind == "llm"
    assert result.provider == "deterministic"
    assert result.rows_written == 2
    assert result.metrics["rows_generated"] == 2
    assert result.metrics["provider_calls"] == 2

    dtypes = _columns(project)
    columns = set(dtypes)
    assert {"document_id", "sentiment", "score"} <= columns
    assert {
        "llm_provider",
        "llm_model",
        "llm_provider_implementation",
        "llm_input_hash",
        "llm_config_hash",
        "generated_at",
    } <= columns
    # score is an integer column, not a string.
    assert "INT" in dtypes["score"].upper()
    provider = _rows(project, f"SELECT DISTINCT llm_provider FROM {_FACTS}")
    assert provider == [("deterministic",)]


def test_llm_model_is_deterministic_and_incremental(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)
    before = _rows(
        project,
        f"SELECT document_id, sentiment, score FROM {_FACTS} ORDER BY document_id",
    )

    # Re-run with no input change: every input skipped, output identical.
    second = run_project(project)
    result = next(r for r in second if r.model_name == "document_facts")
    assert result.documents_skipped == 2
    assert result.documents_processed == 0
    after = _rows(
        project,
        f"SELECT document_id, sentiment, score FROM {_FACTS} ORDER BY document_id",
    )
    assert before == after


def test_llm_model_reprocesses_changed_and_deletes_removed(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)
    original = dict(_rows(project, f"SELECT document_id, score FROM {_FACTS}"))

    # Change one document's body, remove another entirely.
    (project / "data" / "a.json").write_text(
        json.dumps({"title": "a", "body": "employment surged sharply"})
    )
    (project / "data" / "b.json").unlink()
    _seed(project, {"c": "new release"})

    results = run_project(project)
    result = next(r for r in results if r.model_name == "document_facts")
    assert result.documents_deleted == 1  # b removed

    surviving = dict(_rows(project, f"SELECT document_id, score FROM {_FACTS}"))
    assert len(surviving) == 2  # a and c
    changed_a = [
        doc_id
        for doc_id, score in surviving.items()
        if doc_id in original and score != original[doc_id]
    ]
    assert changed_a  # a regenerated with new content


def test_llm_model_fan_out_many(tmp_path: Path) -> None:
    project = _project(tmp_path, cardinality="many")
    results = run_project(project)
    result = next(r for r in results if r.model_name == "document_facts")
    # Deterministic provider fans out 2 items per input row.
    assert result.rows_written == 4
    assert {"llm_row_id", "ordinal"} <= set(_columns(project))
    row_ids = {row[0] for row in _rows(project, f"SELECT llm_row_id FROM {_FACTS}")}
    assert len(row_ids) == 4  # unique deterministic fan-out ids
    per_parent = _rows(
        project,
        f"SELECT document_id, COUNT(*) FROM {_FACTS} GROUP BY document_id",
    )
    assert all(count == 2 for _doc, count in per_parent)


def test_llm_model_fan_out_incremental_replaces_parent_rows(tmp_path: Path) -> None:
    project = _project(tmp_path, cardinality="many")
    run_project(project)
    assert _rows(project, f"SELECT COUNT(*) FROM {_FACTS}") == [(4,)]

    # Changing a body regenerates exactly that parent's fan-out rows (still 2),
    # leaving the total unchanged and no orphaned ordinals.
    (project / "data" / "a.json").write_text(
        json.dumps({"title": "a", "body": "completely different content"})
    )
    run_project(project)
    assert _rows(project, f"SELECT COUNT(*) FROM {_FACTS}") == [(4,)]
    per_parent = _rows(
        project,
        f"SELECT document_id, COUNT(*) FROM {_FACTS} GROUP BY document_id",
    )
    assert all(count == 2 for _doc, count in per_parent)


def test_llm_model_full_refresh_rebuilds(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)
    results = run_project(project, full_refresh=True)
    result = next(r for r in results if r.model_name == "document_facts")
    assert result.documents_processed == 2
    assert result.documents_skipped == 0
    assert _rows(project, f"SELECT COUNT(*) FROM {_FACTS}") == [(2,)]


def test_llm_model_manifest_records_kind_and_identity(tmp_path: Path) -> None:
    project = _project(tmp_path)
    manifest = build_manifest(project)
    node = next(
        model for model in manifest["models"] if model["name"] == "document_facts"
    )
    assert node["kind"] == "llm"
    assert node["llm"]["input_field"] == "body"
    assert node["llm"]["output_cardinality"] == "one"
    assert node["llm_identity"]["provider"] == "deterministic"
    assert node["llm_identity"]["config_hash"]
    # No secrets or api key references anywhere in the node.
    assert "api_key" not in json.dumps(node)


def test_llm_model_run_budget_exceeded_returns_status(tmp_path: Path) -> None:
    project = _project(tmp_path)
    # A run-scope budget the 2-chunk llm model exceeds when it charges its work.
    (project / "profiles.yml").write_text(
        _PROFILES_YML.replace(
            "        schema: docs\n",
            "        schema: docs\n"
            "      llm:\n"
            "        provider: deterministic\n"
            "        model: deterministic-v1\n"
            "        budget:\n          max_documents: 1\n",
        )
    )
    results = run_project(project)
    result = next(r for r in results if r.model_name == "document_facts")
    # Budget exhaustion becomes a model result, not an escaped exception.
    assert result.status == "budget_exceeded"
    assert result.errors
    assert result.rows_written == 0
    # Nothing published: the table was never created (this model writes once).
    tables = _rows(
        project,
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'docs' AND table_name = 'document_facts'",
    )
    assert tables == []


def test_llm_model_missing_upstream_column(tmp_path: Path) -> None:
    project = _project(tmp_path)
    # Drop `body` from the extraction so the llm input_field is absent.
    (project / "models" / "models.yml").write_text(
        (project / "models" / "models.yml")
        .read_text()
        .replace("fields: [title, body]", "fields: [title]")
    )
    with pytest.raises(RunError, match="missing required column"):
        run_project(project)
