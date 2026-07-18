from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from click.testing import CliRunner

from dbt_ml.cli import cli
from dbt_ml.config import SearchConfig, load_project
from dbt_ml.manifest import build_manifest
from dbt_ml.profile import resolve_profile
from dbt_ml.retrieval import create_store
from dbt_ml.runner import run_project
from dbt_ml.search import (
    SearchError,
    SearchFilter,
    SearchFilterOperator,
    SearchMode,
    SearchRequest,
    _rank_table,
    search,
)


def _write_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "dbt_ml_project.yml").write_text(
        "name: economic_search\nversion: '0.1.0'\nprofile: economic_search\n"
    )
    (project / "profiles.yml").write_text(
        "economic_search:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        "        path: target/data.duckdb\n"
        "        schema: analytics\n"
        "      retrieval:\n"
        "        default: local\n"
        "        allow_public_indexes: true\n"
        "        stores:\n"
        "          local:\n"
        "            type: lancedb\n"
        "            path: target/lancedb\n"
    )
    (project / "sources").mkdir()
    (project / "sources" / "documents.yml").write_text(
        "version: 2\n"
        "sources:\n"
        "  - name: releases\n"
        "    path: data\n"
        "    file_pattern: '*.json'\n"
    )
    (project / "models").mkdir()
    (project / "models" / "search.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: release_documents\n"
        "    source: ref('releases')\n"
        "    extraction:\n"
        "      backend: json\n"
        "      options:\n"
        "        fields: [title, body, category]\n"
        "    materialization: incremental\n"
        "  - name: release_chunks\n"
        "    depends_on: [ref('release_documents')]\n"
        "    chunk:\n"
        "      text_field: body\n"
        "      chunk_size: 1000\n"
        "      chunk_overlap: 0\n"
        "    materialization: incremental\n"
        "  - name: release_embeddings\n"
        "    depends_on: [ref('release_chunks')]\n"
        "    embed:\n"
        "      provider: deterministic\n"
        "      model: economic-search-v1\n"
        "      text_field: text\n"
        "      id_field: chunk_id\n"
        "      vector_field: embedding\n"
        "      dimensions: 8\n"
        "    materialization: incremental\n"
        "  - name: release_search\n"
        "    depends_on: [ref('release_embeddings')]\n"
        "    materialization: incremental\n"
        "    search:\n"
        "      access: public\n"
        "      id_field: chunk_id\n"
        "      document_id_field: document_id\n"
        "      chunk_id_field: chunk_id\n"
        "      text_fields: [text]\n"
        "      return_text_fields: [text]\n"
        "      vector:\n"
        "        field: embedding\n"
        "        dimensions: 8\n"
        "        metric: cosine\n"
        "        search: exact\n"
        "        embedding: inherit\n"
        "      full_text:\n"
        "        fields: [text]\n"
        "      attributes:\n"
        "        - name: category\n"
        "          data_type: string\n"
        "          filter_role: user\n"
        "          returned: true\n"
        "      display_fields: [title]\n"
        "      query:\n"
        "        modes: [vector, text, hybrid]\n"
        "        consistency: strong\n"
    )
    data = project / "data"
    data.mkdir()
    for name, payload in {
        "inflation.json": {
            "title": "Consumer prices",
            "body": "Inflation moderated as consumer price growth slowed.",
            "category": "prices",
        },
        "labor.json": {
            "title": "Employment report",
            "body": "Payroll employment increased and unemployment remained stable.",
            "category": "labor",
        },
        "output.json": {
            "title": "GDP report",
            "body": "Economic output expanded during the latest quarter.",
            "category": "growth",
        },
    }.items():
        (data / name).write_text(json.dumps(payload))
    return project


@pytest.fixture
def published_project(tmp_path: Path) -> Path:
    project = _write_project(tmp_path)
    results = run_project(project)
    assert results[-1].model_name == "release_search"
    assert results[-1].serving_resource is not None
    return project


@pytest.mark.parametrize("mode", list(SearchMode))
def test_portable_search_api_supports_all_modes(
    published_project: Path,
    mode: SearchMode,
) -> None:
    results = search(
        published_project,
        SearchRequest(
            model="release_search",
            query="inflation consumer prices",
            mode=mode,
            limit=2,
        ),
    )

    assert results
    assert [result.rank for result in results] == list(range(1, len(results) + 1))
    assert all(0 < result.score <= 1 for result in results)
    assert results[0].provenance.unique_id == (
        "search_index.economic_search.release_search"
    )
    assert results[0].provenance.embedding is not None
    assert results[0].provenance.embedding["provider"] == "deterministic"
    if mode == SearchMode.HYBRID:
        assert set(results[0].contributing_ranks) == {"text", "vector"}
        assert results[0].raw_score is None
    else:
        assert results[0].raw_score is not None
    serialized = json.dumps([result.to_dict() for result in results])
    assert "economic-search-v1" in serialized
    assert "embedding" not in results[0].text


def test_search_contract_repr_and_reserved_columns_are_safe() -> None:
    request = SearchRequest(
        model="release_search",
        query="sensitive query text",
        vector=(0.1, 0.2),
        mode=SearchMode.VECTOR,
        filters=(
            SearchFilter("category", SearchFilterOperator.EQUAL, "secret-category"),
        ),
    )
    assert "sensitive query text" not in repr(request)
    assert "secret-category" not in repr(request)

    with pytest.raises(ValueError, match="reserved retrieval score"):
        SearchConfig.model_validate(
            {
                "id_field": "_score",
                "text_fields": ["text"],
                "full_text": {"fields": ["text"]},
                "query": {"modes": ["text"]},
            }
        )

    ranked = _rank_table(
        pa.Table.from_pylist(
            [{"_id": "record-1", "_source": "archive", "_score": 0.75}]
        ),
        "_id",
    )
    assert ranked[0].record_id == "record-1"
    assert ranked[0].values == {"_id": "record-1", "_source": "archive"}
    assert ranked[0].raw_score == 0.75


def test_search_filters_are_typed_and_authorized(published_project: Path) -> None:
    filtered = search(
        published_project,
        SearchRequest(
            model="release_search",
            query="economic report",
            mode=SearchMode.HYBRID,
            filters=(
                SearchFilter(
                    "category",
                    SearchFilterOperator.IN,
                    ("labor", "growth"),
                ),
            ),
        ),
    )
    assert filtered
    assert {result.metadata["category"] for result in filtered} <= {
        "labor",
        "growth",
    }

    with pytest.raises(SearchError, match="not available for user filtering"):
        search(
            published_project,
            SearchRequest(
                model="release_search",
                query="prices",
                mode=SearchMode.TEXT,
                filters=(
                    SearchFilter("title", SearchFilterOperator.EQUAL, "Consumer prices"),
                ),
            ),
        )


def test_search_cli_emits_stable_json_and_human_provenance(
    published_project: Path,
) -> None:
    runner = CliRunner()
    json_result = runner.invoke(
        cli,
        [
            "--project-dir",
            str(published_project),
            "search",
            "--model",
            "release_search",
            "--query",
            "employment report",
            "--mode",
            "hybrid",
            "--filter",
            "category",
            "eq",
            "labor",
            "--output",
            "json",
        ],
    )
    assert json_result.exit_code == 0, json_result.output
    payload = json.loads(json_result.output)
    assert payload[0]["metadata"]["category"] == "labor"
    assert payload[0]["provenance"]["store_type"] == "lancedb"
    assert "embedding_provider_implementation" not in json_result.output

    human = runner.invoke(
        cli,
        [
            "search",
            "--project-dir",
            str(published_project),
            "--model",
            "release_search",
            "--query",
            "employment report",
            "--mode",
            "text",
            "--limit",
            "1",
        ],
    )
    assert human.exit_code == 0, human.output
    assert "search_index.economic_search.release_search" in human.output
    assert "record_id" in human.output


def test_manifest_records_resolved_inherited_embedding_identity(
    published_project: Path,
) -> None:
    manifest = build_manifest(published_project)
    descriptor = next(
        model
        for model in manifest["models"]
        if model["name"] == "release_search"
    )["output"]["serving_resource"]

    assert descriptor["vector"]["embedding"]["provider"] == "deterministic"
    assert descriptor["vector"]["embedding"]["dimensions"] == 8
    assert descriptor["query"]["modes"] == ["hybrid", "text", "vector"]


def test_same_request_runs_against_mocked_remote_store(
    published_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SearchRequest(
        model="release_search",
        query="inflation consumer prices",
        mode=SearchMode.HYBRID,
        limit=2,
    )
    local = search(published_project, request)
    project, _, _ = load_project(published_project)
    resolved = resolve_profile(project, published_project)
    assert resolved.retrieval is not None
    real_store = create_store(
        resolved.retrieval.stores["local"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="local",
    )
    physical = real_store.physical_collection("release_search")
    with real_store:
        metadata = real_store.inspect_collection(physical)
    assert metadata is not None
    rows = [
        {
            "chunk_id": result.record_id,
            "document_id": result.document_id,
            "text": result.text["text"],
            "category": result.metadata["category"],
            "title": result.display["title"],
        }
        for result in local
    ]

    class MockRemoteStore:
        def __enter__(self) -> MockRemoteStore:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def capabilities(self) -> Any:
            return real_store.capabilities()

        def physical_collection(self, _logical: str) -> str:
            return physical

        def inspect_collection(self, _physical: str) -> Any:
            return metadata

        def vector_search(self, *_args: Any, **_kwargs: Any) -> pa.Table:
            return pa.Table.from_pylist(
                [{**row, "_distance": rank / 10} for rank, row in enumerate(rows, 1)]
            )

        def text_search(self, *_args: Any, **_kwargs: Any) -> pa.Table:
            return pa.Table.from_pylist(
                [{**row, "_score": 1 / rank} for rank, row in enumerate(rows, 1)]
            )

    search_module = importlib.import_module("dbt_ml.search")
    monkeypatch.setattr(search_module, "create_store", lambda *_args, **_kwargs: MockRemoteStore())

    remote = search(published_project, request)
    assert [result.record_id for result in remote] == [
        result.record_id for result in local
    ]
    assert remote[0].contributing_ranks == {"text": 1, "vector": 1}
