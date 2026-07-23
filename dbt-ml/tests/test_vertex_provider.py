from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest
from click.testing import CliRunner

from dbt_ml.cli import cli
from dbt_ml.config import load_project
from dbt_ml.config.model import EmbedConfig
from dbt_ml.embedding import EmbeddingIdentity
from dbt_ml.optional_dependencies import OptionalDependencyError
from dbt_ml.profile import ProfileError, resolve_profile
from dbt_ml.providers import (
    EmbeddingRequest,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderRuntimeOptions,
    get_embedding_provider,
    list_embedding_providers,
)
from dbt_ml.providers import vertex as vertex_module
from dbt_ml.providers.vertex import (
    VertexEmbeddingOptions,
    VertexEmbeddingProvider,
)


class _FakeModels:
    def __init__(self, calls: list[dict[str, Any]], *, malformed: bool = False) -> None:
        self.calls = calls
        self.malformed = malformed

    def embed_content(
        self,
        *,
        model: str,
        contents: list[str],
        config: dict[str, Any],
    ) -> Any:
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        dimensions = config["output_dimensionality"]
        embeddings = [
            SimpleNamespace(
                values=[float(index + 1)] * dimensions,
                statistics=SimpleNamespace(
                    token_count=float(len(text.split())),
                    truncated=False,
                ),
            )
            for index, text in enumerate(contents)
        ]
        if self.malformed:
            embeddings = embeddings[:1]
        return SimpleNamespace(embeddings=embeddings)


class _FakeGenAI:
    def __init__(self, *, malformed: bool = False) -> None:
        self.client_options: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.close_count = 0
        self.malformed = malformed

    def Client(self, **options: Any) -> Any:
        self.client_options.append(options)
        owner = self

        class Client:
            models = _FakeModels(owner.calls, malformed=owner.malformed)

            def close(self) -> None:
                owner.close_count += 1

        return Client()


def _provider(**options: Any) -> VertexEmbeddingProvider:
    provider = get_embedding_provider("vertex", profile_options=options)
    assert isinstance(provider, VertexEmbeddingProvider)
    return provider


def test_vertex_provider_is_registered_with_strict_typed_options() -> None:
    assert "vertex" in list_embedding_providers()
    provider = _provider(
        project="economic-data-prod",
        location="global",
        task_type="RETRIEVAL_DOCUMENT",
        query_task_type="RETRIEVAL_QUERY",
        auto_truncate=False,
    )

    assert isinstance(provider.profile_options, VertexEmbeddingOptions)
    assert provider.profile_options.project == "economic-data-prod"

    with pytest.raises(ProviderConfigurationError, match="rejected provider_options"):
        _provider(unknown=True)


def test_vertex_provider_maps_batch_dimensions_ids_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeGenAI()
    monkeypatch.setattr(vertex_module, "_load_google_genai", lambda: fake)
    provider = _provider(project="economic-data-prod", location="global")
    request = EmbeddingRequest(
        model="text-embedding-005",
        texts=("employment increased", "inflation moderated"),
        dimensions=3,
        input_ids=("chunk-a", "chunk-b"),
    )

    result = provider.embed(
        request,
        credential=None,
        runtime=ProviderRuntimeOptions(max_retries=2, timeout_seconds=12.5),
    )

    assert result.model == request.model
    assert result.dimensions == 3
    assert len(result.vectors) == 2
    assert result.input_ids == request.input_ids
    assert result.usage.input_tokens == 4
    assert fake.client_options == [
        {
            "vertexai": True,
            "project": "economic-data-prod",
            "location": "global",
            "http_options": {
                "api_version": "v1",
                "timeout": 12_500,
                "retry_options": {
                    "attempts": 3,
                    "http_status_codes": [408, 409, 425, 429, 500, 502, 503, 504],
                },
            },
        }
    ]
    assert fake.calls == [
        {
            "model": "text-embedding-005",
            "contents": ["employment increased", "inflation moderated"],
            "config": {
                "task_type": "RETRIEVAL_DOCUMENT",
                "output_dimensionality": 3,
                "auto_truncate": False,
            },
        }
    ]
    assert fake.close_count == 1


def test_vertex_gemini_model_splits_multi_input_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeGenAI()
    monkeypatch.setattr(vertex_module, "_load_google_genai", lambda: fake)
    request = EmbeddingRequest(
        model="publishers/google/models/gemini-embedding-001",
        texts=("employment increased", "inflation moderated"),
        dimensions=3,
        input_ids=("chunk-a", "chunk-b"),
    )

    result = _provider().embed(
        request,
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )

    assert [call["contents"] for call in fake.calls] == [
        ["employment increased"],
        ["inflation moderated"],
    ]
    assert result.input_ids == request.input_ids
    assert len(result.vectors) == 2
    assert result.usage.input_tokens == 4
    assert result.provider_requests == 2
    assert fake.close_count == 1


@pytest.mark.parametrize(
    "model",
    [
        "text-embedding-005",
        "publishers/google/models/text-multilingual-embedding-002",
    ],
)
def test_vertex_text_models_split_batches_at_five_inputs(
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeGenAI()
    monkeypatch.setattr(vertex_module, "_load_google_genai", lambda: fake)
    texts = tuple(f"economic document {index}" for index in range(6))
    input_ids = tuple(f"chunk-{index}" for index in range(6))

    result = _provider().embed(
        EmbeddingRequest(
            model=model,
            texts=texts,
            dimensions=3,
            input_ids=input_ids,
        ),
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )

    assert [len(call["contents"]) for call in fake.calls] == [5, 1]
    assert [
        text
        for call in fake.calls
        for text in call["contents"]
    ] == list(texts)
    assert result.input_ids == input_ids
    assert len(result.vectors) == 6
    assert result.usage.input_tokens == 18
    assert result.provider_requests == 2
    assert fake.close_count == 1


def test_vertex_provider_uses_query_task_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeGenAI()
    monkeypatch.setattr(vertex_module, "_load_google_genai", lambda: fake)
    provider = _provider(query_task_type="SEMANTIC_SIMILARITY")

    provider.embed(
        EmbeddingRequest(
            model="text-embedding-005",
            texts=("latest payroll release",),
            dimensions=4,
            input_type="query",
        ),
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )

    assert fake.calls[0]["config"]["task_type"] == "SEMANTIC_SIMILARITY"


def test_vertex_provider_rejects_api_keys_and_has_actionable_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    with pytest.raises(ProviderConfigurationError, match="Default Credentials"):
        provider.resolve_credential("GOOGLE_API_KEY")

    def missing() -> Any:
        raise OptionalDependencyError(
            "Vertex AI embeddings requires the optional dependency 'google-genai'. "
            "Install it with: pip install 'dbt-ml[vertex]'"
        )

    monkeypatch.setattr(vertex_module, "_load_google_genai", missing)
    with pytest.raises(ProviderConfigurationError, match=r"dbt-ml\[vertex\]"):
        provider.embed(
            EmbeddingRequest(
                model="text-embedding-005",
                texts=("document",),
                dimensions=2,
            ),
            credential=None,
            runtime=ProviderRuntimeOptions(),
        )


def test_vertex_api_key_env_is_rejected_during_profile_resolution(
    tmp_path: Path,
) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text(
        "name: vertex_project\nversion: '0.1.0'\nprofile: vertex_project\n"
    )
    (tmp_path / "profiles.yml").write_text(
        "vertex_project:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        "        path: ./target/db.duckdb\n"
        "        schema: docs\n"
        "      embedding:\n"
        "        provider: vertex\n"
        "        api_key_env: GOOGLE_API_KEY\n"
    )
    project, _, _ = load_project(tmp_path)

    with pytest.raises(
        ProfileError,
        match=r"Application Default Credentials.*do not accept api_key_env",
    ):
        resolve_profile(project, tmp_path)


def test_vertex_malformed_response_is_sanitized_and_accounts_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeGenAI(malformed=True)
    monkeypatch.setattr(vertex_module, "_load_google_genai", lambda: fake)
    provider = _provider()

    with pytest.raises(ProviderResponseError) as excinfo:
        provider.embed(
            EmbeddingRequest(
                model="text-multilingual-embedding-002",
                texts=("first input", "second input"),
                dimensions=2,
            ),
            credential=None,
            runtime=ProviderRuntimeOptions(),
        )

    assert "inputs" in str(excinfo.value)
    assert excinfo.value.failure is not None
    assert excinfo.value.failure.error_code == "invalid_embedding_response"
    assert excinfo.value.failure.usage.input_tokens == 2


def test_vertex_sdk_error_text_does_not_cross_provider_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sensitive-document-fragment"

    class Models:
        def embed_content(self, **kwargs: Any) -> Any:
            del kwargs
            raise RuntimeError(f"upstream echoed {secret} and /private/path")

    class Client:
        models = Models()

        def close(self) -> None:
            return None

    fake = SimpleNamespace(Client=lambda **kwargs: Client())
    monkeypatch.setattr(vertex_module, "_load_google_genai", lambda: fake)

    with pytest.raises(ProviderRequestError) as excinfo:
        _provider().embed(
            EmbeddingRequest(
                model="text-embedding-005",
                texts=(secret,),
                dimensions=2,
            ),
            credential=None,
            runtime=ProviderRuntimeOptions(),
        )

    assert excinfo.value.code == "RuntimeError"
    assert secret not in str(excinfo.value)
    assert "/private/path" not in str(excinfo.value)


def test_vertex_semantic_options_are_part_of_embedding_identity() -> None:
    config = EmbedConfig(
        provider="vertex",
        model="gemini-embedding-001",
        dimensions=3,
    )
    first = EmbeddingIdentity.from_config(
        config,
        profile_options={
            "project": "project-a",
            "location": "us-central1",
            "task_type": "RETRIEVAL_DOCUMENT",
        },
    )
    other_deployment = EmbeddingIdentity.from_config(
        config,
        profile_options={
            "project": "project-b",
            "location": "global",
            "task_type": "RETRIEVAL_DOCUMENT",
        },
    )
    other_task = EmbeddingIdentity.from_config(
        config,
        profile_options={
            "project": "project-a",
            "location": "us-central1",
            "task_type": "CLUSTERING",
        },
    )

    assert first.config_hash == other_deployment.config_hash
    assert first.provider_options_identity == other_deployment.provider_options_identity
    assert first.config_hash != other_task.config_hash
    assert first.provider_options_identity != other_task.provider_options_identity


def test_build_runs_vertex_embed_model_with_mocked_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeGenAI()
    monkeypatch.setattr(vertex_module, "_load_google_genai", lambda: fake)
    project = tmp_path / "vertex_project"
    (project / "models").mkdir(parents=True)
    (project / "sources").mkdir()
    (project / "data").mkdir()
    (project / "dbt_ml_project.yml").write_text(
        "name: vertex_project\nversion: '0.1.0'\nprofile: vertex_project\n"
    )
    (project / "profiles.yml").write_text(
        "vertex_project:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        "        path: ./target/db.duckdb\n"
        "        schema: docs\n"
        "      embedding:\n"
        "        provider: vertex\n"
        "        timeout_seconds: 15\n"
        "        provider_options:\n"
        "          project: economic-data-dev\n"
        "          location: global\n"
        "          task_type: RETRIEVAL_DOCUMENT\n"
    )
    (project / "sources" / "documents.yml").write_text(
        "version: 2\n"
        "sources:\n"
        "  - name: raw_documents\n"
        "    path: data\n"
        "    file_pattern: '*.json'\n"
    )
    (project / "models" / "documents.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: document_registry\n"
        "    source: ref('raw_documents')\n"
        "    extraction:\n"
        "      backend: json\n"
        "      options:\n"
        "        fields: [body]\n"
        "    materialization: full\n"
        "  - name: document_embeddings\n"
        "    depends_on: [ref('document_registry')]\n"
        "    embed:\n"
        "      provider: vertex\n"
        "      model: gemini-embedding-001\n"
        "      text_field: body\n"
        "      id_field: document_id\n"
        "      dimensions: 3\n"
        "      batch_size: 10\n"
        "      max_retries: 1\n"
        "    materialization: full\n"
        "    tests:\n"
        "      - not_null: [document_id, embedding]\n"
        "      - unique: document_id\n"
    )
    for name, body in (
        ("employment.json", "employment increased"),
        ("inflation.json", "inflation moderated"),
    ):
        (project / "data" / name).write_text(json.dumps({"body": body}))

    built = CliRunner().invoke(
        cli,
        ["--project-dir", str(project), "build"],
    )

    assert built.exit_code == 0, built.output
    connection = duckdb.connect(str(project / "target" / "db.duckdb"))
    try:
        rows = connection.execute(
            'SELECT embedding, embedding_provider, embedding_model, '
            'embedding_dimensions FROM "db".docs.document_embeddings '
            "ORDER BY document_id"
        ).fetchall()
    finally:
        connection.close()
    assert len(rows) == 2
    assert all(len(row[0]) == 3 for row in rows)
    assert {row[1] for row in rows} == {"vertex"}
    assert {row[2] for row in rows} == {"gemini-embedding-001"}
    assert {row[3] for row in rows} == {3}
    assert len(fake.calls) == 2
    assert all(len(call["contents"]) == 1 for call in fake.calls)
    assert all(call["config"]["output_dimensionality"] == 3 for call in fake.calls)
    run_results = json.loads((project / "target" / "run_results.json").read_text())
    embed_result = next(
        result
        for result in run_results["results"]
        if result["model_name"] == "document_embeddings"
    )
    assert embed_result["metrics"]["provider_calls"] == 2
    assert embed_result["metrics"]["batches"] == 1
    manifest = json.loads((project / "target" / "manifest.json").read_text())
    serialized_manifest = json.dumps(manifest)
    assert "economic-data-dev" not in serialized_manifest
    embedding = next(
        model["embedding"]
        for model in manifest["models"]
        if model["name"] == "document_embeddings"
    )
    assert embedding["provider"] == "vertex"
    assert embedding["provider_options_identity"]
