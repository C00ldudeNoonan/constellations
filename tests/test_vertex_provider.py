from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest
from click.testing import CliRunner

from stel.cli import cli
from stel.config import load_project
from stel.config.model import EmbedConfig
from stel.embedding import EmbeddingIdentity
from stel.optional_dependencies import OptionalDependencyError
from stel.profile import ProfileError, resolve_profile
from stel.providers import (
    EmbeddingRequest,
    InferenceRequest,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderRuntimeOptions,
    get_embedding_provider,
    get_inference_provider,
    list_embedding_providers,
    list_inference_providers,
    profile_options_fingerprint,
)
from stel.providers import vertex as vertex_module
from stel.providers.vertex import (
    VertexEmbeddingOptions,
    VertexEmbeddingProvider,
    VertexInferenceOptions,
    VertexInferenceProvider,
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


@pytest.fixture(autouse=True)
def _reset_vertex_client_cache() -> Any:
    """Clients are cached process-wide (issue #335).

    Each test substitutes its own fake `genai`, so a client cached by an
    earlier test would be reused here and record its calls against the wrong
    fake.
    """
    vertex_module._CLIENTS.clear()
    yield
    vertex_module._CLIENTS.clear()


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
    # Not closed: the client is shared across requests now, and closing
    # it here would leave the cached entry unusable (issue #335).
    assert fake.close_count == 0


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
    # Not closed: the client is shared across requests now, and closing
    # it here would leave the cached entry unusable (issue #335).
    assert fake.close_count == 0


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
    # Not closed: the client is shared across requests now, and closing
    # it here would leave the cached entry unusable (issue #335).
    assert fake.close_count == 0


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
            "Install it with: pip install 'stel[vertex]'"
        )

    monkeypatch.setattr(vertex_module, "_load_google_genai", missing)
    with pytest.raises(ProviderConfigurationError, match=r"stel\[vertex\]"):
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
    (tmp_path / "stel_project.yml").write_text(
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
    (project / "stel_project.yml").write_text(
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


# --- Vertex AI inference (Gemini structured extraction, issue #17) -----------


class _FakeInferenceModels:
    def __init__(self, calls: list[dict[str, Any]], *, response: Any) -> None:
        self.calls = calls
        self._response = response

    def generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: dict[str, Any],
    ) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._response


class _FakeInferenceGenAI:
    def __init__(self, response: Any) -> None:
        self.client_options: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.close_count = 0
        self._response = response

    def Client(self, **options: Any) -> Any:
        self.client_options.append(options)
        owner = self

        class Client:
            models = _FakeInferenceModels(owner.calls, response=owner._response)

            def close(self) -> None:
                owner.close_count += 1

        return Client()


def _inference_response(
    *,
    output: str = '{"relation_type": "acquired"}',
    finish: str = "STOP",
    prompt_tokens: int = 10,
    output_tokens: int = 5,
    thoughts_tokens: int = 0,
    cached: int = 0,
    response_id: str = "resp-1",
    include_text: bool = True,
) -> Any:
    candidate = SimpleNamespace(
        finish_reason=SimpleNamespace(name=finish),
        content=SimpleNamespace(parts=[SimpleNamespace(text=output)]),
    )
    response = SimpleNamespace(
        candidates=[candidate],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            thoughts_token_count=thoughts_tokens,
            cached_content_token_count=cached,
        ),
        response_id=response_id,
    )
    if include_text:
        response.text = output
    return response


def _inference_provider(**options: Any) -> VertexInferenceProvider:
    provider = get_inference_provider("vertex", profile_options=options)
    assert isinstance(provider, VertexInferenceProvider)
    return provider


def _inference_request(**overrides: Any) -> InferenceRequest:
    fields: dict[str, Any] = {
        "model": "gemini-2.5-flash",
        "content": "Document text.",
        "system_prompt": "Extract relations.",
        "output_schema": {"type": "object", "properties": {}},
    }
    fields.update(overrides)
    return InferenceRequest(**fields)


def test_vertex_inference_provider_is_registered_with_strict_typed_options() -> None:
    assert "vertex" in list_inference_providers()
    provider = _inference_provider(project="economic-data-prod", location="us-central1")

    assert isinstance(provider.profile_options, VertexInferenceOptions)
    assert provider.profile_options.project == "economic-data-prod"

    with pytest.raises(ProviderConfigurationError, match="rejected provider_options"):
        _inference_provider(unknown=True)


def test_vertex_inference_maps_request_runtime_and_parses_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeInferenceGenAI(
        _inference_response(
            output='{"relation_type": "acquired", "confidence": 0.9}',
            prompt_tokens=12,
            output_tokens=7,
            cached=3,
            response_id="resp-42",
        )
    )
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)
    provider = _inference_provider(project="proj", location="us-central1")

    schema = {"type": "object", "properties": {"relation_type": {"type": "string"}}}
    result = provider.complete(
        _inference_request(
            output_schema=schema,
            output_name="relation",
            max_tokens=256,
        ),
        credential=None,
        runtime=ProviderRuntimeOptions(max_retries=2, timeout_seconds=30.0),
    )

    assert result.output == {"relation_type": "acquired", "confidence": 0.9}
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 7
    assert result.usage.cache_read_input_tokens == 3
    assert result.provider_request_id == "resp-42"

    client_options = fake.client_options[0]
    assert client_options["vertexai"] is True
    assert client_options["location"] == "us-central1"
    assert client_options["project"] == "proj"
    assert client_options["http_options"]["timeout"] == 30000
    assert client_options["http_options"]["retry_options"]["attempts"] == 3

    call = fake.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["contents"] == "Document text."
    assert call["config"]["system_instruction"] == "Extract relations."
    assert call["config"]["max_output_tokens"] == 256
    assert call["config"]["response_mime_type"] == "application/json"
    assert call["config"]["response_schema"] == schema
    # Not closed: the client is shared across requests now, and closing
    # it here would leave the cached entry unusable (issue #335).
    assert fake.close_count == 0


def test_vertex_inference_folds_thinking_tokens_into_output_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Gemini thinking models report reasoning tokens separately; they must be
    # counted as output usage so budgets and metrics see the full spend.
    fake = _FakeInferenceGenAI(
        _inference_response(output_tokens=5, thoughts_tokens=40)
    )
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)

    result = _inference_provider().complete(
        _inference_request(),
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )
    assert result.usage.output_tokens == 45
    assert result.usage.thinking_tokens == 40


def test_vertex_inference_thinking_budget_rejects_negative() -> None:
    with pytest.raises(ProviderConfigurationError, match="rejected provider_options"):
        _inference_provider(thinking_budget=-1)


def test_vertex_inference_thinking_budget_participates_in_fingerprint() -> None:
    default = _inference_provider()
    explicit = _inference_provider(thinking_budget=1024)
    assert profile_options_fingerprint(
        default.profile_options
    ) != profile_options_fingerprint(explicit.profile_options)


def test_vertex_inference_defaults_thinking_budget_to_zero_for_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeInferenceGenAI(_inference_response())
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)
    schema = {"type": "object", "properties": {"relation_type": {"type": "string"}}}

    _inference_provider().complete(
        _inference_request(output_schema=schema),
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )

    assert fake.calls[0]["config"]["thinking_config"] == {"thinking_budget": 0}


def test_vertex_inference_omits_thinking_config_without_declared_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeInferenceGenAI(_inference_response())
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)

    _inference_provider().complete(
        _inference_request(),
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )

    assert "thinking_config" not in fake.calls[0]["config"]


@pytest.mark.parametrize(
    "model",
    [
        # 2.5 Pro enforces a minimum thinking budget, pre-2.5 models reject
        # thinking_config outright, and Gemini 3 uses thinking_level — sending
        # an automatic 0 to any of them would break a valid extraction call.
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-3-pro-preview",
    ],
)
def test_vertex_inference_skips_automatic_zero_on_models_that_reject_it(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    fake = _FakeInferenceGenAI(_inference_response())
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)
    schema = {"type": "object", "properties": {"relation_type": {"type": "string"}}}

    _inference_provider().complete(
        _inference_request(model=model, output_schema=schema),
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )

    assert "thinking_config" not in fake.calls[0]["config"]


@pytest.mark.parametrize(
    "model",
    [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-preview-05-20",
        "publishers/google/models/gemini-2.5-flash",
    ],
)
def test_vertex_inference_applies_automatic_zero_across_flash_variants(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    fake = _FakeInferenceGenAI(_inference_response())
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)
    schema = {"type": "object", "properties": {"relation_type": {"type": "string"}}}

    _inference_provider().complete(
        _inference_request(model=model, output_schema=schema),
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )

    assert fake.calls[0]["config"]["thinking_config"] == {"thinking_budget": 0}


def test_vertex_inference_explicit_thinking_budget_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeInferenceGenAI(_inference_response())
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)
    schema = {"type": "object", "properties": {"relation_type": {"type": "string"}}}

    _inference_provider(thinking_budget=1024).complete(
        _inference_request(output_schema=schema),
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )

    assert fake.calls[0]["config"]["thinking_config"] == {"thinking_budget": 1024}


def test_vertex_inference_explicit_budget_is_forwarded_on_any_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An operator-chosen budget is never second-guessed by the capability
    # gate: the provider reports its own error if the model rejects it.
    fake = _FakeInferenceGenAI(_inference_response())
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)

    _inference_provider(thinking_budget=0).complete(
        _inference_request(model="gemini-2.5-pro"),
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )

    assert fake.calls[0]["config"]["thinking_config"] == {"thinking_budget": 0}


def test_vertex_inference_billed_failure_includes_thinking_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeInferenceGenAI(
        _inference_response(output="not json", output_tokens=3, thoughts_tokens=20)
    )
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)

    with pytest.raises(ProviderResponseError) as excinfo:
        _inference_provider().complete(
            _inference_request(),
            credential=None,
            runtime=ProviderRuntimeOptions(),
        )
    assert excinfo.value.failure is not None
    assert excinfo.value.failure.usage.output_tokens == 23


def test_vertex_inference_omits_project_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeInferenceGenAI(_inference_response())
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)

    _inference_provider(location="us-central1").complete(
        _inference_request(),
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )
    assert "project" not in fake.client_options[0]


def test_vertex_inference_reads_text_fallback_when_parts_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _inference_response(output='{"relation_type": "supplies"}')
    response.candidates[0].content = SimpleNamespace(parts=None)
    fake = _FakeInferenceGenAI(response)
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)

    result = _inference_provider().complete(
        _inference_request(),
        credential=None,
        runtime=ProviderRuntimeOptions(),
    )
    assert result.output == {"relation_type": "supplies"}


def test_vertex_inference_rejects_api_keys_and_has_actionable_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _inference_provider()
    with pytest.raises(ProviderConfigurationError, match="Default Credentials"):
        provider.resolve_credential("GOOGLE_API_KEY")

    def missing() -> Any:
        raise OptionalDependencyError(
            "Vertex AI inference requires the optional dependency 'google-genai'. "
            "Install it with: pip install 'stel[vertex]'"
        )

    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", missing)
    with pytest.raises(ProviderConfigurationError, match=r"stel\[vertex\]"):
        provider.complete(
            _inference_request(),
            credential=None,
            runtime=ProviderRuntimeOptions(),
        )


def test_vertex_inference_api_key_env_is_rejected_during_profile_resolution(
    tmp_path: Path,
) -> None:
    (tmp_path / "stel_project.yml").write_text(
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
        "      llm:\n"
        "        provider: vertex\n"
        "        model: gemini-2.5-flash\n"
        "        api_key_env: GOOGLE_API_KEY\n"
    )
    project, _, _ = load_project(tmp_path)

    with pytest.raises(
        ProfileError,
        match=r"Application Default Credentials.*does not accept api_key_env",
    ):
        resolve_profile(project, tmp_path)


def test_vertex_inference_truncation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeInferenceGenAI(_inference_response(finish="MAX_TOKENS"))
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)

    with pytest.raises(ProviderResponseError, match="truncated at max_tokens"):
        _inference_provider().complete(
            _inference_request(max_tokens=8),
            credential=None,
            runtime=ProviderRuntimeOptions(),
        )


def test_vertex_inference_non_stop_finish_reason_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeInferenceGenAI(_inference_response(finish="SAFETY"))
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)

    with pytest.raises(ProviderResponseError, match="SAFETY"):
        _inference_provider().complete(
            _inference_request(),
            credential=None,
            runtime=ProviderRuntimeOptions(),
        )


def test_vertex_inference_malformed_json_is_sanitized_and_accounts_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeInferenceGenAI(
        _inference_response(output="not valid json", prompt_tokens=15, output_tokens=4)
    )
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)

    with pytest.raises(ProviderResponseError) as excinfo:
        _inference_provider().complete(
            _inference_request(),
            credential=None,
            runtime=ProviderRuntimeOptions(),
        )

    assert "malformed JSON" in str(excinfo.value)
    assert excinfo.value.failure is not None
    assert excinfo.value.failure.error_code == "invalid_response"
    assert excinfo.value.failure.usage.input_tokens == 15
    assert excinfo.value.failure.usage.output_tokens == 4


def test_vertex_inference_sdk_error_text_does_not_cross_provider_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sensitive-document-fragment"

    class Models:
        def generate_content(self, **kwargs: Any) -> Any:
            del kwargs
            raise RuntimeError(f"upstream echoed {secret} and /private/path")

    class Client:
        models = Models()

        def close(self) -> None:
            return None

    fake = SimpleNamespace(Client=lambda **kwargs: Client())
    monkeypatch.setattr(vertex_module, "_load_google_genai_inference", lambda: fake)

    with pytest.raises(ProviderRequestError) as excinfo:
        _inference_provider().complete(
            _inference_request(content=secret),
            credential=None,
            runtime=ProviderRuntimeOptions(),
        )

    assert excinfo.value.code == "RuntimeError"
    assert secret not in str(excinfo.value)
    assert "/private/path" not in str(excinfo.value)


def test_vertex_inference_surfaces_http_status_for_retry_triage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 429 (rate limit) must surface as a retryable http_429, distinguishable
    # from a permanent 400/403 — so a caller can build a retry policy (#258).
    class _ClientError(Exception):
        def __init__(self, code: int) -> None:
            super().__init__("quota")
            self.code = code

    def _raising(code: int) -> Any:
        class Models:
            def generate_content(self, **kwargs: Any) -> Any:
                del kwargs
                raise _ClientError(code)

        class Client:
            models = Models()

            def close(self) -> None:
                return None

        return SimpleNamespace(Client=lambda **kwargs: Client())

    monkeypatch.setattr(
        vertex_module, "_load_google_genai_inference", lambda: _raising(429)
    )
    with pytest.raises(ProviderRequestError) as excinfo:
        _inference_provider().complete(
            _inference_request(), credential=None, runtime=ProviderRuntimeOptions()
        )
    assert excinfo.value.code == "http_429"
    assert excinfo.value.retryable is True

    # Clients are cached on their resolved options (issue #335), so swapping
    # the SDK underneath mid-test needs an explicit reset — in production
    # there is one SDK and one client per configuration, which is the point.
    vertex_module._CLIENTS.clear()
    monkeypatch.setattr(
        vertex_module, "_load_google_genai_inference", lambda: _raising(403)
    )
    with pytest.raises(ProviderRequestError) as excinfo:
        _inference_provider().complete(
            _inference_request(), credential=None, runtime=ProviderRuntimeOptions()
        )
    assert excinfo.value.code == "http_403"
    assert excinfo.value.retryable is False


# ─── client reuse (issue #335) ──────────────────────────────────────────────


def test_one_client_serves_many_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The measured defect: one client construction per request.

    Each construction re-runs `google.auth.default()`, which under end-user
    ADC includes a token refresh round trip — 4,252 of them in one production
    inference run, and enough fixed overhead per embed batch to turn a
    3.93M-chunk backfill into days rather than hours.

    Deliberately built through `get_embedding_provider` per batch, the way
    `embed_texts` does: providers are constructed fresh on every call, so a
    cache living on the provider instance would never see a second hit.
    """
    fake = _FakeGenAI()
    monkeypatch.setattr(vertex_module, "_load_google_genai", lambda: fake)

    for _ in range(10):
        provider = get_embedding_provider(
            "vertex",
            profile_options={
                "project": "p",
                "location": "global",
                "task_type": "RETRIEVAL_DOCUMENT",
                "query_task_type": "RETRIEVAL_QUERY",
                "auto_truncate": False,
            },
        )
        provider.embed(
            EmbeddingRequest(
                model="text-embedding-005",
                texts=("a", "b"),
                dimensions=3,
                input_ids=("x", "y"),
            ),
            credential=None,
            runtime=ProviderRuntimeOptions(max_retries=4, timeout_seconds=60.0),
        )

    assert len(fake.client_options) == 1


def test_a_different_runtime_gets_its_own_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse only where reuse is correct.

    `client_options` mixes profile values with per-request timeout and retry
    counts, so a single cached client would serve the first request's timeout
    to every later one.
    """
    fake = _FakeGenAI()
    monkeypatch.setattr(vertex_module, "_load_google_genai", lambda: fake)
    request = EmbeddingRequest(
        model="text-embedding-005",
        texts=("a",),
        dimensions=3,
        input_ids=("x",),
    )

    for timeout in (60.0, 60.0, 12.5):
        _provider(project="p", location="global").embed(
            request,
            credential=None,
            runtime=ProviderRuntimeOptions(max_retries=4, timeout_seconds=timeout),
        )

    timeouts = [
        options["http_options"]["timeout"] for options in fake.client_options
    ]
    assert timeouts == [60_000, 12_500]


def test_concurrent_first_requests_construct_one_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The embedding path runs several requests in flight.

    Without the double-check under the lock, concurrent first requests would
    each construct — reintroducing the cost being removed, just less often.
    """
    import threading

    fake = _FakeGenAI()
    monkeypatch.setattr(vertex_module, "_load_google_genai", lambda: fake)
    request = EmbeddingRequest(
        model="text-embedding-005",
        texts=("a",),
        dimensions=3,
        input_ids=("x",),
    )
    barrier = threading.Barrier(5)

    def _embed() -> None:
        barrier.wait()
        _provider(project="p", location="global").embed(
            request,
            credential=None,
            runtime=ProviderRuntimeOptions(max_retries=4, timeout_seconds=60.0),
        )

    threads = [threading.Thread(target=_embed) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(fake.client_options) == 1
