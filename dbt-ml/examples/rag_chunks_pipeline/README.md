# rag_chunks_pipeline

Document → registry → chunks → native embeddings → LanceDB search,
the local shape RAG pipelines need (issues #86 and #134).

```bash
uv sync --extra lancedb
uv run dbt-ml --project-dir examples/rag_chunks_pipeline run
uv run python examples/rag_chunks_pipeline/search_demo.py
```

- `document_registry` — one row per document; structure-preserving HTML
  extraction (`include_structure`) plus the common contract columns.
- `document_chunks` — one row per chunk, deterministic `chunk_id`, lineage
  carried from the registry. Swap the source `path:` for `gs://…` and the
  warehouse `type:` for `bigquery` to run the same models against the cloud.
- `chunk_embeddings` — a native `embed:` model using deterministic
  eight-dimensional vectors, so the default remains offline and reproducible.
  `chunk_search` inherits the exact provider identity and uses the same
  provider with its query task when text is supplied to vector search.
- `chunk_facts` — a native `llm:` map model (issue #144): one governed
  provider call per chunk turns text into typed rows (`claim`, `topic`),
  `output_cardinality: one`. Provenance columns (`llm_provider`, `llm_model`,
  `llm_input_hash`, `generated_at`, …) are appended for lineage.
- `chunk_entities` — a fan-out `llm:` model (`output_cardinality: many`): one
  call per chunk yields a list of entity rows, each keyed by a deterministic
  `llm_row_id`. Both llm models use the offline `deterministic` provider so the
  example needs no credentials; swap `provider: anthropic` (with an `llm:`
  block in `profiles.yml`) for real extractions.
- `chunk_search` — an incremental, public LanceDB search resource with exact
  vector search, full-text search, and a typed `source_uri` filter.

The public index is an explicit local-development opt-in in `profiles.yml`.
The portable `dbt-ml search` command and governed indexes with trusted
policy-prefilter enforcement are implemented. Full replacement and unsupported
store capabilities still fail closed. This public example uses the same typed
`RetrievalStore` API that a query service can wrap.

To evaluate real retrieval quality with Vertex AI, install the provider extra,
change `chunk_embeddings.embed.provider` to `vertex`, change its model to one
of `gemini-embedding-001`, `text-embedding-005`, or
`text-multilingual-embedding-002`, and add this target configuration:

```bash
uv sync --extra lancedb --extra vertex
gcloud auth application-default login
```

```yaml
embedding:
  provider: vertex
  timeout_seconds: 60
  provider_options:
    project: your-gcp-project
    location: global
    task_type: RETRIEVAL_DOCUMENT
    query_task_type: RETRIEVAL_QUERY
    auto_truncate: false
```

Vertex uses Application Default Credentials, including service-account ADC;
do not configure an API key. Keep `dimensions: 8` for a direct provider swap
in this example, or choose a larger output dimensionality before benchmarking.

## Golden-set retrieval evaluation (issue #137)

`chunk_search` declares two `retrieval_tests:` — deterministic, CI-friendly
checks over hand-labeled queries, run through the same `search()` API a real
caller uses:

```bash
uv run dbt-ml --project-dir examples/rag_chunks_pipeline eval
```

- `chunk_search_quality` (`chunk_search_golden`) — three correctly labeled
  queries ("paid time off", "password manager MFA", "remote work stipend")
  against the two document chunks. **Passes**: `recall_at_2`, `mrr_at_1`, and
  `ndcg_at_2` all hit 1.0, since full-text search correctly ranks each query's
  genuinely relevant chunk first.
- `chunk_search_quality_regression_demo` (`chunk_search_golden_degraded`) — one
  **deliberately mislabeled** query: asserts the security-policy chunk is
  relevant to a paid-time-off query. **Always fails** (`recall_at_1: 0.0`),
  proving the mechanism catches a real mismatch between labeled and retrieved
  relevance — this is the acceptance criterion for "an intentionally degraded
  configuration that fails," not a bug in the example.

`dbt-ml eval` therefore exits `1` on this project by design (both
`retrieval_tests` entries run together — selection is per search model, like
`dbt-ml test`, not per named test). Inspect `target/retrieval_eval.json` for
the full per-query artifact (store provenance, embedding identity, aggregate
metrics, threshold outcomes) to see both the passing and the intentionally
failing result.
