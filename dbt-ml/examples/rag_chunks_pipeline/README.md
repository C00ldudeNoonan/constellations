# rag_chunks_pipeline

Document → registry → chunks → fixture embeddings → LanceDB search,
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
- `chunk_embeddings` — a deterministic eight-dimensional fixture transform.
  It keeps the example offline and reproducible; replace it with a real
  embedding provider before evaluating retrieval quality. The search resource
  declares its full external embedding identity because native `embed:` model
  inheritance remains issue #138.
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
Governed indexes, policy enforcement, online replacement, and the portable
`dbt-ml search` command fail closed or remain follow-up work. The example query
uses the same typed `RetrievalStore` API that a query service can wrap.
