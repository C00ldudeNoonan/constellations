# rag_chunks_pipeline

Document → registry → chunks, the shape RAG pipelines need (issue #86).

```
uv run dbt-ml --project-dir examples/rag_chunks_pipeline run
uv run dbt-ml --project-dir examples/rag_chunks_pipeline show document_chunks
```

- `document_registry` — one row per document; structure-preserving HTML
  extraction (`include_structure`) plus the common contract columns.
- `document_chunks` — one row per chunk, deterministic `chunk_id`, lineage
  carried from the registry. Swap the source `path:` for `gs://…` and the
  warehouse `type:` for `bigquery` to run the same models against the cloud.
