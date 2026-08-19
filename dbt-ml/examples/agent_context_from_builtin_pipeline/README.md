# agent_context_from_builtin_pipeline

Proves that a project built entirely on the built-in `extraction:`/`chunk:`
primitives can become `agent_context/v1`-discoverable (issue #300), by wrapping
their real output in two thin `transform:` models instead of hand-rolling the
contract's ~30 fields per row.

```bash
uv sync --extra lancedb
uv run dbt-ml --project-dir examples/agent_context_from_builtin_pipeline run
```

No network credentials are required — a local duckdb warehouse, a local
LanceDB store, and the offline `deterministic` embed provider.

## DAG

```
raw_research_notes        extraction: backend json          — built-in, no agent_context
        |
        +-----------------------------+
        v                             v
document_registry           research_note_chunks   chunk: strategy recursive, real splitter
transform:, agent_context:          |                — built-in, no agent_context
  grain document_registry           |
        +-------------+-------------+
                       v
               document_chunks      transform:, agent_context: grain document_chunks
                       v
               chunk_embeddings     embed: provider deterministic
                       v
               context_search       search: access governed, store local
```

`raw_research_notes` and `research_note_chunks` are ordinary dbt-ml
primitives — nothing about them changes for this example, and neither can
declare `agent_context:` directly (see
[docs/architecture/agent-context-v1.md](../../docs/architecture/agent-context-v1.md#runtime-and-artifact-integration)
for why). `document_registry` and `document_chunks` are the contract-emitting
wrapper transforms:

```python
# transforms/document_chunks.py (abridged)
from dbt_ml.agent_context import project_document_chunk_row

for chunk in deps["research_note_chunks"].iter_rows(named=True):
    parent = registry_by_source_key[chunk["source_key"]]
    row = project_document_chunk_row(
        parent,
        chunk_index=chunk["chunk_index"],
        text=chunk["text"],
        upstream_unique_id="model.agent_context_from_builtin_pipeline.research_note_chunks",
        invocation_id="agent-context-builtin-pipeline-v1",
        chunker_identity=f"{chunk['chunk_strategy']}:400:50:v1",
    )
```

`project_document_chunk_row` computes `chunk_id`/`context_id`/
`chunk_content_hash` and copies every bitemporal/policy/freshness field from
the parent `document_registry` row verbatim, so the two ~15-20 line transforms
here replace the ~75-90 lines of hand assembly in
[`examples/metric_evidence_agent/`](../metric_evidence_agent/) — which also
predates real `chunk:` splitting (it treats each document as one chunk). This
example's `chunk_size: 400`/`chunk_overlap: 50` produces several real chunks
per document, and `document_chunks` projects one contract row per real chunk
row — not a shortcut.

`research_note_chunks["document_id"]` is the built-in pipeline's own internal
id (`versioning.compute_document_id`) — a different id space than the
contract's `document_id` (`agent_context.make_document_id`), which
`document_registry` computes. They share a column name by coincidence, so
`document_chunks.py` joins the two through `source_key`, never through
`document_id` — see the comment in that file.

`context_entity_links` is intentionally out of scope here; see
`examples/metric_evidence_agent/` for that grain.

## Verify it's MCP-discoverable

After `dbt-ml run`, `target/manifest.json` carries an `agent_context`
descriptor on `document_registry`/`document_chunks`, and `context_search`
resolves both ancestors — the same catalog logic the MCP server's
`list_context_models` uses (`src/dbt_ml/mcp_server/catalog.py`) will surface
it.
