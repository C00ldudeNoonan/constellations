# Governed context over MCP

dbt-ml exposes a read-only MCP stdio server for document context governed by
`agent_context/v1`. It complements the dbt MCP server rather than proxying it:

- use dbt-ml MCP for document discovery, retrieval, citations, and provenance;
- use dbt MCP for governed metrics, Semantic Layer queries, and dbt metadata.

The initial server exposes exactly four tools:

| Tool | Purpose |
| --- | --- |
| `list_context_models` | Discover only context models available to the caller. |
| `search_context` | Query the portable dbt-ml search API and return governed context. |
| `get_document` | Fetch one authorized document version as bounded, paginated chunks. |
| `get_context_lineage` | Trace an authorized document, context, chunk, or search result. |

There are no build, mutation, metric-query, answer-generation, path-read, or
source-URL-read tools.

## Prepare a project

Install the MCP extra plus the retrieval or warehouse extras used by the
project:

```bash
uv add 'dbt-ml[mcp,lancedb]'
```

The server reads `target/manifest.json` and `target/run_results.json`, then
reads authorized rows through the configured warehouse adapter. Run the
project before starting the server so those artifacts and relations agree:

```bash
uv run dbt-ml run
```

An exposed search index must descend from `agent_context/v1`
`document_registry` and `document_chunks` relations. Its `id_field` must be
`context_id` or `chunk_id`. Optional `context_entity_links` descendants add dbt
entity types and links to responses.

Those `document_registry`/`document_chunks` relations must come from
`transform: {type: python}` models that declare `agent_context:` — the
built-in `extraction:`/`chunk:` primitives cannot claim the contract, so a
search index built directly on them will not surface here even once fully
populated. `dbt_ml.agent_context.project_document_registry_row`/
`project_document_chunk_row` turn an existing `extraction:`/`chunk:` pipeline
into a contract-emitting transform in roughly 15-20 lines each; see
`examples/agent_context_from_builtin_pipeline/` for a complete project
(`extraction:` → `chunk:` → two thin `transform:` wrappers → `embed:` →
`search:`) that compiles, runs, and is discoverable through
`list_context_models`. See [Agent context contract
v1](architecture/agent-context-v1.md#runtime-and-artifact-integration) for why
and how to wrap an existing extraction/chunk pipeline in a contract-emitting
transform.

## Local principal

The stdio MVP resolves one deterministic principal from operator-owned
environment variables:

| Variable | Meaning |
| --- | --- |
| `DBT_ML_MCP_PRINCIPAL_ID` | Required opaque local subject identity. |
| `DBT_ML_MCP_TENANT_ID` | Trusted tenant claim. |
| `DBT_ML_MCP_ACCESS_GROUPS` | Optional comma-separated access groups. |
| `DBT_ML_MCP_POLICY_CLAIMS` | Optional JSON object of additional trusted policy claims. |

Every call fails closed without `DBT_ML_MCP_PRINCIPAL_ID`, including calls to
public context models. Authorization values are never accepted as tool
arguments. Governed search predicates are compiled from the principal, and
warehouse rows are independently rechecked after search and on every document
or lineage lookup.

This environment resolver is for deterministic local stdio use. The service
accepts injectable principal and authorization interfaces so a later network
transport can supply authenticated identities without changing tool contracts.

## Generic client configuration

Most stdio MCP clients accept a configuration shaped like this. Replace the
project path and add the extras required by its active profile:

```json
{
  "mcpServers": {
    "dbt-ml-context": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "dbt-ml[mcp,lancedb]",
        "dbt-ml",
        "--project-dir",
        "C:/path/to/document-project",
        "mcp",
        "serve"
      ],
      "env": {
        "DBT_ML_MCP_PRINCIPAL_ID": "local-analyst",
        "DBT_ML_MCP_TENANT_ID": "economic-research",
        "DBT_ML_MCP_ACCESS_GROUPS": "analysts,reviewers",
        "DBT_ML_MCP_POLICY_CLAIMS": "{\"classification\":[\"internal\"]}"
      }
    },
    "dbt": {
      "command": "uvx",
      "args": ["dbt-mcp"]
    }
  }
}
```

Configure dbt MCP credentials and toolsets separately using the
[dbt Labs setup guide](https://docs.getdbt.com/docs/dbt-ai/about-mcp). A client
can then combine dbt metrics with dbt-ml citations without either server
assuming ownership of the other's domain.

For a source checkout, replace the `--with dbt-ml[...]` portion with
`--project C:/path/to/dbt-ml/dbt-ml`.

## Limits and errors

`dbt-ml mcp serve --help` lists controls for operation timeout, requests per
minute, concurrency, response bytes, and warehouse scan rows. Tool requests
also have bounded result and page sizes. Document cursors are tied to the
model, document ID, and exact document-version ID, so they cannot be replayed
against another document.

Every response uses the `mcp_context/v1` schema and returns a structured error
with a stable code. Unauthorized and nonexistent models or records both return
`not_found_or_denied`; the server does not reveal which condition occurred.
Timeout, request-rate, and concurrency errors are retryable. Response and scan
limit errors require a smaller request or an operator limit change.
