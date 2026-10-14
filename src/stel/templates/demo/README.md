# The stel demo: an on-call knowledge base

Acme Cloud is a made-up company with a small pile of documents: three incident
postmortems, three runbooks, two months of release notes, and eight customer
support tickets. This project turns that pile into a tested, searchable,
entity-linked knowledge base that an agent can query with citations. It runs
offline, needs no API keys, and keeps everything in one DuckDB file.

```bash
stel build
```

That command runs 13 models and then every test they declare, in a few
seconds. The rest of this page walks through what it built and what you can
do with it.

## What gets built

```
raw_docs, raw_tickets
  └─ document_registry
       └─ passages
            └─ document_chunks
                 ├─ chunk_embeddings ─ knowledge_search
                 └─ entity_mentions
                      ├─ entity_relations
                      └─ entity_links
                           ├─ context_entity_links
                           ├─ system_health   (SQL)
                           └─ system_names
```

| model | what it is |
|---|---|
| `raw_docs`, `raw_tickets` | Built-in extraction: markdown frontmatter and body, JSON fields. |
| `document_registry` | One row per document, with the provenance an agent needs to cite it. |
| `passages` | The built-in chunker. Each chunk records the `## ` section it came from. |
| `document_chunks` | Citable passages: a chunk plus its document's version, dates, and source. |
| `chunk_embeddings` | One vector per passage. |
| `knowledge_search` | A vector and full-text index, stored inside the warehouse file. |
| `entity_mentions` | Every mention of a system declared in `stel_project.yml`. |
| `entity_links` | Each mention resolved to its canonical system: "checkout API" is the Payments API. |
| `entity_relations` | Two systems named in the same sentence. |
| `context_entity_links` | Which passages are about which system, for agents. |
| `system_health` | Plain SQL over the above: postmortems and tickets per system. |
| `system_names` | Each system's name, description, and class, for the star map. |

## 1. Query the results

```bash
stel show system_health
```

```
│ system             ┆ postmortems ┆ tickets ┆ has_runbook ┆ last_postmortem │
│ Webhook Dispatcher ┆ 2           ┆ 3       ┆ true        ┆ 2026-07-06      │
│ Payments API       ┆ 1           ┆ 2       ┆ true        ┆ 2026-03-14      │
│ Search Service     ┆ 1           ┆ 2       ┆ true        ┆ 2026-05-22      │
│ ...                                                                         │
│ Auth Gateway       ┆ 0           ┆ 1       ┆ false       ┆ null            │
```

That last row is an answer nobody wrote down: the Auth Gateway gets customer
tickets, but no runbook mentions it.

It is all one file, so any DuckDB client can read it:

```bash
duckdb target/warehouse.duckdb "select system, tickets from __PROJECT_NAME__.system_health"
```

## 2. Search it

```bash
stel search --model knowledge_search --query "connection pool exhausted" --mode text
stel search --model knowledge_search --query "webhooks late" --mode text --filter kind eq ticket
```

The first returns the checkout postmortem's root cause, then the Payments API
runbook. The second returns only tickets. Every hit carries its title, its
document kind, and the section it came from.

`--mode text` is BM25 full-text search. The demo's embeddings come from the
offline `deterministic` provider, which is a hash rather than a language
model, so `--mode vector` runs but does not rank by meaning. Point
`chunk_embeddings` at a real provider (see "Embedding models" in the stel
reference) and `--mode vector` and `--mode hybrid` rank by meaning too, with
no other change.

## 3. Change a document and rebuild

Add a paragraph to one file:

```bash
printf '\n## Update\n\nThe Auth Gateway now has its own runbook.\n' >> data/docs/release-notes-2026-08.md
stel run
```

```
model                 kind        mater.         processed   skipped  deleted    rows
raw_docs              extraction  incremental            1         7        0       1
raw_tickets           extraction  incremental            0         8        0       0
passages              chunk       incremental            1        15        0       2
chunk_embeddings      embed       incremental            2        29        2       2
knowledge_search      search      incremental            2        29        2       2
```

One document was re-extracted and two passages were re-embedded and
republished. The other fifteen documents and twenty-nine passages were
skipped. With a paid embedding provider, that is the difference between paying
for two passages and paying for all of them.

`stel plan` says what a run would reprocess before it spends anything.

## 4. Browse the lineage

```bash
stel docs generate
stel docs serve
```

A static site with every model, its columns, its tests, and the DAG.

## 5. See what the corpus is about

```bash
stel concept-cloud --linking-model entity_links --relation-model entity_relations \
  --names-model system_names --dimension class=system_names.class \
  --title "Acme Cloud systems" -o target/star-map.html
```

`target/star-map.html` is a self-contained 3D map of every system the
documents mention, sized by how often they are named and joined where two are
named together. Open it in a browser. Colour it by `class` to separate services
from datastores.

To read the passages behind a star, serve it instead (this needs the `mcp`
extra, below):

```bash
STEL_MCP_PRINCIPAL_ID=me stel concept-cloud --linking-model entity_links \
  --relation-model entity_relations --names-model system_names \
  --dimension class=system_names.class --context-model knowledge_search \
  --serve -o target/star-map.html
```

Click a star, then **Show me a sentence**.

## 6. Give it to an agent

stel serves the index over MCP, with citations and lineage on every result.
Install the extra (`uv tool install 'stel[mcp]'`, or `uv add 'stel[mcp]'` in a
project), then register the server with your client. For Claude Code, from
this directory:

```bash
claude mcp add __PROJECT_NAME__ -e STEL_MCP_PRINCIPAL_ID=me -- stel --project-dir "$PWD" mcp serve
```

Then ask: *"Has checkout broken like this before? Cite the postmortem."* An agent
can also scope a search to what the project declares, such as
`{"entity_scope": {"class": "datastore"}}` for passages about any datastore.

## 7. Make it yours

- Replace `data/docs/` and `data/tickets/` with your own files, and edit the
  `systems` vocabulary in `stel_project.yml` to name what your documents are
  about.
- Swap `deterministic` in `models/chunk_embeddings.yml` for a real embedding
  provider.
- Uncomment the `motherduck` target in `profiles.yml` to build the same
  project into MotherDuck.
- `stel emit-dbt-sources` writes a dbt `sources.yml`, so a dbt-duckdb project
  can read these tables like any other source.
