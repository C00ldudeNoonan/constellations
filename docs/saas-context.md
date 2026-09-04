# SaaS context: land, then render

Status: accepted position (issue #352, [ADR-0006](adr/0006-saas-context-is-landed-then-rendered.md)).
Worked proof: [`examples/notion_landed_pages`](../examples/notion_landed_pages/).

"Can stel read my Notion?" — and Slack, Confluence, Drive, Linear — is the
question the document-source seam gets asked most, and the obvious answer is
another entry in `get_document_source()`'s scheme dispatch. That answer is
wrong, and this page exists so the argument is had once.

## The position

**Ingestion transport is not stel's job. Rendering is.**

Two things are true at the same time:

1. Moving SaaS data into a warehouse is a solved, commodity problem that other
   teams maintain as their whole business. Fivetran, Airbyte, dlt, Nango, and
   every warehouse vendor's native connectors land Notion, Slack, Zendesk, and
   Linear as tables today. Competing there means owning OAuth flows, pagination
   quirks, and rate-limit backoff for a long tail of vendors, permanently, and
   it breaks the lean-core rule the moment the third connector lands.
2. What those tools produce is structurally faithful and semantically useless.
   A Notion connector lands one row per block with a parent pointer and a
   rich-text array. That is the raw material for a document, not a document:
   nothing in the table has reading order, heading structure, or page
   hierarchy, which are exactly the inputs chunking and
   [heading attribution](reference.md#heading-attribution) need.

So SaaS context arrives through the EL tool you already run, lands as a
relation, and stel reads it with a
[`warehouse://` source](reference.md#warehouse-table-sources). Since #322 any
table is a stel source. The gap was never transport; it was the renderer that
turns block rows back into a document. `stel.text.transforms.render_blocks`
is that renderer.

## The recipe

```
EL tool  ─►  landed.notion_page / landed.notion_block   (one row per page, per block)
          ─►  warehouse:// sources + json extraction    (stel models, incremental)
          ─►  render_blocks                              (one markdown document per page)
          ─►  chunk: with headings:                      (section-attributed chunks)
          ─►  embed: / search:                           (the ordinary tail)
```

```yaml
sources:
  - name: notion_page_rows
    path: warehouse://landed.notion_page
    key_column: page_id
  - name: notion_block_rows
    path: warehouse://landed.notion_block
    key_column: block_id

models:
  - name: page_documents
    depends_on: [ref('notion_pages'), ref('notion_blocks')]
    transform:
      type: python
      module: stel.text.transforms.render_blocks
      options:
        pages: notion_pages
        blocks: notion_blocks
        include_fields: [parent_page_id, database_id, properties_json]
    materialization: incremental

  - name: page_chunks
    depends_on: [ref('page_documents')]
    chunk:
      strategy: recursive
      headings: {pattern: '^#{1,6} (.+)$'}
      in_text_metadata: [title]
    materialization: incremental
```

The renderer consumes a **vendor-neutral block contract** — `block_id`,
`page_id`, `parent_block_id`, `position`, `type`, `text`, and optional
`checked` and `language` — and a page relation with `page_id` and `title`.
Your connector's tables map onto it with one SQL staging model; that mapping
is the only vendor-shaped code in the project, and it lives in the project.
The example README documents the contract and the block-type vocabulary.

What comes out is ordered markdown with real heading levels, nested lists,
code fences, quotes, and links to child pages. The page title renders as the
top heading and the page's own headings nest one level under it, so
`headings.pattern` attributes every chunk to a section and text before the
first page heading belongs to the page rather than to nothing. Landed data is
imperfect, so the renderer accounts for it instead of guessing: a block whose
parent was never landed renders at the end of its page and is counted in
`orphan_block_count`; a block type outside the vocabulary renders as text and
is counted in `unknown_block_count`. Both are columns, so `stel test` can
require them to be zero on a corpus that should be clean.

Pages are the incremental parents and blocks a reference keyed to their page,
so editing one block re-extracts one row, re-renders one page, and re-chunks
one page.

## Why the renderer is a core transform

Three places it could have lived:

- **A `python:` transform copied into each project.** Zero core surface, but
  the tree walk — sibling order, nesting depth, heading levels, orphan and
  cycle handling, list numbering — is the part people get wrong, and it is
  the same for every vendor. Copy-paste multiplies the bugs.
- **A template shipped by `stel init`.** The same copy, one step removed.
- **An extraction backend.** Backends read one document from one file; the
  renderer reads many rows and joins two relations. Wrong shape.

It is a core transform because the hard part is vendor-neutral and the
vendor-specific part is a SQL projection. That keeps vendor shape out of the
core entirely, which was the point.

## Notion was the proof for a reason

Notion's block model is the most hostile to naive flattening: a page is a
tree of typed blocks, lists nest arbitrarily, headings sit at any depth, and
a connector lands the tree as flat rows in fetch order. Anything that handles
Notion handles Confluence and Linear comfortably. The example fixture carries
the failure modes a real landing has — rows out of position order, a
three-deep list, a paragraph under a bullet, a toggle with children, a
dangling parent, an unknown block type — and the test suite runs the same
models over local files and over `warehouse://` tables and asserts the
documents are identical.

## What we owe the seam if a first-party API source is ever justified

Not proposed. Recorded so the analysis exists when the question returns.

- `discover()` requires a `content_hash` derived from a listing that changes
  when the object's bytes change. An API gives `last_edited_time`: a change
  *token*, not a hash. It over-triggers (a no-op edit forces re-extraction,
  which is real provider spend under `llm:` extraction) and can under-trigger
  (comments, linked-database contents). A change token should be named as such
  in the contract, not allowed to masquerade as a hash.
- `fetch()` is cheap and byte-verifiable today: GCS pins a generation, local
  sources snapshot into verified scratch. An API fetch is N paginated calls
  rendered into a file, with nothing to verify against what discovery saw. It
  needs the generation-pin analogue — re-read the change token at fetch and
  fail loudly on drift — or the "extraction cannot follow a change after
  discovery" guarantee in `sources/base.py` quietly stops holding.
- `max_objects` as a listing bound does not compose with cursor pagination
  plus rate limits; the bound and the backoff have to be designed together.

If one is ever built, Google Drive is the better first: file-grained, so it
fits the existing object shape with far less new contract. Slack is the worst
candidate — message grain, threading, retention policy, and the most sensitive
corpus of the three.

## Out of scope

Auth and OAuth for third-party SaaS; incremental state for externally managed
EL pipelines (the EL tool's job); any change to the `local`, `gs://`, or
`warehouse://` sources.
