# Notion pages, landed then rendered

The worked proof behind [land, then render](../../docs/saas-context.md)
(issue #352): SaaS context arrives through your EL tool as block rows, and
stel turns those rows back into documents with reading order, real heading
levels, and nesting — the inputs chunking and heading attribution need.

Notion is the proof because its block model is the most hostile to naive
flattening: a page is a tree of typed blocks, lists nest arbitrarily, and a
connector lands the tree as flat rows in fetch order. Anything that handles
Notion handles Confluence and Linear comfortably.

## The pipeline

| model | kind | what it does |
|---|---|---|
| `notion_pages` | `json` extraction | the landed page rows, as a stel model |
| `notion_blocks` | `json` extraction | the landed block rows, in the contract below |
| `page_documents` | `stel.text.transforms.render_blocks` | one markdown document per page |
| `page_chunks` | `chunk:` with `headings:` | section-attributed chunks carrying the page's title, parent, and database |

No new model kind and no connector. The renderer is the only new piece, and
it is vendor-neutral.

```bash
uv run stel --project-dir examples/notion_landed_pages run
uv run stel --project-dir examples/notion_landed_pages test
```

The fixture is three pages and 45 blocks under `landed/`, one JSON object per
file — the row shape a `warehouse://` source serves. It deliberately includes
what a real landing has: block ids out of position order, a three-deep nested
list, a paragraph under a bullet, a toggle with children, a block whose parent
the connector never landed, and a block type the renderer does not know.

## The block contract

`render_blocks` reads two relations. A connector's own tables map onto them
with one SQL staging model; nothing vendor-shaped enters stel.

**Pages** — one row per page: `page_id`, `title`, and any columns to carry
onto the document row (`include_fields`).

**Blocks** — one row per block:

| column | meaning |
|---|---|
| `block_id` | unique across the relation |
| `page_id` | the page the block belongs to |
| `parent_block_id` | the enclosing block, or null at the top level of the page |
| `position` | integer order among siblings |
| `type` | `heading_1..3`, `paragraph`, `bulleted_list_item`, `numbered_list_item`, `to_do`, `toggle`, `quote`, `callout`, `code`, `divider`, `child_page`, `child_database`, `column_list`, `column`, `synced_block` |
| `text` | the block's plain text, or null |
| `checked` | optional, for `to_do` |
| `language` | optional, for `code` |

A type outside the vocabulary renders as a paragraph when it has text and is
counted in `unknown_block_count`, so an unmapped connector type shows up in
the output instead of vanishing. A block whose parent chain never reaches the
page renders at the end of the page and is counted in `orphan_block_count`;
a dropped block is worse than a misplaced one. Both counts are ordinary
columns, so a `stel test` can assert they are zero on a corpus that should be
clean.

## Swapping in the landed tables

Point the two sources at the relations your EL tool lands and keep everything
else:

```yaml
sources:
  - name: notion_page_rows
    path: warehouse://landed.notion_page
    key_column: page_id
  - name: notion_block_rows
    path: warehouse://landed.notion_block
    key_column: block_id
```

`tests/test_notion_landed_example.py` performs exactly this swap over DuckDB
tables and asserts the rendered documents are byte-identical to the local
run.

If the connector's block table is not already in the contract's shape — most
land the block type and a rich-text array per row, and some land one table per
block type — a staging model does the mapping. Illustrative, not a specific
vendor's schema:

```sql
-- sql/notion_blocks.sql: project a connector's block rows onto the contract
select
    b.id                                  as block_id,
    b.page_id,
    nullif(b.parent_block_id, b.page_id)  as parent_block_id,
    b.sibling_index                       as position,
    b.type,
    b.plain_text                          as text,
    b.checked,
    b.code_language                       as language
from {{ ref('raw_notion_blocks') }} as b
where not b.archived
```

## What an edit costs

Pages are the incremental parents; blocks are a reference keyed to their page
(issue #364). Editing one block re-extracts that row, re-renders its page, and
re-chunks that page — nothing else:

```
notion_blocks     extraction  incremental   1 processed   44 skipped
page_documents    transform   incremental   1 processed    2 skipped
page_chunks       chunk       incremental   1 processed    2 skipped
```

## Heading attribution

The renderer emits the page title as `#` and shifts the page's own headings
one level down, so the document's outline sits under its title. With
`headings.pattern: '^#{1,6} (.+)$'` every chunk gets a `section`: the title
for text before the first heading, then each heading in turn. `in_text_metadata:
[title]` also puts the title inside the embedded text, so a chunk of "Encrypt
the disk" still says which runbook it came from.
