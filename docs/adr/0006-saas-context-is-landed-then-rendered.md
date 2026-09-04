# ADR-0006: SaaS context is landed by an EL tool and rendered by stel

Status: accepted (2026-09-04). Issue #352.

## Context

The document-source seam (#84) has three implementations: local files,
`gs://`, and `warehouse://` (#322). The next question is always "what about
Notion, Slack, Confluence, Drive, Linear?", and the pull toward answering it
with another entry in `get_document_source()`'s scheme dispatch is constant.

What a connector lands for Notion is one row per block with a parent pointer
and a rich-text array. That is structurally faithful and semantically useless:
no reading order, no heading structure, no page hierarchy, which are exactly
the inputs chunking and heading attribution (#343) need.

## Decision

stel does not ship first-party SaaS connectors. SaaS context arrives through
the operator's EL tool, lands as a relation, and stel reads it with a
`warehouse://` source. stel owns the **renderer**:
`stel.text.transforms.render_blocks`, a core transform over a vendor-neutral
block contract (page, parent block, sibling position, type, text) that emits
one ordered markdown document per page. The vendor-specific column mapping is
a SQL staging model in the project.

Full position and recipe: [`docs/saas-context.md`](../saas-context.md).

## Alternatives rejected

- **Ship `notion://` and friends in-tree.** Commodity transport that other
  teams maintain as their whole business; OAuth, pagination, and backoff for
  a long tail of vendors, permanently; breaks the lean-core rule at the third
  connector. And it does not address the part that is actually broken, which
  is rendering.
- **stel as an MCP client, using vendor MCP servers as sources.** MCP cannot
  satisfy `discover()`: no bulk listing, no stable content identity, no
  pagination contract, no cheap change signal. It is a synchronous,
  agent-shaped protocol for fetching a few things a human is asking about —
  the opposite of "deterministically list 5,000 objects with stable identity
  without downloading them." stel serves MCP; it does not consume it for
  ingestion.
- **A generic `http://` source with per-source response mapping in YAML.**
  Turns project YAML into a scraping DSL and moves parser complexity into
  configuration, against the strict-Pydantic, validate-early boundary.
- **Do nothing.** The status quo technically worked (any table was already a
  source), but "can stel read my Notion?" was answered ad hoc and differently
  each time, and the renderer stayed unwritten.

Where the renderer lives had three candidates too:

- **A transform copied into each project**, or **a template shipped by
  `stel init`.** Zero core surface, but the tree walk — sibling order,
  nesting, heading levels, orphans, cycles, numbering — is the part people get
  wrong and it is identical across vendors. Copying multiplies the bugs.
- **An extraction backend.** Backends read one document from one file; the
  renderer joins two relations. Wrong shape.

A core transform keeps the hard, vendor-neutral part in one tested place and
the vendor-shaped part in the project's SQL.

## Evidence

`examples/notion_landed_pages` renders a fixture built to be hostile — rows
out of position order, a three-deep list, a paragraph under a bullet, a
dangling parent, an unknown block type — and
`tests/test_notion_landed_example.py` runs the same models over local files
and over DuckDB `warehouse://` tables and asserts byte-identical documents.
An edited block re-renders exactly one page (1 processed, 2 skipped), which
the incremental contract's keyed reference dependency (#364) makes cheap.

## Consequences

- A request for a SaaS source gets one answer: land it, then map its tables
  onto the block contract.
- If a first-party API source is ever justified, `docs/saas-context.md`
  records what it owes the seam: a change token named as such rather than a
  fake content hash, a fetch that re-verifies the token, and a listing bound
  designed together with pagination and backoff. Google Drive would be the
  better first, being file-grained; Slack the worst.
