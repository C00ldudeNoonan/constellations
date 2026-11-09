# ADR-0056: Supersession is a full-materialization step over the relation table, keyed by entity link

- **Status:** accepted
- **Date:** 2026-11-08
- **Prompted by:** #766 (ALE-422), following ADR-0055

## Context

An extracted fact is true for a window ("the Fed is in a hiking cycle"). Relation
rows carried no window, so a later document either added a contradicting row
with no link to the first, or a reprocess replaced it. As-of questions and
backtests need to know what was believed when, and a temporal graph store
(Graphiti, ADR-0055) wants edges it can load without guessing which one a later
fact invalidated.

Two properties of the existing pipeline constrain where this can live.

`extract_relations` is an incremental one-to-many step. Each document's
relation rows are replaced as a unit, keyed on `document_id`, and the step sees
one document's mentions at a time. A newer document closing an *older
document's* row is a write to a row the changed document does not own, which an
incremental contract cannot express.

A relation row names two *mentions*, not entities. "One current object per
subject" needs a subject identity, and only a link table (`link_entities`)
supplies one.

## Decision

**The extractor stamps; a separate step supersedes.** `extract_relations` writes
`valid_from` and `valid_to` when given the new `effective_date_field` option (a
mentions column), and `observed_at` when given `observed_at_field`. A model that
names neither writes none of the columns. `valid_to` is always null from the
extractor. Nothing is read from the clock, so rows stay byte-identical across
reruns and `relation_id` is unchanged.

**`stel.text.transforms.supersede_relations` is a `full` materialization over
the relation table and the link table.** It reads every relation row, never
deletes one, and writes the same rows with `valid_to` closed and a new
`superseded_by` column. Because it recomputes from the whole table each run, a
forced reprocess of the extractor yields the same intervals, and an
incremental extractor run keeps the rows of unchanged documents, so closed rows
survive.

**`functional: true` is declared on the relation type** (`relations:` in
`stel_project.yml`), so the policy lives next to the domain/range pairing it
qualifies. Every pairing declared under one name must agree.

**A functional relation is keyed by `(relation_type, subject entity)`.** The
subject entity is the matched entity link of the subject mention. A row closes
at the earliest strictly later `valid_from` among rows of the same key that
assert a *different* object entity. A restatement of the same object closes
nothing, so corroboration is not mistaken for change. Rows with the same
`valid_from` and different objects stay open: nothing orders them.

**Failures are loud where identity or order is missing.** The step fails when a
link or relation column is absent, when a mention is matched to two entities,
when a functional row has no `valid_from`, and when a functional relation type
is undirected. A row that is not `asserted`, or whose subject or object mention
is unlinked, passes through unchanged and is counted in the run log, because it
names no entity to compare.

**The as-of read is a pure function**, `relations_as_of`, over a relation frame.
An interval is half-open, `[valid_from, valid_to)`, so at the instant a newer
fact begins only the newer fact is valid. Unknown validity (null `valid_from`)
is excluded unless the caller asks for it, by a required keyword.

**Graph export moves to `stel.graph_export/v2`.** Edges carry `valid_from`,
`valid_to`, `observed_at` and `superseded_by`, as ADR-0055 said they would
arrive: a version bump, not null columns in v1. A relation model without the
columns exports them as null, which is what "validity unknown" means.

## Alternatives considered

### Compute intervals at export or read time

No new step, no new table: sort a subject's rows and close them when read. It
loses because the warehouse would hold rows that disagree with the graph an
operator exports, and every consumer (SQL, MCP, the concept cloud) would have to
re-derive the same closing rule. Two sources of truth is the problem ADR-0055
exists to avoid.

### Supersede inside `extract_relations`

One step and one table. It cannot work incrementally: closing document A's row
because document B arrived writes outside B's parent key, so a changed B would
have to re-read and rewrite every other document that mentions the subject. It
would also make the extractor read the link table, which today it does not.

### An incremental supersession step

Closed rows are exactly what an incremental delete-on-parent, upsert-on-child
contract cannot represent: the parent that changed is not the parent whose row
moves. A full rewrite of a derived table is correct by construction and its
cost follows the relation table, not the corpus.

### Key a subject on its normalized text

Works without a link table, and closes the wrong "Fed" on the first collision.
Supersession deletes nothing, but a wrongly closed interval is a wrong answer
to an as-of question that looks like a right one.

### `observed_at` from the run clock

Closer to "when stel saw it", and it makes every rerun a new row value, which
defeats byte-identical republish (ALE-302) and the deterministic reprocess this
design promises. The document's own observation time, when the operator names
it, is the stable version of the same fact.

## Consequences

- Supersession needs `link_entities` upstream. A project without entity links
  can stamp validity and filter as-of, but cannot declare a relation functional.
- The columns are opt-in because adding a column to an incremental table is a
  schema change the run refuses (`on_schema_change: fail`, the default). Always
  emitting them would have failed every existing incremental relation model on
  upgrade; `include_mention_text` already adds columns conditionally for the
  same reason. Opting in is a `--full-refresh`, which reprocesses and stamps
  every document together.
- `valid_to` on the extractor's table is always null. The closed intervals are
  in the supersession model's table, so an as-of read of closed facts must read
  that one.
- Rows with the same `valid_from` and different objects stay open together. A
  same-document double assertion is the usual cause, and the run log counts it.
- `superseded_by` names the row that closed it, not a chain: a row closed by a
  fact that was itself later closed still points at the first.
- Graph export v1 consumers keep working against a v1 directory and must opt in
  to v2 fields.

## Revisit when

- A relation model is incremental on something other than the document, and the
  full rewrite is the cost.
- A consumer needs two valid objects at once for a relation that is *almost*
  functional (a per-class cardinality), which wants `max_current: n` and not a
  boolean.
- Bitemporal belief ("what did we think on date X") is asked of the table
  itself: it needs `observed_at` in the interval logic, which this ADR leaves
  out on purpose.
