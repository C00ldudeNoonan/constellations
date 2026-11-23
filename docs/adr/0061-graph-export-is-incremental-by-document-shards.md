# ADR-0061: Graph export is incremental by document shards, and a vanished document is swept the way a stale key is

- **Status:** accepted; amends 0055
- **Date:** 2026-11-22
- **Prompted by:** #807 (ALE-601)

## Context

ADR-0055 shipped `stel graph-export` as a full rewrite of `nodes.jsonl` and
`edges.jsonl` that read both source tables whole (an `EXCEPTION` row in
`test_bounded_memory`), and named incremental export under "Revisit when": a
correct one has to remove the edges of documents that disappeared, which is a
stale-key problem of its own. That problem now has a design in the repository.
`BoundedReconciler.iter_stale_pages` (#153, #764, ADR-0037) says a published
key is stale when it is absent from the relation *as this publish read it*, and
fixes the rules around that: the stream of keys is validated to be strictly
ascending rather than trusted, an empty upstream makes every published key
stale, and deletions follow a successful read.

## Decision

**A document is the shard.** Its nodes and edges are two files named by the
first 32 hex characters of the SHA-256 of the document id, under `nodes/<xx>/`
and `edges/<xx>/`. The manifest (`stel.graph_export/v3`) holds, for each
document that has anything to export, its shard name, row counts and a
fingerprint (a SHA-256 over the two files' bytes). The fingerprint is of the
*output*, not of the input rows: any change that moves a shard's bytes (an edge
row, a node's class from a changed vocabulary, `--entity-name`) rewrites it, so
no separate "context changed" check exists to forget.

**Both tables stream, ordered by `document_id`, and are joined one document at
a time.** Each is read with `table_snapshot(order_by=("document_id", <key>),
key_column=<key>)`, where the key is `entity_link_id` and `relation_id`. The key
makes the order total and refuses a repeated or null id before the first row
(the old whole-frame duplicate check, kept). The two streams are merged by
`document_id` in Python, which checks that each stream really is ascending in
Python's string order and fails otherwise: a warehouse collation that disagreed
would pair the wrong documents silently. Resident rows are one batch plus one
document's links and relations. The `read_table` call sites are gone and the
`test_bounded_memory` exception with them; the manifest's per-document entries
are the distinct-key term the contract already allows.

**Stale means the reconciler's stale.** A document the previous manifest names
and this read did not produce (absent from both tables, or present with nothing
left to export) has its files and entry removed. Absence is judged against
exactly the rows this export read; an empty source removes every shard; nothing
is removed until the whole read has succeeded and validated.

**Stage, then commit.** Changed shards are written to a scratch directory inside
the output directory; only after the read, the snapshots' own change validation
and every document's checks have passed does the commit unlink the old manifest,
move the shards in, remove the stale ones and write the manifest last. A failed
run leaves the previous export exactly as it was. A directory with a manifest is
still complete. A crash between unlinking the manifest and writing the new one
leaves no manifest, so the next run treats the directory as unmanaged: it
rewrites every shard and removes shard-named files that nothing names, in
two-hex-character directories of the two trees and nowhere else.

**The manifest is the state, and it lives with the output.** Adapter state is
for warehouse outputs (AGENTS.md); this output is a directory, and state kept
anywhere else could disagree with it. Delete the directory and the next export
is a full one, which is the right answer; delete a shard and it is rewritten
because its file is checked, not only its fingerprint.

**An earlier export is protected.** A v1/v2 manifest is read as "legacy": the
new export commits first and only then are `nodes.jsonl` and `edges.jsonl`
removed. A manifest that is unreadable, malformed or from a newer schema is
refused, and nothing in the directory is changed.

## Alternatives rejected

- **Keep `nodes.jsonl` and `edges.jsonl` and rewrite them from shards.** A
  consumer's files would not change shape, but the two files are concatenations
  of every document, so one changed document rewrites both: the done-when (only
  that document's shard and the manifest) is not met, and the I/O is the cost
  this was meant to remove.
- **Global nodes, a node written once.** A node is shared by every document
  that links it, so removing a document means knowing whether another still
  links the entity: a reference count per entity in the manifest, or a second
  pass over all links. Per-document node rows make a shard independent of every
  other, at the price that a loader merges on `entity_id` (the Neo4j example
  already did) and `totals.node_rows` counts rows, not distinct entities.
- **Fingerprint the input rows.** It would skip building an unchanged shard,
  but it is blind to anything that changes output without changing rows
  (vocabulary classes, `entity_name`, the export's own code) and needs a
  context digest to compensate. Building every shard is cheap beside the read
  that feeds it, and the output fingerprint cannot miss a change.
- **State in the adapter's state tables.** It would reuse `StateScope`, but the
  state would then be able to say "unchanged" about a directory that was
  deleted or edited by hand.
- **Delete during the stream.** Simpler, and a failed run would leave a
  directory with documents gone and others stale, under a manifest that no
  longer matched: the failure mode the manifest-last rule exists to prevent.
- **Hold the whole seen-set for the sweep.** The sweep is a set difference
  between the previous manifest's keys and this run's, both already resident
  as manifest entries, so no second structure grows with the corpus.

## Consequences

- The layout changed (schema v3). A loader written for v1 or v2 reads `nodes/`
  and `edges/`; `docs/reference.md` and the CHANGELOG say so.
- An unchanged corpus still costs a full read of both tables (bounded in
  memory, not in time). Skipping that read needs a per-table generation check
  (`TableReadSnapshot.generation_fingerprint` exists where an adapter can
  supply it before the read); it is separate work.
- The manifest is O(documents) and is read and written whole. At millions of
  documents it is hundreds of megabytes; moving it to JSON Lines is the next
  step if that is reached.
- A relation's mentions resolve against its own document's links. That is how
  `extract_relations` produces them; a model that related mentions across
  documents would see those relations counted as `unlinked_mention`.
- Rows must sort as Python strings do. DuckDB and BigQuery order strings by
  bytes, which agrees for the ids stel mints; an adapter that does not fails
  the export with a message, not silently.

## Revisit when

- The manifest, or the full read of an unchanged corpus, is the cost: a JSON
  Lines manifest, and a generation check that skips the read.
- ALE-419 lands IRIs: a schema bump, as 0055 said.
