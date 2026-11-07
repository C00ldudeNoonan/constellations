# ADR-0055: Graph export is a CLI over per-fact edges, keyed by the agent-context `entity_id`

- **Status:** accepted
- **Date:** 2026-11-07
- **Prompted by:** #762 (ALE-423)

## Context

Entity links and relations land as warehouse tables and the concept cloud
renders them for a person, but nothing writes them as a graph a downstream
store (Neo4j, Graphiti, a triple store) can load. Without one, each consumer
joins `entity_links` to the relation table itself and re-derives node
identity, which is where two sources of truth start. Astrolabe wants a
temporal fact graph fed from stel (ALE-418).

Three choices had real alternatives.

## Decision

**A CLI command, `stel graph-export`, not a model kind.** It reads the two
published tables through the adapter and writes `nodes.jsonl`, `edges.jsonl`
and `graph_manifest.json` to a directory, the way `stel concept-cloud` writes
its bundle. The output is a deterministic function of two tables, with no
timestamp, so two exports of unchanged tables are byte-identical.

**One edge per extracted fact, not an aggregate.** Each edge is a relation row
and names its `relation_id`, `document_id`, `extractor`, `extractor_version`
and the resolvers that linked its ends. A graph store wants to ask which
document asserted a fact and with which extractor version; an aggregate
cannot answer that.

**A node's id is the agent-context `entity_id`, and `entity_name` is
required.** `make_entity_id(namespace, entity_name, canonical_json(canonical_id))`
is what `project_entity_link` writes, so an exported node joins to a governed
metric keyed on the same entity. The links table carries the namespace but
not `entity_name`, which is the operator's choice. The command takes it as a
required option instead of defaulting it: a default would mint ids that join to
nothing, and nothing would say so.

An edge exists only for an `asserted` relation whose two mentions resolve to
two different *matched* entities. Anything else is left out and counted by
reason in the manifest. A mention matched to two entities, or a repeated
`relation_id`, fails the export.

The first version writes JSON Lines only, as a full rewrite, and carries no
`iri` or validity interval. The manifest is versioned (`stel.graph_export/v1`).

## Alternatives rejected

- **A `graph_export` model kind.** It would put the output in the warehouse or
  behind a file target. A model's `materialization` is `full` or `incremental`,
  both warehouse tables, so a file target would be a new materialization for
  every adapter, for an output no warehouse query reads. The model kind's one
  advantage, incremental state, is deferred below rather than bought with that
  surface.
- **Reusing the concept cloud's edges.** `_aggregate_edges` collapses relations
  into weighted concept pairs and drops the document, which is the provenance
  this export exists to carry. The two share a rule for which links count; they
  do not share a grain.
- **Defaulting `entity_name` to a constant.** Rejected above: the failure is
  silent.
- **Incremental export in the first version.** A correct one has to remove
  edges whose documents disappeared, which is a stale-key problem of its own.
  No done-when item needs it, so it is separate work.

## Revisit when

- A consumer's graph is large enough that a full rewrite is the cost, which
  starts the incremental design (per-document shards and a sweep of vanished
  documents).
- ALE-419 lands IRIs, or ALE-422 lands validity intervals: add them as a
  schema-version bump, not as null columns.
- A consumer needs JSON-LD, Turtle or the Graphiti format: add each behind its
  own optional extra, importing lazily.
