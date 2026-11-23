# ADR-0065: Graph export for Graphiti is a record format with a loader, behind the `graphiti` extra

- **Status:** accepted; amends 0055, 0061, 0063
- **Date:** 2026-11-23
- **Prompted by:** #815 (ALE-603)

## Context

ADR-0055 named Graphiti as the downstream temporal fact graph and deferred a
Graphiti-shaped output, because its "loads with no hand transformation" claim
could not be proven without loading into one. ADR-0056 has since put the
interval a fact holds over on every edge, and ADR-0061 and 0063 made the export
per-document shards with pluggable formats.

Graphiti has three ways in: `add_episode`, where an LLM reads text and extracts
the graph; `add_triplet`, which takes finished nodes and an edge but still
calls an embedder; and `Model.save(driver)` on its own `EntityNode`,
`EntityEdge`, `EpisodicNode` and `EpisodicEdge`. stel already holds the
extracted graph, so only the last reaches a Graphiti store without a second
extraction, a credential or a vendor call.

## Decision

**`--format graphiti` writes the records Graphiti's models take, and
`stel graph-load` saves them.** The shard layout is unchanged (a document's
`nodes/` and `edges/` files, `.graphiti.jsonl`); each line is one record with a
`record` discriminator. The nodes shard holds the document's entity nodes, one
`episodic_node` (the document is the episode) and a `mentions` edge to each
entity; the edges shard holds one `entity_edge` per fact, its `episodes` naming
the document. A load is `to_model(record).save(driver)`, per shard, nodes
before edges.

**The export validates through the loader's own conversion.** `to_model` builds
the graphiti-core model for a record, with the fields the loader owns filled in.
The serializer calls it on every record before writing, so a graphiti-core
change that no longer accepts the shape fails the export, not a load. This is
why the format needs the extra at export time even though it writes plain JSON.

**The file carries valid time; the loader stamps transaction time.** `valid_at`
is `valid_from`, `invalid_at` is `valid_to`, `reference_time` is `observed_at`,
each null when stel does not know. `created_at`, and `expired_at` for an edge
whose `superseded_by` is set, are the load time: they describe when the graph
learned a fact, which stel did not measure, and a clock in the file would break
the byte-identical, fingerprinted shards. An episode with no observed time is
null in the file and stamped at load, because Graphiti requires one. Nothing is
invented in the file.

**Identity is a `uuid5` of the group id and the ADR-0063 `urn:`.** The same
entity is the same node in every document, a load is idempotent (Graphiti saves
merge on `uuid`), and two projects exported under different groups do not
collide. The group id is the project name restricted to what Graphiti accepts; a name
that needed restricting carries a digest of the original, so `sales/eu` and
`sales_eu` stay two groups.

**The extra is `graphiti`, lazily imported, and the load is credential-free to
test.** `graphiti-core>=0.31.2` (with `httpx`, which 0.31 imports without
declaring). Without it, `--format graphiti` and `graph-load` fail before
reading anything, naming `pip install 'stel[graphiti]'`. The loader reads the
manifest's shard list rather than walking the directory, checks each document's
fingerprint before saving it, refuses symlinks, and holds one shard at a time.
The credential-free test loads into Kuzu, an embedded database graphiti-core
drives from a file; `kuzu` is a dev dependency only. Graphiti's Neo4j save queries call a vector procedure that rejects a
null embedding, so the Neo4j load removes that clause from the query graphiti-core
builds (a unit test pins it against a recording driver). A Neo4j load is
credential-gated (`STEL_NEO4J_TEST_URI`, `_USER`, `_PASSWORD`). `graph-load`
reads `NEO4J_USER` and `NEO4J_PASSWORD` from the environment, never a flag, and
reports a connection failure by exception type alone.

## Alternatives rejected

- **`add_episode` with document text.** Graphiti would re-extract entities and
  edges with its own LLM, discarding stel's links, resolvers, extractor versions
  and intervals, and sending document text to a vendor. It also reads mention
  text stel never exports.
- **`add_triplet`.** Takes finished nodes and edges, but calls Graphiti's
  embedder, which needs a model credential and makes a load non-deterministic.
  Embeddings are left to the store (see Consequences).
- **Write plain JSON and leave loading to the consumer.** That is the claim
  ADR-0055 could not prove; a loader is the proof, and saving through the
  models is what makes a graphiti-core schema change visible.
- **Cypher or CSV for Neo4j.** It would bypass Graphiti's own indices and its
  Kuzu and FalkorDB backends, and encode Graphiti's storage schema in stel.
- **Stamp `created_at` and `expired_at` in the file.** Either a run-clock
  timestamp that rewrites every shard on every export, or a value borrowed from
  `valid_to` that is not when the graph learned anything.

## Consequences

- No embeddings are written. Graphiti's vector search over node names and edge
  facts needs a backfill with the store's own embedder; text search works
  without it.
- Undirected relations (co-occurrence) are one directed Graphiti edge from the
  lower `entity_id`; `directed: false` is in its attributes.
- Kuzu is deprecated upstream in graphiti-core. The test target moves if it is
  removed; the format and the loader do not depend on it.
- A graphiti-core release that changes a model's fields can rewrite every
  shard once on the next export, or fail it with the model's own message.
- Bounded memory is unchanged for the export; the load holds one shard.
