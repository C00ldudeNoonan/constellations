# ADR-0037: A segmented search group declares its partitions

- **Status:** proposed
- **Date:** 2026-10-22
- **Prompted by:** #620

## Context

`for_each` expands a template into ordinary models and tags each variant with
the template name. It does not divide an upstream relation. A `search:` model
still reads every row of its one dependency, and its stale-row probe checks
keys against that whole upstream relation. Putting `for_each` on the SEC
search model today would publish the same corpus into every collection.

The query half is also per model. MCP's catalog creates one resource for each
search index, and `search()` pins and searches one collection. The query lease
guards that collection's publication and retirement; a lease on a synthetic
group name would not protect a member from its own publisher.

## Decision

Build segmentation as an **explicit, compiled search group** over independently
published `search:` models. `for_each` may generate its members, but a tag or
shared name does not make a group. The declaration names the members, their
typed partition field and nonoverlapping bounds, and one logical serving name.
Compilation must reject gaps in the declared domain, overlaps, mixed store or
embedding identities, incompatible schemas, modes, or policy contracts. A
catch-all partition must account for nulls and values outside finite bounds.
Membership and bounds belong in a versioned serving manifest, not a live YAML
read at query time. The partition predicate is part of each member's publish
identity. A group remains unavailable until every member has published under
the manifest's current bounds and any member removed by a boundary change is
retired safely.

Each member's publisher reads only its assigned upstream rows and reconciles
stale keys against **the same partition predicate at the same snapshot
instant**. Its collection, state and serving ledger remain independent. A
changed partition map is a membership migration: publishers must remove rows
that moved out before the new map can serve. The compiler and publisher must
prove the configured adapter can do this with bounded, resumable reads before
accepting a grouped publish.

A group query routes only from validated filters on the declared partition
field, then acquires generation pins for **all selected members** before any
store read. Admission is one all-or-nothing operation over those member scopes;
one group pin that member publishers do not consult is not sufficient. A
selected member that is missing or unreadable fails the whole query. The
query embeds its text once, applies the same mandatory policy and user filters
inside every selected store query, validates all pins after reading, and fuses
member rankings at document grain using rank rather than raw scores. It
preserves each hit's member generation in provenance. An unfiltered query
searches every member; narrowing to recent members requires an explicit
caller filter, never an implicit default window.

The group appears once in `list_context_models`. Its members remain
individually publishable and inspectable but are not additional default MCP
catalog entries. Group freshness is stale if any member is stale under its own
policy; the group's last-success timestamp is the oldest member activation,
not the newest one.

## Alternatives considered

### Infer membership from the `for_each` tag

The tag is for selection, not a partition contract. The loader discards the
template on expansion, and tags express neither row bounds nor coverage.
Inferring a group from them could duplicate a whole corpus or silently omit a
variant. An explicit declaration wins.

### One lease on a group scope

This would pay one lease write, but existing publishers check only their own
scope. They could mutate or retire a member while a group reader uses it.
Batching the existing member pins preserves the proven exclusion boundary;
the implementation may optimize admission to one warehouse round trip.

### Search only recent members when no partition filter is supplied

That changes an unfiltered search into a partial search without telling the
caller. An old filing may be the right answer. The caller must request a date
window explicitly; if all-member fan-out exceeds a configured limit, refuse
with an actionable error rather than return incomplete results.

### Fuse native scores across members

The retrieval contract says scores are not comparable across collections or
generations. Separate ANN training makes this particularly unsafe for
`ivf_pq`. Rank fusion matches the portable contract, with a held-out retrieval
evaluation required before it replaces a monolithic index.

## Consequences

- This is a design decision, not shipped segmented serving. It requires a
  strict partition config, compile checks, a serving-manifest version change,
  a group catalog/search path, and all-or-nothing member pin admission.
- The existing typed `TableReadRequest.predicates` is a useful seam, but the
  current BigQuery adapter sends predicated snapshots through a query job and
  disables its resumable keyed segments. The publish path needs a bounded
  filtered read before a 3.6M-row corpus can use it safely.
- `StateReconciler.iter_stale_pages` currently probes the whole upstream. It
  needs the same partition predicate as the snapshot, or a row moved between
  years survives in its old collection.
- MCP `BusinessFilter` currently accepts only `eq`, `ne`, and `in`, so date
  range routing also needs an additive typed filter contract. An unsupported
  filter may prevent pruning but must still be enforced in every member.
- Group queries need bounded fan-out, stable tie breaking, per-member
  generation provenance and query-log fields, and tests for partial failures,
  concurrent publication, partition moves, authorization, and empty members.
  Before enabling a group, compare its recall and latency with the monolithic
  index on held-out queries, including old and unfiltered questions.

## Evidence

Read from the 2026-10-22 tree: `config/loader.py` expands `for_each` into
tagged variants; `execution/search.py` opens an unfiltered `table_snapshot`
and calls `iter_stale_pages` without a partition; `state_reconciliation.py`
builds a whole-table absence probe; `adapters/bigquery.py` takes the query-job
path when predicates are present. `tests/test_bigquery_adapter.py` pins that
last behavior in `test_a_predicated_snapshot_still_uses_the_query_path`.
`mcp_server/catalog.py` emits one resource per search index, and `search.py`
acquires one member-scope lease. ADR-0035 explains why those pins must remain
valid for the entire read; ADR-0036 establishes document-grain rank fusion.
