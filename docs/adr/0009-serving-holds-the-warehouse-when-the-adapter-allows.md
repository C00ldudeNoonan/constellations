# ADR-0009: The serving session holds its warehouse connection only when the adapter says one may outlive a request

- **Status:** accepted
- **Date:** 2026-09-07
- **Prompted by:** #523

## Context

A served MCP query touches the warehouse several times: once for the lease
that pins the generation it reads, once to re-read each hit's row for text,
authorization and lineage, once per entity relation, and once more to write
the query log. Until this decision every touch opened its own connection.
On BigQuery an open is a credential resolution measured at 2.1–2.25s
(`warehouse_connect`, the #519 breakdown posted on #523), so a served query
paid three or more of them before any index was consulted. PR #534 held the
compile and the retrieval store across requests; the connection was the last
increment, and #523 called it "the fiddliest": DuckDB connections are not
thread-safe, and a cached broken connection has to be detected and replaced.

There is a third constraint #523 did not name. An open DuckDB file is an
exclusive lock: one process may read and write it, or several may read. A
server that held its connection would hold that lock for as long as it ran,
and `stel run` in a second terminal would fail to open the warehouse until
the server exited. Today, with a connection per request, the two only
collide for the duration of a query.

## Decision

`SearchSession` holds one warehouse connection across requests when the
adapter says it may — `WarehouseAdapter.supports_held_connection()`, true
for BigQuery and for DuckDB reached over MotherDuck, false by default and
for a file-backed DuckDB warehouse. The MCP repository's reads and log
writes go through the same session, so every warehouse touch of a served
query shares the one connection. Statements on it are serialized per call
by `SerializedAdapter`, the wrapper the runner already uses under
`--threads`, not per query. An `AdapterError` raised inside any operation
discards the held connection, and the next operation reconnects. A
file-backed warehouse keeps a connection per call and pays
`warehouse_connect` every time, as before.

## Alternatives considered

### Hold the connection for every adapter

The simplest change, and the one the issue's table implies. It lost on the
DuckDB file lock: the local development loop is `stel run` in one terminal
with an MCP server configured in an editor or desktop client in the
background, and holding the file would make every run fail with a lock
error while the server lived. The saving on DuckDB is also nothing worth
having — a local open is milliseconds. The cost is real only where the lock
is not, which is exactly the distinction the adapter can state and
orchestration cannot infer.

### Open the DuckDB warehouse read-only for serving

Would release the lock, since several read-only processes may share a file.
It lost because the serving path writes: the query lease is an INSERT and a
DELETE on the lease table, and the query log is an append. Both are
correctness machinery (#152, #329), not overhead, and a read-only connection
cannot carry them.

### A lock per query rather than per statement

Simpler to reason about than a per-call guard, and what a first draft of the
store guard in #534 looked like. It lost on the lesson recorded against
#432: a lock spanning a whole query serializes the lease round trips and the
provider call too, which is serialized execution rather than serialized I/O.
Concurrent tool threads on a held connection wait for one statement each.

### A connection pool

The general answer, and the wrong size for the problem. The serving process
is one Python process whose tools already run on a bounded worker pool, and
the warehouse cost being removed is the open, not contention. A pool would
add reconnect and health-check policy for a benefit no measurement asked
for. If concurrent serving ever contends on the single connection, that is
the measurement to take first.

## Consequences

- On BigQuery a served query pays `warehouse_connect` on the session's first
  operation only — the warm-up at boot — instead of three or more times per
  request. The lease itself is unchanged and still per request (#541 is the
  design question about where the pin lives).
- A DuckDB file warehouse behaves exactly as before, and `stel run` keeps
  working beside a running local server.
- A new adapter inherits `False` and must opt in. That is the safe default:
  the wrong answer here blocks other processes, and an adapter author knows
  whether their connection is a lock.
- A held connection that breaks costs one failed request; the discard makes
  the next one reconnect rather than inherit the failure. Nothing detects a
  silently dead connection ahead of use, and nothing was asked to.
