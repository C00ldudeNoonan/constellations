# ADR-0035: Session query pins drain cooperatively; a clock never revokes a reader

- **Status:** accepted
- **Date:** 2026-10-16
- **Prompted by:** #541

## Context

Every served query inserted and deleted a warehouse lease. Issue #541 measured
about four seconds in these two BigQuery DML jobs, against about eleven seconds
for a warm text query. Keeping the store and connection resident removed other
setup costs, but left the pin protocol on every request.

The September proposal in the issue chose session pins with renewal and expiry.
It did not establish how an expired reader is stopped. That is a correctness
requirement: the MCP operation limiter returns a timeout without stopping its
worker (`mcp_server/service.py`, `_OperationLimiter.run`). Stores enforce
publisher exclusion, but have no reader fencing or cancellation contract. A
paused worker can therefore resume reading after its pin's expiry, while a
publisher mutates or deletes its collection. A validation after the read cannot
undo that overlap. No finite renewal/expiry pair establishes termination.

## Decision

Share one durable pin per serving session and scope on adapters supporting held
operator connections. Each new query reads the ledger and its exact durable
pin together before joining. It must still name the active generation,
collection and fingerprint, and receives current readiness/error metadata.
Every query also validates its pin after store I/O. Initial admission retains
its conditional INSERT and active-pointer recheck.

Track local holders under a lock covering admission and release, never query
embedding or store I/O. Only the last holder can release a retiring pin. Idle
pins remain reusable for 30 seconds; a background sweep every 5 seconds deletes
eligible idle pins. These intervals bound a healthy owner's idle retention,
not the duration of a running query or the lifetime of a durable lease.

An in-place publisher blocked by pins requests a drain on the ledger, conditional
on its observed fence and the absence of a publisher. `query_drain_until_epoch`
is an additive nullable BIGINT column. For 60 seconds, fresh and reused
admissions are refused; existing readers continue. A reused pin that observes
the request retires after its last holder leaves. An idle session that receives
no requests releases through the idle sweep, normally within 35 seconds of
its last completed query. Publishers retain their fail-fast busy error and
must retry the run; acquiring a claim clears the gate. An abandoned gate expires
without releasing any durable pin. The deadline uses the same process epoch
clock as existing publisher diagnostics; clock skew affects admission delay,
never pin safety. A stale planned publisher does not request a drain.

Pins whose owners crash or lose access remain until explicit recovery after
termination of all previous owners. Shutdown releases idle pins immediately;
an in-flight or timed-out worker releases its pin when it actually finishes.
Failed DELETEs remain registered, non-reusable, for maintenance to retry.
File-backed DuckDB and one-shot queries retain their practical behavior:
file-backed queries release per call, and one-shot sessions close after use.

Keep scope-wide publication exclusion and retirement checks. Narrowing them
to generations is an independent protocol change: `validate_query` currently
rejects all readers during in-place publication, even a pin on an older
generation. Changing only the count would let a publisher proceed while those
readers are still governed by the old exclusion rule. Conservative scope-wide
checks preserve all four guarantees without requiring that redesign.

The resulting guarantees are unchanged: readers pin their physical generation;
in-place publication excludes them; retirement waits for pins; recovery clears
pins and fences surviving workers. No provider, source, authorization result,
query text or filter is cached by the pin manager. Reuse is limited to the
existing operator search path, not caller-scoped warehouse connections.

## Alternatives considered

### Session pins with bounded expiry and renewal

Rejected for the present stores: time passing cannot prove a worker stopped
reading. Safe automatic crash cleanup needs a store-enforced reader fence or
another mechanism that prevents reads after expiry. A longer TTL, a renewal
thread, or a Python request timeout does not supply that mechanism. This
revises the proposed, unfiled "ADR-0011" in the issue; the existing ADR-0011
about entitlement intervals is unrelated and keeps its number.

### Per-query pins with deferred DELETE

Saves the release round trip, but retains an INSERT on every request and
needs background cleanup anyway. Reusing a durable pin removes both writes
from warm requests and retains explicit before/after validation.

### External coordination store

A fast KV service might reduce admission latency, but adds deployment and
failure modes. It also cannot stop a paused reader's independent store SDK
calls merely by expiring a KV key. No such dependency is introduced.

### Conditional pinning only during mutation

Rejected because private generations still need protection from retirement.
A read-only collection can be deleted while a reader is using it.

### Keep idle pins until the session ends

An idle server could block in-place builds indefinitely. Bounded cooperative
idle retention plus an admission gate lets healthy sessions drain without
revoking active readers or starving a retrying publisher with fresh queries.

## Consequences and evidence

Warm remote queries use two serving-ledger reads (readmission and completion)
and no lease DML. Cold admission still pays its existing protocol. Servers
whose queries are more than 30 seconds apart will often reacquire; there is no
claim that every request becomes warm. Maintenance adds an eventual DELETE
per idle pin, not a renewal job per idle session.

In-place publication may require a retry and wait for active work plus idle
retention. A crashed server still costs explicit recovery. Private cutover
remains available while pins exist, with cleanup deferred under ADR-0003.
Deploy serving and publisher processes together to obtain cooperative drain:
older servers still protect their reads, but do not honor the admission gate.

`test_session_query_pins.py` covers overlapping holders, a paused reader across
gate expiry and shutdown, idle cleanup, abandoned gates, cutover, recovery,
current degraded status and retry of failed DELETEs. `test_search.py` counts
real warehouse statements across repeated queries and shutdown, including
the unchanged file-backed path. These deterministic tests establish protocol
behavior, not a new measured BigQuery latency; the production timing cited
above remains the historical measurement in #541.
