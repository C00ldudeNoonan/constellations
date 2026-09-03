# ADR-0003: Reader-safe online publication uses independent generations

- **Status:** accepted
- **Date:** 2026-09-03
- **Prompted by:** #473

## Context

The reported 3,613,979-row exact-to-approximate update exhausted a 20 GiB
container after 84 of 145 batches. The prior empty-target publication completed.
This establishes an OOM incident, not proof of a particular LanceDB leak.
Online publication merged every row into the populated live collection because
row fingerprints include the changed configuration. Its claim disabled serving.
Even private builds excluded readers for their full duration.

## Decision

Keep vector strategy changes compatible, but publish every accepted online
configuration change into a fresh independent generation. Reuse warehouse
vectors; do not invoke providers. Use bounded append writes for LanceDB private
builds, with durable receipt and snapshot validation before activating.
An explicit rebuild policy also accepts compatible changes. Reject unsupported
stores, incompatible online changes, and subset replacements before the claim.
Plan only against a publisher-free ledger and condition the claim on its
observed fencing token, so a competing activation cannot stale the cleanup target.

Private claims preserve the active generation and its fingerprint in the ledger;
the publisher's lease carries the proposed fingerprint. Query admission checks
the active pointer after inserting its pin. Admitted pins then survive private
builds and cutover, and retirement defers while any pins remain. In-place writes
still require exclusive access. Recovery deletes pins and fences old owners.

This supersedes ADR-0002's in-place publication choice, not its compatibility
classification, and ADR-0001's assumption that all publishers exclude readers.
The failure asymmetry remains: a private build retains the old generation;
a failed in-place write cannot safely do so.

## Alternatives considered

### Build the index directly on the live collection

Avoids copying rows, but changes query behavior before activation, especially
when dropping an approximate index. Concurrent warehouse changes may also need
row mutation. The current store interface cannot pin independent index versions.
Retaining a serving pointer alone would advertise safety it does not provide.

### Shallow-clone the LanceDB collection

LanceDB 0.34 supports independent manifests referencing shared source files.
Existing generation retirement deletes whole collections; deleting a source
would break a clone still using its files. This needs reference-aware retention
and a store capability contract first, so it is deferred rather than hidden
inside publication.

### Only remove the strategy from row fingerprints

Changes persisted identities for unchanged installations, and still does not
make live index mutation reader-safe. No existing fingerprint domains or
unchanged collection stamps change in this fix.

### Catch an OOM kill

SIGKILL cannot run Python cleanup. Numeric batch memory samples and external
container OOM diagnostics survive as evidence; Python MemoryError handling is
best effort and is not represented as protection against kernel termination.

## Consequences

- Online changes still copy warehouse rows and require temporary storage for
  two generations. Index construction has its own native memory requirements.
- Old-config readers can continue; new-config queries are refused until their
  generation is activated. A failed configuration update does not make the old
  index answer under the new definition.
- Pins can delay reclamation. Operators must terminate abandoned owners before
  explicit recovery; no timeout-based lease stealing is introduced.
- Deploy publishers and serving processes on the same release after draining
  leases. Older publishers do not honor the reader-aware retirement protocol.
- Ordinary unchanged-config incremental updates remain in-place and exclusive.
- Live-scale verification is still required to establish the memory behavior
  of the reported GCS/BigQuery workload. Small regression tests are not evidence
  that the 20 GiB incident cannot recur.

## Evidence

Issue #473 supplies the container OOM counter and batch progression. Tests in
`tests/test_online_publication.py` exercise real local LanceDB append/index
publication, query pins before/during/after cutover, failure and termination,
unchanged reruns, subset refusal, and reader-aware retirement. The SDK's
[clone contract](https://lancedb.github.io/lancedb/python/python/) documents
shared source files; independent ownership must precede adopting it here.
