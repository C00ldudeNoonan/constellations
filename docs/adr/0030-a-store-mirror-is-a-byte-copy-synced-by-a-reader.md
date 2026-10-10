# ADR-0030: A store mirror is a byte copy, synced by a reader after activation

- **Status:** accepted
- **Date:** 2026-10-10
- **Prompted by:** #666

## Context

ADR-0029 let a retrieval store declare an `identity:`, so a store's bytes can
move without its published collections reading as never published. That made
the deployment #666 asked for possible: a local primary that builds and queries
read, so nothing pays object-store egress per read, plus a copy in a bucket that
a fresh host is restored from. It did not provide the copy. Until this change
that was a `gcloud storage rsync` the operator scheduled after each publish,
which stel knew nothing about: it could be forgotten, it could copy a table
while a publish was writing it, and nothing could say whether the copy in the
bucket was the generation the ledger serves.

The question this records is what "the mirror" is, and how a copy of it is
kept consistent, without making the publish path any less safe than it is.

## Decision

A LanceDB store config takes `mirror:` (a `gs://` or `s3://` URI, or a local
path), allowed only with a local primary `path`. Three decisions shape it.

**The mirror is a byte copy of each Lance table directory.** Lance writes data,
index, deletion and transaction files once and never modifies them; the
manifests that name them are written once too; one small file hints at the
newest version. The copy lists the primary's table directory once, copies every
file the mirror lacks (or holds at a different size) with data files first,
then manifests, then the hint, and then removes from the mirror what the
primary no longer has, manifests before data. The mirror is therefore a
readable table at every instant of a copy, holds exactly the primary's versions
and indices, and converges on rerun.

**A sync is a reader.** It holds a query lease on the active generation for the
length of the copy, mirrors a collection only when its physical generation is
the one the ledger activated, verifies the lease survived the copy, and records
the mirrored generation in the serving ledger with a write conditional on that
generation still being active. The lease is what makes the copy consistent: an
in-place publisher cannot claim the scope while any lease is held, and retiring
a superseded generation waits for leases to drain.

**A sync runs after activation, and a failed one fails the model.** Every
successful publish and `stel serving activate` ends with one. The generation is
already live, so a failure cannot touch the ledger; it surfaces as a model
failure naming `stel serving sync`, which retries it.

`stel serving restore` is the same copy in the other direction. It never
replaces or removes a file at the primary, restores only when the ledger
records the mirror as holding exactly the served generation, and requires the
collection it produced to be that generation.

## Alternatives considered

### Write rows into a second LanceDB store

Open the mirror with `lancedb.connect` and stream the collection's rows into it.
It loses because it rebuilds every index at the mirror -- the ANN build is the
most expensive step a publish has -- and because the
result is a different table holding the same rows: different versions,
different generation fingerprint, so nothing could prove a restore put back the
generation the ledger vouched for.

### Publish to both stores in the page loop

Make the mirror a second publish target. It loses on the property ADR-0001 and
#617 were about: a publish that can fail in two stores has two places to leave
a half-written collection, and an object store reached over the internet is
the one more likely to fail. It also doubles the egress-bearing writes on the
hot path to save a copy that only needs to exist after the fact.

### Hold the publish claim, or the single-host publisher lock, during the sync

The claim is the natural "nobody else is writing" token. It loses because
acquiring it changes the ledger's status and fencing token to copy files that
change nothing a reader sees, and a sync that dies holding it needs
`serving recover` before the next publish. The publisher lock is keyed on the
physical collection, so it does not exclude a rebuild into a new generation
whose activation then retires the collection being copied. A query lease
excludes exactly what must be excluded -- an in-place rewrite and retirement --
and nothing else; a private rebuild may still run alongside, as it may
alongside any query, and the conditional record means its activation wins.

### Warn instead of failing the model when a sync fails

Keeps a build green through an object-store outage. It loses because a mirror
that silently stops tracking the served generation is the failure the mirror
exists to prevent, and is discovered only when a host is lost and the restore
is refused or, worse, restores a stale copy. Failing the model makes an
orchestrator's retry the fix: the rerun reconciles nothing and copies only what
the mirror lacks.

### Use the store's own credential references for the mirror

`storage_options_env` resolves credentials at the LanceDB SDK boundary. The
copy runs through Arrow's filesystem layer, whose option names and credential
model differ per provider; mapping one onto the other is a second credential
surface to keep secret-safe. The mirror uses the environment's default
credentials instead -- Application Default Credentials, the AWS default chain
-- which is what `gcloud storage rsync` used, and is why `mirror:` requires a
local primary: a cloud primary would need its own credentials on that path too.

## Consequences

- **A long first sync holds a query lease for its duration.** An in-place
  publish of the same index is refused as busy until it finishes, as it would
  be for any long query. A sync killed hard leaves its lease behind like any
  reader would, cleared by `serving recover`.
- **Two concurrent syncs of different generations can race on retirement.**
  Retirement runs only after the conditional record lands, but a sync of a
  newer generation that completes between that record and the retirement would
  have its collection retired by the older sync. It needs a full rebuild to
  start and activate inside that window; the next sync restores the mirror.
- **The mirror relies on Lance's directory layout.** The copy addresses a table
  as `<root>/<name>.lance` and refuses to sync a collection that LanceDB stores
  anywhere else, so a LanceDB release that changes the layout fails loudly
  rather than mirroring the wrong directory.
- **The ledger gains three nullable columns**, added in place to an existing
  ledger like #355's and #621's, and moved with the row by `migrate-scope`.
- `az://` mirrors are refused for now: Arrow's Azure filesystem does not take
  the URI forms LanceDB does, and nothing needs one yet.

## Evidence

- `tests/test_store_mirror.py` stops a copy after every possible number of
  files and requires whatever landed to open as the table at some version;
  making every file copy in one pass fails it.
- `tests/test_store_mirror_e2e.py` publishes with a mirror, deletes the primary,
  restores onto a new path under the same `identity:`, and runs again: zero rows
  processed, two skipped.
