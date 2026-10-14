# ADR-0029: A retrieval store's identity may be declared, not only derived from its location

- **Status:** accepted
- **Date:** 2026-10-09
- **Prompted by:** #666

## Context

A retrieval store's identity is a fingerprint over where the store is:
`LanceDBConfig.identity_key()` returns the posix-normalized path or the
canonical cloud URI, `safe_descriptor()` hashes it with the store type and any
non-secret routing, and that digest keys three things — the search model's
`retrieval_publish` state scope, its serving-ledger row, and the single-host
publisher lock.

Deriving identity from location was the right default and remains correct for
the case it was built for: two stores at two paths are two stores, and a changed
endpoint or region must not let state written against one object store be read
against another.

It also made a store's location permanent. astrolabe points its prod store at
`gs://econ-project-general-storage/lancedb` while the Dagster build that
publishes it and the `stel mcp serve` processes that query it all run on one
machine outside GCP, so every build and every query pays GCS internet egress:
1.43 TB over 2026-08-08..10-08, ~$150, about $70 of it per index rebuild. The
reason the store went to GCS — a local-disk store cannot be shared between a
builder and a server in different containers — does not hold for that
deployment, and does not hold for the common shape of a stel project run from a
laptop or a CI runner against a cloud warehouse.

The obvious fix, read a local copy and sync it to the bucket, could not be
done: the copy is at a different path, so it is a different store, with no
ledger row and no publication state. `stel mcp serve` refuses it as never
published and a build against it re-embeds the corpus — 3.67M chunks, ~$100 of
provider spend, on top of the egress it was meant to save.

## Decision

A store config may declare `identity:`, a stable label. When it is declared,
that label *is* the store's identity: the location and its routing are excluded
from the fingerprint entirely, so the same store can be read from a local
primary, a bucket, or a copy restored onto a fresh host and still resolve to one
state scope and one ledger row. When it is absent the fingerprint is
byte-identical to what shipped, including omitting `routing` when it is empty,
so no existing profile re-keys.

A declared identity is keyed under `identity` in the hashed payload rather than
`path`, so a label can never collide with the fingerprint another store derives
from a location that happens to spell the same text. The label is validated as a
label — 1-128 characters of `[A-Za-z0-9._:-]`, starting with a letter or digit —
which refuses a pasted URI or path at the profile boundary rather than letting
it strand a scope.

Existing indexes adopt one through `stel serving migrate-scope --from-path
<old>`, which re-derives the scope the store had at its old location and moves
the ledger row and publication state onto the scope the profile resolves now.

## Alternatives considered

### Keep identity derived, and require every reader to be co-located

What the code already assumed. It is a real deployment discipline, and it is
what astrolabe does today by running builder and servers on one host. It loses
because it makes the location permanent for the life of a published corpus: the
only way to stop paying egress is to re-publish into a new store, which is the
cost the change exists to avoid. It also leaves "restore this generation onto a
fresh host" with no answer at all.

### A `mirror:` URI that is defined as the same identity

The shape #666 asks for first, and the eventual goal: stel writes the primary,
syncs the generation to the mirror after a successful publish, and records it in
the generation ledger. It loses *as the first step* because it cannot work
without this one — a mirror at a second URI is a different store until identity
stops being the path, so `mirror:` would have had to smuggle in the same
decoupling implicitly, as a side effect of a sync feature, where nobody would
find it. Decoupling first, then building `mirror:` on top of a store that can
legitimately be in two places, keeps the invariant reviewable on its own.

### Make the identity default to a hash of the collection template or project name

Tempting because it needs no new field. It loses because it silently re-keys
every store on upgrade: the thing the derived-identity default buys is that
nothing moves, and any new default moves everything.

### Re-key by hand with `rekey_scope`/`rekey_state_scope`

Both already exist for #355. They are not reachable for this case:
`stel serving migrate-scope` computes only the pre-#355 legacy scope and takes
no source location, so an operator would be writing SQL against the state table
and the serving ledger to move a store. That is the opposite of the
"publication state is stel's to manage" invariant.

## Consequences

- **An identity is now a promise the operator makes.** Two genuinely different
  stores given the same label share a state scope and a ledger row, and stel
  cannot tell: it will read one store's publication state against the other.
  The derived default is what protects everyone who declares nothing, which is
  why the default stays.
- **`migrate-scope` is no longer a one-time #355 tool.** It has a second,
  recurring job, and `--from-path` requires the operator to know the location
  the store had. That is in the profile's git history; it is not in the
  warehouse, because the ledger row stores the fingerprint and not the path it
  came from.
- **The declared label is not artifact-visible**, only its fingerprint is, as
  before. A label is operator-chosen text, so keeping it out of manifests and
  logs keeps the artifact-visible surface exactly where it was.
- **The compiler's duplicate-collection check now sees across locations.** It
  keys on `(safe_target_identity, physical_collection)`, so two aliases at
  different paths sharing a declared identity will collide where they used not
  to. That is the correct answer — they are one store — but it is a new way for
  a previously valid project to fail compilation.
- `identity:` alone does not copy any bytes. It makes a local primary with a
  cloud copy *possible*; the copy is still the operator's to run until #666's
  `mirror:` lands.

## Evidence

- Cloud Monitoring on `gs://econ-project-general-storage`, 2026-08-08..10-08:
  `network/sent_bytes_count` 1.43 TB (by week: 223 GB, 361 GB, 613 GB for the
  week of 09-28 which held one index rebuild and a `serving activate`
  reconcile, 3-60 GB on quiet weeks); `api/request_count` ReadObject 11.8M. At
  GCS internet-egress rates, ~$140-165. Recorded in #666 and in astrolabe's
  Aug-Oct cost review.
- The derived fingerprints are pinned as literals in
  `tests/test_store_identity.py`, read off an unmodified v0.21.x tree:
  `90500be31b4af309f72535d7435b7ab7` for a local path,
  `1f3aaf7e4170eed0990d02f101969d36` for a cloud URI with routing,
  `c179fcc74d30f736f989e8ee0846a5b5` for one without. A payload change fails
  those tests rather than stranding a live index.
