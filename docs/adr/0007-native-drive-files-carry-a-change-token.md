# ADR-0007: Native Drive files carry a change token, not a content hash

Status: accepted (2026-09-04). Issue #514.

## Context

The document-source contract (`sources/base.py`) asks `discover()` for a
`content_hash` that "changes when the object's bytes change", derived from a
listing rather than from content. Local files hash their bytes; GCS lists an
md5. Google Drive lists an md5 for uploaded binaries only. Native Docs and
Slides are not stored as bytes, and no listing field is a function of their
content.

## Decision

For native files, `content_hash` is `mtime:<modifiedTime>` and is documented
as a **change token**. The prefix keeps it from ever colliding with an
`md5:` value, `source_metadata.identity` names which kind a row has, and
Drive's monotonic `version` is recorded beside it. `fetch()` re-reads the
listing fields and refuses a file whose token moved after discovery. Uploaded
binaries keep `md5:<checksum>` and are verified byte-for-byte on download.

The token's semantics are stated where they matter: every content edit moves
it, so it never under-triggers on content; a no-op save also moves it, so it
can re-extract an unchanged document. Under `llm:` extraction that is
provider spend, which is why it is named rather than hidden.

## Alternatives rejected

- **Hash the exported bytes at discovery.** Honest, but it exports every
  native file on every run, which is the download the contract exists to
  avoid. A folder of a thousand Docs would pay a thousand exports to learn
  that nothing changed.
- **Use Drive's `version` counter as the token.** Monotonic and cheap, but it
  advances on changes that are not content — a permission change, a comment —
  so it over-triggers strictly more than `modifiedTime` does, with nothing
  gained.
- **Use `headRevisionId`.** Only populated for files with binary content; it
  is absent for the native files that need it.
- **Store `modifiedTime` in the `content_hash` field without saying so.** The
  incremental machinery works identically, and #352 recorded why this is the
  wrong move: a token that masquerades as a hash lets the "changes when the
  bytes change" guarantee silently stop holding for one source, and nobody
  reading the contract can tell.

## Consequences

- A first-party API source's identity contract is now explicit: an md5 where
  the listing has one, a named change token where it does not, and a fetch
  that verifies against whichever it was.
- A future source with the same shape (Confluence pages, if one is ever
  justified) follows the same rule rather than inventing a third.
- Reducing over-triggering further would mean a content-derived identity
  computed *after* export and compared before materialization. That is a
  different contract (fetch-then-skip) and would need its own decision.
