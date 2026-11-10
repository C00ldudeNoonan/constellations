# ADR-0059 — Scoped embeddings bound warehouse vector reuse

- **Status** Accepted
- **Date** 2026-11-10
- **Prompted by** [#761](https://github.com/C00ldudeNoonan/constellations/issues/761)

A BigQuery embedding run filtered to recent filings billed 68.3 GiB in
vector lookups before a budget guard stopped it. The lookup resolves text
hashes to rows in the existing target. It builds partition and cluster
predicates from those rows, so a recent chunk can pull the range back
years when its text matches an older filing.

For a run with `--read-filter`, apply the filters to the target key scan
and vector reads. Use the current input window's partition range and
distinct clustering values where those values can be expressed as
predicates. Keep one candidate per text hash and distinct layout so a
duplicate in another layout cannot hide a match inside the window.
The local response cache still looks up vectors by content.
Unfiltered runs retain corpus-wide warehouse reuse for re-keying.

Keeping corpus-wide reuse for scoped runs would avoid some provider
calls but leave BigQuery scans dependent on where earlier copies of the
text landed. Turning off warehouse reuse in scoped runs would also give
up vectors already available inside the slice. Limiting lookup to the
slice and window preserves those hits and can require a provider call
when the only available match falls outside them.
