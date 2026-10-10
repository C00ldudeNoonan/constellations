# ADR-0027: the serving catalog reads the declaration the artifact was compiled with, not the live project file

- **Status:** accepted
- **Date:** 2026-10-10
- **Prompted by:** #669 (found by review on #668; recorded as a known limitation in [ADR-0025](0025-a-declared-entity-scope-filters-after-retrieval-and-reports-what-it-excluded.md))

## Context

`search_context`'s `entity_scope` and `list_context_models`' `entity_types`
resolve against the project's `vocabularies:` declaration: which terms a class
holds, which terms sit beneath another. The facts they interpret are
`context_entity_links` rows, published by an earlier `stel run` under the
declaration in force at that time.

Until this decision `ArtifactCatalog.load` read `vocabularies:` from the
current `stel_project.yml` while reading the models and DAG from the compiled
`manifest.json` beside it. The declaration was live and the data was
published, and nothing tied the two together. An operator who edited a term's
`class:` or `broader`, or renamed a term, and started the server without
recompiling and republishing had old link rows read under the new declaration.
The result is wrong, not stale, and silent: a term moved between classes
changes which scope a chunk answers; a renamed term stops matching its own
published rows, which reads as "the corpus is thin" rather than "the
declaration moved".

The fix had to be an artifact contract change, which is why #668 recorded the
limitation and left it: the manifest was `manifest_version: 2` and did not
carry the declaration.

## Decision

`stel compile`, `run` and `build` write the project's `classes:` and
`vocabularies:` into the serving manifest as a top-level `declarations` block,
with every term field as authored. The manifest version moves to 3. The
catalog builds its vocabularies from that block and never from the project
file; it refuses a manifest at any other version with "run `stel compile`",
and refuses a declaration that does not validate the same way. `load` still
reads the project file for `target_path`, and when the live `vocabularies:`
differs from the compiled one it logs a warning naming the remedy, because
that is the one state an operator cannot otherwise see.

## Alternatives considered

### A fingerprint only, with the mismatch reported to the caller

Write a canonical fingerprint of the declaration into the manifest, keep
reading the live file, and refuse an `entity_scope` (or flag it in
`entity_scope_applied`) when the two fingerprints disagree. Cheapest change to
the artifact and it turns the silent wrong answer into a visible refusal.
Declined because it keeps the wrong source of truth and adds a failure mode:
the server would refuse scopes for as long as the file and the artifact
disagree, when the artifact alone already holds the right answer. Persisting
the declaration costs a few lines more and has no mismatch to report.

### Tie the declaration to the serving generation rather than the manifest

The precise answer: a republish of one search model under a new declaration
would not invalidate scopes on the others, and a `stel compile` run after an
edit but before a republish could not put the manifest ahead of the published
rows. Declined for now because it needs a per-generation store of the
declaration and a reader for it at serve time, for a hazard the operational
rule already covers: `run` and `build` rewrite the manifest after publishing,
so the declaration and the rows move together whenever the rows move. The
residual, a `compile` without a `run`, puts the edited declaration in the
manifest ahead of the rows for as long as the operator leaves it there, which
is the same exposure every other compile-time field has. If that proves
sharp, this is the ADR to supersede.

### Treat a manifest without the block as declaring nothing

Keep `manifest_version: 2`, read `declarations` when present, and fall back to
an empty declaration when absent. No version bump, no recompile forced on
upgrade. Declined because an older artifact beside a project with a vocabulary
would silently downgrade `entity_types` to the scanned answer and refuse every
`entity_scope` with `capability_unavailable`, a reason that is not true. A
contract that cannot tell "declared none" from "predates the field" is the
shape of the bug this fixes.

## Consequences

- Editing `vocabularies:` has no effect on a running or restarted server until
  the project is recompiled, and no effect on scope answers until it is also
  republished. The reference states the rule and the warning names it.
- Every deployment recompiles once on upgrade: a v2 manifest is refused. The
  downstream Dagster launcher reads `depends_on` from the manifest and does not
  pin its version, so the bump reaches it as a new key only.
- The manifest now carries term labels, descriptions and aliases. These are
  operator-authored schema already published to every caller through
  `entity_types`, not corpus content, so nothing new becomes artifact-visible.
- Tests that build a catalog from a hand-written manifest must carry the
  `declarations` block; `from_payloads` no longer accepts vocabularies beside
  the manifest, so the persisted shape is what every test exercises.
