# ADR-0042: Relation evaluation scores distinct facts and retains unresolved predictions

- **Status:** accepted
- **Date:** 2026-10-29
- **Prompted by:** constellations#731, ALE-425

## Context

A relation extractor emits evidence rows at mention grain. Several mentions
may resolve to the same canonical triple. Joining predictions to gold only on
matched endpoints would hide extra assertions and unresolved links, inflating
precision. Counting repeated evidence as additional facts would make quality
depend on how often a document mentions an entity.

## Decision

Score exact directed canonical triples with set semantics for complete rows.
An optional document column makes document identity part of the fact; omitting
it explicitly chooses corpus-level facts. Overall precision, recall and F1 are
micro scores, with independent per-type rows and explicit labels retained.

Each prediction with a missing canonical endpoint costs a false positive and
is counted in unresolved-endpoint diagnostics. Do not deduplicate these rows:
two unknown subjects cannot be assumed to denote the same entity. Gold must
contain complete, nonempty string triples and at least one fact. Invalid gold
fails before publication. Invalid types or configured document IDs also fail;
diagnostics name columns, never row values.

Project only scoring columns through adapter snapshots. Deduplication still
retains O(distinct triples) memory, documented as a benchmark-size limitation.
Reuse existing eval metric IDs, schema, history behavior and build gates.

## Alternatives considered

- **Evidence-row scoring:** repeated mentions change the score without changing
  the facts extracted. Exact sets are the unit the canonical-ID contract names.
- **Drop unresolved predictions:** an extractor with linking failures would
  report precision only over its successes. Preserve the loss in the denominator.
- **Treat missing gold endpoints as unmatched predictions:** broken annotation
  would become a model-quality measurement. Reject malformed gold instead.
- **Infer inverse matches:** the project declaration has no inverse contract
  today (ALE-421). Guessing from relation names would create false matches.
  Inverse support belongs with that future declaration and its validation.

## Evidence

The offline economic NLP fixture runs the actual rule extractor against four
independent gold facts. Breaking one of its two rules changes overall F1 from
1.0 to 2/3 and fails the existing `min_metric` gates. Unit tests pin extra and
missing facts, duplicate evidence, direction, document scope, unresolved links,
zero denominators and validation before metric publication.
