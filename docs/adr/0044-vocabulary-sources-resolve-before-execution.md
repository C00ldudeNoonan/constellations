# 0044 — Vocabulary sources resolve before execution

Status: accepted. Issue: constellations#718 / ALE-431.

## Decision

Resolve CSV vocabulary seeds while loading project configuration. Defer
warehouse vocabulary reads until structural project and adapter validation
has passed, then take one bounded snapshot through the active adapter before
discovery, code-version calculation, or provider calls. The adapter's
`preflight_reads()` connection omits schema/state initialization. Normal
execution opens its own initialized connection afterward.

Source rows sort by label; authored extra terms follow in declared order.
Consumers receive the same resolved vocabulary and enum models as inline
declarations. Artifacts preserve their existing resolved declaration shape.
`compile` needs warehouse access for a warehouse-backed vocabulary. Offline
export and embedded execution refuse unresolved warehouse declarations.

## Alternatives

Resolving during model execution would discover missing inputs after earlier
models had published, and planning could miss changes to the closed label
set. Resolving at config load would make every metadata-only command connect
to a warehouse, before the project contract had been validated.

An offline manifest with an empty or synthetic enum could be mistaken for an
executable schema or a valid code identity. Refuse unresolved identities and
provider schemas instead. Structural backend-option validation may use a
private validation copy with a non-empty set; it never changes the original
reference or reaches execution. This is revalidated against real labels after
resolution.

Keeping source row order would make a relation's arbitrary order a paid
reprocess trigger. Canonical ordering makes equal snapshots equal; inline
vocabularies continue preserving their authored order.

## Consequences

Relations must exist before a run and cannot depend on models built later in
that run. Missing columns, empty or duplicate labels, and bounded-read overflow
are safe configuration errors before publication. There is no persisted cache
or auto-refresh of a vocabulary during an invocation. Label sets retain the
same artifact visibility as authored enum values.
