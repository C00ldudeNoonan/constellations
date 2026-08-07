# dbt-ml review lessons

Use these as failure-mode prompts, not as a replacement for current source and maintained documentation. They summarize concrete Codex findings and follow-up fixes from the July--August 2026 PR review period.

## Secrets, identity, and deployment scope

- [PR #278](https://github.com/C00ldudeNoonan/dbt-ml/pull/278): `SecretStr` redaction was not sufficient because the resolved value still lived in the config model. Store credential *references* in a dedicated `*_env` field and resolve only at `lancedb.connect()` or another native SDK boundary. Test all serialization surfaces with a planted secret.
- [PR #278](https://github.com/C00ldudeNoonan/dbt-ml/pull/278): a `tempfile` lock cannot promise host-wide exclusion because containers/processes can choose different temp roots. Use a fixed shared location for the stated scope and expose an explicit shared-lock directory when deployments need one.
- [PR #278](https://github.com/C00ldudeNoonan/dbt-ml/pull/278): include non-secret routing in safe store identity; canonicalize URI aliases and trailing slashes before deriving identity or locks. Preserve exact legacy identity when the effective target has not changed.
- [PR #278](https://github.com/C00ldudeNoonan/dbt-ml/pull/278): reject malformed cloud URIs during profile preflight, before a lease or remote connection.

## CLI, logging, and concurrency

- [PR #276](https://github.com/C00ldudeNoonan/dbt-ml/pull/276): verbosity must not enable raw tracebacks or unsafe exception text. Cap built-in verbose output at the safe level and leave deep diagnostics to an operator-controlled handler.
- [PR #276](https://github.com/C00ldudeNoonan/dbt-ml/pull/276): choose exactly one stderr progress channel. A TTY can use a reporter; captured stderr must use atomic log lines. Test that the other channel is absent.
- [PR #279](https://github.com/C00ldudeNoonan/dbt-ml/pull/279): when models execute in parallel, independent progress bars share stderr and corrupt each other. Disable bars and use log lines. Log the final selected count, not the backend's pre-filter discovery count.

## Compiler and generated model contracts

- [PR #236](https://github.com/C00ldudeNoonan/dbt-ml/pull/236): expand `for_each` templates on raw YAML before Pydantic validation and SQL dependency discovery; typed fields and paths can carry placeholders.
- [PR #236](https://github.com/C00ldudeNoonan/dbt-ml/pull/236): calculate Cartesian-product size before calling `itertools.product`; enforce the limit without allocating an unbounded expansion.
- [PR #232](https://github.com/C00ldudeNoonan/dbt-ml/pull/232): inspect the actual branch diff before merging. A feature branch based on another feature can accidentally ship unrelated work.

## Data semantics and typed empty output

- [PR #232](https://github.com/C00ldudeNoonan/dbt-ml/pull/232): null sentence indices cannot safely form multi-token phrases; reject the condition with an actionable repair path while permitting behavior that does not need sentence boundaries.
- [PR #230](https://github.com/C00ldudeNoonan/dbt-ml/pull/230): token-derived transforms need an optional documents spine to retain documents with zero tokens. Preserve passthrough types when the resulting frame is empty, and reject non-finite numeric inputs.

## Incremental state and warehouse publication

- [PR #228](https://github.com/C00ldudeNoonan/dbt-ml/pull/228): an existing target without compatible incremental state requires a safe full replacement to establish a baseline. A partial-parent batch cannot use partition-wide `insert_overwrite` if it could remove unchanged parents.
- [PR #228](https://github.com/C00ldudeNoonan/dbt-ml/pull/228) and [PR #234](https://github.com/C00ldudeNoonan/dbt-ml/pull/234): a delete-then-upsert availability window is cross-cutting. Do not add a transform-only workaround; move atomic parent-scoped replacement into an adapter-owned contract and track it until implemented.
- [PR #120](https://github.com/C00ldudeNoonan/dbt-ml/pull/120): validate a full-refresh replacement, including warehouse layout, before dropping or swapping the current target. Test failure preserves the last good relation.
- [PRs #259 and #270](https://github.com/C00ldudeNoonan/dbt-ml/pull/270): replace unbounded BigQuery array parameters with bounded batches or temporary staging while retaining the adapter's transaction/publication semantics.

## Resource and integration safety

- [PR #274](https://github.com/C00ldudeNoonan/dbt-ml/pull/274): cleanup at run exit is insufficient after kill/crash and unbounded staging grows with corpus size. Clean per document/batch and narrowly sweep stale dbt-ml-owned directories at startup.
- [PR #265](https://github.com/C00ldudeNoonan/dbt-ml/pull/265): give cloud-source handles explicit cleanup and sanitize cloud SDK errors before they cross the source boundary.
- [PR #257](https://github.com/C00ldudeNoonan/dbt-ml/pull/257): when a generated artifact needs an offline runtime, make the dependency packaging decision explicit and test the built wheel or produced artifact, not just source-tree behavior.
