---
name: stel-engineering
description: Engineer, extend, refactor, or review the stel Python CLI safely. Use for any stel feature, bug fix, configuration, source, backend, provider, adapter, execution, retrieval, CLI, artifact, state, or documentation change; especially when preserving validation-before-side-effects, secret safety, incremental correctness, warehouse publication, or public-contract compatibility matters.
---

# stel Engineering

Build from the repository's maintained contracts, then prove the changed behavior at its boundaries. Treat config, artifacts, state, logs, and public CLI output as compatibility and security surfaces.

## Start with scope

1. Read the repository's `AGENTS.md` and any more-specific instructions before editing.
2. Inspect the worktree and preserve unrelated changes.
3. Identify the user-visible contract and the owning seam. Read the relevant shipped README, `CONTRIBUTING.md`, config examples/templates, and adjacent tests before proposing the design.
4. Write down the invariants that must remain true on success, invalid input, cancellation, retry, and upgrade. Read [review lessons](references/pr-review-lessons.md) when the change touches a listed risk area.

Use this ownership map. Do not solve a problem by adding an integration-specific branch to orchestration.

| Change | Own it in | Verify |
| --- | --- | --- |
| Project/profile YAML, config semantics | compiler and strict Pydantic v2 models | Invalid config fails before discovery, SDK calls, or mutation |
| Document discovery/fetch | `sources/` | No symlink following; verified scratch copies; bounded cleanup |
| Extraction | `backends/`; project derivation in `post_extract.py` | Registry/options contract, typed/empty results, warning behavior, pre-publication field boundary |
| Inference/embeddings | provider registry/contract | Lazy optional import, sanitized errors, retry classification |
| SQL, quoting, materialization, state | `adapters/` | Adapter capability and publication guarantee, not DuckDB assumptions |
| Scheduling/model execution | `execution/` | State transitions and error paths across every model kind |
| Search/retrieval stores | retrieval/store seam | Canonical identity, honest lock scope, safe serving/publication |
| CLI and machine output | Click command/services boundary | Stable stdout JSON and deliberate stderr behavior |

## Apply the non-negotiable checks

### Configuration, credentials, and observability

- Validate public configuration with strict models before source access, credential resolution, remote calls, or warehouse mutation.
- Keep credentials as references through models, errors, dumps, fingerprints, cache keys, logs, and artifacts. Resolve their value only at the native SDK construction boundary.
- Treat raw input, provider response bodies/headers, warehouse error text, and exception chains as unsafe. Surface only stable, diagnostic-safe categories where output can be logged or serialized.
- Route project-configured paths through `paths.py`. Require a deliberate external-access opt-in; reject symlink traversal for discovered configuration and local sources.
- Make a new non-secret routing option part of the safe identity if changing it reaches a different physical target. Preserve existing fingerprints where the effective identity has not changed.

### State, publication, and resources

- Define the state baseline before choosing an incremental path. Rebuild safely when an existing target has no compatible baseline; reject an incremental strategy that cannot preserve untouched rows in a partial batch.
- Validate keys, layout, and input contracts before mutation. Stage and validate a replacement before swapping it into service whenever an adapter supports that guarantee.
- Advance state only after successful publication. Make failure/retry behavior explicit; do not claim atomicity beyond the active adapter's documented capability.
- Canonicalize aliases before deriving state scope, cache identity, or lock names. Ensure locks are shared across the deployment boundary their capability claims to protect; expose configuration when a shared mount is required.
- Bound scratch space and release fetched/staged resources per item or batch. Sweep only stel-owned stale artifacts using a narrow, verifiable predicate.
- For `post_extract`, apply the hook to ordinary and native-batch successes while the verified snapshot still exists and before row construction. The returned mapping is the publication boundary: preserve backend warnings/metrics outside it, sanitize hook failures, omit hook options from artifacts, sever validation exception chains, and include hook source/options in incremental code identity.

### Data and CLI behavior

- Preserve the declared row universe: explicitly decide how zero-token, empty, null-metadata, and first-run inputs behave. Preserve declared/passthrough dtypes even for empty output.
- Reject unsafe or semantically incomplete inputs early, such as non-finite weights or missing sentence boundaries for multi-token phrases.
- Expand templates before typed validation and dependency discovery when generated values can occupy typed fields or paths. Compute matrix cardinality before materializing a Cartesian product.
- Keep `--json` stdout to one valid payload. Write human progress to stderr. Select one progress channel per environment; avoid terminal bars in captured or parallel execution, and report the final post-filter selection count.

## Test the dangerous path first

Add focused regression tests for the exact defect and its nearest boundary. Favor these evidence patterns:

- Plant a sentinel secret and assert it is absent from `repr`, Pydantic dumps, errors, logs, artifacts, fingerprints, and caches.
- Assert invalid configuration causes zero discovery, SDK, adapter, and mutation calls.
- Simulate failure immediately before and after publication; assert target visibility and state advancement match the documented guarantee.
- Exercise no-prior-state/existing-target, partial-parent batches, aliases, different routing, empty inputs, invalid values, non-TTY output, and concurrent/parallel execution where applicable.
- Use fake clients and deterministic unit tests for cloud/provider behavior. Gate real credentials and live integration tests.
- Update user docs, config examples/templates, fixtures, and artifact schema/version tests whenever a public behavior or persisted shape changes.

## Validate and hand off

Run commands from `stel/`. Use focused tests while iterating; before handoff, run the maintained suite unless an environmental limitation prevents it:

```powershell
uv sync --all-extras --dev --locked
uv run pip-audit --skip-editable
uv run ruff check
uv run ty check
uv run pytest -q
```

Run `uv build` for packaging or release changes. If Windows symlink privileges or credential-gated integration tests prevent coverage, distinguish the known environment limitation from the changed-path results; do not call the suite clean without that distinction.

In the handoff, name the contract changed, the security/publication/state behavior, tests run, known limitations, and documentation/config artifacts updated.
