# Contributing to dbt-ml

dbt-ml is a small, focused Python project. Contributions welcome.

## Setup

```bash
git clone https://github.com/C00ldudeNoonan/dbt-ml
cd dbt-ml/dbt-ml
uv sync --all-extras --dev --locked
uv run pytest -q
```

## Local checks (required before opening a PR)

```bash
uv run pip-audit --skip-editable  # dependency advisories
uv run ruff check                 # lint
uv run ty check                   # required type checking
uv run mypy                       # temporary migration parity
uv run pytest -q                  # tests
```

ty is the primary type checker and checks `src/dbt_ml` plus focused static
compatibility fixtures in `typecheck/`. Ruff's ANN and PYI rules preserve
annotation discipline for package source; explicit `Any` remains permitted at
validated dynamic, SDK, and optional-dependency boundaries. Do not add broad
suppressions to keep the command clean.

During the migration tracked in issue #49, CI and release validation also run
mypy as a parity check. Remove that second checker only after the issue's
side-by-side validation period is complete. The workflows live in the
repository-level `.github/workflows/` directory.

## Adding a new backend

1. New file in `src/dbt_ml/backends/` inheriting from `BaseBackend`.
2. Decorate the class with `@register`.
3. Implement `name()`, `supported_formats()`, `extract(path, options)`.
4. Register the import in `src/dbt_ml/backends/__init__.py` (the side-effect
   import is what triggers `@register`).
5. Add a synth generator under `src/dbt_ml/synth/` if you want to support
   `dbt-ml seed --type <name>`.
6. Add tests under `tests/test_<backend>_backend.py`.
7. Add an init template under `src/dbt_ml/templates/<backend>/` if a fresh-
   project starter makes sense.

## Adding a built-in Python transform

1. Put the module under the appropriate package, such as
   `src/dbt_ml/text/transforms/`, and expose `run(deps, ctx)` with one or two
   positional arguments.
2. For transforms with options, expose `validate_options(options) -> None`.
   The compiler calls this hook before source discovery, credentials, optional
   SDK initialization, model loading, or warehouse mutation.
3. Parse runtime options with the same strict Pydantic model used by the
   validation hook. Keep optional dependencies lazy and provide an actionable
   extra-install command when they are absent.
4. Use a provider protocol when execution depends on an external SDK or model.
   Unit tests must use a deterministic fake and must not require downloads,
   credentials, or network access.
5. Document the input and output schemas, sensitive-field behavior, optional
   extra, and a runnable example whenever the transform is public.

## Adding a new model kind

Model kinds are `ModelConfig` sub-blocks; exactly one per model. To add one
(the native `llm:` map model, issue #144, is the reference):

1. Add a strict Pydantic block to `src/dbt_ml/config/model.py`, wire it into
   `ModelConfig`, `_validate_single_kind`, `kind_block_count`, and the
   `ModelFile` missing-kind message.
2. Add an execution function in `src/dbt_ml/runner.py` and a dispatch branch in
   `_run_model`. Reuse warehouse-neutral adapter primitives (`read_table`,
   `materialize_full`/`materialize_incremental`, `delete_rows`, state helpers);
   advance state only after a successful publish.
3. Preflight the kind in `src/dbt_ml/compiler.py` (`_kind_label`, dependency and
   provider validation, warehouse capability requirements).
4. Fold its code identity into `src/dbt_ml/versioning.py` and surface it in
   `src/dbt_ml/manifest.py`, `src/dbt_ml/cli.py`, and the docs template.
5. Keep provider/cache/retry logic in one shared core rather than duplicating it
   — `llm:` and `backend: llm` both route through
   `llm_backend.extract_fields_with_usage`.
6. Add tests: config validation, the runner path (credential-free via the
   `deterministic` provider), compiler failures, versioning, and manifest.

## Adding a new schema test

1. Add the name to `SUPPORTED_TESTS` in `src/dbt_ml/checks/schema.py`.
2. Implement a helper function returning `TestResult`(s).
3. Wire into `_run_named_test`.
4. Add tests under `tests/test_checks.py`.

## Adding a new CLI command

1. Add the click subcommand in `src/dbt_ml/cli.py`.
2. Wire through `ctx.obj["project_dir"]` / `profiles_dir` / `target` like the
   existing commands do.
3. Raise `click.ClickException` from any `*Error` exception you catch.
4. Update README's CLI section.

## Security and correctness invariants

- Route paths from project YAML through the boundary helpers in `paths.py`.
  External access must be an explicit, reviewable opt-in.
- Local document discovery and fetch must not follow symlinks. Preserve the
  no-follow walk and verified scratch-copy boundary in `sources/local.py`.
- Parse credentials into the shared protected reference/value types before
  generic interpolation. Neither credential values nor environment-variable
  names may enter reprs, dumps, equality/hashing, artifacts, or diagnostics;
  reveal a value only at the native SDK boundary. Do not log secrets or raw
  document content.
- Validate incremental keys before mutation and keep each adapter write atomic.
- Raw PII evidence is opt-in. A redacted output does not make retained input
  columns safe; tests and examples must project sensitive originals away.
- User-facing cleanup commands remove owned local artifacts only. Warehouse-wide
  reset behavior must not hide behind a familiar dbt command name.

## Scope discipline

The current implementation boundaries are:

- Python 3.12+ only; no Rust or PyO3.
- DuckDB and BigQuery are the implemented warehouse adapters. State follows the
  active adapter; do not introduce new DuckDB assumptions into orchestration.
- Local filesystem and GCS are the implemented document-source types.
- Anthropic is the only implemented LLM provider.
- dbt-ml is dbt-shaped but standalone. Its artifacts and test semantics must be
  documented explicitly rather than assumed to be dbt-core contracts.

Extend these boundaries through the existing adapter/provider seams and include
contract tests for every supported implementation.

Warehouse consumers that can scale with relation size must use the typed
`table_snapshot()` contract rather than assembling SQL or collecting
`read_table()`. Keep projection, predicates, snapshot consistency, generation
validation, and key-domain checks inside the adapter. Add explicit capabilities
only after contract tests cover empty schemas, multi-batch reads, early close,
mid-stream failure, and source-generation changes; never derive an adapter's
capabilities from every enum member.

Publication-state consumers that can scale with scope size must use the
bounded reconciliation contract (`fetch_state_subset()`, `state_page_reader()`,
`replace_state_scope()`) rather than `fetch_state()`. Keep key ordering,
snapshot consistency across pages, opaque cursor validation, and fence
verification inside the adapter, and advertise
`paged_state_reconciliation` / `atomic_state_scope_replace` only after
contract tests cover multi-page domains, empty scopes, cursor misuse,
interleaved deletions, and fenced-replacement rollback.

## Commit style

Conventional commits not required, but tight subject lines please:

```
add html backend
fix: pdf backend silent failure on scanned PDFs
test: cover profile lookup with $DBT_ML_PROFILES_DIR
docs: rewrite README quickstart for end users
```

## Releasing

Follow [`docs/release.md`](docs/release.md). Update `pyproject.toml` and the
changelog, then push a matching `vX.Y.Z` tag. The release workflow verifies the
tag, audits, lints, type-checks, tests, builds the distributions, publishes to
PyPI, and creates the matching GitHub Release. Do not publish manually from a
developer workstation.
