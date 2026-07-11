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
uv run mypy                       # strict type checking
uv run pytest -q                  # tests
```

CI runs the dependency audit, Ruff, and pytest on every push and PR. Release
validation also runs mypy and builds the distributions; see the workflows in
the repository-level `.github/workflows/` directory.

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
- Configuration and artifacts may carry credential environment-variable names,
  never resolved secret values. Do not log secrets or raw document content.
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
