# AGENTS.md — stel

## Scope and sources of truth

This file applies to the whole repository. A more specific `AGENTS.md`
supplements these rules and overrides only conflicting guidance in its subtree.

stel is a standalone, dbt-shaped Python CLI for turning unstructured data
into warehouse tables. It is not a dbt package or adapter, and similarly named
artifacts are not dbt-core contracts unless that compatibility is explicit.

The Python project lives at the repository root. Run every command — Git,
GitHub, Python, uv, test, lint, type-check, build — from there.

Use these maintained references rather than copying volatile feature lists:

- `README.md` (landing page) and `docs/reference.md` (full reference) —
  shipped behavior and user guidance.
- `CONTRIBUTING.md` — extension contracts and contributor workflow.
- `docs/release.md` — release process.
- GitHub issues labeled `roadmap` — planning context. Verify every claim against
  current code, tests, and user docs before describing it as implemented.

## Architecture and product boundaries

- Python 3.12+ only. Do not introduce Rust or PyO3 without an explicitly
  accepted design and scoped task.
- Keep warehouse-specific SQL/dialect behavior, materialization, quoting, and
  incremental state behind `src/stel/adapters/`. State belongs to the active
  adapter; do not add new DuckDB assumptions to orchestration.
- Keep document discovery and fetch behind `src/stel/sources/`, extraction
  behind `src/stel/backends/`, and vendor inference behavior behind a provider
  registry/contract. Establish that seam before adding another vendor; avoid
  integration-specific branches in `runner.py`.
- Configuration uses strict Pydantic v2 models. Compiler/preflight validation
  should reject bad configuration before source discovery, credentials, remote
  calls, or warehouse mutation.
- Keep the core installation lean. Optional integrations must import lazily,
  declare the appropriate extra, and fail with an actionable install command
  when the extra is absent.
- Treat stel artifact shapes as explicit contracts and version schema
  changes. Do not imply generic dbt artifact, test, state, or selector
  compatibility that is not implemented.
- Update user docs, config examples, templates, and artifact fixtures when a
  public CLI/config/behavior contract changes.

## Security and correctness invariants

- Python transforms and custom tests execute as trusted code; profiles are
  operator-controlled. Project YAML and source documents still cross strict
  validation, path, and parser boundaries. None of these inputs are sandboxed.
- Route paths from project YAML through `paths.py`. External access must be an
  explicit, reviewable opt-in; profile paths remain operator-controlled.
- Configuration discovery must accept regular, non-symlink files only. Local
  source discovery and fetch must not follow symlinks; preserve the no-follow
  walk and verified scratch-copy boundary in `sources/local.py`.
- Never expose resolved credentials or credential environment-variable names in
  logs, caches, artifacts, config dumps, hashing, or diagnostics. Preserve
  references through validation and reveal values only at native SDK
  construction. Raw documents, provider response bodies/headers, and sensitive
  exception text must not enter logs or artifacts. Warehouses and caches may
  contain intended
  outputs; configured prompts and cached values can be sensitive, so minimize
  and document artifact-visible fields and preserve owner-only cache storage.
- Validate incremental keys before mutation. Use each adapter's declared
  publication guarantees, and advance state only after successful publication;
  do not assume cross-operation transactions where the adapter lacks them.
- Cleanup commands may remove only stel-owned local artifacts. Do not hide
  warehouse-wide reset behavior behind a familiar dbt command.
- PII redaction is not sufficient when raw sensitive input columns remain in the
  output. Tests, examples, and docs must project or drop retained originals.

## Development workflow

From the repository root:

```bash
uv sync --all-extras --dev --locked
uv run pip-audit --skip-editable
uv run ruff check
uv run ty check
uv run pytest -q
```

Use targeted tests while iterating, then run the full audit/lint/type/test set
before handing off implementation, configuration, template, or dependency
changes. Run `uv build` for packaging or release changes. The default suite must
not require live provider or cloud credentials; opt-in integration tests must be
credential-gated and have deterministic unit coverage. Update `uv.lock` only
when required by an intentional `pyproject.toml` metadata or dependency change,
and exclude unrelated resolution churn.

## Change and GitHub hygiene

- Inspect the worktree first. Preserve unrelated user changes, use an isolated
  worktree when branches conflict, and stage explicit files only.
- Never commit `target/`, database/WAL files, caches, virtual environments,
  generated example data, `dist/`, root `docs/research/`, root `docs/private/`,
  or any `_scratch/` directory. Put reusable review findings in issue/PR
  comments instead of committing temporary notes.
- Search open and closed issues before creating one. Comment on an existing
  tracker when the scope overlaps; create a new issue only for distinct work.
- Link implementation PRs to their related issues. Use `Closes #…` only when
  the PR fully satisfies that issue's acceptance criteria; keep parent/design
  issues open when follow-up work remains.
- Keep PRs focused, document validation performed, and record intentionally
  deferred findings on the relevant issue.
- Never paste or commit access tokens or resolved credentials. Do not work
  around authentication failures by writing secrets to repository files,
  temporary notes, command history, or issue/PR comments.

## Code and writing style

- Use type hints throughout; ty is the required and only static type checker.
  Ruff's ANN and PYI rules enforce annotation presence. Ruff targets Python 3.12
  with a 100-column line length.
- Use Pydantic v2 for configuration and Click for CLI behavior.
- Comments explain non-obvious reasons or constraints, not line-by-line
  mechanics.
- Keep `dbt` and `stel` lowercase, including at the start of a sentence.
  Write `dbt Labs` for the company.
