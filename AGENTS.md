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
- `docs/adr/` — decision records: what was ruled out and why. Read the index
  before revisiting a design that looks arbitrary, and add one when a decision
  had a real alternative a contributor would plausibly try.
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
changes. `uv run pytest -q -m "not e2e"` is the fast loop -- 2,521 tests in
~82s against ~396s for everything -- and CI runs the whole suite regardless. Run `uv build` for packaging or release changes. The default suite must
not require live provider or cloud credentials; opt-in integration tests must be
credential-gated and have deterministic unit coverage. Update `uv.lock` only
when required by an intentional `pyproject.toml` metadata or dependency change,
and exclude unrelated resolution churn.

## Tests

The suite is the main safety net for a project with one user and no staging
environment, so it is worth stating what it is for. These are habits the suite
already follows; writing them down is what keeps a new test held to them rather
than added by accretion (issue #518).

- **Every documented behavior gets a pin.** Docs cannot drift from code, and
  error messages are product: a user reads them, so a change to one is a change
  to the product.
- **Every incident gets its test.** The fix is a claim; the test is what keeps
  it fixed. Cite the issue in the docstring so the next reader knows what the
  case is defending.
- **Contracts as data.** `test_reentry_contract.py` and `test_frozen_names.py`
  fail when a step or a name appears without a row. A contract nobody can add
  to silently is worth more than a document.
- **Examples run.** Every README claim is executed, never only asserted in
  prose.
- **Doubles, not mocks of the network.** `FakeRepository`, `FakeSearch`,
  `_FakeStorageClient` and `FakeDrive` implement the same protocol as the real
  client, so a signature change breaks them.

**Two tiers.** A file that calls `run_project`, `create_store` or
`export_concept_cloud` declares `pytestmark = pytest.mark.e2e`. Those 59 of 158
files carry roughly 80% of the suite's wall clock, so `-m "not e2e"` is the
loop worth running between edits and the full suite is what you run before
handing off. `test_test_tiers.py` fails if a file that runs a project is
missing the marker, and again if the e2e files ever become the majority --
the split is a contract, not a convention, because a fast tier nobody trusts
is one everybody stops using.

**One test per distinct failure mode, never one per line of a description.**
Two tests that fail together for the same reason are one test and one
maintenance cost.

**A test that cannot fail is worse than no test**, because it reports safety it
does not provide. When a test guards something load-bearing -- a security
property, an ordering guarantee, a rule the code comments argue for -- break
the code deliberately and watch it fail before trusting it. Restore by copying
a backup, never with `git checkout --`, which discards uncommitted work. Real
examples this caught: a query-log test that asserted rows written *after* the
buffer flushed on close, so it passed against the very bug it was written for;
and a concept-cloud test whose period axis assertion held for both the correct
and the broken derivation.

**Isolate process-global state.** Logging configuration, module-level caches
and environment variables outlive the test that set them, and the resulting
failure surfaces in an unrelated file -- or, worse, only in a subset run that
CI never performs. Restore such state in an autouse fixture in
`tests/conftest.py` rather than in the test that happens to touch it, because
the next test to touch it will not know to.

## Change, GitHub, and Linear hygiene

Linear tracks **themes**; GitHub tracks **work**. One Linear issue (team prefix
`ALE`, project `Constellations` for this repo, `Astrolabe` for the downstream
data project) spans many GitHub issues and many PRs. Never mirror the two
one-to-one — a Linear issue per GitHub issue makes both lists worthless.

- **Every GitHub issue names its theme.** Put `Theme: ALE-nn — <title>` and the
  Linear URL in the body. Find the theme by listing open Linear issues in the
  project rather than from a list kept here, which would go stale; if none
  fits, say so in the issue and ask instead of inventing one.
- **Every PR names the theme too**, as a bare identifier (`ALE-nn`) in the
  body. **Never use a Linear closing keyword** — `Fixes ALE-nn`, `Closes
  ALE-nn`, `Resolves ALE-nn` — because a theme outlives any single PR, and
  auto-completing it hides every remaining GitHub issue underneath.
- **Post the back-link.** When a GitHub issue is filed under a theme, or a PR
  merges under one, add a one-line comment on the Linear theme naming the
  issue or PR and what it did. Branch names here are `feat/<issue>-<slug>` and
  carry no `ALE-` identifier, so Linear's automatic branch linking does not
  fire; the comment is the link that is actually guaranteed to exist.
- Keep the traffic proportionate: a theme wants a line per GitHub issue and per
  merged PR, not a running log of every commit.

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

### Python standards

These are enforced by review, not by ruff, so they need stating. Where a rule
has a standing exception in this repository, the exception is named — treat an
unnamed deviation as a defect.

- **Look before you leap.** Check conditions rather than catching exceptions
  for control flow: membership tests over `except KeyError`, `.exists()`
  before `.resolve()`. Exceptions belong at three places only — an error
  boundary (CLI, MCP, runner), wrapping a third-party call that offers no
  alternative, and adding context before re-raising. The provider and store
  layers wrap broad `except Exception` deliberately: the security invariants
  above require that native exception text never reach logs or artifacts, so
  those handlers re-raise a sanitized error rather than swallowing one.
- **Always pass `encoding="utf-8"` to text I/O.** `read_text()`,
  `write_text()`, and text-mode `open()` otherwise use the platform locale,
  which is not UTF-8 on Windows — the same file then reads differently on two
  developers' machines, and a non-ASCII character in project YAML, a source
  document, or a manifest becomes a decode failure that reproduces nowhere
  else. Binary handles take no encoding.
- **Use `pathlib`, not `os.path`** — with one deliberate exception:
  `os.path.abspath` is used where normalization must stay *lexical*, because
  `Path.resolve()` follows symlinks and the source-discovery invariants above
  require not following them. Keep that reasoning at the call site.
- **Imports at module level.** Inline imports are legitimate only for lazily
  loading an optional dependency (see the extras rule above), breaking a
  circular import, or `TYPE_CHECKING`. Modules that import lazily by design —
  `cli_services/*`, the CLI command bodies — say so in their docstring; follow
  the local convention rather than hoisting those.
- **Prefer required parameters to defaults.** A default silently encodes an
  assumption, and adding one to an existing signature does not fail any
  existing call site. Give a parameter a default only when the default is
  right for essentially every caller; when a caller passing the wrong value
  would be a silent bug, make it required so the decision is visible at the
  call site.
- **One canonical import path.** Do not re-export a symbol from a package
  `__init__` unless something actually imports it from there.
- **Four levels of indentation maximum**; extract a helper instead. Properties
  and magic methods stay O(1) — anything doing I/O or iteration gets an
  explicit method name.
