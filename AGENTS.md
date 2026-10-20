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
- Linear project `Constellations` — the backlog, its themes and milestones, and
  what is next (see "Change, GitHub, and Linear hygiene"). Verify every claim
  against current code, tests, and user docs before describing it as
  implemented.

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
  Known gap (issue #648): a Rust panic inside a native extension (LanceDB)
  prints to the process's stderr through Rust's default panic hook before any
  Python handler runs, so stderr is outside this guarantee; the store's
  sanitizer still governs what Python sees. Observed panic text carries the
  thread name, the crate's source location and the panic message, not the
  store path, but no handler of stel's can promise that.
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
changes. `uv run pytest -q -m "not e2e"` is the fast loop -- 2,683 tests in
~66s against ~248s for everything, serial on four cores -- and CI runs the
whole suite regardless. `-n auto --dist loadfile` spreads either run across
every core (the whole suite in ~90s on four); CI runs that way, and it is
what to reach for when the fast tier is not enough. Run `uv build` for
packaging or release changes. The default suite must
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

**Linear holds the backlog; GitHub holds the work.** Linear (team prefix `ALE`,
project `Constellations` for this repo, `Astrolabe` for the downstream data
project) is the source of truth for what exists, where it sits and what is
next. GitHub is where a piece of work gets its branch, PR and review. Decided
2026-10-20 (issue #705, ALE-311), following astrolabe (astrolabe#1181,
ALE-228). Before then GitHub `triage:*` labels picked the next issue and Linear
held only themes, so "next" rested on labels nobody maintained and six themes
sat In Progress with nothing open under them. Those labels are retired.

**Where an issue goes in Linear.**

| Level | Use it for | Example |
|---|---|---|
| Project | A separately shippable product or repo | Constellations, Astrolabe |
| Milestone | A sequenced phase with a done-when and a real target date | Phase 1 — Dogfood |
| Parent issue (a *theme*) | One body of work, with what "done" means for it | ALE-48 Retrieval quality |
| Leaf issue | One piece of work: one GitHub issue, one or a few PRs | ALE-305 Snowflake adapter |

Do not nest more than one level. A group of more than six issues that will
take longer than a week gets a milestone. In Linear text, write a GitHub
reference with its repo (`constellations#701`): Linear turns a bare `#701` into
a link to whatever has that number in another repo.

**Filing an issue.** File it in Linear (`save_issue`), not with `gh issue
create`: a title, a problem statement and a done-when, placed under its parent
theme with a priority, in **Todo** (next up) or **Backlog** (not yet). Do not
create its GitHub issue yet. If no theme fits, say so and ask rather than
inventing one: a new theme is a decision about the project's shape.

**Picking the next issue** is a Linear query, not a scan of GitHub:

- **State Todo, leaf issues only**, ordered by the target date of the issue's
  milestone (none sorts last), then by priority. A parent is never picked:
  take its next child.
- **Keep In Progress short.** At most five leaf issues In Progress in the
  project; at the limit, finish one before starting another. In Progress means
  a branch or an open PR exists.
- **A parent is never In Progress** and never counts toward that limit. Its
  state is its children's.
- **Skip an issue claimed on GitHub** (`scripts/check_issue_claim.py` exit 1).

**Starting work on ALE-nn** (the `start-issue` skill runs all of this):

1. Find or create its GitHub twin: `python scripts/linear_twin.py ALE-nn`.
   Exit 0 prints it; exit 1 means none, so create it with `--create --title
   "<title>" --body-file <file>`; exit 2 means it could not check. The body's
   first line is `Theme: ALE-<parent> — <title>`, which names the parent; the
   script adds the `Synced from Linear [ALE-nn](url)` footer, which is the only
   thing that marks a twin. Never search GitHub for the identifier: search
   matches substrings.
2. Attach the GitHub issue in Linear (`save_issue` `links`).
3. Claim it: `python scripts/check_issue_claim.py N`, then `--claim --branch
   feat/<N>-<slug>`. Every session runs as one GitHub account, and the claim
   comment naming the branch is what tells sessions apart. Exit 1 means stop
   and ask; exit 2 is not a pass. Re-check with `--mine <branch>` just before
   `gh pr create`, and `--release` a claim whose work is dead.
4. Move the Linear issue to In Progress.

**Opening the PR.** `Closes #N` closes the GitHub issue on merge; `Part of #N`
does not. Put `Fixes ALE-nn` on the **leaf** issue in the body too, and attach
the PR in Linear (`save_issue` `links`) so the link exists even if Linear's
GitHub app misses it. **Never put a closing keyword on a theme** (`Fixes`,
`Closes`, `Resolves` against a parent): a theme outlives any single PR, and
completing it on merge hides everything still open under it. Name the theme as
`Theme: ALE-nn` instead.

**On merge**, check that the leaf issue went to Done. Done means shipped and
verified. If something is still owed after the merge (a release, a check
against astrolabe's prod), move it to **In Review** with a comment saying what
is owed, and to Done once that is recorded. In Review never means "PR open":
that stays In Progress.

An issue that arrived GitHub-first (from outside, or an older one) gets its
Linear issue now: file it in Linear, add the footer to the GitHub body, and
attach the GitHub link.

- Inspect the worktree first. Preserve unrelated user changes, use an isolated
  worktree when branches conflict, and stage explicit files only.
- Never commit `target/`, database/WAL files, caches, virtual environments,
  generated example data, `dist/`, root `docs/research/`, root `docs/private/`,
  or any `_scratch/` directory. Put reusable review findings in issue/PR
  comments instead of committing temporary notes.
- Search Linear, and open and closed GitHub issues, before filing one. Comment
  on an existing issue when the scope overlaps; file a new one only for
  distinct work.
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
