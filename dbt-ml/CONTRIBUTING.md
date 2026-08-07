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
uv run ty check                   # type checking
uv run pytest -q                  # tests
```

ty is the required and only static type checker, and it checks the whole
project — `src/dbt_ml`, `tests`, `examples`, `scripts`, and the focused static
compatibility fixtures in `typecheck/`. Ruff's ANN and PYI rules enforce
annotation *presence* on package source only (tests and examples are exempt so
fixture code needn't be fully annotated, but ty still type-checks their bodies).
Explicit `Any` remains permitted at validated dynamic, SDK, and
optional-dependency boundaries; in test doubles, prefer `cast(Any, obj)` over a
suppression comment. Do not add broad suppressions to keep the command clean. CI
and release validation run the same checks; the workflows live in the
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

## Adding a project-local post-extract hook

Use `extraction.post_extract` when a backend can parse the source envelope but
the useful warehouse representation must be derived before publication (for
example, JSON containing a large HTML field). Put the dotted module path in
model YAML and the matching `.py` file inside the project.

1. Expose `run(fields)` or `run(fields, ctx)` and return a mapping whose keys are
   strings. The returned mapping replaces the backend fields, so explicitly
   retain every field that belongs in the output and omit raw payloads.
2. Treat `fields` as untrusted document data. Do not log it or interpolate it
   into exceptions. Hook failures are intentionally surfaced without their raw
   exception detail.
3. Use `ctx.local_path` only when the verified fetched bytes are needed;
   `ctx.document_id`, `source_name`, `source_path`, `source_uri`,
   `source_metadata`, and configured `options` carry safe execution context.
4. Expose `validate_options(options) -> None` when the hook takes options. The
   compiler calls it before source discovery, credentials, or warehouse access.
   Option values affect incremental code identity but are omitted from generated
   manifests; validation failures surface only a stable, value-free diagnostic.
5. Keep imports needed only by the hook lazy and document the required dbt-ml
   extra. The hook is trusted project code, not a sandbox or provider boundary.
6. Test that the raw input field is absent from the materialized schema, hook
   source/options change `code_version`, and native batch mode applies the hook
   before fetched-file cleanup when the model supports batching.

Backend warnings and metrics are preserved by the runner; hooks only own the
field mapping. Use a built-in backend plus this hook for project/domain-specific
derivation. Add a package backend only when the parser is reusable and belongs
in dbt-ml's installed backend registry.

## Adding a built-in Python transform

1. Put the module under the appropriate package, such as
   `src/dbt_ml/text/transforms/`, and expose `run(deps, ctx)` with one or two
   positional arguments.
2. For transforms with options, expose `validate_options(options) -> None`.
   The compiler calls this hook before source discovery, credentials, optional
   SDK initialization, model loading, or warehouse mutation.
3. When options name the transform's upstream models, also expose
   `declared_dependencies(options) -> Iterable[str]` returning the complete set
   of model names those options require. Implementing it asserts that the
   options fully determine the inputs, so the compiler enforces that
   `depends_on` matches exactly and rejects a misspelled or stale reference
   before any model is materialized. Omit the hook when a transform accepts a
   variable dependency set.
4. A one-to-many transform (many stable child rows per input parent) may opt
   into `materialization: incremental` by exposing
   `declared_incremental_contract(options) -> IncrementalContract`. The contract
   names the output `parent_key` (delete scope) and `child_key` (upsert scope),
   the `parent_source` dependency (and its key column) whose rows define the
   parents, and any whole-table `reference_deps`. The runner then skips
   unchanged parents, invokes the transform only on changed and new parents, and
   replaces a changed parent's children by deleting on the parent key and
   upserting on the child key. Emit the same deterministic `child_key` for the
   same input, carry the `parent_key` on every output row, and process parents
   independently (a transform needing cross-parent state cannot be incremental).
   Without the hook a transform stays `full`; declaring it for
   `materialization: incremental` is required and validated against `depends_on`
   at compile time.
5. Parse runtime options with the same strict Pydantic model used by the
   validation hook. Keep optional dependencies lazy and provide an actionable
   extra-install command when they are absent.
6. Use a provider protocol when execution depends on an external SDK or model.
   Unit tests must use a deterministic fake and must not require downloads,
   credentials, or network access. A transform that calls an LLM resolves the
   provider from `ctx.llm` (the profile's `llm:` block) — reuse
   `backends.llm_backend.extract_fields_with_usage` so caching/retries/credential
   resolution match the `llm:` kind — and should expose `requires_llm(options) ->
   bool`. The compiler enforces `transform.uses_llm: true` for such a model, and
   `versioning.py` then folds the resolved provider identity into its code
   version. The `extract_relations` `model_assertion` extractor is the reference.
7. Document the input and output schemas, sensitive-field behavior, optional
   extra, and a runnable example whenever the transform is public.

### Adding an entity-linking resolver

The `link_entities` transform dispatches to a resolver selected by its
`resolver:` option. To add one:

1. Add a strict options model in `src/dbt_ml/text/linking.py` that subclasses
   `_EntityLinkBaseOptions` (shared mention identity, privacy, and projection
   fields) with a `resolver: Literal["<name>"]` discriminator, then add it to
   the `EntityLinkConfig` union.
2. Implement an `EntityResolver` (declaring the mention columns it reads and how
   to extract a per-mention match signal) whose `build_reference` validates and
   indexes the alias frame and exposes a `fingerprint` for `alias_set_version`.
   Register the instance in `RESOLVERS` and give it a `version` constant that is
   bumped whenever its matching semantics change.
3. Keep resolution deterministic and offline. Reuse existing model kinds for
   anything needing credentials or a provider — the vector-similarity resolver
   consumes vectors produced by the `embed` kind rather than embedding text
   itself — so the transform seam stays credential-free and unit-testable with
   hand-authored inputs.
4. The shared driver in `text/transforms/_linking.py` owns row shaping,
   ambiguity policy, and the fixed output schema; resolvers only return
   per-namespace `NamespaceResolution` outcomes, so `match_score` and the status
   columns stay consistent across resolvers.

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
