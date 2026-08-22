# Release process

stel publishes to [PyPI](https://pypi.org/project/stel/) from GitHub
Actions when a version tag is pushed.

## One-time setup

Create a secret named `PYPI_API_TOKEN` holding a PyPI API token that can publish
the `stel` package.

It must be an **environment** secret on the `pypi` environment, not a plain
repository secret: the publish job declares `environment: name: pypi`
(`release.yml:72`), so it resolves secrets from that environment. A token stored
only at the repository level is not visible to it.

- Repository settings -> Environments -> `pypi` -> Environment secrets

The token must be scoped to the `stel` project. A token scoped to a different
project fails at the publish step, after the whole build and test matrix has
already run.

## The `dbt-ml` redirect package (one-time)

`compat/dbt-ml/` is not part of the `stel` release and is not built by the
release workflow. It is a standalone redirect published once to the old PyPI
project so an old pin or a stale link resolves to something that says where the
project went.

It carries no functionality by design. A shim that aliased stel's submodules so
`import dbt_ml.adapters` kept working would hide the rename behind a facade that
then has to be maintained and eventually removed, for a user base of one.

Publishing it needs a PyPI token scoped to **`dbt-ml`**, not the `stel` token
the release workflow uses:

```bash
cd compat/dbt-ml
uv build
uv publish --token <dbt-ml-scoped-token>
```

Do not yank `dbt-ml` 0.8.0. It still works, and yanking breaks anyone pinned to
it — which is precisely the person this redirect is for.

## Cut a release

1. Merge all release-bound PRs to `master`.
2. Update `pyproject.toml` with the new package version. Re-run
   `uv sync` so `uv.lock` records the same version.
3. Move the top `CHANGELOG.md` entries from `Unreleased` to a dated
   version heading, for example `## v0.2.7 - 2026-07-10`. Confirm every merged
   PR since the previous tag has an entry — a feature PR that skipped the
   changelog is easy to miss.
4. Run the local checks from the repository root:

   ```bash
   uv sync --all-extras --dev --locked
   uv run pip-audit --skip-editable
   uv run ruff check
   uv run ty check
   uv run pytest -q
   uv build
   ```

5. **BigQuery pre-release smoke test.** The default `pytest` run skips the live
   BigQuery integration tests, so warehouse-adapter, capability-contract,
   incremental, or materialization changes are only exercised against a fake
   client and DuckDB. When the release touches any of those, run the live tests
   once against a real project — a declared-capability or materialization gap on
   BigQuery is otherwise invisible until a consumer hits it (this is what shipped
   the v0.2.9 → v0.2.10 hotfix). Authentication is Application Default
   Credentials — stel does not read a `.env` file; export real environment
   variables:

   ```bash
   # ADC via gcloud, or point at a service-account key (keep it out of the repo):
   gcloud auth application-default login
   # or: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json

   STEL_BQ_TEST_PROJECT=your-gcp-project uv run pytest -q tests/test_bigquery_adapter.py
   ```

   The tests create and drop their own scratch datasets. If you cannot run them,
   note in the release PR that the BigQuery live path was not exercised.

   **A new BigQuery operation needs a live test in that file before it ships.**
   The gate is only worth running if it touches the code the release changed.
   `append_rows` and `read_relation` shipped in v0.10.0 with no entry there, so
   the gate passed while neither had ever run against BigQuery — reporting
   success on code it never executed, which is worse than having no gate,
   because this one is trusted. Neither is abstract on `WarehouseAdapter`, so
   implementing the ABC never forced coverage.

   `test_every_bigquery_operation_is_live_tested_or_listed_as_debt` now
   enforces this in the ordinary suite: adding a BigQuery-specific method
   without a live test fails, and the operations still lacking one are listed
   explicitly rather than being invisible. That list may only shrink.

6. Commit the release prep, then tag and push. The tag must match the package
   version in `pyproject.toml` with a leading `v`:

   ```bash
   git tag v0.2.7
   git push origin master
   git push origin v0.2.7
   ```

The `release` workflow builds the distributions, publishes them to PyPI, and
creates the matching GitHub Release with the built artifacts attached.
