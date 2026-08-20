# Release process

stel publishes to [PyPI](https://pypi.org/project/stel/) from GitHub
Actions when a version tag is pushed.

## One-time setup

Create a GitHub Actions secret named `PYPI_API_TOKEN` with a PyPI API token that
can publish the `stel` package.

Recommended location:

- Repository settings -> Secrets and variables -> Actions -> Repository secrets

If you want extra release controls, create a `pypi` environment in GitHub and
move the secret there as an environment secret.

## Cut a release

1. Merge all release-bound PRs to `master`.
2. Update `stel/pyproject.toml` with the new package version. Re-run
   `uv sync` so `uv.lock` records the same version.
3. Move the top `stel/CHANGELOG.md` entries from `Unreleased` to a dated
   version heading, for example `## v0.2.7 - 2026-07-10`. Confirm every merged
   PR since the previous tag has an entry — a feature PR that skipped the
   changelog is easy to miss.
4. Run the local checks from `stel/`:

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

6. Commit the release prep, then tag and push. The tag must match the package
   version in `pyproject.toml` with a leading `v`:

   ```bash
   git tag v0.2.7
   git push origin master
   git push origin v0.2.7
   ```

The `release` workflow builds the distributions, publishes them to PyPI, and
creates the matching GitHub Release with the built artifacts attached.
