# Release process

dbt-ml publishes to [PyPI](https://pypi.org/project/dbt-ml/) from GitHub
Actions when a version tag is pushed.

## One-time setup

Create a GitHub Actions secret named `PYPI_API_TOKEN` with a PyPI API token that
can publish the `dbt-ml` package.

Recommended location:

- Repository settings -> Secrets and variables -> Actions -> Repository secrets

If you want extra release controls, create a `pypi` environment in GitHub and
move the secret there as an environment secret.

## Cut a release

1. Merge all release-bound PRs to `master`.
2. Update `dbt-ml/pyproject.toml` with the new package version. Re-run
   `uv sync` so `uv.lock` records the same version.
3. Move the top `dbt-ml/CHANGELOG.md` entries from `Unreleased` to a dated
   version heading, for example `## v0.2.7 - 2026-07-10`. Confirm every merged
   PR since the previous tag has an entry — a feature PR that skipped the
   changelog is easy to miss.
4. Run the local checks from `dbt-ml/`:

   ```bash
   uv sync --all-extras --dev --locked
   uv run pip-audit --skip-editable
   uv run ruff check
   uv run ty check
   uv run mypy
   uv run pytest -q
   uv build
   ```

5. Commit the release prep, then tag and push. The tag must match the package
   version in `pyproject.toml` with a leading `v`:

   ```bash
   git tag v0.2.7
   git push origin master
   git push origin v0.2.7
   ```

The `release` workflow builds the distributions, publishes them to PyPI, and
creates the matching GitHub Release with the built artifacts attached.
