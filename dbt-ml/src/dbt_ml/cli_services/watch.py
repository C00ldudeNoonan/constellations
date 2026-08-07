"""Watch execution service (issue #190, Workstream D).

The `run --watch` loop, factored out of the CLI command so it is importable and
testable without invoking Click. It still emits progress via `click.echo` (the
loop is inherently interactive), but owns no command declaration.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import click

from ..adapters import create_adapter
from ..compiler import validate_project_contract, validate_warehouse_capabilities
from ..config import ConfigError
from ..manifest import write_manifest, write_run_results
from ..paths import resolve_within_project
from ..profile import apply_source_path_overrides, resolve_profile
from ..runner import RunError, run_project
from .context import CONFIG_ERRORS, ConfigClickError, load_project_or_click


def run_watch(
    project_dir: Path,
    *,
    profiles_dir: Path | None,
    target: str | None,
    full_refresh: bool,
    select: str | None,
    exclude: str | None,
    threads: int = 1,
    source_filter: Sequence[str] = (),
) -> None:
    """Watch source paths and re-run on changes. Blocking; Ctrl-C to exit."""
    from watchfiles import watch

    project, sources, models = load_project_or_click(project_dir)
    try:
        dag = validate_project_contract(project, sources, models, project_dir)
        selected = dag.select_models(select=select, exclude=exclude)
        if not selected:
            click.echo("No models selected.")
            return
        required_sources = set(dag.required_sources(selected))
        resolved = resolve_profile(
            project, project_dir, target=target, profiles_dir=profiles_dir
        )
        selected_names = set(selected)
        adapter = create_adapter(resolved.warehouse, project_dir=project_dir)
        validate_warehouse_capabilities(
            [model for model in models if model.name in selected_names],
            adapter,
        )
        sources = apply_source_path_overrides(sources, resolved)
    except CONFIG_ERRORS as e:
        raise ConfigClickError(str(e)) from e
    watch_paths = []
    for s in sources:
        if s.name not in required_sources:
            continue
        try:
            candidate = resolve_within_project(
                s.path,
                project_dir,
                surface=f"Source '{s.name}' path",
                external=s.external,
                hint="Set `external: true` on the source to allow it.",
            )
        except ConfigError as e:
            raise ConfigClickError(str(e)) from e
        if candidate.exists():
            watch_paths.append(candidate)
    if not watch_paths:
        raise click.ClickException(
            "No source paths exist on disk yet. Create them (or run `dbt-ml seed`) "
            "and try `dbt-ml run --watch` again."
        )

    click.echo(f"watching {len(watch_paths)} source path(s); Ctrl-C to stop")

    def _do_run() -> None:
        try:
            results = run_project(
                project_dir,
                full_refresh=full_refresh,
                select=select,
                exclude=exclude,
                target=target,
                profiles_dir=profiles_dir,
                threads=threads,
                source_filter=source_filter,
            )
        except (*CONFIG_ERRORS, RunError) as e:
            click.echo(f"error: {e}", err=True)
            return
        write_manifest(project_dir, target=target, profiles_dir=profiles_dir)
        write_run_results(
            project_dir, results, target=target, profiles_dir=profiles_dir
        )
        for r in results:
            click.echo(
                f"  {r.model_name:<22} {r.kind:<12} "
                f"processed={r.documents_processed:<5} skipped={r.documents_skipped:<5} "
                f"rows={r.rows_written}"
            )

    _do_run()
    try:
        for _ in watch(*watch_paths, debounce=500, recursive=True):
            click.echo("change detected, re-running...")
            _do_run()
    except KeyboardInterrupt:
        click.echo("watch stopped.")
