"""Watch execution service (issue #190, Workstream D).

The `run --watch` loop, factored out of the CLI command so it is importable and
testable without invoking Click. It still emits progress via `click.echo` (the
loop is inherently interactive), but owns no command declaration.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import click

from ..adapters import create_adapter
from ..compiler import validate_project_contract, validate_warehouse_capabilities
from ..config import ConfigError
from ..cost_guard import CostConfirmation
from ..execution import ModelRunResult
from ..manifest import write_run_results
from ..paths import resolve_within_project
from ..profile import apply_source_path_overrides, resolve_profile
from ..runner import RunError, run_project
from ..table_layout import RERUN_NAME_MINIMUM, column_width
from .context import CONFIG_ERRORS, ConfigClickError, load_project_or_click

# One directory walk per interval: a created, modified or deleted file is
# noticed within about this long, then the walk repeats until the tree is quiet.
POLL_INTERVAL_SECONDS = 1.0

Snapshot = dict[str, tuple[int, int]]


def snapshot_paths(paths: Sequence[Path]) -> Snapshot:
    """Map every file under `paths` to its (mtime_ns, size).

    Symlinks are neither followed nor recorded, matching the no-follow rule for
    local source discovery in `sources/local.py`: a link inside a watched tree
    must not make stel observe (or re-run for) files outside it.
    """
    state: Snapshot = {}
    pending = list(paths)
    while pending:
        current = pending.pop()
        try:
            if current.is_symlink():
                continue
            info = current.lstat()
            if not current.is_dir():
                state[str(current)] = (info.st_mtime_ns, info.st_size)
                continue
            pending.extend(current.iterdir())
        except OSError:
            continue  # vanished or unreadable mid-walk
    return state


def watch_changes(
    paths: Sequence[Path],
    *,
    interval: float,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[None]:
    """Yield once each time the files under `paths` change, forever.

    A change is held until the tree is unchanged for one full interval, so a
    burst of writes (a copy in progress) yields once. KeyboardInterrupt raised
    by `sleep` propagates to the caller.
    """
    previous = snapshot_paths(paths)
    while True:
        sleep(interval)
        current = snapshot_paths(paths)
        if current == previous:
            continue
        while True:
            sleep(interval)
            settled = snapshot_paths(paths)
            if settled == current:
                break
            current = settled
        previous = current
        yield


def format_rerun_lines(results: Sequence[ModelRunResult]) -> list[str]:
    """One line per model after a watch re-run (issue #822)."""
    width = column_width([r.model_name for r in results], minimum=RERUN_NAME_MINIMUM)
    return [
        f"  {r.model_name:<{width}}{r.kind:<12} "
        f"processed={r.documents_processed:<5} skipped={r.documents_skipped:<5} "
        f"rows={r.rows_written}"
        for r in results
    ]


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
    read_filter: Sequence[tuple[str, str, str]] = (),
    accept_reprocess: bool = False,
    accept_cost: bool = False,
    confirm_cost: CostConfirmation | None = None,
) -> None:
    """Watch source paths and re-run on changes. Blocking; Ctrl-C to exit."""
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
            "No source paths exist on disk yet. Create them (or run `stel seed`) "
            "and try `stel run --watch` again."
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
                read_filter=read_filter,
                accept_reprocess=accept_reprocess,
                accept_cost=accept_cost,
                confirm_cost=confirm_cost,
                persist_manifest=True,
            )
        except (*CONFIG_ERRORS, RunError) as e:
            click.echo(f"error: {e}", err=True)
            return
        write_run_results(
            project_dir, results, target=target, profiles_dir=profiles_dir
        )
        for line in format_rerun_lines(results):
            click.echo(line)

    _do_run()
    try:
        for _ in watch_changes(watch_paths, interval=POLL_INTERVAL_SECONDS):
            click.echo("change detected, re-running...")
            _do_run()
    except KeyboardInterrupt:
        click.echo("watch stopped.")
