"""Filesystem boundary policy for project-configured paths (issue #65).

Paths declared in project YAML ship with a repo, so a third-party project
could otherwise read, write, or delete anywhere the operator can. Policy:

- Project-YAML paths are confined to the project directory. Escaping is a
  ConfigError (exit 2) unless the declaring block opts in with
  `external: true` (sources, ml.artifact) — layout paths and model-level llm
  cache paths have no opt-in.
- profiles.yml paths (warehouse, llm cache) are trusted operator-local
  config, like dbt's; `dbt-ml clean` additionally requires --force when the
  warehouse file lives outside the project directory.
"""
from __future__ import annotations

from pathlib import Path

from .config.loader import ConfigError


def resolve_within_project(
    path: Path | str,
    project_dir: Path,
    *,
    surface: str,
    external: bool = False,
    hint: str | None = None,
) -> Path:
    """Resolve `path` against `project_dir` and enforce the boundary.

    Absolute inputs pass through pathlib join semantics untouched; `.resolve()`
    canonicalizes `..` and symlinks, so a link escaping the project is caught
    the same as a literal traversal."""
    resolved = (project_dir / Path(path)).resolve()
    if external:
        return resolved
    if not resolved.is_relative_to(project_dir.resolve()):
        raise ConfigError(
            f"{surface} '{path}' resolves outside the project directory: "
            f"{resolved}. dbt-ml confines project-configured paths to the "
            f"project. {hint or 'Move it inside the project directory.'}"
        )
    return resolved


def is_within_project(path: Path, project_dir: Path) -> bool:
    return path.resolve().is_relative_to(project_dir.resolve())
