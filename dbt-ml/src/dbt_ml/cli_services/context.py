"""Shared CLI bootstrap and the exit-code error contract (issue #190).

One project/profile bootstrap path reused across commands, plus the
configuration-error taxonomy that maps setup failures to exit code 2. Kept out
of ``cli.py`` so both the command edge and the internal services (watch, …)
depend on a single bootstrap without importing the command module.
"""

from __future__ import annotations

from pathlib import Path

import click

from ..config import ConfigError, load_project
from ..config.model import ModelConfig
from ..config.project import ProjectConfig
from ..config.source import SourceConfig
from ..dag import DAGError, ProjectDAG, SelectionError
from ..manifest import StateError
from ..optional_dependencies import OptionalDependencyError
from ..profile import ProfileError
from ..providers import ProviderConfigurationError, ProviderNotFoundError


class ConfigClickError(click.ClickException):
    """A configuration/usage error the run never got past. Exits 2 so an
    orchestrator (issue #87) can tell a broken project apart from a run that
    started but had a model fail (exit 1)."""

    exit_code = 2


# Errors that mean the project couldn't be coherently set up → exit 2. RunError
# (a run that started but a model failed hard) stays a plain ClickException → 1.
CONFIG_ERRORS = (
    ConfigError,
    DAGError,
    SelectionError,
    ProfileError,
    ProviderConfigurationError,
    ProviderNotFoundError,
    StateError,
    OptionalDependencyError,
)


def load_project_or_click(
    project_dir: Path,
) -> tuple[ProjectConfig, list[SourceConfig], list[ModelConfig]]:
    try:
        return load_project(project_dir)
    except ConfigError as e:
        raise ConfigClickError(str(e)) from e


def build_dag_or_click(
    sources: list[SourceConfig], models: list[ModelConfig]
) -> ProjectDAG:
    try:
        return ProjectDAG(sources, models)
    except DAGError as e:
        raise ConfigClickError(str(e)) from e
