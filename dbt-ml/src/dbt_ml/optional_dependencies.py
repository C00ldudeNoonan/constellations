from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any


class OptionalDependencyError(ImportError):
    pass


def optional_dependency_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def import_optional_dependency(
    module: str,
    *,
    extra: str,
    feature: str,
    distribution: str | None = None,
) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        package = distribution or module.split(".", maxsplit=1)[0]
        raise OptionalDependencyError(
            f"{feature} requires the optional dependency '{package}'. "
            f"Install it with: pip install 'dbt-ml[{extra}]'"
        ) from error
