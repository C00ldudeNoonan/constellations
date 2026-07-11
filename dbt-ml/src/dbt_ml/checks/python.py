"""Custom Python tests: user-supplied modules that return None (pass) or a
failure message string (fail).

User contract is `def run(con, table_ref) -> str | None`, where `con` is the
underlying warehouse connection (e.g. duckdb.DuckDBPyConnection). For
DuckDB this is `adapter.connection`. Other adapters expose their own
raw connection via `adapter.raw_connection`.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from ..adapters import WarehouseAdapter
from ..versioning import resolve_module_file


class CustomTestError(Exception):
    pass


def load_python_test(module_path: str, project_dir: Path) -> Callable[..., Any]:
    file_path = resolve_module_file(module_path, project_dir)
    if not file_path.exists():
        raise CustomTestError(f"Custom test module not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_path, file_path)
    if spec is None or spec.loader is None:
        raise CustomTestError(f"Could not load test module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = module
    try:
        spec.loader.exec_module(module)
    except (Exception, SystemExit) as e:
        raise CustomTestError(
            f"Custom test module '{module_path}' could not be imported: "
            f"{type(e).__name__}: {e}"
        ) from e

    run_fn: Any = getattr(module, "run", None)
    if run_fn is None or not callable(run_fn):
        raise CustomTestError(
            f"Custom test '{module_path}' must define `run(con, table_ref) -> str | None`"
        )
    if inspect.iscoroutinefunction(run_fn):
        raise CustomTestError("Async custom test functions are not supported")
    try:
        inspect.signature(run_fn).bind(object(), object())
    except (TypeError, ValueError) as e:
        raise CustomTestError(
            f"Custom test '{module_path}' must accept `(con, table_ref)` positional "
            "arguments"
        ) from e
    return cast(Callable[..., Any], run_fn)


def run_python_test(
    module_path: str,
    project_dir: Path,
    adapter: WarehouseAdapter,
    table_ref: str,
) -> str | None:
    """Load `module_path` and call its `run(con, table_ref)`.

    Returns None for pass, a string message for fail.
    """
    run_fn = load_python_test(module_path, project_dir)

    # Hand the underlying warehouse connection to the user-supplied function.
    con = getattr(adapter, "raw_connection", None)
    if con is None:
        # Adapter doesn't expose a raw connection — pass the adapter itself
        # and let the user's test work against its public API.
        con = adapter
    result = run_fn(con, table_ref)
    if result is None:
        return None
    if not isinstance(result, str):
        return str(result)
    return result
