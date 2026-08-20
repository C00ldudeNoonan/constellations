"""Compilation for warehouse-native SQL transform models (issues #143, #142).

See docs/architecture/sql-models.md for the accepted design. A SQL transform is
`transform.type: sql` with an external `.sql` file whose only template surface is
`ref('literal')`, a read-only `target`, and — for `materialization: incremental`
models only — `is_incremental()` and `this`. Compilation is two-phase:

1. discover — statically read the `ref('…')` calls (AST, no execution) so the DAG
   can be built and validated before any warehouse access;
2. compile — render each ref into its adapter-quoted target relation.

The rendered statement must be a single `SELECT`; stel owns target creation and
replacement via the adapter (never core-assembled CTAS).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, nodes
from jinja2.sandbox import SandboxedEnvironment

# Bump when the template contract or compiled-SQL shape changes in a way that
# should re-select existing SQL models under --state.
SQL_COMPILER_CONTRACT_VERSION = "sql/v1"

_SQL_SUFFIX = ".sql"

# Zero-arg template calls that are not refs but are still recognized by
# discover_refs; presence/absence of their runtime value is enforced by
# compile_sql's StrictUndefined rendering, not here.
_INCREMENTAL_CALLS = frozenset({"is_incremental"})


class SqlModelError(Exception):
    """Raised for invalid SQL-model source, refs, or statement shape.

    Carries no file/line context itself; the compiler wraps it into a
    ConfigError with the model's YAML location.
    """


def _sandbox() -> SandboxedEnvironment:
    # A sandboxed env blocks underscore/internal attribute access and unsafe
    # callables; we additionally clear the default globals and filters so a
    # template can reach nothing but the names we inject explicitly.
    env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
    env.globals.clear()
    env.filters.clear()
    env.tests.clear()
    return env


class _Target:
    """Frozen, string-only `target` exposed to templates (no methods/paths)."""

    __slots__ = ("name", "type")

    def __init__(self, name: str, type_: str) -> None:
        self.name = name
        self.type = type_


def discover_refs(sql_text: str, *, model_name: str) -> list[str]:
    """Return the deduplicated, order-preserving list of model names referenced
    by literal `ref('name')` calls. Rejects dynamic/non-literal refs by parsing
    the template AST rather than executing it."""
    env = _sandbox()
    try:
        ast = env.parse(sql_text)
    except Exception as e:  # jinja2.TemplateSyntaxError and friends
        raise SqlModelError(
            f"SQL model '{model_name}' has invalid template syntax: {e}"
        ) from e

    refs: list[str] = []
    seen: set[str] = set()
    for call in ast.find_all(nodes.Call):
        func = call.node
        if isinstance(func, nodes.Name) and func.name in _INCREMENTAL_CALLS:
            if call.args or call.kwargs or call.dyn_args or call.dyn_kwargs:
                raise SqlModelError(
                    f"SQL model '{model_name}' calls '{func.name}(...)' with "
                    "arguments; it takes none."
                )
            continue
        if not (isinstance(func, nodes.Name) and func.name == "ref"):
            # Any other callable in the template is unsupported (no macros/
            # filters/functions beyond ref/is_incremental); flag it rather than
            # ignoring it.
            name = getattr(func, "name", None)
            if isinstance(func, nodes.Name) and name not in {"ref"}:
                raise SqlModelError(
                    f"SQL model '{model_name}' calls unsupported '{name}(...)'; "
                    "only ref('model') and is_incremental() are available."
                )
            continue
        if (
            len(call.args) != 1
            or call.kwargs
            or call.dyn_args is not None
            or call.dyn_kwargs is not None
            or not isinstance(call.args[0], nodes.Const)
            or not isinstance(call.args[0].value, str)
        ):
            raise SqlModelError(
                f"SQL model '{model_name}' has a non-literal ref(); ref() takes a "
                "single quoted model name (dynamic refs are unsupported)."
            )
        target = call.args[0].value
        if target not in seen:
            seen.add(target)
            refs.append(target)
    return refs


def read_sql_source(path: Path, *, model_name: str) -> str:
    """Read a resolved (already confined) `.sql` file, validating extension and
    existence. Path confinement is the caller's responsibility."""
    if path.suffix.lower() != _SQL_SUFFIX:
        raise SqlModelError(
            f"SQL model '{model_name}' path must end in .sql, got '{path.name}'"
        )
    if not path.is_file():
        raise SqlModelError(
            f"SQL model '{model_name}' SQL file not found: {path}"
        )
    return path.read_text(encoding="utf-8")


# Strip line (`-- …`) and block (`/* … */`) comments before the statement guard.
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def validate_single_select(sql_text: str, *, model_name: str) -> None:
    """Lightweight, dialect-agnostic guard: the (comment-stripped) statement must
    be a single `SELECT`/`WITH … SELECT`, with no trailing second statement.
    This is not a full parser — the adapter dry-run is authoritative — but it
    cheaply rejects multi-statement scripts and leading DDL/DML before any
    connection."""
    stripped = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql_text)).strip()
    if not stripped:
        raise SqlModelError(f"SQL model '{model_name}' is empty")
    # Allow exactly one optional trailing semicolon.
    body = stripped[:-1] if stripped.endswith(";") else stripped
    if ";" in body:
        raise SqlModelError(
            f"SQL model '{model_name}' must be a single statement; multiple "
            "';'-separated statements are unsupported."
        )
    first = body.split(None, 1)[0].lower()
    if first not in {"select", "with"}:
        raise SqlModelError(
            f"SQL model '{model_name}' must be a single SELECT (or WITH … SELECT); "
            f"found a statement starting with '{first.upper()}'."
        )


def compile_sql(
    sql_text: str,
    *,
    model_name: str,
    relations: dict[str, str],
    target_name: str,
    target_type: str,
    this: str | None = None,
    is_incremental: bool | None = None,
) -> str:
    """Render the SQL with `ref('m')` → `relations['m']` (an adapter-quoted
    relation) and a read-only `target`. `relations` must cover every discovered
    ref. Returns the compiled single-statement SELECT.

    `this` (the target's own quoted relation) and `is_incremental` are only
    meaningful for `materialization: incremental` SQL models — pass both or
    neither. Left `None`, referencing either in the template raises (a `full`
    model has no incremental branch to compile)."""

    def ref(name: str) -> str:
        try:
            return relations[name]
        except KeyError:  # pragma: no cover - guarded by discover_refs upstream
            raise SqlModelError(
                f"SQL model '{model_name}' references unresolved model '{name}'"
            ) from None

    env = _sandbox()
    template = env.from_string(sql_text)
    render_kwargs: dict[str, Any] = {
        "ref": ref,
        "target": _Target(target_name, target_type),
    }
    if this is not None:
        render_kwargs["this"] = this
    if is_incremental is not None:
        render_kwargs["is_incremental"] = lambda: is_incremental
    compiled = template.render(**render_kwargs)
    validate_single_select(compiled, model_name=model_name)
    return compiled.strip()


def build_key_check_sql(select_sql: str, key_column: str) -> str:
    """A portable, read-only diagnostic query: counts null and duplicate values
    of `key_column` in `select_sql`'s result without materializing or returning
    any row payload. `key_column` must already be adapter-quoted by the caller.
    Used by adapters to validate the unique key before mutating the target."""
    return (
        "select "
        f"sum(case when {key_column} is null then 1 else 0 end) as null_count, "
        f"count(*) - count(distinct {key_column}) as duplicate_count "
        f"from ({select_sql}) as _stel_key_check"
    )
