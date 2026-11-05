"""Which part of `backends/options.py` a backend's identity depends on (issue #572).

Every backend's option model lives in one module, so hashing that module whole
re-keys every backend when any one of them changes: a release whose only edit
was the LLM `temperature` default re-extracted a JSON-backed corpus that never
reads it. A backend's identity instead covers the definitions it can reach by
name: its own options model, the helpers and base classes that model names,
and the stel modules those definitions import.

The reach is static name resolution inside one file, the same standard
`_imported_package_modules` already applies to whole modules; a definition
found by `getattr` or a string is invisible to both. A definition is followed
when named anywhere in another definition, including a decorator, annotation
or default, so the closure errs toward hashing too much.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from dataclasses import dataclass

from ..hashing import HASH_DIGEST_SIZE

OPTIONS_MODULE = f"{__package__}.options"

# The one entry in a backend's dependency digests that stands for the options
# module. Not a module name, so it cannot collide with one.
CLOSURE_KEY = f"{OPTIONS_MODULE}#closure"

# The validation path every backend runs through. Hashed by its own source
# only: `validate_backend_options` names `LLMBackendOptions` to special-case
# its credential fields, so following that reference would hand every backend
# the LLM model again. An edit to this code is shared code and re-keys all of
# them, which is right.
SHARED_VALIDATION_PATH = frozenset(
    {
        "BackendOptionContract",
        "BackendOptionsError",
        "get_backend_option_contract",
        "register_backend_option_contract",
        "validate_backend_options",
    }
)


@dataclass(frozen=True)
class OptionsImports:
    """What a module takes from the options module."""

    names: frozenset[str]
    whole: bool


@dataclass(frozen=True)
class OptionsClosure:
    digest: str
    #: Other stel modules the followed definitions import by name. The caller
    #: hashes them, and what they import, as it does any dependency.
    modules: frozenset[str]


def options_imports(
    source: str, *, module_package: str, options_module: str = OPTIONS_MODULE
) -> OptionsImports:
    """Names `source` imports from the options module, or `whole` when it binds
    the module itself (`from . import options`, `import ...options`, `*`)."""
    names: set[str] = set()
    whole = False
    parent, _, leaf = options_module.rpartition(".")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            whole = whole or any(alias.name == options_module for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve(node, module_package)
            if resolved == options_module:
                for alias in node.names:
                    if alias.name == "*":
                        whole = True
                    else:
                        names.add(alias.name)
            elif resolved == parent:
                whole = whole or any(alias.name == leaf for alias in node.names)
    return OptionsImports(frozenset(names), whole)


def options_closure(
    source: str,
    *,
    module_package: str,
    package: str,
    deep: frozenset[str],
    shallow: frozenset[str],
) -> OptionsClosure:
    """Digest of the definitions in `source` reachable from `deep` (followed)
    and `shallow` (their own text only), plus the module-level statements that
    are neither definitions nor imports, which run whatever is asked for."""
    lines = source.splitlines()
    index = _index(source, module_package=module_package, package=package)
    wanted: set[str] = set()
    modules: set[str] = set()
    pending = list(deep)
    for statement in index.effects:
        pending.extend(_names_in(statement))
    while pending:
        name = pending.pop()
        if name in wanted:
            continue
        wanted.add(name)
        if name in index.definitions:
            pending.extend(_names_in(index.definitions[name]))
        elif name in index.imports:
            modules.add(index.imports[name])
    wanted.update(shallow)

    parts: list[tuple[str, str]] = []
    for name in sorted(wanted):
        node = index.definitions.get(name)
        # A name that is not defined here still perturbs the digest, so
        # dropping a definition re-keys what used to reach it.
        parts.append((name, _segment(lines, node) if node is not None else ""))
    parts.extend(("<statement>", _segment(lines, node)) for node in index.effects)
    return OptionsClosure(_digest(parts), frozenset(modules))


@dataclass(frozen=True)
class _Index:
    definitions: dict[str, ast.stmt]
    effects: list[ast.stmt]
    imports: dict[str, str]


def _index(source: str, *, module_package: str, package: str) -> _Index:
    definitions: dict[str, ast.stmt] = {}
    effects: list[ast.stmt] = []
    imports: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign | ast.AnnAssign) and _assigned(node):
            for name in _assigned(node):
                definitions[name] = node
        elif isinstance(node, ast.ImportFrom):
            _bind_from_import(node, module_package, package, imports)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and _within(alias.name, package):
                    imports[alias.asname] = alias.name
        elif not _is_docstring(node):
            effects.append(node)
    return _Index(definitions, effects, imports)


def _bind_from_import(
    node: ast.ImportFrom, module_package: str, package: str, imports: dict[str, str]
) -> None:
    resolved = _resolve(node, module_package)
    if resolved is None or not _within(resolved, package):
        return
    for alias in node.names:
        if alias.name == "*":
            continue
        # `from x import y` binds the submodule x.y when there is one.
        candidate = f"{resolved}.{alias.name}"
        imports[alias.asname or alias.name] = (
            candidate if _is_module(candidate) else resolved
        )


def _is_module(name: str) -> bool:
    # find_spec imports the parent and raises when it is a module, not a
    # package: there is no existence check that avoids the import.
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _resolve(node: ast.ImportFrom, module_package: str) -> str | None:
    if not node.level:
        return node.module
    try:
        return importlib.util.resolve_name(
            "." * node.level + (node.module or ""), module_package
        )
    except ImportError:
        return None


def _within(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def _assigned(node: ast.Assign | ast.AnnAssign) -> list[str]:
    """Names an assignment binds. None for `registry["k"] = v`, which binds no
    name: it is a statement that runs, so it counts as one."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [name for target in targets for name in _target_names(target)]


def _target_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple | ast.List):
        return [name for item in target.elts for name in _target_names(item)]
    return []


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _names_in(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _segment(lines: list[str], node: ast.stmt) -> str:
    decorators = getattr(node, "decorator_list", [])
    first = min([node.lineno, *(decorator.lineno for decorator in decorators)])
    return "\n".join(lines[first - 1 : node.end_lineno])


def _digest(parts: list[tuple[str, str]]) -> str:
    hasher = hashlib.blake2b(digest_size=HASH_DIGEST_SIZE)
    for name, text in parts:
        hasher.update(f"{name}\0{text}\0".encode())
    return hasher.hexdigest()
