"""The tier split is a contract, not a convention (issue #518).

`-m "not e2e"` is only a useful local loop while the marker is actually on
every file that earns it. A file that runs a whole project without the marker
does not fail anything -- it just quietly makes the fast tier slower, until
nobody trusts it and everyone runs the whole suite again.

So this is a gate in the same shape as `test_reentry_contract.py` and
`test_frozen_names.py`: the rule is checked, not documented and hoped for.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent

# Calling any of these means the test builds a project, runs it, or opens a
# retrieval store -- the work that separates the two tiers.
E2E_CALLS = frozenset({"run_project", "create_store", "export_concept_cloud"})


def _calls(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def _declares_e2e(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            return "e2e" in ast.dump(node.value)
    return False


def test_every_project_running_file_declares_the_e2e_tier() -> None:
    """A file that runs a project belongs to `e2e`, and says so."""
    missing = []
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _calls(tree) & E2E_CALLS and not _declares_e2e(tree):
            missing.append(path.name)

    assert not missing, (
        "these files run a whole project or open a store but are not marked "
        "`pytestmark = pytest.mark.e2e`, so `-m \"not e2e\"` would run them "
        f"and stop being the fast tier: {missing}"
    )


def test_the_fast_tier_is_actually_most_of_the_files() -> None:
    """A guard against the split inverting.

    If nearly everything ends up marked `e2e`, the fast tier stops being worth
    running and the marker stops being worth maintaining. Measured when this
    landed: 59 of 158 files were e2e, carrying ~80% of the wall clock. This
    fails if the *count* ever goes the other way, which is the cheap signal
    that the suite has drifted back toward whole-project tests by default.
    """
    total = e2e = 0
    for path in sorted(TESTS.glob("test_*.py")):
        total += 1
        if _declares_e2e(ast.parse(path.read_text(encoding="utf-8"))):
            e2e += 1

    assert e2e < total / 2, (
        f"{e2e} of {total} test files are e2e; the fast tier is no longer the "
        "majority, so either the split needs revisiting or new tests are "
        "reaching for a whole project when a unit would do"
    )


# Cross-module imports that predate the rule below. Both are the same shape as
# the retrieval one this issue fixed -- a sibling test module's helpers used as
# an API -- and both are the next to move into support modules. Listed rather
# than hidden so the remaining work is visible and no *new* one can appear.
_KNOWN_CROSS_IMPORTS = {
    ("test_google_drive_context_example.py", "tests.test_gdrive_source"),
    ("test_mcp_rate_limits.py", "tests.test_mcp_server"),
}


def _imported_test_modules(node: ast.AST) -> list[str]:
    """Every sibling test module one import statement reaches, in any form.

    Three spellings reach the same place and only one of them was caught
    (Codex review, #518):

        from tests.test_x import helper    ImportFrom, module="tests.test_x"
        import tests.test_x                Import,     alias="tests.test_x"
        from tests import test_x           ImportFrom, module="tests", alias

    A contract that only rejects the first is worse than none, because it
    reads as enforcement while two ordinary spellings walk past it.
    """
    if isinstance(node, ast.Import):
        return [a.name for a in node.names if a.name.startswith("tests.test_")]
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module.startswith("tests.test_"):
            return [module]
        if module == "tests":
            return [
                f"tests.{a.name}" for a in node.names if a.name.startswith("test_")
            ]
    return []


def test_no_new_test_module_imports_another_test_module() -> None:
    """Shared fixtures live in a support module, not in a sibling test (#518).

    Four files used to import private helpers out of `test_retrieval.py` and
    `test_online_publication.py`, two of them from *inside test bodies*. That
    is fragile in a specific way: renaming a leading-underscore function is
    normally a free local edit, and there it silently broke other files. The
    inline placement also hid the dependency from anyone reading the imports,
    and breaks the module-level import rule in AGENTS.md.

    `support_retrieval.py` is where such a helper goes now. This fails on any
    cross-module import that is not one of the two already known.
    """
    offenders = []
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for target in _imported_test_modules(node):
                if (path.name, target) in _KNOWN_CROSS_IMPORTS:
                    continue
                offenders.append(f"{path.name} -> {target}:{node.lineno}")

    assert not offenders, (
        "test modules must not import from each other; move the shared helper "
        f"into a support module beside `support_retrieval.py`: {offenders}"
    )


def test_the_known_cross_imports_have_not_quietly_grown() -> None:
    """The allowlist is a debt register, not a licence.

    If one of these is fixed the entry should go; if a file stops importing
    that way this fails and says so, which is the prompt to delete the line
    rather than let the list outlive the problem.
    """
    stale = []
    for name, module in sorted(_KNOWN_CROSS_IMPORTS):
        tree = ast.parse((TESTS / name).read_text(encoding="utf-8"))
        if not any(
            isinstance(n, ast.ImportFrom) and n.module == module
            for n in ast.walk(tree)
        ):
            stale.append(f"{name} -> {module}")

    assert not stale, (
        f"these are no longer cross-importing, so drop them from "
        f"_KNOWN_CROSS_IMPORTS: {stale}"
    )


def test_no_new_bare_exception_pins() -> None:
    """`pytest.raises(Exception, match=...)` is a weak assertion (#518).

    It passes when *any* exception carries that text, so a test meant to prove
    "config rejects this" also passes when an unrelated crash happens to
    mention the same word. Twenty-three of these were tightened to the type
    actually raised, discovered by instrumenting the runs rather than by
    reading the code.

    Two remain and say why in a comment beside them: one asserts DuckDB's own
    `BinderException`, which is a vendored internal with no compatibility
    promise, and one only runs where symlinks can be created, so its real type
    was never observed and narrowing on a guess would be worse than an
    honestly broad pin.
    """
    # This file names the pattern in prose and in the check below, so it
    # would otherwise report itself.
    allowed = {"test_identifier_quoting.py", "test_promotion.py",
               Path(__file__).name}
    offenders = []
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name in allowed:
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "pytest.raises(Exception" in line:
                offenders.append(f"{path.name}:{number}")

    assert not offenders, (
        "pin the exception type actually raised rather than `Exception`; if it "
        "genuinely has to be broad, say why beside it and add the file to "
        f"`allowed` here: {offenders}"
    )
