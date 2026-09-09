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
