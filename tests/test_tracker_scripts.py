"""The pure parts of the tracker scripts (issue #705, ALE-311).

`scripts/linear_twin.py` decides which GitHub issue is a Linear issue's twin,
and `scripts/check_issue_claim.py` decides which branches and PRs name an
issue. Both talk to `gh` and `git` for everything else, so only these
decisions run here: a wrong one either makes a second twin or misses another
session's work, and neither failure raises anything.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


twin = _load("linear_twin")
claim = _load("check_issue_claim")


def test_only_the_footer_makes_a_twin() -> None:
    """A `Theme:` line names the parent and a bare mention is a mention; and
    ALE-31 must not match ALE-311's footer, the substring trap GitHub search
    fell into in astrolabe."""
    issues = [
        {"number": 1, "body": "Theme: ALE-31 — a theme\nwork"},
        {"number": 2, "body": "See ALE-31 for context."},
        {"number": 3, "body": "work\n\n" + twin.footer("ALE-311")},
        {"number": 4, "body": "work\n\n" + twin.footer("ALE-31")},
        {"number": 5, "body": None},
    ]

    assert twin.twins(issues, "ALE-31") == [4]
    assert twin.twins(issues, "ALE-311") == [3]


def test_the_footer_is_added_once() -> None:
    body = twin.with_footer("Problem.\n\n", "ALE-311")

    assert body == (
        "Problem.\n\nSynced from Linear [ALE-311](https://linear.app/alex-noonan/issue/ALE-311)\n"
    )
    assert twin.with_footer(body, "ALE-311") == body


@pytest.mark.parametrize(
    ("branch", "named"),
    [
        ("feat/705-linear-first-triage", True),
        ("origin/fix/705-slug", True),
        ("feat/issue-705-slug", True),
        ("feat/705", True),
        ("feat/7050-slug", False),
        ("feat/70-slug", False),
        ("feat/1705-slug", False),
        ("release/v0.705.0", False),
        ("claude/constellations-705abc", False),
    ],
)
def test_a_branch_names_an_issue_only_by_its_leading_number(branch: str, named: bool) -> None:
    """stel's branches are `feat/<N>-<slug>`, not astrolabe's `issue-<N>`; the
    astrolabe pattern matched none of them, so the port would have reported
    every branch as clear."""
    assert claim.branch_names_issue(branch, 705) is named


def test_references_match_the_whole_number() -> None:
    assert claim.references("Closes #705", 705)
    assert claim.references("see issues/705.", 705)
    assert not claim.references("Closes #7050", 705)
    assert not claim.references("Closes #70", 705)
    assert not claim.references(None, 705)


def test_a_claim_names_every_branch_it_mentions() -> None:
    """One live branch keeps a hand-written two-branch claim live, so
    `--release` has to see both."""
    body = "Claimed on branch `feat/705-a`; continued for branch `feat/705-b`."

    assert claim.claimed_branches(body) == ["feat/705-a", "feat/705-b"]
    assert claim.claimed_branches("claimed, no branch named") == []
