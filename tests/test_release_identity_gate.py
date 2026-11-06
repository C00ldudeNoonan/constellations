"""A release may not move a model's code_version unannounced (issue #750).

Bumping stel 0.23.0 to 0.24.0 re-keyed 20,281 published rows although no
backend read the module that changed. Nothing in CI could have said so: the
identity is a digest of source, and a digest only fails a test that compares it
with something. This compares it with the previous release.

`fixtures/release_identity.json` holds, per built-in backend and transform, the
identity fields of the model's code_version as the previous release computed
them. The candidate's must match, or each difference must be listed under
`intended_changes` with a reason -- the changelog entry that tells an operator
to run `stel plan` (and perhaps `stel adopt`) after upgrading. A release then
re-baselines with `STEL_UPDATE_RELEASE_IDENTITY=1 pytest
tests/test_release_identity_gate.py` (docs/release.md).

`extraction.backend_parser` is not compared: it is a third-party library's
version, which moves with the lockfile and not with stel.
"""
from __future__ import annotations

import importlib
import json
import os
import pkgutil
from collections.abc import Iterator
from pathlib import Path

import pytest

import stel.backends.base as backend_base
import stel.text.transforms as builtin_transforms
from stel.backends import get_backend, list_backends
from stel.config.model import ExtractionConfig, ModelConfig, TransformConfig
from stel.config.project import ProjectConfig
from stel.versioning import compute_model_code_identity

BASELINE = Path(__file__).parent / "fixtures" / "release_identity.json"
UPDATE_ENV = "STEL_UPDATE_RELEASE_IDENTITY"
# Moves with the lockfile (beautifulsoup4, pypdf), not with a stel release.
_NOT_COMPARED = frozenset({"extraction.backend_parser"})

_LLM_BACKEND_FIELDS = [{"name": "title", "type": "string"}]
# Transforms whose options have required fields; the rest take none.
_TRANSFORM_OPTIONS: dict[str, dict[str, object]] = {
    "extract_relations": {"extractor": "co_occurrence", "mentions": "mentions"},
    "link_entities": {"mentions": "mentions", "aliases": "aliases"},
}


def _builtin_backends() -> list[str]:
    return [
        name
        for name in list_backends()
        if type(get_backend(name)).__module__.startswith("stel.backends.")
    ]


def _builtin_transforms() -> list[str]:
    return sorted(
        info.name
        for info in pkgutil.iter_modules(builtin_transforms.__path__)
        if not info.name.startswith("_")
    )


def _cases() -> dict[str, ModelConfig]:
    cases: dict[str, ModelConfig] = {}
    for name in _builtin_backends():
        options = {"fields": _LLM_BACKEND_FIELDS} if name == "llm" else {}
        cases[f"backend:{name}"] = ModelConfig(
            name="m",
            source="ref('docs')",
            extraction=ExtractionConfig(backend=name, options=options),
            materialization="incremental",
        )
    for name in _builtin_transforms():
        cases[f"transform:{name}"] = ModelConfig(
            name="m",
            source="ref('docs')",
            transform=TransformConfig(
                type="python", module=f"stel.text.transforms.{name}",
                options=_TRANSFORM_OPTIONS.get(name, {}),
            ),
            materialization="incremental",
        )
    return cases


def _identities(project_dir: Path) -> dict[str, dict[str, str]]:
    project = ProjectConfig(name="p")
    found: dict[str, dict[str, str]] = {}
    for case, model in _cases().items():
        identity = compute_model_code_identity(model, project, project_dir)
        found[case] = {
            field: digest
            for field, digest in identity.components.items()
            if field not in _NOT_COMPARED
        }
    return found


def unannounced_moves(
    baseline: dict[str, dict[str, str]],
    current: dict[str, dict[str, str]],
    intended: list[dict[str, str]],
) -> list[str]:
    """Every (case, field) that differs from the baseline and is not listed,
    and every listed change that did not happen."""
    listed = {(entry["case"], entry["field"]) for entry in intended}
    moved: set[tuple[str, str]] = set()
    for case in baseline.keys() | current.keys():
        before, after = baseline.get(case, {}), current.get(case, {})
        for name in before.keys() | after.keys():
            if before.get(name) != after.get(name):
                moved.add((case, name))
    problems = [
        f"{case}: {name} moved with no intended_changes entry"
        for case, name in sorted(moved - listed)
    ]
    problems += [
        f"{case}: {name} is listed in intended_changes but did not move"
        for case, name in sorted(listed - moved)
    ]
    return problems


@pytest.fixture(autouse=True)
def _fresh_dependency_digests() -> Iterator[None]:
    backend_base._dependency_source_digests.cache_clear()
    yield
    backend_base._dependency_source_digests.cache_clear()


def test_a_release_moves_no_identity_it_has_not_announced(tmp_path: Path) -> None:
    current = _identities(tmp_path)
    if os.environ.get(UPDATE_ENV):
        BASELINE.write_text(
            json.dumps({"intended_changes": [], "cases": current}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        pytest.skip("release identity baseline rewritten")
    recorded = json.loads(BASELINE.read_text(encoding="utf-8"))

    problems = unannounced_moves(recorded["cases"], current, recorded["intended_changes"])

    assert not problems, (
        "This change moves the code_version of models that use built-in "
        "backends or transforms, which re-keys every operator's published rows "
        "on upgrade:\n  "
        + "\n  ".join(problems)
        + "\nIf the output really changes, add an entry to intended_changes in "
        + f"{BASELINE.name} with the reason. If it does not, narrow what the "
        + "identity hashes (ADR-0050)."
    )


def test_every_builtin_backend_and_transform_has_a_baseline_row() -> None:
    """A backend added without a row is a backend the gate cannot see."""
    recorded = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert set(recorded["cases"]) == set(_cases()), (
        f"re-baseline with {UPDATE_ENV}=1 so new backends and transforms get a row"
    )


def test_the_gate_fails_when_a_backends_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check has to be able to fail (issue #750): seed a real change to the
    JSON backend's source, through the same hashing a release goes through, and
    require the comparison to name it."""
    baseline = _identities(tmp_path)
    json_module = importlib.import_module("stel.backends.json_backend")
    original = backend_base._source_text

    def seeded(obj: object) -> str | None:
        text = original(obj)
        if text is not None and obj is json_module:
            return text + "\n# a behavior change\n"
        return text

    monkeypatch.setattr(backend_base, "_source_text", seeded)
    backend_base._dependency_source_digests.cache_clear()

    problems = unannounced_moves(baseline, _identities(tmp_path), [])

    assert problems == [
        "backend:json: extraction.backend_implementation moved with no intended_changes entry"
    ]


def test_an_announced_move_passes_and_a_stale_announcement_does_not() -> None:
    before = {"backend:json": {"extraction.backend_implementation": "a"}}
    after = {"backend:json": {"extraction.backend_implementation": "b"}}
    entry = {
        "case": "backend:json",
        "field": "extraction.backend_implementation",
        "reason": "x",
    }

    assert unannounced_moves(before, after, [entry]) == []
    assert unannounced_moves(before, before, [entry]) == [
        "backend:json: extraction.backend_implementation is listed in "
        "intended_changes but did not move"
    ]
