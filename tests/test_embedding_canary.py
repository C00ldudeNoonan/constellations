"""Embedding drift canary against a blessed baseline (issue #305).

The gap this covers is precise: a hosted alias re-resolving to a new model
snapshot with our code, config, and input text byte-identical. Every hash
matches, every structural check passes, and retrieval degrades silently. The
canary re-embeds frozen probes and compares cosine against a committed
baseline — and per the issue, the non-negotiable test is that the threshold
actually trips, because a monitor that can only pass is worse than none.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import duckdb
import pytest

from stel.checks import run_project_tests
from stel.runner import run_project

PROBES = (
    "inflation rose sharply in the third quarter",
    "the labor market cooled",
    "tariffs and trade policy",
)


def _project(tmp_path: Path, *, min_similarity: float = 0.999) -> Path:
    """An embed model plus a committed canary baseline, offline throughout.

    The baseline is an ordinary extracted model — text plus a blessed vector
    per probe — exactly the shape `drift`/`golden` use for `to:`. Committed
    and git-reviewable, never an implicit last-run store.
    """
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "stel_project.yml").write_text(
        "name: canary\nversion: '0.1.0'\nprofile: canary\n", encoding="utf-8"
    )
    (project / "profiles.yml").write_text(
        "canary:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: docs\n",
        encoding="utf-8",
    )
    (project / "sources").mkdir()
    (project / "sources" / "s.yml").write_text(
        "version: 2\nsources:\n"
        "  - name: probes\n    path: baseline\n    file_pattern: '*.json'\n",
        encoding="utf-8",
    )
    (project / "models").mkdir()
    (project / "models" / "m.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: canary_baseline\n"
        "    source: ref('probes')\n"
        "    extraction:\n      backend: json\n"
        "      options:\n        fields: [text, embedding]\n"
        "    materialization: full\n"
        "  - name: chunk_embeddings\n"
        "    depends_on: [ref('canary_baseline')]\n"
        "    embed:\n      provider: deterministic\n      model: contract-v1\n"
        "      text_field: text\n      id_field: document_id\n"
        "      vector_field: embedding_out\n      dimensions: 4\n"
        "    materialization: incremental\n"
        "    tests:\n"
        "      - embedding_canary:\n"
        "          enabled: true\n"
        "          to: ref('canary_baseline')\n"
        f"          min_similarity: {min_similarity}\n",
        encoding="utf-8",
    )
    (project / "baseline").mkdir()
    return project


def _blessed_vectors(project: Path) -> list[list[float]]:
    """What the deterministic provider currently returns for the probes —
    obtained by embedding through the real provider, the way an operator
    blesses a baseline in the first place."""
    from stel.config.model import EmbedConfig
    from stel.embedding import EmbeddingIdentity, embed_texts

    del project
    identity = EmbeddingIdentity.from_config(
        EmbedConfig(provider="deterministic", model="contract-v1", dimensions=4)
    )
    return [list(v) for v in embed_texts(list(PROBES), identity).vectors]


def _write_baseline(project: Path, vectors: list[list[float]]) -> None:
    for index, (text, vector) in enumerate(zip(PROBES, vectors, strict=True)):
        (project / "baseline" / f"p{index}.json").write_text(
            json.dumps({"text": text, "embedding": vector}), encoding="utf-8"
        )


def _rotate(vector: list[float], cosine: float) -> list[float]:
    """A vector at exactly `cosine` similarity to `vector`.

    Rotates within the plane spanned by the vector and an orthogonal
    direction, so the fixture's distance is known by construction rather
    than eyeballed — the issue's requirement is a threshold proven to trip
    at a known distance.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    unit = [x / norm for x in vector]
    # Gram-Schmidt an axis into an orthogonal unit vector.
    axis = [1.0, 0.0, 0.0, 0.0]
    dot = sum(a * u for a, u in zip(axis, unit, strict=True))
    ortho = [a - dot * u for a, u in zip(axis, unit, strict=True)]
    ortho_norm = math.sqrt(sum(x * x for x in ortho))
    if ortho_norm < 1e-9:
        axis = [0.0, 1.0, 0.0, 0.0]
        dot = sum(a * u for a, u in zip(axis, unit, strict=True))
        ortho = [a - dot * u for a, u in zip(axis, unit, strict=True)]
        ortho_norm = math.sqrt(sum(x * x for x in ortho))
    ortho = [x / ortho_norm for x in ortho]
    sine = math.sqrt(1.0 - cosine * cosine)
    return [
        norm * (cosine * u + sine * o) for u, o in zip(unit, ortho, strict=True)
    ]


def _canary_results(project: Path) -> list[Any]:
    results = run_project_tests(project)
    return [r for r in results if r.test_name == "embedding_canary"]


# ─── the monitor can pass and, crucially, can fail ──────────────────────────


def test_a_faithful_provider_passes(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _write_baseline(project, _blessed_vectors(project))
    run_project(project)

    [result] = _canary_results(project)

    assert result.status == "pass", result.message


def test_a_drifted_provider_trips_the_threshold(tmp_path: Path) -> None:
    """The issue's non-negotiable: baseline vectors at a *known* cosine below
    the threshold must fail. Otherwise we ship a monitor that can only pass."""
    project = _project(tmp_path, min_similarity=0.99)
    blessed = _blessed_vectors(project)
    drifted = [_rotate(vector, 0.90) for vector in blessed]
    _write_baseline(project, drifted)
    run_project(project)

    [result] = _canary_results(project)

    assert result.status == "fail"
    assert "drifted below" in result.message
    assert "0.99" in result.message


def test_drift_within_the_noise_floor_passes(tmp_path: Path) -> None:
    """Cosine, not exact match: replica-level numeric noise below the
    threshold is benign by definition of the threshold."""
    project = _project(tmp_path, min_similarity=0.99)
    blessed = _blessed_vectors(project)
    noisy = [_rotate(vector, 0.999) for vector in blessed]
    _write_baseline(project, noisy)
    run_project(project)

    [result] = _canary_results(project)

    assert result.status == "pass", result.message


def test_a_dimension_change_is_drift_not_an_exception(tmp_path: Path) -> None:
    """A provider swap that changes dimensionality is the loudest possible
    drift; it must fail the check, not crash the test run."""
    project = _project(tmp_path)
    blessed = _blessed_vectors(project)
    _write_baseline(project, [[*vector, 0.0] for vector in blessed])
    run_project(project)

    [result] = _canary_results(project)

    assert result.status == "fail"
    assert "dimensions" in result.message


def test_an_empty_baseline_fails_rather_than_passing(tmp_path: Path) -> None:
    """A canary with no probes can only ever pass — worse than no canary."""
    project = _project(tmp_path)
    _write_baseline(project, _blessed_vectors(project))
    run_project(project)
    database = project / "target" / "db.duckdb"
    connection = duckdb.connect(str(database))
    try:
        connection.execute('DELETE FROM "db".docs.canary_baseline')
    finally:
        connection.close()

    [result] = _canary_results(project)

    assert result.status == "fail"
    assert "no usable probe" in result.message


# ─── spec validation ────────────────────────────────────────────────────────


def test_min_similarity_is_required() -> None:
    """No shipped default: the right threshold is the provider's measured
    noise floor, and a default would be a guess wearing authority."""
    from stel.test_specs import TestSpecError, parse_test_spec

    with pytest.raises(TestSpecError, match="min_similarity"):
        parse_test_spec({"embedding_canary": {"to": "ref('b')"}})


@pytest.mark.parametrize("value", [0.0, -0.5, 1.5])
def test_min_similarity_must_be_a_ratio(value: float) -> None:
    from stel.test_specs import TestSpecError, parse_test_spec

    with pytest.raises(TestSpecError, match="min_similarity"):
        parse_test_spec(
            {"embedding_canary": {"to": "ref('b')", "min_similarity": value}}
        )


def test_the_baseline_is_a_dag_dependency() -> None:
    """`to:` must build the baseline before the test runs, like drift/golden."""
    from stel.test_specs import parse_test_spec

    parsed = parse_test_spec(
        {"embedding_canary": {"to": "ref('canary_baseline')", "min_similarity": 0.9}}
    )

    assert parsed.ref_target == "canary_baseline"


def test_a_non_embed_model_is_refused(tmp_path: Path) -> None:
    """The canary re-embeds with the tested model's own provider identity;
    on a model with no embed block there is nothing to monitor."""
    project = _project(tmp_path)
    _write_baseline(project, _blessed_vectors(project))
    # The canary declared on a plain extraction model, which has no embed.
    # (It cannot sit on the baseline itself: `to:` is a DAG edge, and a model
    # pointing at itself is a cycle the compiler rightly rejects.)
    (project / "models" / "m.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: canary_baseline\n"
        "    source: ref('probes')\n"
        "    extraction:\n      backend: json\n"
        "      options:\n        fields: [text, embedding]\n"
        "    materialization: full\n"
        "  - name: plain_docs\n"
        "    source: ref('probes')\n"
        "    extraction:\n      backend: json\n"
        "      options:\n        fields: [text]\n"
        "    materialization: full\n"
        "    tests:\n"
        "      - embedding_canary:\n"
        "          enabled: true\n"
        "          to: ref('canary_baseline')\n"
        "          min_similarity: 0.9\n",
        encoding="utf-8",
    )
    run_project(project)

    results = run_project_tests(project)

    assert any(
        r.status == "fail"
        and "only applies to a model with an `embed:`" in (r.message or "")
        for r in results
    ), [(r.test_name, r.status, r.message) for r in results]


def test_an_oversized_baseline_is_refused(tmp_path: Path) -> None:
    """Probes are billed every run; a corpus-sized baseline is a misuse that
    should fail loudly rather than bill quietly."""
    project = _project(tmp_path)
    vectors = _blessed_vectors(project)
    for index in range(70):
        (project / "baseline" / f"bulk{index}.json").write_text(
            json.dumps({"text": f"bulk probe {index}", "embedding": vectors[0]}),
            encoding="utf-8",
        )
    _write_baseline(project, vectors)
    run_project(project)

    results = run_project_tests(project)

    # An oversized baseline is a spec misuse, surfaced through the runner's
    # unknown-test boundary rather than as a canary verdict — so match on the
    # message, not the test name.
    assert any(
        r.status == "fail" and "cap is 64" in (r.message or "") for r in results
    ), [(r.test_name, r.status, r.message) for r in results]


# ─── off by default, and the spend gate is real ─────────────────────────────


def test_disabled_by_default_skips_visibly_and_bills_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`stel build` runs model tests automatically, so an always-on canary
    would bill on every dev build. Off by default -- and the skip is a
    visible result, because a silently-passing disabled canary would be the
    "monitor that can only pass" the design forbids."""
    from stel.providers.deterministic import DeterministicEmbeddingProvider

    project = _project(tmp_path)
    _write_baseline(project, _blessed_vectors(project))
    model_path = project / "models" / "m.yml"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace(
            "          enabled: true\n", ""
        ),
        encoding="utf-8",
    )
    run_project(project)

    calls = {"n": 0}
    original = DeterministicEmbeddingProvider._embed

    def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DeterministicEmbeddingProvider, "_embed", counting)
    results = run_project_tests(project)
    [result] = [r for r in results if r.test_name == "embedding_canary"]

    assert result.status == "skipped"
    assert "enabled: true" in result.message
    assert calls["n"] == 0
