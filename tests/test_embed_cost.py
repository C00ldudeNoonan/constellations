"""An embed never pays twice for a text it already has a vector for (issue #665).

astrolabe's SEC corpus was embedded about 1.9 times. #671 fixed the id-space
half (reuse keyed by text, not row id); these tests are the other half, which
target-table reuse cannot see because the vector is not in the target:

- a crash before a flush loses vectors embedded since the last one;
- duplicate text in one run is embedded once per occurrence;
- a rebuilt or emptied target has nothing left to reuse.

The response cache (`embedding.cache_path`) and in-window deduplication close
those. `embedding.pricing` turns the same run into a dollar figure, the run
budget's `max_cost_usd` into a cap an embed honours, and `stel plan` into a
number a launch comment can carry (ask 3).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import pytest
from click.testing import CliRunner

from stel.cli import cli
from stel.execution.contracts import RunError
from stel.execution.embed_cache import EmbeddingResponseCache
from stel.plan import plan_project
from stel.providers.deterministic import DeterministicEmbeddingProvider
from stel.runner import run_project

# Runs a whole project, so it belongs to the `e2e` tier (issue #518).
pytestmark = pytest.mark.e2e


def _project(
    tmp_path: Path,
    *,
    bodies: list[str],
    flush_every: int,
    embedding_profile: str = "",
    llm_profile: str = "",
    run_log: bool = False,
) -> Path:
    project = tmp_path / "project"
    project.mkdir(parents=True)
    (project / "stel_project.yml").write_text(
        "name: embeddings\nversion: '0.1.0'\nprofile: embeddings\n",
        encoding="utf-8",
    )
    profile = (
        "embeddings:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: docs\n"
    )
    if embedding_profile:
        profile += "      embedding:\n        provider: deterministic\n" + embedding_profile
    profile += llm_profile
    if run_log:
        profile += "      run-log:\n        enabled: true\n"
    (project / "profiles.yml").write_text(profile, encoding="utf-8")
    (project / "sources").mkdir()
    (project / "sources" / "documents.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data\n"
        "    file_pattern: '*.json'\n",
        encoding="utf-8",
    )
    (project / "models").mkdir()
    (project / "models" / "documents.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: document_registry\n"
        "    source: ref('documents')\n"
        "    extraction:\n      backend: json\n"
        "      options:\n        fields: [title, body]\n"
        "    materialization: incremental\n"
        "  - name: document_chunks\n"
        "    depends_on: [ref('document_registry')]\n"
        "    chunk:\n      text_field: body\n      chunk_size: 1000\n"
        "      chunk_overlap: 0\n"
        "    materialization: incremental\n"
        "  - name: document_embeddings\n"
        "    depends_on: [ref('document_chunks')]\n"
        "    embed:\n      provider: deterministic\n"
        "      model: contract-v1\n      text_field: text\n"
        "      id_field: chunk_id\n      vector_field: embedding\n"
        "      dimensions: 4\n      batch_size: 1\n"
        f"      flush_every: {flush_every}\n"
        "      on_code_change: reprocess\n"
        "    materialization: incremental\n",
        encoding="utf-8",
    )
    data = project / "data"
    data.mkdir()
    for index, body in enumerate(bodies):
        (data / f"doc{index}.json").write_text(
            json.dumps({"title": f"Release {index}", "body": body}),
            encoding="utf-8",
        )
    run_project(project, select="document_registry")
    run_project(project, select="document_chunks")
    return project


_CACHE = "        cache_path: ./target/embedding_cache.duckdb\n"


def _distinct(count: int) -> list[str]:
    return [f"body text {index}" for index in range(count)]


class _Provider:
    """Counts the texts that reach the provider, and can fail after `limit`."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, limit: int | None = None):
        self.texts: list[str] = []
        original = DeterministicEmbeddingProvider._embed

        def counting(provider: Any, request: Any, **kwargs: Any) -> Any:
            if limit is not None and len(self.texts) >= limit:
                raise RuntimeError("provider exhausted")
            self.texts.extend(request.texts)
            return original(provider, request, **kwargs)

        monkeypatch.setattr(DeterministicEmbeddingProvider, "_embed", counting)


def _vectors(project: Path) -> dict[str, list[float]]:
    connection = duckdb.connect(str(project / "target" / "db.duckdb"), read_only=True)
    try:
        exists = connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'docs' AND table_name = 'document_embeddings'"
        ).fetchone()
        if exists is None or exists[0] == 0:
            return {}
        rows = connection.execute(
            'SELECT chunk_id, embedding FROM "db".docs.document_embeddings'
        ).fetchall()
    finally:
        connection.close()
    return {chunk_id: list(vector) for chunk_id, vector in rows}


def _embed_result(results: list[Any]) -> Any:
    return next(r for r in results if r.model_name == "document_embeddings")


# ─── the three ways a corpus paid twice ──────────────────────────────────────


def test_a_retry_after_a_crash_does_not_resend_unflushed_texts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidate 2 in #665: everything embedded after the last flush and
    before a crash was paid for and lost. With one flush for the whole corpus
    nothing publishes before the failure, so only the cache can remember."""
    project = _project(
        tmp_path, bodies=_distinct(7), flush_every=100, embedding_profile=_CACHE
    )
    _Provider(monkeypatch, limit=5)
    with pytest.raises(RunError):
        run_project(project, select="document_embeddings")
    assert _vectors(project) == {}

    monkeypatch.undo()
    retry = _Provider(monkeypatch)
    result = _embed_result(run_project(project, select="document_embeddings"))

    assert len(retry.texts) == 2
    assert result.metrics["response_cache_hits"] == 5
    assert len(_vectors(project)) == 7


def test_duplicate_text_in_one_window_is_embedded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boilerplate repeated across filings: one provider text per distinct
    text, every row still gets its vector. Needs no cache."""
    project = _project(
        tmp_path, bodies=["shared boilerplate"] * 4 + ["unique"], flush_every=100
    )
    provider = _Provider(monkeypatch)
    result = _embed_result(run_project(project, select="document_embeddings"))

    assert sorted(provider.texts) == ["shared boilerplate", "unique"]
    assert result.metrics["duplicate_text_hits"] == 3
    vectors = _vectors(project)
    assert len(vectors) == 5
    assert len({tuple(vector) for vector in vectors.values()}) == 2


def test_duplicate_text_across_windows_is_embedded_once_with_a_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The target index is read before the first window, so it cannot know a
    vector the same run produced a window ago; the cache can."""
    project = _project(
        tmp_path,
        bodies=["shared boilerplate"] * 4,
        flush_every=1,
        embedding_profile=_CACHE,
    )
    provider = _Provider(monkeypatch)
    result = _embed_result(run_project(project, select="document_embeddings"))

    assert provider.texts == ["shared boilerplate"]
    assert result.metrics["response_cache_hits"] == 3
    assert len(_vectors(project)) == 4


def test_a_full_refresh_rebuild_reuses_cached_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--full-refresh turns target reuse off, which used to mean paying for
    the corpus again. The rebuild must publish identical vectors for free."""
    project = _project(
        tmp_path, bodies=_distinct(3), flush_every=100, embedding_profile=_CACHE
    )
    run_project(project, select="document_embeddings")
    before = _vectors(project)

    provider = _Provider(monkeypatch)
    run_project(project, select="document_embeddings", full_refresh=True)

    assert provider.texts == []
    assert _vectors(project) == before


def test_a_changed_embedding_configuration_does_not_reuse_cached_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache is keyed on the configuration as well as the text: a vector
    from another model is a wrong answer, not a cheap one."""
    project = _project(
        tmp_path, bodies=_distinct(3), flush_every=100, embedding_profile=_CACHE
    )
    run_project(project, select="document_embeddings")
    models = project / "models" / "documents.yml"
    models.write_text(
        models.read_text(encoding="utf-8").replace("contract-v1", "contract-v2"),
        encoding="utf-8",
    )

    provider = _Provider(monkeypatch)
    run_project(project, select="document_embeddings", full_refresh=True)

    assert len(provider.texts) == 3


# ─── the cache's own boundaries ──────────────────────────────────────────────


def test_a_cache_path_that_cannot_hold_a_cache_fails_before_spending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(
        tmp_path, bodies=_distinct(2), flush_every=100, embedding_profile=_CACHE
    )
    (project / "target" / "embedding_cache.duckdb").mkdir()
    provider = _Provider(monkeypatch)

    with pytest.raises(RunError, match="Embed model 'document_embeddings'"):
        run_project(project, select="document_embeddings")
    assert provider.texts == []


def test_a_cache_write_failure_mid_run_keeps_the_paid_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Raising would discard exactly the spend the cache exists to protect:
    the window is about to publish vectors already paid for. The run carries
    on without the cache and says so once, without DuckDB's own text."""
    project = _project(
        tmp_path, bodies=_distinct(3), flush_every=100, embedding_profile=_CACHE
    )

    def failing(*args: Any, **kwargs: Any) -> None:
        raise duckdb.IOException("disk full at /secret/path")

    monkeypatch.setattr(EmbeddingResponseCache, "_write", failing)
    with caplog.at_level(logging.WARNING, logger="stel.execution.embed_cache"):
        run_project(project, select="document_embeddings")

    assert len(_vectors(project)) == 3
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "IOException" in warnings[0]
    assert "/secret/path" not in warnings[0]


# ─── pricing ─────────────────────────────────────────────────────────────────


def test_pricing_estimates_embed_cost_from_characters_and_tokens(
    tmp_path: Path,
) -> None:
    """Vertex bills characters, token-metered providers tokens; the estimate
    adds whichever rates are set, counted over texts actually sent."""
    project = _project(
        tmp_path,
        bodies=["one two three", "four five"],
        flush_every=100,
        embedding_profile=(
            "        pricing:\n"
            "          input_usd_per_mchar: 1000000\n"
            "          input_usd_per_mtok: 2000000\n"
        ),
    )
    result = _embed_result(run_project(project, select="document_embeddings"))

    characters = len("one two three") + len("four five")
    tokens = 5  # the deterministic provider counts words
    assert result.metrics["input_characters"] == characters
    assert result.metrics["estimated_cost_usd"] == pytest.approx(characters + 2 * tokens)


def test_a_priced_embed_honours_the_run_cost_budget(tmp_path: Path) -> None:
    """max_cost_usd could not see an embed before embeddings had a price."""
    project = _project(
        tmp_path,
        bodies=_distinct(5),
        flush_every=1,
        embedding_profile="        pricing:\n          input_usd_per_mchar: 1000000\n",
        llm_profile=(
            "      llm:\n        provider: deterministic\n"
            "        budget:\n          max_cost_usd: 20\n"
        ),
    )
    result = _embed_result(run_project(project, select="document_embeddings"))

    # Each text is 11 characters ($11): the second window takes spend to $22,
    # and the third is refused before its provider call.
    assert result.status == "budget_exceeded"
    assert len(_vectors(project)) == 2


# ─── stel plan: rows become dollars ──────────────────────────────────────────

_PRICED = "        pricing:\n          input_usd_per_mchar: 1000000\n"


def _change_embedding_model(project: Path) -> None:
    models = project / "models" / "documents.yml"
    models.write_text(
        models.read_text(encoding="utf-8").replace("contract-v1", "contract-v2"),
        encoding="utf-8",
    )


def _add_document(project: Path, index: int, body: str) -> None:
    (project / "data" / f"doc{index}.json").write_text(
        json.dumps({"title": f"Release {index}", "body": body}), encoding="utf-8"
    )


def test_plan_prices_reprocessing_rows_at_the_cost_per_provider_row(
    tmp_path: Path,
) -> None:
    """The per-row rate divides by rows that reached the provider, not rows
    processed: a run whose rows were half reuse would otherwise price a full
    reprocess at half what it costs."""
    project = _project(
        tmp_path,
        bodies=_distinct(3),
        flush_every=100,
        embedding_profile=_PRICED,
        run_log=True,
    )
    # $11 per 11-character text: three paid rows, $33.
    run_project(project, select="document_embeddings")
    # Two more rows, one of them text the target already holds: $11 paid for
    # two rows processed.
    _add_document(project, 3, "body text 0")
    _add_document(project, 4, "body text 9")
    run_project(project)

    _change_embedding_model(project)
    plan = plan_project(project, select="document_embeddings")
    (embed,) = plan.models

    assert embed.status == "changed"
    assert embed.rows_to_reprocess == 5
    # ($33 + $11) / (3 + 1) provider rows = $11 a row; x5 rows. Dividing by
    # rows processed instead would give $8.80 a row and $44.
    assert embed.estimated_cost_usd == pytest.approx(55.0)
    assert plan.cost_source == "stel_run_log"

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "plan"])
    assert result.exit_code == 0, result.output
    assert "<=$55.00" in result.output
    assert "Provider cost: at most $55.00" in result.output


def test_plan_without_a_run_log_says_cost_is_not_estimated(tmp_path: Path) -> None:
    """A blank cost on a paid reprocess would read as free."""
    project = _project(
        tmp_path, bodies=_distinct(2), flush_every=100, embedding_profile=_PRICED
    )
    run_project(project, select="document_embeddings")
    _change_embedding_model(project)

    plan = plan_project(project, select="document_embeddings")
    assert plan.models[0].estimated_cost_usd is None
    assert plan.cost_source is None

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "plan"])
    assert result.exit_code == 0, result.output
    assert "Provider cost is not estimated" in result.output
