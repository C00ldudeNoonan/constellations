"""A run whose projected provider cost is over the target's threshold asks first (issue #586).

The reprocess guard (ADR-0008) counts rows; this one counts dollars, priced
from run-log history the way `stel plan` prices them (ADR-0033). Over
`confirm_cost_above_usd`, a run at a terminal asks and any other run refuses,
both before the first provider call; `--accept-cost` is yes in advance.

The project embeds at $1 per character, so each 11-character body costs $11
and three of them $33 -- comfortably over a $20 threshold.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from stel.adapters import AdapterError
from stel.cli import cli
from stel.cost_guard import CostProjection, project_cost
from stel.execution.contracts import RunError
from stel.plan import plan_project
from stel.profile import ProfileError
from stel.providers.deterministic import DeterministicEmbeddingProvider
from stel.runner import build_project, run_project

# Runs a whole project, so it belongs to the `e2e` tier (issue #518).
pytestmark = pytest.mark.e2e

_PRICED = "        pricing:\n          input_usd_per_mchar: 1000000\n"


def _project(
    tmp_path: Path,
    *,
    threshold: str = "20",
    pricing: str = _PRICED,
    run_log: bool = True,
) -> Path:
    """Three documents, chunked, embedded once at $33 into the run log."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "stel_project.yml").write_text(
        "name: costs\nversion: '0.1.0'\nprofile: costs\n", encoding="utf-8"
    )
    profile = (
        "costs:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: ./target/db.duckdb\n"
        "        schema: docs\n"
        "      embedding:\n        provider: deterministic\n" + pricing
    )
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
        # The reprocess guard is answered for every test here, so what stops
        # a run is this guard alone.
        "      on_code_change: reprocess\n"
        "    materialization: incremental\n",
        encoding="utf-8",
    )
    data = project / "data"
    data.mkdir()
    for index in range(3):
        (data / f"doc{index}.json").write_text(
            json.dumps({"title": f"Release {index}", "body": f"body text {index}"}),
            encoding="utf-8",
        )
    run_project(project)
    if threshold:
        # Set after the first run: a first build has no history to price.
        profiles = project / "profiles.yml"
        profiles.write_text(
            profiles.read_text(encoding="utf-8")
            + f"      confirm_cost_above_usd: {threshold}\n",
            encoding="utf-8",
        )
    return project


def _change_embedding_model(project: Path) -> None:
    models = project / "models" / "documents.yml"
    models.write_text(
        models.read_text(encoding="utf-8").replace("contract-v1", "contract-v2"),
        encoding="utf-8",
    )


class _Provider:
    """Counts the texts that reach the provider."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.texts: list[str] = []
        original = DeterministicEmbeddingProvider._embed

        def counting(provider: Any, request: Any, **kwargs: Any) -> Any:
            self.texts.extend(request.texts)
            return original(provider, request, **kwargs)

        monkeypatch.setattr(DeterministicEmbeddingProvider, "_embed", counting)


def test_a_run_over_the_threshold_refuses_before_any_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    _change_embedding_model(project)
    provider = _Provider(monkeypatch)

    with pytest.raises(RunError) as refused:
        run_project(project)

    assert provider.texts == []
    message = str(refused.value)
    assert message.startswith("Refusing to start: the projected provider cost")
    assert "about $33.00, over confirm_cost_above_usd ($20.00)" in message
    assert "document_embeddings (embed): ~$33.00 for 3 row(s)" in message
    assert "`--accept-cost`" in message


def test_the_prompt_sees_the_projection_and_its_answer_decides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    _change_embedding_model(project)
    provider = _Provider(monkeypatch)
    asked: list[CostProjection] = []

    def no(projection: CostProjection) -> bool:
        asked.append(projection)
        return False

    with pytest.raises(RunError, match="Not started: the projected provider cost was declined"):
        run_project(project, confirm_cost=no)
    assert provider.texts == []
    (projection,) = asked
    assert projection.total_usd == pytest.approx(33.0)
    assert [plan.name for plan in projection.priced] == ["document_embeddings"]

    run_project(project, confirm_cost=lambda _: True)
    assert len(provider.texts) == 3


def test_within_the_threshold_a_run_never_asks(tmp_path: Path) -> None:
    project = _project(tmp_path, threshold="50")
    _change_embedding_model(project)

    def must_not_ask(projection: CostProjection) -> bool:
        raise AssertionError("a run within its threshold asked")

    results = run_project(project, confirm_cost=must_not_ask)
    assert all(not result.errors for result in results)


def test_accept_cost_proceeds_without_asking(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _change_embedding_model(project)

    results = build_project(project, accept_cost=True)
    assert all(not result.errors for result in results.run_results)


@pytest.mark.parametrize("full_refresh", [True, False], ids=["full-refresh", "accept-reprocess"])
def test_flags_that_release_the_reprocess_guard_do_not_release_this_one(
    tmp_path: Path, full_refresh: bool
) -> None:
    """Both say yes to a reprocess; neither says yes to its price. A full
    refresh with nothing changed is priced at every published row, because
    that is what it sends."""
    project = _project(tmp_path)
    if not full_refresh:
        _change_embedding_model(project)

    with pytest.raises(RunError, match=r"about \$33\.00, over"):
        run_project(
            project, full_refresh=full_refresh, accept_reprocess=not full_refresh
        )


def test_reprocess_all_prices_an_embed_model_at_what_it_pays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An embed model under `--reprocess-all` reads its vectors back by input
    hash (issue #655), so the recovery run is free and must not be asked
    about. A changed embedding model under the same flag is not free."""
    project = _project(tmp_path)
    provider = _Provider(monkeypatch)

    plan = plan_project(project, select="document_embeddings", reprocess_all=True)
    (embed,) = plan.models
    assert embed.rows_to_reprocess == 3
    assert embed.priced_rows == 0
    assert embed.estimated_cost_usd == 0
    # The price is for the rows that reach the provider, and the line says
    # so rather than reading as $0 a row for all three (review on #695).
    projection = project_cost(plan.models, threshold_usd=20, source="stel_run_log")
    assert "~$0 for 3 row(s), 0 of them reaching the provider." in "\n".join(
        projection.describe()
    )

    results = run_project(project, reprocess_all=True)
    assert all(not result.errors for result in results)
    assert provider.texts == []

    _change_embedding_model(project)
    with pytest.raises(RunError, match=r"about \$33\.00, over"):
        run_project(project, reprocess_all=True)


def test_reprocess_all_prices_every_row_when_an_upstream_model_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chunk model whose code moved may have changed every text below it, so
    the embed model's rows are priced in full even though its own code did
    not move. Pricing its stale rows alone would project $0 and start the
    expensive run unasked (review on #695)."""
    project = _project(tmp_path)
    models = project / "models" / "documents.yml"
    models.write_text(
        models.read_text(encoding="utf-8").replace(
            "chunk_size: 1000", "chunk_size: 900"
        ),
        encoding="utf-8",
    )
    provider = _Provider(monkeypatch)

    plans = {m.name: m for m in plan_project(project, reprocess_all=True).models}
    assert plans["document_embeddings"].stale_rows == 0
    assert plans["document_embeddings"].priced_rows == 3

    with pytest.raises(RunError, match=r"about \$33\.00, over"):
        run_project(project, reprocess_all=True)
    assert provider.texts == []


def test_an_unpriced_model_is_named_and_does_not_stop_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With no priced history the projection cannot say anything, and a guard
    that refused every unpriced reprocess would refuse forever on a target
    that never set `pricing:`. The warning is what keeps that from being
    silent."""
    project = _project(tmp_path, pricing="")
    _change_embedding_model(project)

    with caplog.at_level(logging.WARNING, logger="stel.runner"):
        results = run_project(project)

    assert all(not result.errors for result in results)
    assert (
        "cost guard: no priced history in stel_run_log for document_embeddings"
        in caplog.text
    )


def test_an_unreadable_run_log_refuses_rather_than_projecting_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unreadable history is not absent history. Pricing nothing would project
    $0 and start a run configured to stop above its threshold (Codex review
    on #695); `stel plan` keeps reporting, since it spends nothing."""
    project = _project(tmp_path)
    _change_embedding_model(project)
    provider = _Provider(monkeypatch)

    def unreadable(*args: Any, **kwargs: Any) -> dict[str, float]:
        raise AdapterError("permission denied on dataset docs")

    monkeypatch.setattr("stel.plan._cost_per_provider_row", unreadable)

    with pytest.raises(RunError) as refused:
        run_project(project)
    assert provider.texts == []
    message = str(refused.value)
    assert "run log stel_run_log could not be read" in message
    assert "[AdapterError]" in message
    assert "permission denied" not in message

    (embed,) = plan_project(project, select="document_embeddings").models
    assert embed.estimated_cost_usd is None

    results = run_project(project, accept_cost=True)
    assert all(not result.errors for result in results)


def test_a_threshold_without_a_run_log_is_rejected_before_the_warehouse(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path, run_log=False, threshold="")
    profiles = project / "profiles.yml"
    profiles.write_text(
        profiles.read_text(encoding="utf-8") + "      confirm_cost_above_usd: 20\n",
        encoding="utf-8",
    )
    (project / "target" / "db.duckdb").unlink()

    with pytest.raises(ProfileError, match="enable run_log on this target"):
        run_project(project)
    assert not (project / "target" / "db.duckdb").exists()


def test_cli_refuses_without_a_terminal_and_asks_with_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    _change_embedding_model(project)
    runner = CliRunner()
    args = ["--project-dir", str(project), "run"]

    refused = runner.invoke(cli, args)
    assert refused.exit_code != 0
    assert "Refusing to start: the projected provider cost" in refused.output

    monkeypatch.setattr("stel.cli._stdin_is_interactive", lambda: True)
    declined = runner.invoke(cli, args, input="n\n")
    assert declined.exit_code != 0
    assert "Proceed? [y/N]" in declined.output
    assert "was declined" in declined.output

    confirmed = runner.invoke(cli, args, input="y\n")
    assert confirmed.exit_code == 0, confirmed.output
    assert "document_embeddings" in confirmed.output


def test_cli_json_never_prompts_and_accept_cost_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    _change_embedding_model(project)
    monkeypatch.setattr("stel.cli._stdin_is_interactive", lambda: True)
    runner = CliRunner()

    piped = runner.invoke(cli, ["--project-dir", str(project), "run", "--json"])
    assert piped.exit_code != 0
    assert "Proceed?" not in piped.output

    accepted = runner.invoke(
        cli, ["--project-dir", str(project), "build", "--accept-cost"]
    )
    assert accepted.exit_code == 0, accepted.output
    assert "Proceed?" not in accepted.output


def test_plan_says_when_a_run_would_ask(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _change_embedding_model(project)

    plan = plan_project(project)
    assert plan.confirm_cost_above_usd == 20
    assert plan.to_dict()["metadata"]["confirm_cost_above_usd"] == 20

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "plan"])
    assert result.exit_code == 0, result.output
    assert "`stel run` would ask before starting" in result.output
    assert "about $33.00, over confirm_cost_above_usd ($20.00)" in result.output
