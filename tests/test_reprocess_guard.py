"""`on_code_change: fail` refuses to spend before the first model runs (issue #530)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from stel.cli import cli
from stel.config.model import EmbedConfig, LLMTransformConfig
from stel.plan import ModelPlan, PlanStatus, plan_project
from stel.reprocess_guard import format_refusals, guard_reprocess
from stel.runner import RunError, build_project, run_project
from stel.versioning import compute_code_version

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def rag_project(tmp_path: Path) -> Path:
    pytest.importorskip("lancedb")
    dst = tmp_path / "rag_chunks_pipeline"
    shutil.copytree(
        EXAMPLES / "rag_chunks_pipeline",
        dst,
        ignore=shutil.ignore_patterns("target", "__pycache__"),
    )
    return dst


def _replace_in_model(project_dir: Path, filename: str, old: str, new: str) -> None:
    path = project_dir / "models" / filename
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new), encoding="utf-8")


def _set_policy(project_dir: Path, filename: str, block: str, policy: str) -> None:
    """Insert `on_code_change`/`reprocess_limit` lines under the kind block."""
    _replace_in_model(project_dir, filename, f"    {block}:\n", f"    {block}:\n{policy}")


def _plan(
    *,
    name: str = "m",
    kind: str = "embed",
    status: PlanStatus,
    state_rows: int = 10,
    rows: int = 10,
    upper: bool = False,
    policy: str | None = "fail",
    limit: int | None = 0,
) -> ModelPlan:
    return ModelPlan(
        name=name,
        kind=kind,
        materialization="incremental",
        status=status,
        code_version="cv",
        state_rows=state_rows,
        stale_rows=rows if status == "changed" else 0,
        rows_to_reprocess=rows,
        reprocess_is_upper_bound=upper,
        caused_by=("up",) if status == "cascade" else (),
        estimated_provider_calls=rows,
        provider="deterministic",
        provider_model="deterministic-v1",
        reason="because",
        reprocess_policy=policy,
        reprocess_limit=limit,
    )


# ─── the rule ───────────────────────────────────────────────────────────────


def test_changed_over_limit_is_refused() -> None:
    assert [r.plan.name for r in guard_reprocess([_plan(status="changed")])] == ["m"]


def test_cascade_upper_bound_is_refused() -> None:
    refusals = guard_reprocess([_plan(status="cascade", upper=True)])
    assert len(refusals) == 1
    assert "up to 10" in refusals[0].describe()


def test_within_limit_new_unchanged_and_full_pass() -> None:
    plans = [
        _plan(status="changed", rows=5, limit=5),
        _plan(status="new", rows=0),
        _plan(status="unchanged", rows=0),
        _plan(status="full", rows=0),
    ]
    assert guard_reprocess(plans) == []


def test_reprocess_policy_and_unguarded_kinds_pass() -> None:
    plans = [
        _plan(status="changed", policy="reprocess"),
        _plan(status="changed", kind="chunk", policy=None, limit=None),
    ]
    assert guard_reprocess(plans) == []


def test_refusal_message_names_every_way_forward() -> None:
    message = format_refusals(guard_reprocess([_plan(status="changed")]))
    assert "Refusing to start" in message
    assert "m (embed, deterministic/deterministic-v1)" in message
    for way in ("--accept-reprocess", "--full-refresh", "on_code_change: reprocess"):
        assert way in message


# ─── policy is not identity ─────────────────────────────────────────────────


def test_policy_fields_do_not_change_code_version(tmp_path: Path) -> None:
    strict = EmbedConfig(provider="deterministic", model="m", dimensions=4)
    relaxed = EmbedConfig(
        provider="deterministic",
        model="m",
        dimensions=4,
        on_code_change="reprocess",
        reprocess_limit=500,
    )
    assert compute_code_version(
        extraction=None, transform=None, embed=strict, project_dir=tmp_path
    ) == compute_code_version(
        extraction=None, transform=None, embed=relaxed, project_dir=tmp_path
    )
    strict_llm = LLMTransformConfig(input_field="text", prompt="p")
    relaxed_llm = LLMTransformConfig(
        input_field="text", prompt="p", on_code_change="reprocess", reprocess_limit=9
    )
    assert compute_code_version(
        extraction=None, transform=None, llm=strict_llm, project_dir=tmp_path
    ) == compute_code_version(
        extraction=None, transform=None, llm=relaxed_llm, project_dir=tmp_path
    )


# ─── the runner, end to end on the rag example ──────────────────────────────


def test_chunk_change_is_refused_and_every_way_forward_works(rag_project: Path) -> None:
    run_project(rag_project)
    _replace_in_model(
        rag_project, "document_chunks.yml", "chunk_size: 800", "chunk_size: 400"
    )

    with pytest.raises(RunError) as excinfo:
        run_project(rag_project)
    message = str(excinfo.value)
    # The chunk model is what changed; the paid models downstream are what
    # the guard refuses, and it names all three.
    assert "3 model(s)" in message
    for name in ("chunk_embeddings", "chunk_facts", "chunk_entities"):
        assert name in message
    assert "upstream document_chunks changed" in message

    # Nothing ran: the chunk model still holds its old code_version.
    plans = {m.name: m for m in plan_project(rag_project).models}
    assert plans["document_chunks"].status == "changed"

    # --accept-reprocess lets the same run through, and afterwards the plan
    # is clean again.
    results = run_project(rag_project, accept_reprocess=True)
    assert all(not r.errors for r in results)
    plans = {m.name: m for m in plan_project(rag_project).models}
    assert plans["document_chunks"].status == "unchanged"
    assert plans["chunk_embeddings"].status == "unchanged"


def test_full_refresh_bypasses_the_guard(rag_project: Path) -> None:
    run_project(rag_project)
    _replace_in_model(
        rag_project, "document_chunks.yml", "chunk_size: 800", "chunk_size: 400"
    )
    results = run_project(rag_project, full_refresh=True)
    assert all(not r.errors for r in results)


def test_embed_identity_change_refuses_only_that_model(rag_project: Path) -> None:
    run_project(rag_project)
    _replace_in_model(
        rag_project,
        "chunk_embeddings.yml",
        "model: deterministic-v1",
        "model: deterministic-v2",
    )
    with pytest.raises(RunError) as excinfo:
        run_project(rag_project)
    message = str(excinfo.value)
    assert "1 model(s)" in message
    assert "chunk_embeddings" in message
    assert "chunk_facts" not in message


def test_model_policy_and_limit_let_the_run_through(rag_project: Path) -> None:
    run_project(rag_project)
    _replace_in_model(
        rag_project, "document_chunks.yml", "chunk_size: 800", "chunk_size: 400"
    )
    _set_policy(
        rag_project, "chunk_embeddings.yml", "embed", "      on_code_change: reprocess\n"
    )
    # The example corpus is tiny; a limit above it makes the llm models pass.
    for filename in ("chunk_facts.yml", "chunk_entities.yml"):
        _set_policy(rag_project, filename, "llm", "      reprocess_limit: 100000\n")
    results = run_project(rag_project)
    assert all(not r.errors for r in results)


def test_selection_without_a_guarded_model_never_plans(rag_project: Path) -> None:
    """A selection of unguarded kinds must not pay for the plan's queries, and
    must not be refused for a change it does not reprocess."""
    run_project(rag_project)
    _replace_in_model(
        rag_project, "document_chunks.yml", "chunk_size: 800", "chunk_size: 400"
    )
    results = run_project(rag_project, select="document_chunks")
    assert [r.model_name for r in results] == ["document_chunks"]


def test_build_is_guarded_too(rag_project: Path) -> None:
    build_project(rag_project)
    _replace_in_model(
        rag_project, "document_chunks.yml", "chunk_size: 800", "chunk_size: 400"
    )
    with pytest.raises(RunError, match="Refusing to start"):
        build_project(rag_project)
    result = build_project(rag_project, accept_reprocess=True)
    assert all(not r.errors for r in result.run_results)


# ─── the CLI ────────────────────────────────────────────────────────────────


def test_cli_refuses_then_accepts(rag_project: Path) -> None:
    runner = CliRunner()
    project = ["--project-dir", str(rag_project)]
    assert runner.invoke(cli, [*project, "run"]).exit_code == 0
    _replace_in_model(
        rag_project, "document_chunks.yml", "chunk_size: 800", "chunk_size: 400"
    )

    plan = runner.invoke(cli, [*project, "plan"])
    assert plan.exit_code == 0, plan.output
    assert "`stel run` would refuse to start: 3 model(s)" in plan.output

    refused = runner.invoke(cli, [*project, "run"])
    assert refused.exit_code == 1
    assert "Refusing to start" in refused.output
    assert "--accept-reprocess" in refused.output

    accepted = runner.invoke(cli, [*project, "run", "--accept-reprocess"])
    assert accepted.exit_code == 0, accepted.output

    clean = runner.invoke(cli, [*project, "plan"])
    assert "would refuse" not in clean.output
