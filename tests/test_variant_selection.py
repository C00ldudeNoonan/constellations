"""Variant selection by quality, cost and latency (ALE-286, #795). Offline: the
evidence is supplied data, so nothing here calls a provider."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from stel.variant_selection import (
    Candidate,
    DocumentScore,
    ExperimentSpend,
    Measurement,
    VariantSelectionError,
    build_variant_selection_artifact,
    compare_development,
    config_diff,
    recommend,
    validate_holdout,
    write_variant_selection_artifact,
)

SPEND = ExperimentSpend(budget_usd=5.0, spent_usd=2.5, basis="measured")


def _m(cost: float | None = 1.0, secs: float = 10.0, complete: bool = True) -> Measurement:
    basis = "unknown" if cost is None else "measured"
    return Measurement(cost, basis, secs, complete, usage={"input_tokens": 100})


def _docs(
    prefix: str, n: int, found: int, emitted: int, gold: int = 10
) -> tuple[DocumentScore, ...]:
    return tuple(
        DocumentScore(f"{prefix}{i}", f"src-{prefix}{i}", gold, found, emitted) for i in range(n)
    )


def _cand(
    name: str,
    *,
    found: int = 9,
    emitted: int = 9,
    cost: float | None = 1.0,
    secs: float = 10.0,
    n: int = 40,
    prefix: str = "d",
    corpus: str = "dev-v1",
    config: str | None = None,
    complete: bool = True,
) -> Candidate:
    return Candidate(
        name=name,
        provider="prov",
        model=f"model-{name}",
        implementation_identity="impl-1",
        strategy="whole",
        prompt_identity="p1",
        schema_identity="s1",
        config_identity=config or f"cfg-{name}",
        corpus_identity=corpus,
        judgment_identity="j1",
        pricing_assumption="usd-per-mtok-v1",
        documents=_docs(prefix, n, found, emitted),
        cold=_m(cost, secs, complete),
        warm=_m(0.0, 1.0),
        changed=_m(0.1, 2.0),
    )


def _compare(*cands: Candidate, p: float = 0.8, r: float = 0.8):
    return compare_development(
        cands, precision_floor=p, recall_floor=r, seed=0, resamples=200, experiment=SPEND
    )


def test_selects_cheapest_candidate_whose_lower_bound_clears_the_floor() -> None:
    cheap_bad = _cand("cheap", found=5, emitted=9, cost=0.1)
    mid = _cand("mid", cost=1.0)
    dear = _cand("dear", cost=3.0)
    dev = _compare(cheap_bad, mid, dear)
    assert dev.selected == "mid"
    assert not dev.results[0].feasible
    assert "below its floor" in (dev.results[0].infeasible_reason or "")


def test_frontier_is_nondominated_and_dominated_candidate_is_dropped() -> None:
    dev = _compare(
        _cand("a", cost=1.0, secs=10), _cand("b", cost=2.0, secs=20), _cand("c", cost=3.0, secs=5)
    )
    assert "b" not in dev.frontier
    assert set(dev.frontier) == {"a", "c"}


def test_unknown_or_incomplete_cold_cost_is_never_selected_or_on_frontier() -> None:
    dev = _compare(
        _cand("known", cost=5.0), _cand("unknown", cost=None), _cand("partial", complete=False)
    )
    assert dev.selected == "known"
    assert dev.frontier == ("known",)


def test_overlapping_intervals_are_reported_as_tied_not_better() -> None:
    # Same counts give identical intervals; cost vs speed keeps both on the frontier.
    dev = _compare(_cand("a", cost=1.0, secs=20), _cand("b", cost=2.0, secs=5))
    assert dev.tied == (("a", "b"),)


def test_candidates_must_share_documents_and_judgments() -> None:
    with pytest.raises(VariantSelectionError, match="different documents"):
        _compare(_cand("a"), _cand("b", n=39))
    other = replace(_cand("b"), judgment_identity="j2")
    with pytest.raises(VariantSelectionError, match="judgment identity"):
        _compare(_cand("a"), other)


def test_experiment_spend_over_budget_is_refused() -> None:
    with pytest.raises(VariantSelectionError, match="exceeds its budget"):
        ExperimentSpend(budget_usd=1.0, spent_usd=1.5, basis="measured")


def test_unknown_cost_must_carry_unknown_basis() -> None:
    with pytest.raises(VariantSelectionError):
        Measurement(None, "measured", 1.0, True)
    with pytest.raises(VariantSelectionError):
        Measurement(1.0, "unknown", 1.0, True)


def test_holdout_is_scored_for_the_frozen_selection_only() -> None:
    dev = _compare(_cand("a", cost=1.0), _cand("b", cost=2.0))
    assert dev.selected == "a"
    with pytest.raises(VariantSelectionError, match="frozen"):
        validate_holdout(dev, _cand("b", prefix="h", corpus="hold-v1"))
    with pytest.raises(VariantSelectionError, match="config_identity differs"):
        validate_holdout(dev, _cand("a", prefix="h", corpus="hold-v1", config="cfg-other"))
    with pytest.raises(VariantSelectionError, match="overlap"):
        validate_holdout(dev, _cand("a", prefix="d", corpus="hold-v1"))
    with pytest.raises(VariantSelectionError, match="different corpus"):
        validate_holdout(dev, _cand("a", prefix="h", corpus="dev-v1"))


def _adopt_inputs(hold_found: int = 9):
    dev = _compare(_cand("a", cost=1.0, secs=20), _cand("b", cost=2.0, secs=5))
    hold = validate_holdout(
        dev, _cand("a", prefix="h", corpus="hold-v1", found=hold_found, emitted=hold_found)
    )
    return dev, hold


BASE = {"model": "old", "strategy": "whole"}
PICK = {"model": "new", "strategy": "whole"}


def test_adopt_emits_a_config_diff_and_names_the_ties() -> None:
    dev, hold = _adopt_inputs()
    rec = recommend(dev, hold, baseline_config=BASE, selected_config=PICK, min_documents=30)
    assert rec.decision == "adopt"
    assert '-  "model": "old"' in rec.config_diff and '+  "model": "new"' in rec.config_diff
    assert any("statistically tied with b" in r for r in rec.reasons)


def test_defer_when_holdout_fails_is_missing_or_sample_is_thin() -> None:
    dev, failed = _adopt_inputs(hold_found=5)
    assert not failed.passed

    def rec_for(holdout, minimum: int = 30):
        return recommend(
            dev, holdout, baseline_config=BASE, selected_config=PICK, min_documents=minimum
        )

    rec = rec_for(failed)
    assert rec.decision == "defer" and rec.config_diff == ""
    assert any("holdout failed" in r for r in rec.reasons)
    assert any("not been validated" in r for r in rec_for(None).reasons)
    assert any("fewer than 100" in r for r in rec_for(failed, 100).reasons)


def test_defer_when_nothing_clears_the_floor() -> None:
    dev = _compare(_cand("a", found=5, emitted=9), _cand("b", found=5, emitted=9))
    rec = recommend(dev, None, baseline_config=BASE, selected_config=PICK, min_documents=30)
    assert rec.decision == "defer" and rec.selected is None


def test_config_diff_refuses_credentials_and_credential_names() -> None:
    with pytest.raises(VariantSelectionError, match="credentials"):
        config_diff({"llm": {"api_key_env": "X"}}, {"llm": {}}, label="x")
    with pytest.raises(VariantSelectionError, match="credentials"):
        config_diff({}, {"nested": [{"token": "t"}]}, label="x")


def test_artifact_records_identities_money_basis_and_leaks_nothing(tmp_path: Path) -> None:
    dev, hold = _adopt_inputs()
    rec = recommend(dev, hold, baseline_config=BASE, selected_config=PICK, min_documents=30)
    artifact = build_variant_selection_artifact(dev, hold, rec)
    out = write_variant_selection_artifact(tmp_path / "target", artifact)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["version"] == 1
    assert loaded["experiment"] == {"budget_usd": 5.0, "spent_usd": 2.5, "basis": "measured"}
    first = loaded["candidates"][0]
    assert first["identity"]["pricing_assumption"] == "usd-per-mtok-v1"
    assert first["cold"]["cost_basis"] == "measured" and first["cold"]["usage"] == {
        "input_tokens": 100
    }
    assert {"cold", "warm", "changed"} <= first.keys()
    assert loaded["holdout"]["passed"] is True
    assert loaded["recommendation"]["decision"] == "adopt"
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["variant_selection.json"]


def test_max_tokens_is_not_a_credential_but_a_token_field_is() -> None:
    diff = config_diff({"llm": {"max_tokens": 10}}, {"llm": {"max_tokens": 20}}, label="x")
    assert '"max_tokens": 20' in diff
    with pytest.raises(VariantSelectionError, match="credentials"):
        config_diff({}, {"llm": {"access_token": "t"}}, label="x")


def test_holdout_must_match_every_execution_identity() -> None:
    dev = _compare(_cand("a", cost=1.0), _cand("b", cost=2.0))
    for field_name in ("implementation_identity", "model", "prompt_identity", "strategy"):
        other = replace(_cand("a", prefix="h", corpus="hold-v1"), **{field_name: "changed"})
        with pytest.raises(VariantSelectionError, match=field_name):
            validate_holdout(dev, other)


def test_a_holdout_from_another_comparison_cannot_support_adoption() -> None:
    _, hold = _adopt_inputs()
    looser = compare_development(
        (_cand("a", cost=1.0, secs=20), _cand("b", cost=2.0, secs=5)),
        precision_floor=0.5,
        recall_floor=0.5,
        seed=0,
        resamples=200,
        experiment=SPEND,
    )
    assert looser.selected == "a"
    with pytest.raises(VariantSelectionError, match="different development comparison"):
        recommend(looser, hold, baseline_config=BASE, selected_config=PICK, min_documents=30)
