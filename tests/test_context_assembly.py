"""ALE-283: budget, citation, and authorization boundaries of context experiments."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from examples.context_assembly.compare import build_comparison
from stel.context_assembly import AssemblyError, AssemblyPolicy, Evidence, assemble_context
from stel.context_assembly_eval import (
    AnswerReview,
    AssemblyCase,
    ClaimReview,
    GeneratedAnswer,
    answer_identity,
    compare_context_policies,
    score_answer,
)
from stel.mcp_server.authorization import ClaimAuthorizationProvider, Principal


def _evidence(index=0, text="abcde", *, start=0, document=1, version=2,
              tenant=None, registry_tenant=None, context=None):
    row = {
        "document_id": f"{document:032x}", "document_version_id": f"{version:032x}",
        "context_id": f"{context if context is not None else index + 10:032x}",
        "chunk_id": f"{index + 100:032x}", "chunk_index": index, "text": text,
        "source_uri": "secret://sensitive-locator", "authorization_resolved": True,
        "is_public": tenant is None, "tenant_id": tenant,
        "citation_char_start": start,
        "citation_char_end": start + len(text) if start is not None else None,
    }
    registry = {**row, "is_public": registry_tenant is None, "tenant_id": registry_tenant}
    return Evidence.from_rows(registry, row)


def _assemble(candidates, *, neighbors=(), policy=None, counter=len, prefix="", suffix=""):
    return assemble_context(
        candidates, neighbors=neighbors, policy=policy or AssemblyPolicy(token_budget=10000),
        principal=Principal("test", tenant_id="allowed"),
        authorization=ClaimAuthorizationProvider(), policy_attributes=(),
        tokenizer_identity="test/codepoints/v1", count_tokens=counter, prefix=prefix, suffix=suffix,
    )


def _case(**kwargs):
    return replace(AssemblyCase(
        "case-1", "group-1", "holdout", "SENSITIVE QUESTION", (_evidence(),), (),
        ("fact-a", "fact-b", "fact-c"), False, "reviewed-facts/v1",
    ), **kwargs)


def _compare(cases=None, *, policies=None, generator=None, allow_live=False, reviews=None):
    return compare_context_policies(
        cases if cases is not None else [_case()],
        policies=policies or {"baseline": AssemblyPolicy(token_budget=20000)},
        benchmark_id="fixture/v1", corpus_id="corpus/v1", judgments_id="judgments/v1",
        principal=Principal("test", tenant_id="allowed"),
        authorization=ClaimAuthorizationProvider(),
        policy_attributes=(), tokenizer_identity="test/codepoints/v1", count_tokens=len,
        instructions="Use evidence and cite context IDs.", generator=generator,
        allow_live=allow_live, reviews=reviews or {},
    )


class FixedGenerator:
    identity = "offline-fixture/v1"
    live = False

    def __init__(self):
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return GeneratedAnswer(text="SENSITIVE ANSWER", input_tokens=None,
                               output_tokens=None, cost_usd=None)


def test_budget_counts_complete_serialized_prompt_and_metadata():
    """A body-only count would pass while the IDs and long locator exceed the cap."""
    item = _evidence(text="evidence with quotes \" and Unicode Ω")
    full = _assemble([item], prefix="QUESTION\n", suffix="\nANSWER")
    assert full.token_count == len(full.render())
    assert full.metadata_tokens > full.text_tokens
    exact = _assemble([item], policy=AssemblyPolicy(token_budget=full.token_count),
                      prefix="QUESTION\n", suffix="\nANSWER")
    assert exact.render() == full.render()
    too_small = _assemble([item], policy=AssemblyPolicy(token_budget=full.token_count - 1),
                          prefix="QUESTION\n", suffix="\nANSWER")
    assert not too_small.evidence
    assert too_small.omissions[0].reason == "budget"
    assert too_small.token_count <= full.token_count - 1
    with pytest.raises(AssemblyError, match="prompt envelope"):
        _assemble([], policy=AssemblyPolicy(token_budget=1))


def test_greedy_packing_skips_large_chunk_and_retains_later_evidence():
    small, large = _evidence(1), _evidence(0, "x" * 2000)
    budget = _assemble([small]).token_count
    context = _assemble([large, small], policy=AssemblyPolicy(token_budget=budget))
    assert [block.chunk_index for block in context.evidence] == [1]
    assert context.omissions[0].reason == "budget"


def test_document_order_and_policy_identity_are_explicit():
    roots = [_evidence(2, start=20), _evidence(0)]
    relevance = _assemble(roots)
    document = _assemble(roots, policy=AssemblyPolicy(token_budget=10000, order="document"))
    assert [block.chunk_index for block in relevance.evidence] == [2, 0]
    assert [block.chunk_index for block in document.evidence] == [0, 2]
    assert relevance.snapshot_id == document.snapshot_id
    assert relevance.policy_id != document.policy_id
    assert relevance.prompt_id != document.prompt_id


def test_overlap_requires_identical_versioned_offsets_and_preserves_locators():
    roots = [_evidence(), _evidence(1, "defg", start=3), _evidence(2, "abcd", start=0)]
    context = _assemble(roots, policy=AssemblyPolicy(token_budget=10000, remove_overlap=True))
    assert [block.text for block in context.evidence] == ["abcde", "fg"]
    assert context.evidence[1].citation["char_start"] == 5
    assert context.evidence[1].citation["char_end"] == 7
    assert context.evidence[1].context_id == roots[1].rows()[1]["context_id"]
    assert context.omissions[0].reason == "overlap"
    assert context.omissions[0].covered_by == (context.evidence[0].context_id,)
    baseline = _assemble(roots)
    assert len(context.render()) < len(baseline.render())


@pytest.mark.parametrize("other", [
    _evidence(1, "abcde", start=50),
    _evidence(1, "abcde", version=3),
    _evidence(1, "different", start=0),
    _evidence(1, "abcde", start=None),
])
def test_dedup_does_not_infer_overlap_from_similarity_or_adjacency(other):
    context = _assemble([_evidence(), other],
                        policy=AssemblyPolicy(token_budget=10000, remove_overlap=True))
    assert len(context.evidence) == 2
    assert not context.omissions


def test_bounded_neighbors_stay_within_document_version_and_keep_citation_ids():
    root = _evidence(1, "boundary", start=10)
    neighbors = [_evidence(0), _evidence(2, "continuation", start=20),
                 _evidence(3, "too far", start=40), _evidence(0, document=3, context=50),
                 _evidence(0, version=3, context=51)]
    policy = AssemblyPolicy(token_budget=10000, neighbor_radius=1, max_neighbors=2,
                            order="document")
    first = _assemble([root], neighbors=neighbors, policy=policy)
    second = _assemble([root], neighbors=list(reversed(neighbors)), policy=policy)
    assert first.render() == second.render()
    assert first.snapshot_id == second.snapshot_id
    assert [block.chunk_index for block in first.evidence] == [0, 1, 2]
    assert sum(block.expanded for block in first.evidence) == 2
    capped = _assemble([root], neighbors=neighbors,
                       policy=policy.model_copy(update={"max_neighbors": 1}))
    assert sum(block.expanded for block in capped.evidence) == 1


@pytest.mark.parametrize("denied", [
    _evidence(tenant="denied"), _evidence(registry_tenant="denied"),
])
@pytest.mark.parametrize("neighbor", [False, True])
def test_denied_root_or_neighbor_fails_before_tokenizer_or_generation(denied, neighbor):
    """ALE-283: budget omission and shared-document adjacency never bypass policy."""
    def forbidden_counter(_):
        pytest.fail("Unauthorized evidence reached the tokenizer")

    with pytest.raises(AssemblyError, match="unavailable or denied"):
        _assemble([_evidence(1)] if neighbor else [denied],
                  neighbors=[denied] if neighbor else [], counter=forbidden_counter,
                  policy=AssemblyPolicy(token_budget=1, neighbor_radius=1, max_neighbors=1)
                  if neighbor else AssemblyPolicy(token_budget=1))


def test_snapshots_are_copied_and_conflicting_ids_are_refused():
    item = _evidence()
    registry, chunk = item.rows()
    evidence = Evidence.from_rows(registry, chunk)
    chunk["text"] = "mutated"
    registry["tenant_id"] = "mutated"
    assert _assemble([evidence]).evidence[0].text == "abcde"
    conflict = _evidence(text="different")
    with pytest.raises(AssemblyError, match="conflicting rows"):
        _assemble([item], neighbors=[conflict])
    with pytest.raises(AssemblyError, match="unique"):
        _assemble([item, item])


def test_artifact_and_repr_exclude_sensitive_payloads():
    context = _assemble([_evidence(text="RAW-SENTINEL")])
    report = _compare([_case(candidates=(_evidence(text="RAW-SENTINEL"),))],
                      generator=FixedGenerator()).artifact()
    safe = json.dumps([context.artifact(), report]) + repr(context) + repr(context.evidence)
    for value in ("RAW-SENTINEL", "SENSITIVE ANSWER", "SENSITIVE QUESTION", "secret://"):
        assert value not in safe
    assert "RAW-SENTINEL" in context.render()


def test_comparison_holds_roots_constant_and_keeps_unreviewed_scores_null():
    report = _compare(policies={
        "relevance": AssemblyPolicy(token_budget=20000),
        "document": AssemblyPolicy(token_budget=20000, order="document"),
    }).artifact()
    assert len({row["snapshot_id"] for row in report["rows"]}) == 1
    assert len({tuple(row["retrieved_context_ids"]) for row in report["rows"]}) == 1
    assert all(row["answer_scores"] is None for row in report["rows"])
    scores = report["aggregate"]["relevance"]["holdout"]["answer_scores"]
    assert scores["fact_precision"] == {"mean": None, "scored_cases": 0}
    assert report["decision"].startswith("defer:")


def test_all_arms_are_authorized_before_any_generator_call():
    generator = FixedGenerator()
    case = _case(neighbors=(_evidence(1, tenant="denied"),))
    with pytest.raises(AssemblyError, match="denied"):
        _compare([case], policies={"plain": AssemblyPolicy(token_budget=20000),
                                  "expanded": AssemblyPolicy(token_budget=20000,
                                                             neighbor_radius=1, max_neighbors=1)},
                 generator=generator)
    assert generator.calls == []


def test_live_generation_requires_explicit_opt_in():
    generator = FixedGenerator()
    generator.live = True
    with pytest.raises(AssemblyError, match="opt-in"):
        _compare(generator=generator)
    assert not generator.calls
    _compare(generator=generator, allow_live=True)
    assert len(generator.calls) == 1


@pytest.mark.parametrize("other", [
    _case(case_id="case-2", split="development"),
    _case(case_id="case-2", group_id="different-group", split="development"),
])
def test_benchmark_groups_and_documents_cannot_cross_holdout(other):
    with pytest.raises(AssemblyError, match="crosses development and holdout"):
        _compare([_case(), other])


def _review(prompt_id="prompt", answer_id="answer", **kwargs):
    values = dict(reviewer="human-review/v1", fact_inventory_id="reviewed-facts/v1",
                  prompt_id=prompt_id, answer_id=answer_id, abstained=False,
                  scope_correct=True, answer_relevant=True, duplicate_claims=0, claims=())
    values.update(kwargs)
    return AnswerReview.model_validate(values)


def test_fact_evidence_scoring_matches_reviewed_rubric_and_counts_each_fact_once():
    context_id = _evidence().rows()[1]["context_id"]
    claims = (
        ClaimReview(claim_id="a", required_fact_id="fact-a", correct=True, supported=True,
                    citation_context_ids=(context_id,)),
        ClaimReview(claim_id="b", required_fact_id="fact-b", correct=False, supported=False,
                    citation_context_ids=()),
        ClaimReview(claim_id="extra", required_fact_id=None, correct=False, supported=False,
                    citation_context_ids=()),
    )
    scores = score_answer(_case(), _review(claims=claims), prompt_id="prompt", answer_id="answer",
                          context_ids={context_id})
    assert scores["fact_precision"] == scores["fact_recall"] == scores["evidence_support"] == 1 / 3
    assert scores["required_facts"] == scores["asserted_facts"] == 3
    assert scores["false_abstention"] == 0


def test_empty_answer_and_absence_have_distinct_scoring_denominators():
    review = _review(abstained=True)
    scores = score_answer(_case(), review, prompt_id="prompt", answer_id="answer",
                          context_ids=set())
    assert scores["fact_precision"] is scores["evidence_support"] is None
    assert scores["fact_recall"] == 0
    assert scores["false_abstention"] == 1
    absent = _case(unanswerable=True, required_fact_ids=())
    scores = score_answer(absent, review, prompt_id="prompt", answer_id="answer", context_ids=set())
    assert scores["fact_recall"] is None
    assert scores["abstention_success"] == 1
    wrong_scope = review.model_copy(update={"scope_correct": False})
    assert score_answer(absent, wrong_scope, prompt_id="prompt", answer_id="answer",
                        context_ids=set())["abstention_success"] == 0


def test_reviews_bind_exact_prompt_answer_inventory_and_cited_evidence():
    generator = FixedGenerator()
    row = _compare(generator=generator).artifact()["rows"][0]
    review = _review(prompt_id=row["prompt_id"], answer_id=answer_identity("SENSITIVE ANSWER"))
    key = (_case().case_id, row["policy_id"])
    report = _compare(generator=generator, reviews={key: review}).artifact()
    assert report["aggregate"]["baseline"]["holdout"]["reviewed_cases"] == 1
    for field in ("prompt_id", "answer_id", "fact_inventory_id"):
        stale = review.model_copy(update={field: "stale"})
        with pytest.raises(AssemblyError, match="does not match"):
            _compare(generator=generator, reviews={key: stale})
    claim = ClaimReview(claim_id="a", required_fact_id="fact-a", correct=True, supported=True,
                        citation_context_ids=("not-in-prompt",))
    with pytest.raises(AssemblyError, match="absent from"):
        _compare(generator=generator, reviews={key: review.model_copy(update={"claims": (claim,)})})


@pytest.mark.parametrize("values", [
    {"neighbor_radius": 1}, {"max_neighbors": 1}, {"token_budget": True},
    {"neighbor_radius": 3}, {"max_neighbors": 65}, {"order": "arbitrary"},
])
def test_policy_rejects_invalid_or_unbounded_configuration(values):
    with pytest.raises(ValidationError):
        AssemblyPolicy.model_validate({"token_budget": 1000, **values})


def test_documented_offline_example_is_reproducible():
    """Every example claim runs: no model scores, fixed evidence, bounded input."""
    first = build_comparison(counter=len, tokenizer_identity="fixture/characters/v1",
                             token_budget=1300).artifact()
    second = build_comparison(counter=len, tokenizer_identity="fixture/characters/v1",
                              token_budget=1300).artifact()
    assert first == second
    assert len(first["rows"]) == 20
    assert all(row["token_count"] <= 1300 for row in first["rows"])
    boundary = next(row for row in first["rows"]
                    if row["case_id"] == "boundary" and row["policy"] == "neighbors")
    assert len(boundary["context_ids"]) == 2
    assert len(boundary["expanded_ids"]) == 1
    insufficient = [row for row in first["rows"] if row["case_id"] == "insufficient-budget"]
    assert all(not row["context_ids"] for row in insufficient)
    assert all(row["answer_scores"] is None for row in first["rows"])


def test_native_counter_and_generator_failures_are_sanitized():
    def fail(_):
        raise ValueError("SENSITIVE NATIVE EXCEPTION")

    with pytest.raises(AssemblyError, match="Token counting failed") as counted:
        _assemble([_evidence()], counter=fail)
    assert "SENSITIVE" not in str(counted.value)
    assert counted.value.__suppress_context__

    class FailingGenerator(FixedGenerator):
        def generate(self, prompt):
            return fail(prompt)

    with pytest.raises(AssemblyError, match="Answer generation failed") as generated:
        _compare(generator=FailingGenerator())
    assert "SENSITIVE" not in str(generated.value)
    assert generated.value.__suppress_context__


def test_known_review_mismatch_fails_before_paid_generation():
    row = _compare().artifact()["rows"][0]
    generator = FixedGenerator()
    stale = _review(prompt_id="stale", answer_id="future-answer")
    with pytest.raises(AssemblyError, match="does not match"):
        _compare(generator=generator, reviews={(_case().case_id, row["policy_id"]): stale})
    assert generator.calls == []


def test_non_json_citation_is_rejected_at_the_input_boundary():
    registry, chunk = _evidence().rows()
    chunk["citation_locator"] = {"unsupported": float("nan")}
    with pytest.raises(AssemblyError, match="invalid citation"):
        Evidence.from_rows(registry, chunk)
