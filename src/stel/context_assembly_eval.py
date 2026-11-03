"""Compare packing policies on frozen evidence; reviewed answer scores stay separate.

No retrieval rerun or implicit answer judge. A trusted caller supplies an
explicit answer generator and human claim judgments bound to each exact
prompt/answer pair. Missing judgments remain null, never manufactured zeroes.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .context_assembly import AssemblyError, AssemblyPolicy, Evidence, assemble_context
from .hashing import canonical_fingerprint
from .mcp_server.authorization import AuthorizationProvider, PolicyAttribute, Principal


@dataclass(frozen=True, repr=False)
class AssemblyCase:
    case_id: str
    group_id: str
    split: Literal["development", "holdout"]
    question: str
    candidates: tuple[Evidence, ...]
    neighbors: tuple[Evidence, ...]
    required_fact_ids: tuple[str, ...]
    unanswerable: bool
    fact_inventory_id: str


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    text: str = Field(repr=False)
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    cost_usd: float | None = Field(ge=0)


class AnswerGenerator(Protocol):
    """Trusted experiment driver; live drivers must enforce their own spend cap."""

    identity: str
    live: bool

    def generate(self, prompt: str) -> GeneratedAnswer: ...


class ClaimReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    claim_id: str = Field(min_length=1)
    required_fact_id: str | None
    correct: bool
    supported: bool
    # Human-reviewed entailment, not merely a matching document ID.
    citation_context_ids: tuple[str, ...]


class AnswerReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    reviewer: str = Field(min_length=1)
    fact_inventory_id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    answer_id: str = Field(min_length=1)
    abstained: bool
    scope_correct: bool
    answer_relevant: bool
    duplicate_claims: int = Field(ge=0)
    claims: tuple[ClaimReview, ...]

    @model_validator(mode="after")
    def _distinct_claims(self) -> AnswerReview:
        if len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError("Review requires distinct normalized claim IDs")
        if self.abstained and self.claims:
            raise ValueError("An abstention cannot also assert factual claims")
        if any(claim.supported and not claim.citation_context_ids for claim in self.claims):
            raise ValueError("Supported claims require passage citations")
        return self


def answer_identity(text: str) -> str:
    return canonical_fingerprint(text, domain="stel-context-assembly-answer")


def score_answer(
    case: AssemblyCase, review: AnswerReview, *, prompt_id: str, answer_id: str,
    context_ids: set[str],
) -> dict[str, Any]:
    """Apply ALE-281's companion fact/evidence rubric to an exact human review."""
    if (review.prompt_id, review.answer_id, review.fact_inventory_id) != (
        prompt_id, answer_id, case.fact_inventory_id
    ):
        raise AssemblyError("Answer review does not match this prompt, answer, and fact inventory")
    required = set(case.required_fact_ids)
    for claim in review.claims:
        if claim.required_fact_id is not None and claim.required_fact_id not in required:
            raise AssemblyError("Claim review names an unknown required fact")
        if not set(claim.citation_context_ids) <= context_ids:
            raise AssemblyError("Claim review cites evidence absent from the assembled prompt")
    count = len(review.claims)
    correct = sum(claim.correct for claim in review.claims)
    supported = sum(claim.supported for claim in review.claims)
    recovered = {
        claim.required_fact_id for claim in review.claims
        if claim.correct and claim.required_fact_id is not None
    }
    return {
        "fact_precision": correct / count if count else None,
        "fact_recall": len(recovered) / len(required) if required else None,
        "evidence_support": supported / count if count else None,
        "answer_relevance": int(review.answer_relevant),
        "abstention_success": int(review.abstained and review.scope_correct)
        if case.unanswerable else None,
        "false_abstention": int(review.abstained) if not case.unanswerable else None,
        "asserted_facts": count, "correct_facts": correct,
        "supported_facts": supported, "required_facts": len(required),
        "recovered_facts": len(recovered), "duplicate_claims": review.duplicate_claims,
    }


@dataclass(frozen=True)
class ContextComparison:
    benchmark_id: str
    corpus_id: str
    judgments_id: str
    tokenizer_identity: str
    generator_identity: str | None
    policies: Mapping[str, Mapping[str, Any]]
    rows: tuple[Mapping[str, Any], ...]
    decision: str = field(default="defer: held-out reviewed model comparison required", init=False)

    def artifact(self) -> dict[str, Any]:
        """No question, answer, document content, policy row, or citation URI."""
        return {
            "schema_version": "context_assembly_compare/v1",
            "benchmark_id": self.benchmark_id, "corpus_id": self.corpus_id,
            "judgments_id": self.judgments_id, "tokenizer_identity": self.tokenizer_identity,
            "generator_identity": self.generator_identity, "policies": dict(self.policies),
            "decision": self.decision, "rows": list(self.rows),
            "aggregate": _aggregate(self.rows),
        }


def compare_context_policies(
    cases: Sequence[AssemblyCase], *, policies: Mapping[str, AssemblyPolicy],
    benchmark_id: str, corpus_id: str, judgments_id: str,
    principal: Principal, authorization: AuthorizationProvider,
    policy_attributes: Sequence[PolicyAttribute], tokenizer_identity: str,
    count_tokens: Callable[[str], int], instructions: str,
    generator: AnswerGenerator | None, allow_live: bool,
    reviews: Mapping[tuple[str, str], AnswerReview],
) -> ContextComparison:
    """Compare every arm on identical frozen roots and a bounded neighbor pool.

    All arms and cases are assembled/authorized before any generation. Review
    keys are (case_id, policy_id), binding the reviewer to the exact rendering.
    Instructions and question are included in every arm's measured budget.
    Generation never occurs unless a caller explicitly supplies a driver.
    """
    _validate_cases(cases)
    if not policies or not all((benchmark_id, corpus_id, judgments_id)):
        raise AssemblyError("Comparison requires policies and versioned benchmark provenance")
    if len({policy.token_budget for policy in policies.values()}) != 1:
        raise AssemblyError("Compared policies must use the same token budget")
    if generator is not None and (not generator.identity or (generator.live and not allow_live)):
        raise AssemblyError("Live answer generation requires explicit opt-in and a driver identity")
    assembled = [
        (case, name, assemble_context(
            case.candidates, neighbors=case.neighbors, policy=policy,
            principal=principal, authorization=authorization, policy_attributes=policy_attributes,
            tokenizer_identity=tokenizer_identity, count_tokens=count_tokens,
            prefix=instructions + "\nQuestion: " + case.question + "\nEvidence:\n", suffix="\n",
        ))
        for case in cases for name, policy in policies.items()
    ]
    keys = {(case.case_id, context.policy_id) for case, _, context in assembled}
    if not set(reviews) <= keys or (reviews and generator is None):
        raise AssemblyError("Answer reviews require matching generated comparison arms")
    for case, _, context in assembled:
        review = reviews.get((case.case_id, context.policy_id))
        if review is not None:
            # The actual answer binding can only be checked after generation;
            # reject every already-known mismatch before making any paid call.
            score_answer(
                case, review, prompt_id=context.prompt_id, answer_id=review.answer_id,
                context_ids={item.context_id for item in context.evidence},
            )
    rows: list[dict[str, Any]] = []
    for case, name, context in assembled:
        row = {"case_id": case.case_id, "group_id": case.group_id, "split": case.split,
               "policy": name, "fact_inventory_id": case.fact_inventory_id,
               "retrieved_context_ids": [item.rows()[1]["context_id"] for item in case.candidates],
               **context.artifact(), "answer_id": None, "answer_scores": None,
               "generation_seconds": None, "input_tokens": None,
               "output_tokens": None, "cost_usd": None, "reviewer": None}
        if generator is not None:
            answer, seconds = _generate(generator, context.render())
            answer_id = answer_identity(answer.text)
            row.update(answer_id=answer_id, generation_seconds=seconds,
                       input_tokens=answer.input_tokens, output_tokens=answer.output_tokens,
                       cost_usd=answer.cost_usd)
            review = reviews.get((case.case_id, context.policy_id))
            if review is not None:
                row["answer_scores"] = score_answer(
                    case, review, prompt_id=context.prompt_id, answer_id=answer_id,
                    context_ids={item.context_id for item in context.evidence},
                )
                row["reviewer"] = review.reviewer
        rows.append(row)
    return ContextComparison(
        benchmark_id, corpus_id, judgments_id, tokenizer_identity,
        generator.identity if generator is not None else None,
        {name: policy.model_dump() for name, policy in policies.items()}, tuple(rows),
    )


def _validate_cases(cases: Sequence[AssemblyCase]) -> None:
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise AssemblyError("Comparison requires non-empty cases with unique IDs")
    groups: dict[str, str] = {}
    document_splits: dict[str, str] = {}
    for case in cases:
        if not all((case.case_id, case.group_id, case.question, case.fact_inventory_id)):
            raise AssemblyError("Cases require question, group, and fact inventory identities")
        if case.split not in {"development", "holdout"}:
            raise AssemblyError("Unknown benchmark split")
        if case.group_id in groups and groups[case.group_id] != case.split:
            raise AssemblyError("A benchmark group crosses development and holdout")
        groups[case.group_id] = case.split
        for evidence in (*case.candidates, *case.neighbors):
            document_id = evidence.rows()[1]["document_id"]
            if document_id in document_splits and document_splits[document_id] != case.split:
                raise AssemblyError("A source document crosses development and holdout")
            document_splits[document_id] = case.split
        if case.unanswerable == bool(case.required_fact_ids):
            raise AssemblyError("Answerable cases need required facts; absences must have none")
        if len(set(case.required_fact_ids)) != len(case.required_fact_ids):
            raise AssemblyError("Required fact IDs must be unique")


def _generate(generator: AnswerGenerator, prompt: str) -> tuple[GeneratedAnswer, float]:
    started = time.monotonic()
    try:
        answer = GeneratedAnswer.model_validate(generator.generate(prompt))
    except Exception:
        raise AssemblyError("Answer generation failed") from None
    return answer, time.monotonic() - started


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Separate splits and scored denominators, keeping missing reviews visible."""
    grouped: dict[str, Any] = {}
    for row in rows:
        group = grouped.setdefault(row["policy"], {}).setdefault(row["split"], [])
        group.append(row)
    return {
        policy: {split: _aggregate_group(items) for split, items in splits.items()}
        for policy, splits in grouped.items()
    }


def _aggregate_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reviewed = [row["answer_scores"] for row in rows if row["answer_scores"] is not None]
    metrics = ("fact_precision", "fact_recall", "evidence_support", "answer_relevance",
               "abstention_success", "false_abstention")
    scores = {}
    for metric in metrics:
        values = [review[metric] for review in reviewed if review[metric] is not None]
        scores[metric] = {"mean": sum(values) / len(values) if values else None,
                          "scored_cases": len(values)}
    measurements = {}
    for metric in ("generation_seconds", "input_tokens", "output_tokens", "cost_usd"):
        values = [row[metric] for row in rows if row[metric] is not None]
        measurements[metric] = {
            "mean": math.fsum(values) / len(values) if values else None,
            "measured_cases": len(values),
        }
    return {"cases": len(rows), "reviewed_cases": len(reviewed), "answer_scores": scores,
            "measurements": measurements,
            "mean_context_tokens": sum(row["token_count"] for row in rows) / len(rows),
            "mean_metadata_tokens": sum(row["metadata_tokens"] for row in rows) / len(rows)}
