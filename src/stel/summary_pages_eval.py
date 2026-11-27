"""Passages versus passages plus summary pages on frozen governed evidence (ALE-285).

Builds on ``context_assembly``: both arms run through ``assemble_context`` with
the same token budget, tokenizer and authorization, so a page pays for its
tokens out of the budget it competes in. Retrieval is not rerun.

The score here is *evidence presence*: whether a case's required literal
appears in the assembled prompt. It is a mechanics proxy, not answer quality;
like ADR 0049 the comparison always records ``defer`` until a reviewed
held-out set scores answers.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .context_assembly import AssemblyError, AssemblyPolicy, Evidence, assemble_context
from .context_assembly_eval import AssemblyCase
from .mcp_server.authorization import AuthorizationProvider, PolicyAttribute, Principal
from .summary_pages import (
    PageError,
    Summarizer,
    SummaryPage,
    build_page,
    page_identity,
    resolve_claims,
)

QuestionClass = Literal["broad", "multi_section", "exact_identifier"]
ARMS = ("passages", "passages_plus_pages")


@dataclass(frozen=True, repr=False)
class PageCase:
    case: AssemblyCase
    question_class: QuestionClass
    #: fact ID -> literal text that must appear in the prompt to carry the fact.
    facts: Mapping[str, str]


@dataclass(frozen=True)
class PageComparison:
    benchmark_id: str
    corpus_id: str
    tokenizer_identity: str
    summarizer_identity: str
    token_budget: int
    build: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    decision: str = field(
        default="defer: reviewed held-out broad, multi-section and exact-identifier "
        "answer scores are required", init=False,
    )

    def artifact(self) -> dict[str, Any]:
        """IDs, counts and timings only: no question, page, or document text."""
        return {
            "schema_version": "summary_page_compare/v1",
            "benchmark_id": self.benchmark_id, "corpus_id": self.corpus_id,
            "tokenizer_identity": self.tokenizer_identity,
            "summarizer_identity": self.summarizer_identity,
            "token_budget": self.token_budget, "build": dict(self.build),
            "decision": self.decision, "rows": list(self.rows),
            "aggregate": _aggregate(self.rows),
        }


def _validate(cases: Sequence[PageCase]) -> None:
    if not cases or len({item.case.case_id for item in cases}) != len(cases):
        raise AssemblyError("Comparison requires non-empty cases with unique IDs")
    for item in cases:
        if set(item.facts) != set(item.case.required_fact_ids) or not all(item.facts.values()):
            raise AssemblyError("Case facts must name every required fact with a literal")
        if item.question_class not in ("broad", "multi_section", "exact_identifier"):
            raise AssemblyError("Unknown question class")


def _by_document(case: AssemblyCase) -> list[list[Evidence]]:
    """Frozen candidates plus neighbors, grouped per document version, in rank order."""
    groups: dict[tuple[str, str], dict[str, Evidence]] = {}
    for item in (*case.candidates, *case.neighbors):
        row = item.rows()[1]
        groups.setdefault((row["document_id"], row["document_version_id"]), {})[
            row["context_id"]
        ] = item
    return [list(group.values()) for group in groups.values()]


def compare_page_arms(
    cases: Sequence[PageCase], *, summarizer: Summarizer, allow_live: bool,
    token_budget: int, benchmark_id: str, corpus_id: str, principal: Principal,
    authorization: AuthorizationProvider, policy_attributes: Sequence[PolicyAttribute],
    tokenizer_identity: str, count_tokens: Callable[[str], int], instructions: str,
) -> PageComparison:
    """Run both arms on every case; pages are cached by identity across cases.

    A page whose sources span scopes aborts the comparison before any arm is
    scored, as does a denied page or chunk: authorization is never a metric.
    """
    _validate(cases)
    if not all((benchmark_id, corpus_id)) or not summarizer.identity:
        raise AssemblyError("Comparison requires versioned benchmark and summarizer identities")
    if summarizer.live and not allow_live:
        raise AssemblyError("A live summarizer requires explicit opt-in")
    policy = AssemblyPolicy(token_budget=token_budget)
    cache: dict[str, SummaryPage] = {}
    calls = 0
    built_seconds = 0.0
    plans: list[tuple[PageCase, list[SummaryPage]]] = []
    for item in cases:
        pages = []
        for chunks in _by_document(item.case):
            try:
                key = page_identity(chunks, summarizer.identity)
                if key not in cache:
                    started = time.monotonic()
                    cache[key] = build_page(chunks, summarizer)
                    built_seconds += time.monotonic() - started
                    calls += 1
            except PageError as exc:
                raise AssemblyError(str(exc)) from None
            pages.append(cache[key])
        plans.append((item, pages))
    rows = [
        _row(item, pages, arm, policy, principal, authorization, policy_attributes,
             tokenizer_identity, count_tokens, instructions)
        for item, pages in plans for arm in ARMS
    ]
    source_chars = sum(page.source_chars for page in cache.values())
    page_chars = sum(len(page.text) for page in cache.values())
    kept = sum(len(page.claims) for page in cache.values())
    rejected = sum(page.rejected_claims for page in cache.values())
    build = {
        "pages": len(cache), "summarizer_calls": calls, "reused_builds": sum(
            len(pages) for _, pages in plans) - calls,
        "source_chars": source_chars, "page_chars": page_chars,
        "page_to_source_ratio": page_chars / source_chars if source_chars else None,
        "claims_kept": kept, "claims_rejected": rejected,
        "claim_faithfulness": kept / (kept + rejected) if kept + rejected else None,
        "build_seconds": built_seconds,
        "pages_by_id": {page_id: page.artifact() for page_id, page in cache.items()},
    }
    return PageComparison(
        benchmark_id, corpus_id, tokenizer_identity, summarizer.identity, token_budget,
        build, tuple(rows),
    )


def _row(
    item: PageCase, pages: Sequence[SummaryPage], arm: str, policy: AssemblyPolicy,
    principal: Principal, authorization: AuthorizationProvider,
    policy_attributes: Sequence[PolicyAttribute], tokenizer_identity: str,
    count_tokens: Callable[[str], int], instructions: str,
) -> dict[str, Any]:
    case = item.case
    page_evidence = [page.evidence() for page in pages] if arm == "passages_plus_pages" else []
    started = time.monotonic()
    context = assemble_context(
        [*page_evidence, *case.candidates], neighbors=case.neighbors, policy=policy,
        principal=principal, authorization=authorization,
        policy_attributes=policy_attributes, tokenizer_identity=tokenizer_identity,
        count_tokens=count_tokens,
        prefix=instructions + "\nQuestion: " + case.question + "\nEvidence:\n", suffix="\n",
    )
    seconds = time.monotonic() - started
    packed_pages = {block.context_id for block in context.evidence} & {
        page.page_id for page in pages}
    chunk_text = "\n".join(
        block.text for block in context.evidence if block.context_id not in packed_pages)
    served = [page for page in pages if page.page_id in packed_pages]
    covered = {fact for fact, literal in item.facts.items()
               if literal in chunk_text or any(literal in page.text for page in served)}
    page_only = {fact for fact in covered if item.facts[fact] not in chunk_text}
    # Facts reached only through a page must resolve to the cited source chunk.
    sources = [c for chunks in _by_document(case) for c in chunks]
    resolved = {page.page_id: resolve_claims(page, sources) for page in served}
    unresolved = sum(
        not any(item.facts[fact] in claim.quote and claim.context_id in resolved[page.page_id]
                for page in served for claim in page.claims)
        for fact in page_only
    )
    required = len(item.facts)
    return {
        "case_id": case.case_id, "group_id": case.group_id, "split": case.split,
        "question_class": item.question_class, "arm": arm,
        "policy_id": context.policy_id, "prompt_id": context.prompt_id,
        "token_count": context.token_count,
        "context_ids": [block.context_id for block in context.evidence],
        "page_ids": sorted(packed_pages),
        "omitted_ids": [omission.context_id for omission in context.omissions],
        "required_facts": required, "covered_facts": len(covered),
        "page_only_facts": len(page_only), "unresolved_page_facts": unresolved,
        "fact_coverage": len(covered) / required if required else None,
        "assemble_seconds": seconds,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("all", "broad", "multi_section", "exact_identifier"):
        group: dict[str, Any] = {}
        for arm in ARMS:
            selected = [row for row in rows if row["arm"] == arm
                        and key in ("all", row["question_class"])]
            scored = [row["fact_coverage"] for row in selected if row["fact_coverage"] is not None]
            group[arm] = {
                "cases": len(selected), "scored_cases": len(scored),
                "mean_fact_coverage": sum(scored) / len(scored) if scored else None,
                "mean_token_count": (sum(row["token_count"] for row in selected) / len(selected))
                if selected else None,
                "mean_assemble_seconds": (
                    sum(row["assemble_seconds"] for row in selected) / len(selected)
                ) if selected else None,
                "page_only_facts": sum(row["page_only_facts"] for row in selected),
                "unresolved_page_facts": sum(row["unresolved_page_facts"] for row in selected),
            }
        out[key] = group
    return out

