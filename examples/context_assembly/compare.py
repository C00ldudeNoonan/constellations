"""Reproducible synthetic packing comparison; no warehouse or model calls."""
from __future__ import annotations

import json
from collections.abc import Callable
from importlib.metadata import version

from stel.context_assembly import AssemblyPolicy, Evidence
from stel.context_assembly_eval import AssemblyCase, ContextComparison, compare_context_policies
from stel.mcp_server.authorization import ClaimAuthorizationProvider, Principal
from stel.text.tokens import count_tokens


def _chunk(document: int, index: int, text: str, start: int, *, long_metadata: bool) -> Evidence:
    row = {
        "document_id": f"{document:032x}", "document_version_id": f"{document + 100:032x}",
        "context_id": f"{document * 100 + index:032x}",
        "chunk_id": f"{document * 1000 + index:032x}",
        "chunk_index": index, "text": text,
        "citation_char_start": start, "citation_char_end": start + len(text),
        "source_uri": "fixture://" + ("long-metadata/" * 1000 if long_metadata else str(document)),
        "authorization_resolved": True, "is_public": True,
    }
    return Evidence.from_rows(row, row)


def fixture_cases() -> tuple[AssemblyCase, ...]:
    """Synthetic cases exercise mechanics; these are not ALE-281 answer golds."""
    boundary = (
        _chunk(1, 0, "The monthly limit is ", 0, long_metadata=False),
        _chunk(1, 1, "12 requests, approved by the owner.", 21, long_metadata=False),
    )
    text = "The monthly limit is 12 requests. Approval requires the project owner."
    overlap = (
        _chunk(2, 0, text[:46], 0, long_metadata=False),
        _chunk(2, 1, text[21:], 21, long_metadata=False),
    )
    long = _chunk(3, 0, "A short fact with a very long locator.", 0, long_metadata=True)
    short = _chunk(3, 1, "Approval requires the project owner.", 40, long_metadata=False)
    oversized = _chunk(4, 0, "Unrelated detail. " * 300, 0, long_metadata=False)
    return (
        AssemblyCase("boundary", "doc-1", "development", "What is the monthly limit?",
                     boundary[:1], boundary[1:], ("monthly-limit",), False, "synthetic-facts/v1"),
        AssemblyCase("overlap", "doc-2", "development", "Who approves the monthly limit?",
                     tuple(reversed(overlap)), (), ("approver",), False, "synthetic-facts/v1"),
        AssemblyCase("metadata", "doc-3", "holdout", "Who approves?",
                     (long, short), (), ("approver",), False, "synthetic-facts/v1"),
        AssemblyCase("insufficient-budget", "doc-4", "holdout", "What is the limit?",
                     (oversized,), (), ("monthly-limit",), False, "synthetic-facts/v1"),
        AssemblyCase("absent", "absent-scope", "holdout", "What is the limit in an absent scope?",
                     (), (), (), True, "synthetic-facts/v1"),
    )


def build_comparison(
    *, counter: Callable[[str], int], tokenizer_identity: str, token_budget: int,
) -> ContextComparison:
    policies = {
        "relevance": AssemblyPolicy(token_budget=token_budget),
        "document": AssemblyPolicy(token_budget=token_budget, order="document"),
        "overlap": AssemblyPolicy(token_budget=token_budget, remove_overlap=True),
        "neighbors": AssemblyPolicy(token_budget=token_budget, order="document",
                                    remove_overlap=True, neighbor_radius=1, max_neighbors=2),
    }
    return compare_context_policies(
        fixture_cases(), policies=policies, benchmark_id="synthetic-assembly/v1",
        corpus_id="synthetic-documents/v1", judgments_id="no-answer-judgments/v1",
        principal=Principal("offline-example"), authorization=ClaimAuthorizationProvider(),
        policy_attributes=(), tokenizer_identity=tokenizer_identity, count_tokens=counter,
        instructions="Answer only from the evidence. Cite context_id for each factual claim. "
                     "Abstain if evidence is insufficient.",
        generator=None, allow_live=False, reviews={},
    )


def main() -> None:
    comparison = build_comparison(
        counter=count_tokens,
        tokenizer_identity=f"tiktoken/{version('tiktoken')}/cl100k_base",
        token_budget=650,
    )
    print(json.dumps(comparison.artifact(), indent=2))


if __name__ == "__main__":
    main()
