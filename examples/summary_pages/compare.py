"""Reproducible synthetic passages-versus-pages comparison; no warehouse or model calls."""
from __future__ import annotations

import json
from collections.abc import Callable
from importlib.metadata import version

from stel.context_assembly import Evidence
from stel.context_assembly_eval import AssemblyCase
from stel.mcp_server.authorization import ClaimAuthorizationProvider, Principal
from stel.summary_pages import LeadSentenceSummarizer, Maintenance, build_page, reconcile
from stel.summary_pages_eval import PageCase, PageComparison, QuestionClass, compare_page_arms
from stel.text.tokens import count_tokens

_DOCUMENTS = {
    1: [
        "Supply concentration is the largest risk. Three vendors make 80% of parts.",
        "Demand was flat in 2025. Unit sales moved less than 1%.",
        "Currency exposure is the second risk. Euro revenue is 35% of the total.",
        "Litigation is a minor risk. One patent case is pending.",
    ],
    2: [
        "The retention policy ZX-4471-B keeps logs 90 days. Exports are excluded.",
        "Backups follow policy QQ-1002-A. They are kept 30 days.",
    ],
}


def _chunk(document: int, index: int) -> Evidence:
    text = _DOCUMENTS[document][index]
    row = {
        "document_id": f"{document:032x}", "document_version_id": f"{document + 100:032x}",
        "context_id": f"{document * 100 + index:032x}",
        "chunk_id": f"{document * 1000 + index:032x}", "chunk_index": index, "text": text,
        "source_uri": f"fixture://{document}", "authorization_resolved": True, "is_public": True,
    }
    return Evidence.from_rows(row, row)


def fixture_cases() -> tuple[PageCase, ...]:
    """Synthetic mechanics only; these are not ALE-281 reviewed golds."""
    def case(case_id: str, group: str, question: str, kind: QuestionClass,
             roots: list[Evidence], documents: tuple[int, ...],
             facts: dict[str, str]) -> PageCase:
        neighbors = tuple(_chunk(d, i) for d in documents for i in range(len(_DOCUMENTS[d])))
        return PageCase(
            AssemblyCase(case_id, group, "holdout", question, tuple(roots), neighbors,
                         tuple(facts), False, "synthetic-facts/v1"),
            kind, facts,
        )
    risks = {"supply": "Supply concentration is the largest risk",
             "currency": "Currency exposure is the second risk",
             "legal": "Litigation is a minor risk"}
    return (
        case("broad-risks", "doc-1", "What are the risks across the report?", "broad",
             [_chunk(1, 1)], (1,), risks),
        case("multi-section", "doc-1-2", "Which risk and which log policy apply?",
             "multi_section", [_chunk(1, 3), _chunk(2, 1)], (1, 2),
             {"currency": risks["currency"], "policy": "QQ-1002-A"}),
        case("identifier-retention", "doc-2", "Which policy keeps logs 90 days?",
             "exact_identifier", [_chunk(2, 0)], (2,), {"policy": "ZX-4471-B"}),
    )


def build_comparison(
    *, counter: Callable[[str], int], tokenizer_identity: str, token_budget: int,
) -> PageComparison:
    return compare_page_arms(
        fixture_cases(), summarizer=LeadSentenceSummarizer(), allow_live=False,
        token_budget=token_budget, benchmark_id="synthetic-pages/v1",
        corpus_id="synthetic-documents/v1", principal=Principal("offline-example"),
        authorization=ClaimAuthorizationProvider(), policy_attributes=(),
        tokenizer_identity=tokenizer_identity, count_tokens=counter,
        instructions="Answer only from the evidence. Cite context_id for each factual claim.",
    )


def maintenance_after_edit() -> Maintenance:
    """Edit one chunk of document 1; document 2's page must be kept."""
    summarizer = LeadSentenceSummarizer()
    pages = [build_page([_chunk(d, i) for i in range(len(_DOCUMENTS[d]))], summarizer)
             for d in _DOCUMENTS]
    edited = _chunk(1, 1)
    row = {**edited.rows()[1], "text": "Demand fell in 2025. Unit sales dropped 4%."}
    current = {f"{d:032x}": [_chunk(d, i) for i in range(len(_DOCUMENTS[d]))] for d in _DOCUMENTS}
    current[f"{1:032x}"][1] = Evidence.from_rows(row, row)
    return reconcile(pages, current, summarizer_identity=summarizer.identity)


def main() -> None:
    comparison = build_comparison(
        counter=count_tokens, tokenizer_identity=f"tiktoken/{version('tiktoken')}/cl100k_base",
        token_budget=650,
    )
    artifact = comparison.artifact()
    artifact["maintenance_after_one_edit"] = maintenance_after_edit().artifact()
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
