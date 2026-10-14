"""The sentence behind a concept comes from the governed service (issue #678).

Item 7 of #555 closes the loop between the aggregate map and the corpus it
sits on. These tests pin the two things that make the loop trustworthy: the
request goes through `ContextService` with the scope the concept actually
warrants, and the answer says which of the two scopes it got -- "linked to
FERC" and "contains the word FERC" are different claims, and a card that
reported one as the other would be worse than no card.

The doubles live in `support_concept_cloud` and implement the real
`search_context` contract, so a change to the request or response models
breaks these tests rather than passing against a mock that agreed with itself.
"""

from __future__ import annotations

import pytest

from stel.concept_cloud.sentences import (
    VIEWER_CALLER,
    Sentence,
    SentenceUnavailable,
    concept_sentence,
    declared_labels,
)
from stel.config.vocabulary import Vocabulary, VocabularyTerm
from stel.mcp_server.contracts import (
    MCPErrorCode,
    SearchContextResponse,
    ToolError,
)
from tests.support_concept_cloud import (
    FakeContextService,
    matched,
    search_result,
)


def _ask(
    service: FakeContextService,
    *,
    canonical_id: str = "Federal Reserve",
    display: str = "Federal Reserve",
    declared: frozenset[str] = frozenset({"Federal Reserve"}),
) -> Sentence | None:
    return concept_sentence(
        service,
        model="sec_context_chunks",
        canonical_id=canonical_id,
        display=display,
        declared=declared,
        mode="hybrid",
    )


def test_a_declared_concept_is_fetched_by_the_term_it_is_linked_to() -> None:
    """A concept the project declares is scoped, so the chunk is one the
    pipeline linked to it rather than one that merely says its name."""
    service = FakeContextService(
        SearchContextResponse(
            results=(
                search_result(
                    matched_terms=(matched(),)
                ),
            )
        )
    )

    sentence = _ask(service)

    request = service.requests[0]
    assert request.entity_scope is not None
    assert request.entity_scope.term == "Federal Reserve"
    assert request.model == "sec_context_chunks"
    assert request.limit == 1
    assert sentence is not None
    assert sentence.scope == "term"
    assert sentence.matched_terms == ("Federal Reserve",)
    assert sentence.snippet.startswith("The Federal Reserve raised rates")


def test_an_undeclared_concept_is_searched_as_text_and_says_so() -> None:
    """A fuzzy resolver's concept, or any project with no `vocabularies:`,
    cannot be scoped. The request carries no scope and the answer is labelled
    `query`, because a text match is a weaker claim than a link."""
    service = FakeContextService(SearchContextResponse(results=(search_result(),)))

    sentence = _ask(service, canonical_id="Acme", display="Acme Corp", declared=frozenset())

    assert service.requests[0].entity_scope is None
    assert service.requests[0].query == "Acme Corp"
    assert sentence is not None
    assert sentence.scope == "query"
    assert sentence.matched_terms == ()


def test_nothing_matching_is_an_answer_not_an_error() -> None:
    """A concept can be in the map and have no chunk the caller may read --
    the policy filter runs inside the service. The card says "no sentence"
    rather than showing a failure."""
    service = FakeContextService(SearchContextResponse(results=()))

    assert _ask(service) is None


def test_a_refusal_carries_the_services_own_sanitized_pair() -> None:
    """The service has already turned whatever went wrong into a `ToolError`;
    this raises that code and message rather than inventing its own, and no
    native exception text is in reach to be leaked onward."""
    service = FakeContextService(
        SearchContextResponse(
            error=ToolError(
                code=MCPErrorCode.NOT_FOUND_OR_DENIED,
                message="The context model is not available to this caller",
            )
        )
    )

    with pytest.raises(SentenceUnavailable) as caught:
        _ask(service)

    assert caught.value.code == "not_found_or_denied"
    assert caught.value.message == (
        "The context model is not available to this caller"
    )


def test_a_degraded_generation_is_carried_onto_the_sentence() -> None:
    """A degraded index answers exactly as a ready one does (issue #617), so
    the flag has to travel or the reader cannot tell a stale sentence from a
    current one."""
    service = FakeContextService(
        SearchContextResponse(results=(search_result(),), degraded=True)
    )

    sentence = _ask(service)

    assert sentence is not None
    assert sentence.degraded is True


def test_the_viewer_identifies_itself_to_the_query_log() -> None:
    """#528's caller fields exist to tell which client asked. A reader
    clicking a star is a different client from an agent, and the log row is
    where that distinction has to land."""
    service = FakeContextService(SearchContextResponse(results=(search_result(),)))

    _ask(service)

    assert service.callers == [VIEWER_CALLER]
    assert VIEWER_CALLER.client_name == "stel concept-cloud"


def test_the_citation_and_truncation_reach_the_payload() -> None:
    """The card shows where a sentence came from and whether it was cut, so
    both survive into the JSON the viewer renders."""
    service = FakeContextService(
        SearchContextResponse(results=(search_result(truncated=True),))
    )

    sentence = _ask(service)

    assert sentence is not None
    payload = sentence.as_payload()
    assert payload["truncated"] is True
    assert payload["document_id"] == "d1"
    assert payload["source_uri"] == "s3://filings/d1.htm"
    assert payload["section_path"] == ["Item 1A", "Interest rates"]


def test_declared_labels_spans_every_vocabulary() -> None:
    """A concept's id is a term label, and a scope accepts that label in each
    vocabulary declaring it -- so the membership test is on labels alone."""
    labels = declared_labels(
        {
            "central_banks": Vocabulary(
                terms=[
                    VocabularyTerm(label="Federal Reserve"),
                    VocabularyTerm(label="Bank of England"),
                ]
            ),
            "regulators": Vocabulary(terms=[VocabularyTerm(label="FERC")]),
        }
    )

    assert labels == frozenset({"Federal Reserve", "Bank of England", "FERC"})
    assert declared_labels({}) == frozenset()
