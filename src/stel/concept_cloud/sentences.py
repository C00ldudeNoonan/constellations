"""The sentence behind a concept, fetched through the governed MCP (issue #678).

The star map is an aggregate: it says a concept was named in 300 documents and
by whom, never what was said. This module is the one place that asks, and it
asks `ContextService` -- the same object `stel mcp serve` wraps -- so a
sentence a reader sees has passed the same authorization, the same generation
pin and the same policy filter an agent's would, and leaves the same
query-log row. Nothing here reads a store or a warehouse directly, and no
sentence is written to disk or into the bundle.
"""
from __future__ import annotations

from collections.abc import Container, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from ..config.vocabulary import Vocabulary, declared_terms
from ..mcp_server.contracts import (
    EntityScope,
    SearchContextRequest,
    SearchContextResponse,
)
from ..mcp_server.service import CallerInfo

# How the query log records a call from the viewer. The concept cloud is not
# an MCP transport and does not borrow one of their names: #528 added the
# caller fields to tell which client asked, and a reader clicking a star is a
# different client from an agent, which is exactly the distinction worth
# keeping in the log.
VIEWER_CALLER = CallerInfo(client_name="stel concept-cloud", transport="loopback")

# The card asks "says who?", not "show me the corpus": one sentence, with no
# way to page. A scoped search still retrieves a pool and filters it before
# this limit applies (ADR-0025), so asking for one does not mean a linked
# chunk gets dropped for an unlinked one that outranked it.
_LIMIT = 1

# `SearchContextRequest.query` caps its length, and a display name long
# enough to trip that cap would surface as a validation failure rather than
# as the answer "no sentence". Concepts are named by people, so this is a
# guard, not a path anything is expected to take.
_MAX_QUERY = 32_768

# Why a sentence is the one it is. `term` means the chunk is *linked* to this
# concept through a declared vocabulary term; `query` means it matched the
# concept's name as text. The card says which, because "linked to FERC" and
# "contains the word FERC" are different claims about the corpus.
SentenceScope = Literal["term", "query"]


class SentenceUnavailable(Exception):
    """The governed service refused the request or could not serve it.

    Carries the service's own `ToolError` code and message, which are already
    the sanitized pair the MCP server hands an agent -- no native exception
    text reaches here to be leaked onward.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class Sentence:
    """One chunk of a document that names a concept, and where it came from."""

    snippet: str
    truncated: bool
    document_id: str
    document_version_id: str
    scope: SentenceScope
    # The declared terms the chunk is actually linked to, empty on a `query`
    # scope. More than one when a term is declared in several vocabularies.
    matched_terms: tuple[str, ...]
    # The pinned generation's own status, not inferred from the result: a
    # degraded index answers exactly as a ready one does (issue #617).
    degraded: bool
    source_uri: str | None
    section_path: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        """The JSON the viewer renders. Field names are the viewer's contract."""
        return {
            "snippet": self.snippet,
            "truncated": self.truncated,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "scope": self.scope,
            "matched_terms": list(self.matched_terms),
            "degraded": self.degraded,
            "source_uri": self.source_uri,
            "section_path": list(self.section_path),
        }


class SentenceService(Protocol):
    """The slice of `ContextService` a sentence lookup needs.

    Narrow on purpose: a test double implements this one method against the
    real request and response contracts, so a change to either breaks the
    double rather than passing against a mock that agreed with itself.
    """

    def search_context(
        self,
        request: SearchContextRequest,
        *,
        caller: CallerInfo | None = None,
    ) -> SearchContextResponse: ...


def declared_labels(vocabularies: Mapping[str, Vocabulary]) -> frozenset[str]:
    """Every term label the project declares, in any vocabulary.

    A concept's `canonical_id` is the label of the term it linked to, which is
    what makes this set the test for whether a concept can be scoped. Keyed by
    label alone rather than by (vocabulary, label): a scope names a term, and
    `_resolve_entity_scope` accepts it in every vocabulary that declares it.
    """
    return frozenset(label for _, label in declared_terms(vocabularies))


def concept_sentence(
    service: SentenceService,
    *,
    model: str,
    canonical_id: str,
    display: str,
    declared: Container[str],
    mode: Literal["vector", "text", "hybrid"],
) -> Sentence | None:
    """Fetch one sentence naming `canonical_id`, or None if nothing matched.

    A concept whose id is a declared vocabulary term is fetched with an
    `entity_scope`, so the result is a chunk the pipeline actually linked to
    the concept. A concept no vocabulary declares -- a fuzzy or
    hand-maintained resolver's output, or a project with no `vocabularies:`
    at all -- can only be searched as text, and the returned `scope` says so.

    Membership is tested rather than scoping unconditionally and recovering
    from the refusal: an undeclared term is refused by the service with
    `INVALID_REQUEST`, which is the right answer to an agent's typo and the
    wrong one here, where the caller is a button and the fallback is known in
    advance.

    None is a real answer, not a failure: a concept can be in the map and have
    no chunk the reader is authorized to see, and the card says so rather than
    showing an error.
    """
    scoped = canonical_id in declared
    response = service.search_context(
        SearchContextRequest(
            model=model,
            query=display[:_MAX_QUERY],
            mode=mode,
            limit=_LIMIT,
            entity_scope=EntityScope(term=canonical_id) if scoped else None,
        ),
        caller=VIEWER_CALLER,
    )
    if response.error is not None:
        raise SentenceUnavailable(response.error.code.value, response.error.message)
    if not response.results:
        return None
    top = response.results[0]
    return Sentence(
        snippet=top.snippet,
        truncated=top.snippet_truncated,
        document_id=top.document_id,
        document_version_id=top.document_version_id,
        scope="term" if scoped else "query",
        matched_terms=tuple(term.label for term in top.matched_terms),
        degraded=response.degraded,
        source_uri=top.citation.source_uri,
        section_path=tuple(top.citation.section_path or ()),
    )
