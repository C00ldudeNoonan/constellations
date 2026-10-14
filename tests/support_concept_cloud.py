"""Shared doubles for the concept-cloud reach-the-source tests (issue #678).

Both the governed-lookup tests and the loopback-server tests need a context
service that answers without a warehouse, and a `SearchContextResult` to
answer with. They live here rather than in whichever test file needed them
first, which is what `test_test_tiers.py` enforces (#518).

`FakeContextService` implements the real `search_context` contract against the
real request and response models, so a change to either breaks the tests using
it rather than passing against a mock that agreed with itself.
"""

from __future__ import annotations

from datetime import UTC, datetime

from stel.mcp_server.contracts import (
    CitationDescriptor,
    CompactLineage,
    ContextInterval,
    FreshnessDescriptor,
    MatchedTerm,
    SearchContextRequest,
    SearchContextResponse,
    SearchContextResult,
)
from stel.mcp_server.service import CallerInfo

NOW = datetime(2026, 10, 10, tzinfo=UTC)

SNIPPET = "The Federal Reserve raised rates in March."


def search_result(
    *,
    snippet: str = SNIPPET,
    truncated: bool = False,
    matched_terms: tuple[MatchedTerm, ...] = (),
    source_uri: str | None = "s3://filings/d1.htm",
    section_path: tuple[str, ...] | None = ("Item 1A", "Interest rates"),
) -> SearchContextResult:
    """One search hit, with every required descriptor filled in."""
    return SearchContextResult(
        rank=1,
        score=0.82,
        document_id="d1",
        document_version_id="d1:v2",
        context_id="c1",
        chunk_id="c1:0",
        snippet=snippet,
        snippet_truncated=truncated,
        source_version="sv1",
        entities=(),
        interval=ContextInterval(
            validity_known=True,
            valid_from=NOW,
            valid_to=None,
            recorded_from=NOW,
            recorded_to=None,
        ),
        freshness=FreshnessDescriptor(
            status="fresh",
            source_updated_at=NOW,
            source_observed_at=NOW,
            freshness_checked_at=NOW,
            refresh_due_at=None,
            stale_after=None,
        ),
        citation=CitationDescriptor(
            source_uri=source_uri, section_path=section_path, chunk_index=0
        ),
        lineage=CompactLineage(
            resources=("sec_context_chunks",),
            upstream_unique_id="model.stel.sec_document_text",
            invocation_id="inv1",
            parser_identity=None,
            transform_identity=None,
            provider_identity=None,
            model_identity=None,
            prompt_fingerprint=None,
            schema_fingerprint=None,
            provenance_fingerprint="pf1",
            search_index_unique_id="search_index.stel.sec_context_chunks",
            store_type="lancedb",
        ),
        matched_terms=matched_terms,
    )


def matched(label: str = "Federal Reserve") -> MatchedTerm:
    """Why a hit satisfied an entity scope: an exact match on `label`."""
    return MatchedTerm(
        namespace="central_banks",
        label=label,
        requested=label,
        relation="exact",
    )


class FakeContextService:
    """A `SentenceService` that records the request and replays a response."""

    def __init__(self, response: SearchContextResponse) -> None:
        self._response = response
        self.requests: list[SearchContextRequest] = []
        self.callers: list[CallerInfo | None] = []

    def search_context(
        self,
        request: SearchContextRequest,
        *,
        caller: CallerInfo | None = None,
    ) -> SearchContextResponse:
        self.requests.append(request)
        self.callers.append(caller)
        return self._response
