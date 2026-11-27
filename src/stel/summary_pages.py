"""Source-linked summary pages for offline retrieval experiments (ALE-285).

A page is one source document's extractive claims, each quoting a cited chunk
verbatim. This is a RAPTOR-inspired baseline (one level, no clustering), not
serving behavior: nothing here is read by MCP search, and no warehouse or
provider is touched. See docs/architecture/summary-pages.md.

Three rules carry the experiment:

* A page is built from one document version in one authorization scope.
  Mixed scope is refused before the summarizer sees any text, so a summary
  cannot carry information across a boundary.
* Every claim is checked against the chunk it cites at build time. A claim
  that is not a verbatim span of that chunk is dropped and counted.
* A page's identity covers its inputs (version, scope, ordered source chunk
  text hashes, summarizer identity). Any change gives a new identity, which is
  the whole invalidation rule: ``reconcile`` compares identities and names the
  cause, and only pages it reports as kept are servable.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .context_assembly import Evidence
from .hashing import canonical_fingerprint

PAGE_SCHEMA_VERSION = 1
# Fields the serving authorization providers read off a row. A page copies
# them from its sources and refuses sources that disagree.
SCOPE_FIELDS = ("authorization_resolved", "is_public", "tenant_id", "access_groups")


class PageError(ValueError):
    """Safe diagnostic: never contains document, policy, or provider values."""


@dataclass(frozen=True, repr=False)
class Claim:
    """A quote and the context ID of the chunk it must appear in verbatim."""

    context_id: str
    quote: str


class Summarizer(Protocol):
    """Trusted driver. A live one must enforce its own spend cap."""

    identity: str
    live: bool

    def summarize(self, chunks: Sequence[tuple[str, str]]) -> Sequence[Claim]:
        """Given ordered (context_id, text) pairs, return claims."""
        ...


class LeadSentenceSummarizer:
    """Deterministic offline mock: the first sentence of every chunk."""

    identity = "mock-lead-sentence/v1"
    live = False

    def summarize(self, chunks: Sequence[tuple[str, str]]) -> Sequence[Claim]:
        claims = []
        for context_id, text in chunks:
            match = re.match(r"\s*(.+?[.!?])(?:\s|$)", text, flags=re.DOTALL)
            quote = match.group(1) if match else text.strip()
            if quote:
                claims.append(Claim(context_id, quote))
        return claims


@dataclass(frozen=True, repr=False)
class SummaryPage:
    page_id: str
    document_id: str
    document_version_id: str
    scope_id: str
    summarizer_identity: str
    #: Ordered (context_id, text hash) of every source chunk.
    sources: tuple[tuple[str, str], ...]
    claims: tuple[Claim, ...]
    rejected_claims: int
    source_chars: int
    _scope: Mapping[str, Any]

    def render(self) -> str:
        """The page text; iterates every claim, so it is a method, not a property."""
        return "\n".join(f"[{claim.context_id}] {claim.quote}" for claim in self.claims)

    def size_ratio(self) -> float:
        return len(self.render()) / self.source_chars

    def evidence(self) -> Evidence:
        """The page as one governed row pair, carrying its sources' scope."""
        page_document = canonical_fingerprint(
            {"page_of": self.document_id, "scope": self.scope_id},
            domain="stel-summary-page-document",
        )
        row = {
            "document_id": page_document, "document_version_id": self.page_id,
            "context_id": self.page_id,
            "chunk_id": canonical_fingerprint(self.page_id, domain="stel-summary-page-chunk"),
            "chunk_index": 0, "text": self.render(), "source_uri": f"summary-page://{self.page_id}",
            **self._scope,
        }
        return Evidence.from_rows(row, row)

    def artifact(self) -> dict[str, Any]:
        """IDs and counts only; never claim text."""
        return {
            "schema_version": f"summary_page/v{PAGE_SCHEMA_VERSION}",
            "page_id": self.page_id, "document_id": self.document_id,
            "document_version_id": self.document_version_id, "scope_id": self.scope_id,
            "summarizer": self.summarizer_identity,
            "source_context_ids": [source[0] for source in self.sources],
            "claim_context_ids": [claim.context_id for claim in self.claims],
            "rejected_claims": self.rejected_claims,
            "source_chars": self.source_chars, "page_chars": len(self.render()),
        }


@dataclass(frozen=True)
class _Inputs:
    document_id: str
    document_version_id: str
    scope_id: str
    scope: Mapping[str, Any]
    sources: tuple[tuple[str, str], ...]
    texts: tuple[tuple[str, str], ...]


def _inputs(chunks: Sequence[Evidence]) -> _Inputs:
    """Validate one document version in one scope; order by chunk index."""
    if not chunks:
        raise PageError("A page requires source chunks")
    rows = sorted((item.rows() for item in chunks), key=lambda pair: pair[1]["chunk_index"])
    first = rows[0][1]
    ids = [chunk["context_id"] for _, chunk in rows]
    if len(set(ids)) != len(ids) or len({chunk["chunk_index"] for _, chunk in rows}) != len(rows):
        raise PageError("Source chunks must be unique")
    document = (first["document_id"], first["document_version_id"])
    if any((chunk["document_id"], chunk["document_version_id"]) != document for _, chunk in rows):
        raise PageError("A page is built from one document version")
    scopes = {
        canonical_fingerprint(
            {name: row.get(name) for name in SCOPE_FIELDS}, domain="stel-summary-page-scope"
        )
        for pair in rows for row in pair
    }
    if len(scopes) != 1:
        raise PageError("Source chunks span more than one authorization scope")
    scope = {name: first.get(name) for name in SCOPE_FIELDS}
    sources = tuple(
        (chunk["context_id"], canonical_fingerprint(chunk["text"], domain="stel-summary-page-text"))
        for _, chunk in rows
    )
    return _Inputs(
        *document, scopes.pop(), scope, sources,
        tuple((chunk["context_id"], chunk["text"]) for _, chunk in rows),
    )


def _page_id(inputs: _Inputs, summarizer_identity: str) -> str:
    if not summarizer_identity.strip():
        raise PageError("A summarizer identity is required")
    return canonical_fingerprint(
        {"document": inputs.document_id, "version": inputs.document_version_id,
         "scope": inputs.scope_id, "sources": inputs.sources,
         "summarizer": summarizer_identity, "schema": PAGE_SCHEMA_VERSION},
        domain="stel-summary-page",
    )


def page_identity(chunks: Sequence[Evidence], summarizer_identity: str) -> str:
    """The identity ``build_page`` would give these chunks, without summarizing."""
    return _page_id(_inputs(chunks), summarizer_identity)


def build_page(chunks: Sequence[Evidence], summarizer: Summarizer) -> SummaryPage:
    inputs = _inputs(chunks)  # scope check precedes any summarizer call
    page_id = _page_id(inputs, summarizer.identity)
    try:
        proposed = list(summarizer.summarize(inputs.texts))
    except Exception:
        raise PageError("Summarization failed") from None
    text_by_id = dict(inputs.texts)
    kept = tuple(
        claim for claim in proposed
        if isinstance(claim, Claim) and claim.quote
        and claim.quote in text_by_id.get(claim.context_id, "")
    )
    if not kept:
        raise PageError("No claim survived verification against its cited chunk")
    return SummaryPage(
        page_id, inputs.document_id, inputs.document_version_id, inputs.scope_id,
        summarizer.identity, inputs.sources, kept, len(proposed) - len(kept),
        sum(len(text) for _, text in inputs.texts), inputs.scope,
    )


def resolve_claims(page: SummaryPage, chunks: Sequence[Evidence]) -> dict[str, Evidence]:
    """Map each claim's context ID back to the source chunk that holds its quote."""
    by_id = {item.rows()[1]["context_id"]: item for item in chunks}
    resolved: dict[str, Evidence] = {}
    for claim in page.claims:
        item = by_id.get(claim.context_id)
        if item is None or claim.quote not in item.rows()[1]["text"]:
            raise PageError("A page claim no longer resolves to its source chunk")
        resolved[claim.context_id] = item
    return resolved


Reason = Literal["source_changed", "permission_changed", "summarizer_changed"]


@dataclass(frozen=True)
class Maintenance:
    """What a corpus change costs: pages to add, rebuild, prune; the rest are kept."""

    add: tuple[str, ...]
    rebuild: Mapping[str, Reason]
    prune: tuple[str, ...]
    keep: tuple[str, ...]

    def artifact(self) -> dict[str, Any]:
        return {"add": list(self.add), "rebuild": dict(self.rebuild),
                "prune": list(self.prune), "keep": list(self.keep)}


def reconcile(
    pages: Sequence[SummaryPage], current: Mapping[str, Sequence[Evidence]],
    *, summarizer_identity: str,
) -> Maintenance:
    """Compare stored pages with the live corpus (document ID -> its chunks).

    ``keep`` is the only servable set: a page whose source changed, lost
    readability under its recorded scope, or was deleted is never served,
    whether or not it has been rebuilt yet. A document whose chunks now span
    scopes cannot be paged at all, so its page is reported as a permission
    change and nothing replaces it.
    """
    add: list[str] = []
    rebuild: dict[str, Reason] = {}
    prune: list[str] = []
    keep: list[str] = []
    paged = {page.document_id for page in pages}
    for page in pages:
        chunks = current.get(page.document_id)
        if not chunks:
            prune.append(page.page_id)
            continue
        try:
            inputs = _inputs(chunks)
        except PageError:
            rebuild[page.page_id] = "permission_changed"
            continue
        if inputs.scope_id != page.scope_id:
            rebuild[page.page_id] = "permission_changed"
        elif _page_id(inputs, summarizer_identity) == page.page_id:
            keep.append(page.page_id)
        elif page.summarizer_identity != summarizer_identity:
            rebuild[page.page_id] = "summarizer_changed"
        else:
            rebuild[page.page_id] = "source_changed"
    add.extend(document for document, chunks in current.items() if chunks and document not in paged)
    return Maintenance(tuple(sorted(add)), rebuild, tuple(prune), tuple(keep))
