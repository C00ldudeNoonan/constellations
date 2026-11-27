"""ALE-285: scope, identity, invalidation and faithfulness of summary pages."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from examples.summary_pages.compare import (
    build_comparison,
    fixture_cases,
    maintenance_after_edit,
)
from stel.context_assembly import AssemblyError, AssemblyPolicy, Evidence, assemble_context
from stel.mcp_server.authorization import ClaimAuthorizationProvider, Principal
from stel.summary_pages import (
    Claim,
    LeadSentenceSummarizer,
    PageError,
    build_page,
    page_identity,
    reconcile,
    resolve_claims,
)
from stel.summary_pages_eval import compare_page_arms


def _chunk(index=0, text="Alpha is first. Detail one.", *, tenant=None, version=2, document=1):
    row = {
        "document_id": f"{document:032x}", "document_version_id": f"{version:032x}",
        "context_id": f"{document * 100 + index:032x}",
        "chunk_id": f"{document * 1000 + index:032x}", "chunk_index": index, "text": text,
        "source_uri": "secret://locator", "authorization_resolved": True,
        "is_public": tenant is None, "tenant_id": tenant,
    }
    return Evidence.from_rows(row, row)


def _doc(**kwargs):
    return [_chunk(0, "Alpha is first. Detail one.", **kwargs),
            _chunk(1, "Beta is second. Detail two.", **kwargs)]


SUMMARIZER = LeadSentenceSummarizer()


class _Scripted:
    live = False

    def __init__(self, claims, identity="scripted/v1", live=False):
        self.claims, self.identity, self.live, self.calls = claims, identity, live, 0

    def summarize(self, chunks):
        self.calls += 1
        return self.claims(chunks)


def test_identity_is_deterministic_and_covers_every_input():
    """Any input change must change the page, or invalidation misses it."""
    base = page_identity(_doc(), SUMMARIZER.identity)
    assert base == page_identity(list(reversed(_doc())), SUMMARIZER.identity)
    assert build_page(_doc(), SUMMARIZER).page_id == base
    edited = [_chunk(0, "Alpha is first. Detail ONE."), _chunk(1, "Beta is second. Detail two.")]
    assert page_identity(edited, SUMMARIZER.identity) != base
    assert page_identity(_doc(tenant="a"), SUMMARIZER.identity) != base
    assert page_identity(_doc(version=3), SUMMARIZER.identity) != base
    assert page_identity(_doc(), "other/v1") != base


def test_mixed_scope_is_refused_before_the_summarizer_sees_text():
    """Information must not cross an authorization boundary through a summary."""
    spy = _Scripted(lambda chunks: [Claim(chunks[0][0], "x")])
    mixed = [_chunk(0, tenant="a"), _chunk(1, "Beta is second.", tenant="b")]
    with pytest.raises(PageError, match="authorization scope"):
        build_page(mixed, spy)
    with pytest.raises(PageError, match="one document version"):
        build_page([_chunk(0), _chunk(1, document=2)], spy)
    assert spy.calls == 0


def test_a_claim_not_quoting_its_cited_chunk_is_dropped_and_counted():
    chunks = _doc()
    summarizer = _Scripted(lambda c: [
        Claim(c[0][0], "Alpha is first."), Claim(c[0][0], "Invented fact."),
        Claim(c[1][0], "Alpha is first."), Claim("f" * 32, "Beta"),
    ])
    page = build_page(chunks, summarizer)
    assert [claim.quote for claim in page.claims] == ["Alpha is first."]
    assert page.rejected_claims == 3
    with pytest.raises(PageError, match="survived"):
        build_page(chunks, _Scripted(lambda c: [Claim(c[0][0], "Invented fact.")]))


def test_claims_resolve_to_source_chunks_until_a_source_changes():
    page = build_page(_doc(), SUMMARIZER)
    resolved = resolve_claims(page, _doc())
    assert set(resolved) == {claim.context_id for claim in page.claims}
    edited = [_chunk(0, "Alpha changed. Detail one."), _chunk(1, "Beta is second. Detail two.")]
    with pytest.raises(PageError, match="no longer resolves"):
        resolve_claims(page, edited)


def test_reconcile_names_the_cause_and_only_kept_pages_are_servable():
    page = build_page(_doc(), SUMMARIZER)
    other = build_page(_doc(document=2), SUMMARIZER)
    ident = SUMMARIZER.identity
    current = {page.document_id: _doc(), other.document_id: _doc(document=2)}
    same = reconcile([page, other], current, summarizer_identity=ident)
    assert set(same.keep) == {page.page_id, other.page_id} and not same.rebuild

    edited = {**current, page.document_id: [_chunk(0, "Alpha changed."), _chunk(1)]}
    work = reconcile([page, other], edited, summarizer_identity=ident)
    assert work.rebuild == {page.page_id: "source_changed"} and work.keep == (other.page_id,)

    # A permission change leaves the text alone; the page must still be withdrawn.
    moved = {**current, page.document_id: _doc(tenant="a")}
    work = reconcile([page, other], moved, summarizer_identity=ident)
    assert work.rebuild == {page.page_id: "permission_changed"}
    assert work.keep == (other.page_id,)

    # Scope now mixed inside one document: unpageable, never served.
    split = {**current, page.document_id: [_chunk(0, tenant="a"), _chunk(1)]}
    work = reconcile([page, other], split, summarizer_identity=ident)
    assert work.rebuild == {page.page_id: "permission_changed"}

    gone = {other.document_id: current[other.document_id]}
    work = reconcile([page, other], gone, summarizer_identity=ident)
    assert work.prune == (page.page_id,) and work.keep == (other.page_id,)

    fresh = {**current, f"{9:032x}": _doc(document=9)}
    work = reconcile([page, other], fresh, summarizer_identity=ident)
    assert work.add == (f"{9:032x}",)

    work = reconcile([page], current, summarizer_identity="other/v1")
    assert work.rebuild == {page.page_id: "summarizer_changed"}


def test_one_edit_rebuilds_one_page():
    work = maintenance_after_edit()
    assert len(work.rebuild) == 1 and len(work.keep) == 1 and not work.add and not work.prune


def test_a_page_is_denied_to_a_principal_outside_its_sources_scope():
    page = build_page(_doc(tenant="a"), SUMMARIZER)

    def assemble(tenant):
        return assemble_context(
            [page.evidence()], neighbors=(), policy=AssemblyPolicy(token_budget=10000),
            principal=Principal("t", tenant_id=tenant), authorization=ClaimAuthorizationProvider(),
            policy_attributes=(), tokenizer_identity="t/v1", count_tokens=len, prefix="", suffix="",
        )

    assert assemble("a").evidence[0].context_id == page.page_id
    with pytest.raises(AssemblyError, match="denied"):
        assemble("b")


def _compare(cases=None, summarizer=SUMMARIZER, allow_live=False, principal=None):
    return compare_page_arms(
        cases or fixture_cases(), summarizer=summarizer, allow_live=allow_live,
        token_budget=20000, benchmark_id="fixture/v1", corpus_id="corpus/v1",
        principal=principal or Principal("t"), authorization=ClaimAuthorizationProvider(),
        policy_attributes=(), tokenizer_identity="t/v1", count_tokens=len,
        instructions="Use evidence.",
    )


def test_comparison_reports_both_arms_cost_and_defers():
    artifact = _compare().artifact()
    assert artifact["decision"].startswith("defer")
    arms = artifact["aggregate"]["holdout"]["all"]
    pages, raw = arms["passages_plus_pages"], arms["passages"]
    assert pages["mean_fact_coverage"] > raw["mean_fact_coverage"]
    assert pages["mean_token_count"] > raw["mean_token_count"]
    assert pages["unresolved_page_facts"] == 0
    classes = artifact["aggregate"]["holdout"]
    assert all(classes[key]["passages"]["cases"] == 1
               for key in ("broad", "multi_section", "exact_identifier"))
    # Exact identifiers gain nothing from pages and pay their tokens anyway.
    ident = classes["exact_identifier"]
    assert ident["passages_plus_pages"]["mean_fact_coverage"] == (
        ident["passages"]["mean_fact_coverage"])
    assert ident["passages_plus_pages"]["mean_token_count"] > ident["passages"]["mean_token_count"]
    build = artifact["build"]
    assert build["reused_builds"] > 0 and build["claim_faithfulness"] == 1.0


def test_artifact_carries_no_question_page_or_document_text():
    text = json.dumps(_compare().artifact())
    for needle in ("Supply concentration", "ZX-4471-B", "risks across", "fixture://"):
        assert needle not in text


def test_a_live_summarizer_needs_opt_in_and_a_mixed_scope_case_aborts():
    live = _Scripted(lambda c: [], identity="live/v1", live=True)
    with pytest.raises(AssemblyError, match="opt-in"):
        _compare(summarizer=live)
    assert live.calls == 0
    case = fixture_cases()[0]
    mixed = (_chunk(0, tenant="a"), _chunk(1))  # readable by tenant a, different scope
    leaky = replace(case, case=replace(case.case, candidates=mixed, neighbors=()))
    with pytest.raises(AssemblyError, match="authorization scope"):
        _compare([leaky], principal=Principal("t", tenant_id="a"))


def test_denied_sources_never_reach_the_summarizer():
    """A live summarizer must not see text the principal cannot read."""
    spy = _Scripted(lambda c: [Claim(c[0][0], "Alpha is first.")])
    case = fixture_cases()[0]
    denied = replace(case, case=replace(
        case.case, candidates=(_chunk(0, tenant="a"),), neighbors=(_chunk(1, tenant="a"),)))
    with pytest.raises(AssemblyError, match="denied"):
        _compare([denied], summarizer=spy)
    assert spy.calls == 0


def test_aggregates_keep_development_and_holdout_apart():
    cases = fixture_cases()
    dev = replace(cases[0], case=replace(cases[0].case, case_id="dev", split="development"))
    aggregate = _compare([dev, *cases[1:]]).artifact()["aggregate"]
    assert set(aggregate) == {"development", "holdout"}
    assert aggregate["development"]["all"]["passages"]["cases"] == 1
    assert aggregate["holdout"]["all"]["passages"]["cases"] == 2


def test_example_comparison_runs_offline():
    comparison = build_comparison(counter=len, tokenizer_identity="t/v1", token_budget=20000)
    assert comparison.artifact()["aggregate"]["holdout"]["all"]["passages"]["cases"] == 3
