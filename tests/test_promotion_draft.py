"""Drafting a golden set from candidate judgments (#380, #329 phase 3).

`stel promote` is a drafting aid, and every test here is about the difference
between drafting and deciding. It must never turn an observation into a label
a human did not choose, and it must never produce a file that *looks* promoted
while carrying a question nobody confirmed.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from stel.cli_services.promote import render_golden_set
from stel.promotion.contract import UNCONFIRMED_QUERY_TEXT, PromotionError
from stel.promotion.draft import CandidateRow, draft_golden_set


def _row(**overrides: object) -> CandidateRow:
    fields: dict[str, object] = {
        "session_id": "sess-1",
        "harness": "claude-code",
        "context_model": "context_search",
        "query_fingerprint": "a" * 32,
        "query_text": "consumer prices inflation",
        "id_space": "context_id",
        "context_id": "ctx-1",
        "judgment": "cited",
    }
    fields.update(overrides)
    return CandidateRow(**fields)  # type: ignore[arg-type]


def _draft(rows: list[CandidateRow]):
    return draft_golden_set(rows, promoted_by="alex", promoted_at=date(2026, 9, 2))


# ─── only a citation becomes a label ────────────────────────────────────────


def test_only_cited_ids_become_relevant_ids() -> None:
    """`returned_not_cited` is absence of evidence, not evidence of
    irrelevance — an agent may use a chunk without naming its id. Promoting it
    either way would invent a judgment the corpus never made."""
    draft = _draft(
        [
            _row(context_id="ctx-1", judgment="cited"),
            _row(context_id="ctx-2", judgment="returned_not_cited"),
            _row(context_id="ctx-3", judgment="cited"),
        ]
    )

    assert len(draft.golden_set.queries) == 1
    query = draft.golden_set.queries[0]
    assert query.relevant_ids == ("ctx-1", "ctx-3")
    # And never as a negative either: excluded_ids is a human's assertion.
    assert query.excluded_ids == ()


def test_a_query_with_no_citation_is_skipped_and_reported() -> None:
    """Reported rather than dropped: a reviewer deciding what to promote needs
    to see what was left behind, and a zero-result query is a real signal that
    simply needs a human to say what should have matched."""
    with pytest.raises(PromotionError, match="nothing to draft"):
        _draft(
            [
                _row(query_fingerprint="b" * 32, context_id=None, judgment="zero_result"),
            ]
        )

    draft = _draft(
        [
            _row(query_fingerprint="a" * 32, judgment="cited"),
            _row(query_fingerprint="b" * 32, context_id=None, judgment="zero_result"),
            _row(query_fingerprint="c" * 32, judgment="returned_not_cited"),
        ]
    )

    assert len(draft.golden_set.queries) == 1
    reasons = {skipped.query_fingerprint: skipped.reason for skipped in draft.skipped}
    assert set(reasons) == {"b" * 32, "c" * 32}
    assert "zero_result" in reasons["b" * 32]
    assert "returned_not_cited" in reasons["c" * 32]


# ─── query text: auto-filled, never assumed ─────────────────────────────────


def test_captured_query_text_is_filled_in_and_marked_as_from_the_corpus() -> None:
    """A transcribed query is more faithful than a remembered one, so it is
    filled — and flagged, because the reviewer still confirms it."""
    draft = _draft([_row(query_text="consumer prices inflation")])

    assert draft.golden_set.queries[0].query_text == "consumer prices inflation"
    assert draft.drafted[0].text_from_corpus is True
    assert draft.needs_text == ()


def test_a_query_with_no_captured_text_drafts_a_placeholder_that_will_not_load() -> None:
    """The sensitivity default is fingerprint-only, so most corpora land here.
    The draft must fail loudly rather than run as a test asking the wrong
    question -- a golden set is replayed through `search()` (#380 c2)."""
    draft = _draft([_row(query_text=None)])

    query = draft.golden_set.queries[0]
    assert query.query_text == UNCONFIRMED_QUERY_TEXT
    assert draft.needs_text == draft.drafted


def test_a_drafted_placeholder_is_refused_on_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Drafting it is allowed -- writing the draft is the point. Loading it is
    not: the check lives at load so an unreviewed file fails when something
    tries to run it, naming the queries that still need a question."""
    from stel.promotion import load_golden_set

    draft = _draft([_row(query_text=None)])
    path = tmp_path / "golden.yml"
    path.write_text(render_golden_set(draft), encoding="utf-8")

    with pytest.raises(PromotionError, match="placeholder query text"):
        load_golden_set(path)


# ─── provenance and id space ────────────────────────────────────────────────


def test_every_drafted_query_names_its_sessions_and_fingerprint() -> None:
    draft = _draft(
        [
            _row(session_id="sess-2"),
            _row(session_id="sess-1"),
            _row(session_id="sess-1", context_id="ctx-9"),
        ]
    )

    evidence = draft.golden_set.queries[0].evidence
    assert evidence.sessions == ("sess-1", "sess-2")
    assert evidence.query_fingerprint == "a" * 32
    assert evidence.harness == "claude-code"


def test_a_query_seen_through_two_harnesses_claims_neither() -> None:
    draft = _draft(
        [_row(harness="claude-code"), _row(harness="codex", context_id="ctx-2")]
    )

    assert draft.golden_set.queries[0].evidence.harness is None


def test_mixed_id_spaces_are_refused_rather_than_picked_between() -> None:
    """#380 constraint 3, and the reason it is a hard error: ids in the wrong
    space match nothing, so the eval reports zero recall as though retrieval
    were broken rather than the golden set mislabelled."""
    with pytest.raises(PromotionError, match="mix id spaces"):
        _draft(
            [
                _row(id_space="context_id"),
                _row(id_space="chunk_id", context_id="ctx-2"),
            ]
        )


def test_the_drafted_id_space_is_carried_from_the_candidates() -> None:
    draft = _draft([_row(id_space="chunk_id")])

    assert draft.golden_set.id_space == "chunk_id"


# ─── one index per golden set (PR #451 review) ──────────────────────────────


def test_candidates_spanning_two_context_models_are_refused() -> None:
    """`query_fingerprint` hashes the query string alone, so the same question
    asked of two indexes shares one fingerprint. Merging them would put ids
    from index B into a golden set run against index A — and the `id_space`
    guard cannot catch it, because both commonly key on the same space. The
    eval would then report zero recall as though retrieval were broken."""
    with pytest.raises(PromotionError, match="more than one context model"):
        _draft(
            [
                _row(context_model="context_search", context_id="ctx-1"),
                _row(context_model="release_search", context_id="ctx-2"),
            ]
        )


def test_a_context_model_filter_draws_from_one_index() -> None:
    rows = [
        _row(context_model="context_search", context_id="ctx-1"),
        _row(context_model="release_search", context_id="ctx-2"),
    ]

    draft = draft_golden_set(
        rows,
        promoted_by="alex",
        promoted_at=date(2026, 9, 2),
        context_model="release_search",
    )

    assert draft.context_model == "release_search"
    assert draft.golden_set.queries[0].relevant_ids == ("ctx-2",)
    # And the reviewer is told which index the file is for.
    assert "Context model: release_search" in render_golden_set(draft)


def test_filtering_to_a_model_with_no_candidates_is_an_error() -> None:
    with pytest.raises(PromotionError, match="No candidate judgments for"):
        draft_golden_set(
            [_row(context_model="context_search")],
            promoted_by="alex",
            promoted_at=date(2026, 9, 2),
            context_model="absent_search",
        )


# ─── the rendered artifact ──────────────────────────────────────────────────


def test_the_rendered_file_says_it_is_a_draft_and_reloads() -> None:
    """The header is the artifact's own warning that it is not yet a
    promotion; the body must still be a valid golden set."""
    draft = _draft([_row()])

    rendered = render_golden_set(draft)

    assert "DRAFTED, NOT PROMOTED" in rendered
    document = yaml.safe_load(rendered)
    assert document["id_space"] == "context_id"
    assert document["queries"][0]["relevant_ids"] == ["ctx-1"]
    assert document["queries"][0]["promoted_by"] == "alex"
    assert document["queries"][0]["promoted_at"] == "2026-09-02"


def test_drafting_from_no_candidates_is_an_error_not_an_empty_set() -> None:
    """An empty golden set would pass every eval it was given."""
    with pytest.raises(PromotionError, match="No candidate judgments"):
        _draft([])


# ─── writing the file ───────────────────────────────────────────────────────


def _service_draft(tmp_path, monkeypatch, rows: list[CandidateRow], **kwargs):  # type: ignore[no-untyped-def]
    """Drive the service with the warehouse read stubbed: the read itself is
    ordinary adapter plumbing, and what matters here is the file handling."""
    from stel.cli_services import promote as service

    monkeypatch.setattr(service, "read_candidates", lambda *a, **k: rows)
    return service.promote_from_candidates(
        tmp_path,
        profiles_dir=None,
        target=None,
        relation="analytics.candidates",
        output=tmp_path / "golden_sets" / "search.yml",
        promoted_by="alex",
        today=date(2026, 9, 2),
        **kwargs,
    )


def test_nothing_is_written_without_write(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    rendered, _draft_result = _service_draft(tmp_path, monkeypatch, [_row()])

    assert "DRAFTED, NOT PROMOTED" in rendered
    assert not (tmp_path / "golden_sets" / "search.yml").exists()


def test_writing_creates_the_file_and_it_loads(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from stel.promotion import load_golden_set

    _service_draft(tmp_path, monkeypatch, [_row()], write=True)

    path = tmp_path / "golden_sets" / "search.yml"
    golden = load_golden_set(path)
    assert golden.queries[0].relevant_ids == ("ctx-1",)
    assert golden.queries[0].evidence.sessions == ("sess-1",)


def test_an_existing_golden_set_is_never_silently_overwritten(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """The file is human-owned. Re-drafting over it would discard the review
    that is the entire value of the artifact."""
    from stel.cli_services.context import ConfigClickError

    _service_draft(tmp_path, monkeypatch, [_row()], write=True)
    path = tmp_path / "golden_sets" / "search.yml"
    reviewed = path.read_text(encoding="utf-8").replace(
        "consumer prices inflation", "a question a human wrote"
    )
    path.write_text(reviewed, encoding="utf-8")

    with pytest.raises(ConfigClickError, match="already exists"):
        _service_draft(tmp_path, monkeypatch, [_row()], write=True)

    assert "a question a human wrote" in path.read_text(encoding="utf-8")

    # --force is the deliberate escape hatch.
    _service_draft(tmp_path, monkeypatch, [_row()], write=True, force=True)
    assert "a question a human wrote" not in path.read_text(encoding="utf-8")


def test_an_output_outside_the_project_is_refused(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A promoted golden set is a project file that gets committed and read by
    a transform; one written elsewhere is something nothing will load."""
    from stel.cli_services import promote as service
    from stel.config.loader import ConfigError

    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "elsewhere" / "golden.yml"
    monkeypatch.setattr(service, "read_candidates", lambda *a, **k: [_row()])

    with pytest.raises(ConfigError, match="outside the project"):
        service.promote_from_candidates(
            project,
            profiles_dir=None,
            target=None,
            relation="analytics.candidates",
            output=outside,
            promoted_by="alex",
            today=date(2026, 9, 2),
            write=True,
        )

    assert not outside.exists()


@pytest.mark.skipif(
    not hasattr(__import__("os"), "symlink"), reason="platform has no symlinks"
)
def test_a_symlinked_output_is_refused_before_anything_is_written(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """A dangling link is the sharp case: `exists()` reports false, so the
    overwrite guard sees a free path and writes straight through the link.
    The loader refuses to read a symlinked golden set, so writing one would
    also produce a file nothing loads (PR #451 review)."""
    import os

    from stel.cli_services import promote as service
    from stel.cli_services.context import ConfigClickError

    project = tmp_path / "project"
    (project / "golden_sets").mkdir(parents=True)
    victim = tmp_path / "victim.yml"
    link = project / "golden_sets" / "search.yml"
    try:
        os.symlink(victim, link)
    except OSError as error:  # Windows without the privilege
        pytest.skip(f"cannot create symlink: {error}")

    monkeypatch.setattr(service, "read_candidates", lambda *a, **k: [_row()])
    with pytest.raises(ConfigClickError, match="symlink"):
        service.promote_from_candidates(
            project,
            profiles_dir=None,
            target=None,
            relation="analytics.candidates",
            output=Path("golden_sets/search.yml"),
            promoted_by="alex",
            today=date(2026, 9, 2),
            write=True,
        )

    # The dangling target was never created: refused before the write, and
    # without needing --force to have been withheld.
    assert not victim.exists()
