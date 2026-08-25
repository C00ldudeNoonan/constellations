"""The promoted golden-set file (issue #380, #329 phase 3).

Promotion is the one genuinely new mechanism phase 3 needs, and #380 settled
its form: **a reviewed file in the project, not a warehouse write.** A
promotion is a human judgement, and a human judgement wants git review, blame,
and revert. A command that wrote rows into the warehouse would put the
reviewable artifact in the place nobody opens and turn "who promoted this, and
why" into a query instead of a diff.

So this file is the artifact. `stel.promotion.golden_set` materializes it into
the ordinary relation `retrieval_tests.golden_set` already refs — the evals
need no changes at all, which is what #329 predicted and what makes promotion
a thin step rather than a new subsystem.

Every promoted query names the sessions it came from. That is not decoration:
the first question a reviewer asks is "where did this come from?", and a
promoted golden that cannot answer it is indistinguishable from one somebody
invented. A golden set written by hand from scratch is a perfectly good
model — it just is not a *promotion*, and does not belong in this file.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

GOLDEN_SET_VERSION = 1


class PromotionError(Exception):
    """A promoted golden-set file that cannot be trusted as written."""


class PromotionEvidence(BaseModel):
    """Where a promoted query came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sessions: tuple[str, ...]
    # Joins the promoted query back to the candidate rows and to the MCP query
    # log; absent for a query whose text a reviewer rewrote from scratch.
    query_fingerprint: str | None = None
    harness: str | None = None

    @model_validator(mode="after")
    def _require_sessions(self) -> PromotionEvidence:
        if not self.sessions:
            raise ValueError(
                "evidence.sessions must name at least one session; a promoted "
                "row that cannot say where it came from is not a promotion"
            )
        return self


class PromotedQuery(BaseModel):
    """One reviewed golden query.

    `query_text` is required and human-owned. The corpus records only a query
    *fingerprint* unless the operator opted into capturing text, and a golden
    set has to be re-runnable — `retrieval_tests` replays each query through
    `search()`. Making the reviewer supply or confirm the text is therefore
    not friction, it is the step that turns an observation into a test
    (issue #380, constraint 2).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    relevant_ids: tuple[str, ...] = ()
    required_ids: tuple[str, ...] = ()
    excluded_ids: tuple[str, ...] = ()
    mode: Literal["vector", "text", "hybrid"] | None = None
    promoted_by: str = Field(min_length=1)
    promoted_at: date
    evidence: PromotionEvidence

    @model_validator(mode="after")
    def _must_assert_something(self) -> PromotedQuery:
        if not (self.relevant_ids or self.required_ids or self.excluded_ids):
            raise ValueError(
                f"promoted query '{self.query_id}' asserts nothing: give it "
                "relevant_ids, required_ids, or excluded_ids"
            )
        return self


class GoldenSetFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = GOLDEN_SET_VERSION
    # The id space every id below is expressed in. A search index keys on its
    # own `id_field`, and `search_context` results carry both a `context_id`
    # and a `chunk_id` — so a set promoted in the wrong space matches nothing
    # and scores a silent zero recall. Declared here and checked against the
    # target index (issue #380, constraint 3).
    id_space: str = Field(min_length=1)
    queries: tuple[PromotedQuery, ...]

    @model_validator(mode="after")
    def _unique_query_ids(self) -> GoldenSetFile:
        seen: set[str] = set()
        for query in self.queries:
            if query.query_id in seen:
                raise ValueError(f"duplicate query_id '{query.query_id}'")
            seen.add(query.query_id)
        return self


def load_golden_set(path: Path) -> GoldenSetFile:
    """Read and validate a promoted golden-set file."""
    if path.is_symlink() or not path.is_file():
        raise PromotionError(
            f"Promoted golden set must be a regular file: {path}"
        )
    text = path.read_text(encoding="utf-8")
    try:
        document: Any = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise PromotionError(f"{path} is not valid YAML: {error}") from None
    if not isinstance(document, dict):
        raise PromotionError(f"{path} must contain a YAML mapping")
    try:
        return GoldenSetFile.model_validate(document)
    except ValidationError as error:
        raise PromotionError(f"{path} is not a valid golden set: {error}") from None
