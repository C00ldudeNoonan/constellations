"""Deterministic, governed context packing for offline experiments (ALE-283).

This is an explicit library experiment, not a change to MCP serving. Inputs
are frozen candidate/neighbor snapshots read through the existing governed
repository. Both registry and chunk policy are checked again before packing.
No provider, repository, file, or logging side effects occur here.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent_context import citation_locator
from .hashing import canonical_fingerprint
from .mcp_server.authorization import AuthorizationProvider, PolicyAttribute, Principal

_ID = re.compile(r"^[0-9a-f]{32}$")


class AssemblyError(ValueError):
    """Safe diagnostic: never contains document, policy, or provider values."""


class AssemblyPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    token_budget: int = Field(ge=1)
    order: Literal["relevance", "document"] = "relevance"
    remove_overlap: bool = False
    neighbor_radius: int = Field(default=0, ge=0, le=2)
    max_neighbors: int = Field(default=0, ge=0, le=64)

    @model_validator(mode="after")
    def _neighbor_bounds(self) -> AssemblyPolicy:
        if bool(self.neighbor_radius) != bool(self.max_neighbors):
            raise ValueError("neighbor_radius and max_neighbors must both be zero or positive")
        return self

    def identity(self, tokenizer_identity: str) -> str:
        if not tokenizer_identity.strip():
            raise AssemblyError("A versioned tokenizer identity is required")
        return canonical_fingerprint(
            {"policy": self.model_dump(), "tokenizer": tokenizer_identity, "renderer": 1},
            domain="stel-context-assembly-policy",
        )


@dataclass(frozen=True, repr=False)
class Evidence:
    """A snapshot of the two rows the serving policy authorizes as a pair.

    Copy rows internally so caller mutations cannot change the
    text, identity, or authorization data between arms of an experiment.
    """

    _registry: dict[str, Any]
    _chunk: dict[str, Any]

    @classmethod
    def from_rows(cls, registry: Mapping[str, Any], chunk: Mapping[str, Any]) -> Evidence:
        for name in ("document_id", "document_version_id", "context_id", "chunk_id"):
            value = chunk.get(name)
            if not isinstance(value, str) or not _ID.fullmatch(value):
                raise AssemblyError("Evidence requires stable agent-context IDs")
        if any(registry.get(key) != chunk[key] for key in ("document_id", "document_version_id")):
            raise AssemblyError("Evidence has an unresolved document version")
        index, text = chunk.get("chunk_index"), chunk.get("text")
        if type(index) is not int or index < 0 or not isinstance(text, str) or not text:
            raise AssemblyError("Evidence requires an index and non-empty text")
        start, end = chunk.get("citation_char_start"), chunk.get("citation_char_end")
        if (start is not None or end is not None) and (
            type(start) is not int or type(end) is not int or start < 0 or end - start != len(text)
        ):
            raise AssemblyError("Evidence character offsets must describe its exact text")
        # This is the boundary for repository rows, which may carry datetime
        # fields but must never turn a parser diagnostic into document output.
        try:
            locator = citation_locator(chunk)
            if not locator:
                raise AssemblyError("Evidence requires a citation locator")
            _json(locator)
            return cls(deepcopy(dict(registry)), deepcopy(dict(chunk)))
        except (TypeError, ValueError):
            raise AssemblyError("Evidence contains invalid citation or snapshot data") from None

    def rows(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return deepcopy(self._registry), deepcopy(self._chunk)


@dataclass(frozen=True, repr=False)
class PackedEvidence:
    context_id: str
    document_id: str
    document_version_id: str
    chunk_id: str
    chunk_index: int
    text: str
    citation: Mapping[str, Any]
    expanded: bool

    def payload(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "document_id": self.document_id,
            "document_version_id": self.document_version_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "citation": dict(self.citation),
            "text": self.text,
        }


@dataclass(frozen=True)
class Omission:
    context_id: str
    reason: Literal["budget", "overlap"]
    covered_by: tuple[str, ...] = ()


@dataclass(frozen=True, repr=False)
class AssembledContext:
    policy_id: str
    snapshot_id: str
    prompt_id: str
    token_count: int
    metadata_tokens: int
    text_tokens: int
    evidence: tuple[PackedEvidence, ...]
    omissions: tuple[Omission, ...]
    _prompt: str = field(repr=False)

    def render(self) -> str:
        """Return sensitive model input; do not write it to logs or artifacts."""
        return self._prompt

    def artifact(self) -> dict[str, Any]:
        """A deliberate allowlist: IDs and counts, never text or locators."""
        return {
            "schema_version": "context_assembly/v1",
            "policy_id": self.policy_id,
            "snapshot_id": self.snapshot_id,
            "prompt_id": self.prompt_id,
            "token_count": self.token_count,
            "metadata_tokens": self.metadata_tokens,
            "text_tokens": self.text_tokens,
            "context_ids": [item.context_id for item in self.evidence],
            "expanded_ids": [item.context_id for item in self.evidence if item.expanded],
            "omissions": [
                {"context_id": item.context_id, "reason": item.reason,
                 "covered_by": list(item.covered_by)} for item in self.omissions
            ],
        }


def assemble_context(
    candidates: Sequence[Evidence],
    *,
    neighbors: Sequence[Evidence],
    policy: AssemblyPolicy,
    principal: Principal,
    authorization: AuthorizationProvider,
    policy_attributes: Sequence[PolicyAttribute],
    tokenizer_identity: str,
    count_tokens: Callable[[str], int],
    prefix: str,
    suffix: str,
) -> AssembledContext:
    """Pack whole chunks greedily, skipping oversized chunks without truncation.

    The budget counts ``prefix + JSON evidence array + suffix`` in one call.
    Caller-supplied tokenizer identity must include implementation/version.
    Neighbor snapshots are bounded to 1000 rows and expansion to 64 chunks;
    neighbors outside a root's document version and index radius never enter
    a prompt. Disallowed selected evidence fails the whole assembly.
    """
    policy_id = policy.identity(tokenizer_identity)
    if len(candidates) > 1000 or len(neighbors) > 1000:
        raise AssemblyError("Evidence snapshot exceeds the 1000-row limit")
    selected = _select(candidates, neighbors, policy)
    # Authorize before constructing a prompt or invoking even a token counter.
    for item, _ in selected:
        _authorize(item, principal, authorization, policy_attributes)
    snapshot_id = canonical_fingerprint(
        {"candidates": [item.rows() for item in candidates],
         "neighbors": sorted((item.rows() for item in neighbors), key=_row_key)},
        domain="stel-context-assembly-snapshot",
    )
    empty_prompt = _render((), prefix, suffix)
    if _count(count_tokens, empty_prompt) > policy.token_budget:
        raise AssemblyError("Token budget cannot fit the prompt envelope")
    packed: list[PackedEvidence] = []
    omissions: list[Omission] = []
    for item, expanded in selected:
        block = _block(item, expanded)
        block, covered_by = _remove_overlap(block, packed) if policy.remove_overlap else (block, ())
        if not block.text:
            omissions.append(Omission(block.context_id, "overlap", covered_by))
        elif _count(count_tokens, _render([*packed, block], prefix, suffix)) <= policy.token_budget:
            packed.append(block)
        else:
            omissions.append(Omission(block.context_id, "budget"))
    prompt = _render(packed, prefix, suffix)
    metadata = [replace(item, text="") for item in packed]
    return AssembledContext(
        policy_id=policy_id, snapshot_id=snapshot_id,
        prompt_id=canonical_fingerprint(prompt, domain="stel-context-assembly-prompt"),
        token_count=_count(count_tokens, prompt),
        metadata_tokens=_count(count_tokens, _render(metadata, prefix, suffix)),
        text_tokens=sum(_count(count_tokens, item.text) for item in packed),
        evidence=tuple(packed), omissions=tuple(omissions), _prompt=prompt,
    )


def _row_key(rows: tuple[dict[str, Any], dict[str, Any]]) -> str:
    return rows[1]["context_id"]


def _select(
    candidates: Sequence[Evidence], neighbors: Sequence[Evidence], policy: AssemblyPolicy,
) -> list[tuple[Evidence, bool]]:
    roots = [(item, False) for item in candidates]
    root_rows = [item.rows()[1] for item in candidates]
    seen = {row["context_id"] for row in root_rows}
    if len(seen) != len(root_rows):
        raise AssemblyError("Retrieved candidate IDs must be unique")
    expanded: list[tuple[Evidence, bool]] = []
    by_id = {item.rows()[1]["context_id"]: item for item in candidates}
    for item in neighbors:
        key = item.rows()[1]["context_id"]
        if key in by_id and by_id[key].rows() != item.rows():
            raise AssemblyError("Evidence snapshot has conflicting rows for one context ID")
        by_id[key] = item
    pool = sorted(neighbors, key=_document_key)
    for item in pool:
        row = item.rows()[1]
        if row["context_id"] in seen or not _is_neighbor(row, root_rows, policy.neighbor_radius):
            continue
        if len(expanded) >= policy.max_neighbors:
            break
        seen.add(row["context_id"])
        expanded.append((item, True))
    selected = roots + expanded
    if policy.order == "document":
        selected.sort(key=lambda pair: _document_key(pair[0]))
    return selected


def _document_key(item: Evidence) -> tuple[str, str, int, str]:
    row = item.rows()[1]
    return row["document_id"], row["document_version_id"], row["chunk_index"], row["context_id"]


def _is_neighbor(row: Mapping[str, Any], roots: Sequence[Mapping[str, Any]], radius: int) -> bool:
    return radius > 0 and any(
        row["document_id"] == root["document_id"]
        and row["document_version_id"] == root["document_version_id"]
        and 0 < abs(row["chunk_index"] - root["chunk_index"]) <= radius
        for root in roots
    )


def _authorize(
    item: Evidence, principal: Principal, authorization: AuthorizationProvider,
    attributes: Sequence[PolicyAttribute],
) -> None:
    registry, chunk = item.rows()
    try:
        permitted = authorization.can_read(principal, registry, attributes=attributes)
        permitted = permitted and authorization.can_read(principal, chunk, attributes=attributes)
    except Exception:
        raise AssemblyError("Evidence authorization failed") from None
    if not permitted:
        raise AssemblyError("Evidence is unavailable or denied")


def _block(item: Evidence, expanded: bool) -> PackedEvidence:
    _, row = item.rows()
    return PackedEvidence(
        context_id=row["context_id"], document_id=row["document_id"],
        document_version_id=row["document_version_id"], chunk_id=row["chunk_id"],
        chunk_index=row["chunk_index"], text=row["text"],
        citation=citation_locator(row), expanded=expanded,
    )


def _remove_overlap(
    block: PackedEvidence, packed: Sequence[PackedEvidence],
) -> tuple[PackedEvidence, tuple[str, ...]]:
    """Trim only a proven identical prefix at the same version/character offsets.

    Text similarity or adjacent chunk numbers cannot prove overlap. Preserve
    the original chunk/context IDs and advance the locator of a trimmed span.
    Already packed text remains the citation target for a fully covered chunk.
    """
    start, end = block.citation.get("char_start"), block.citation.get("char_end")
    if start is None or end is None:
        return block, ()
    covered_by: list[str] = []
    cursor = start
    for prior in sorted(packed, key=lambda item: item.citation.get("char_start", -1)):
        if (prior.document_id, prior.document_version_id) != (
            block.document_id, block.document_version_id
        ):
            continue
        left, right = prior.citation.get("char_start"), prior.citation.get("char_end")
        if left is None or right is None or not left <= cursor < right:
            continue
        stop = min(end, right)
        if prior.text[cursor - left:stop - left] == block.text[cursor - start:stop - start]:
            cursor = stop
            covered_by.append(prior.context_id)
    if cursor == start:
        return block, ()
    return replace(
        block, text=block.text[cursor - start:],
        citation={**block.citation, "char_start": cursor},
    ), tuple(covered_by)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _render(evidence: Sequence[PackedEvidence], prefix: str, suffix: str) -> str:
    return prefix + _json([item.payload() for item in evidence]) + suffix


def _count(counter: Callable[[str], int], value: str) -> int:
    try:
        result = counter(value)
    except Exception:
        raise AssemblyError("Token counting failed") from None
    if type(result) is not int or result < 0:
        raise AssemblyError("Token counter must return a non-negative integer")
    return result
