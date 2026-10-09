"""Declared domain vocabularies (issue #625).

A `type: enum` field's label set lived inline in `values:`, so two models
classifying into the same set repeated it, and nothing noticed when the two
copies drifted apart — the problem #304 solved for one field's three
consumers (the provider schema, the `accepted_values` test, the prompt
fallback), one level up. A `vocabularies:` block in `stel_project.yml`
declares a label set once; a field points at it with `values_from:
vocab.<name>` instead of listing `values:` inline.

Deliberately SKOS-level, not OWL: a term is a preferred label, an optional
definition, optional alternative labels, and an optional broader term. No
reasoner, no class/relation layer — those are separate, larger follow-ups
(issues #626-#629) that build on this declaration rather than this module
growing to anticipate them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# `values_from: vocab.<name>` — the only source kind today. Prefixed rather
# than a bare name so a later `values_from:` source (e.g. a warehouse column)
# can be added without a breaking re-parse of existing project files.
VALUES_FROM_PATTERN = re.compile(r"^vocab\.([A-Za-z_]\w*)$")


class VocabularyTerm(BaseModel):
    """One concept in a vocabulary: a preferred label, plus SKOS-style extras.

    `broader` names another term's `label` in the *same* vocabulary (SKOS
    `broader`); `Vocabulary` validates that it resolves and that no chain of
    `broader` references cycles back on itself.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    label: str
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    broader: str | None = None
    # The entity class this term is an instance of (issue #629). Authored as
    # `class:`; a keyword in Python, so the attribute is `entity_class`. Must
    # name a class declared under `classes:` (checked by ProjectConfig).
    entity_class: str | None = Field(default=None, alias="class")

    @field_validator("label")
    @classmethod
    def _validate_label(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("vocabulary term label must not be empty")
        return v

    @field_validator("entity_class")
    @classmethod
    def _validate_entity_class(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.strip():
            raise ValueError("vocabulary term class must not be empty")
        return v.strip()

    @field_validator("aliases")
    @classmethod
    def _validate_aliases(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        for alias in v:
            if not alias.strip():
                raise ValueError("vocabulary term alias must not be empty")
            if alias in seen:
                raise ValueError(f"vocabulary term alias '{alias}' listed twice")
            seen.add(alias)
        return v


class Vocabulary(BaseModel):
    """One closed, ordered label set, declared once under `vocabularies:`.

    Term order is preserved end to end: it is what an enum field's provider
    schema, `accepted_values` test, and prompt fallback all see as `values`.
    """

    model_config = ConfigDict(extra="forbid")

    terms: list[VocabularyTerm] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_terms(self) -> Vocabulary:
        seen: set[str] = set()
        for term in self.terms:
            if term.label in seen:
                raise ValueError(f"vocabulary term label '{term.label}' declared twice")
            seen.add(term.label)
        by_label = {term.label: term for term in self.terms}
        for term in self.terms:
            if term.broader is None:
                continue
            if term.broader == term.label:
                raise ValueError(
                    f"vocabulary term '{term.label}' declares itself as its own "
                    "`broader` term"
                )
            if term.broader not in by_label:
                raise ValueError(
                    f"vocabulary term '{term.label}' declares `broader: "
                    f"{term.broader}`, which is not a term in this vocabulary"
                )
        self._reject_broader_cycles(by_label)
        return self

    @staticmethod
    def _reject_broader_cycles(by_label: dict[str, VocabularyTerm]) -> None:
        for start in by_label:
            seen: set[str] = set()
            current: str | None = start
            while current is not None:
                if current in seen:
                    raise ValueError(
                        f"vocabulary term '{start}' has a cyclic `broader` chain"
                    )
                seen.add(current)
                current = by_label[current].broader

    def labels(self) -> list[str]:
        """Preferred labels in declared order — an enum field's effective `values:`.

        Alternative labels (`aliases`) never appear here: they widen what an
        entity linker matches (issue #627), not what a classifier may output.
        """
        return [term.label for term in self.terms]

    def descriptions(self) -> dict[str, str]:
        """Preferred label -> definition, for terms that declare one."""
        return {
            term.label: term.description for term in self.terms if term.description
        }

    def labels_in_class(self, entity_class: str) -> list[str]:
        """Preferred labels whose term declares `entity_class`, in declared order.

        Exact match on the class name, which `ProjectConfig` has already
        checked against `classes:` for every term -- so an unknown class here
        yields nothing rather than an error, and the caller decides whether
        asking for an undeclared class is a mistake or an empty answer.
        """
        return [term.label for term in self.terms if term.entity_class == entity_class]

    def broader_chain(self, label: str) -> list[str]:
        """The labels above `label`, nearest first, excluding `label` itself.

        Terminates because `broader` cycles are rejected at validation; the
        `seen` guard is kept anyway so a future construction path that skips
        validation cannot hang a server (this walks on a request).
        """
        by_label = {term.label: term for term in self.terms}
        if label not in by_label:
            return []
        out: list[str] = []
        seen = {label}
        current = by_label[label].broader
        while current is not None and current not in seen:
            out.append(current)
            seen.add(current)
            current = by_label[current].broader if current in by_label else None
        return out

    def narrower_labels(self, label: str) -> list[str]:
        """Every label under `label`, transitively, in declared order.

        Derived from `broader` rather than declared separately: one direction
        is the single source of truth, so a term cannot be its parent's child
        without its parent being its parent. Breadth-first, so nearer terms
        come first, and `label` itself is never included.
        """
        children: dict[str, list[str]] = {}
        for term in self.terms:
            if term.broader is not None:
                children.setdefault(term.broader, []).append(term.label)
        out: list[str] = []
        seen = {label}
        frontier = list(children.get(label, ()))
        while frontier:
            current = frontier.pop(0)
            if current in seen:
                continue
            seen.add(current)
            out.append(current)
            frontier.extend(children.get(current, ()))
        return out


def declared_terms(
    vocabularies: Mapping[str, Vocabulary],
) -> dict[tuple[str, str], VocabularyTerm]:
    """Every declared term keyed by (vocabulary name, term label).

    That pair is how a `link_entities` output row names a declaration: the row
    carries the vocabulary's name as its `entity_namespace` (issue #627) and
    the term's label as the canonical id it linked to. Written inline for the
    concept cloud first (issue #661) and lifted here when the MCP server
    needed the identical lookup -- two resolvers disagreeing about what a link
    row means is the drift worth one shared function.

    A namespace that names no declared vocabulary simply contributes no keys,
    which is what makes the lookup safe on rows a fuzzy or hand-maintained
    resolver produced: those carry an `entity_namespace` too, and it is not a
    declaration (issue #661).
    """
    return {
        (vocab_name, term.label): term
        for vocab_name, vocabulary in vocabularies.items()
        for term in vocabulary.terms
    }


class RelationTypeDef(BaseModel):
    """One allowed (subject_class, object_class) pairing for a named relation
    type (issue #626) — a domain/range constraint for `RelationRule.relation_type`
    and `ModelAssertionExtractorOptions.relation_types` to be checked against.

    The same `name` may repeat with a different pairing: a relation can be
    polymorphic, e.g. `located_in` holding between both (company, country) and
    (person, country), so declared pairings are a set per name, not one each.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    subject_class: str
    object_class: str

    @field_validator("name", "subject_class", "object_class")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        normalized = v.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized
