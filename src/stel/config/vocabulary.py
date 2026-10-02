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

    model_config = ConfigDict(extra="forbid")

    label: str
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    broader: str | None = None

    @field_validator("label")
    @classmethod
    def _validate_label(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("vocabulary term label must not be empty")
        return v

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
