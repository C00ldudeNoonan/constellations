"""The `ontologies:` declaration: import a SKOS/OWL file into the domain declaration.

`vocabularies:`, `classes:` and `relations:` are otherwise written by hand, so
a standard ontology such as FIBO had to be retyped, lost its IRIs, and drifted
from upstream (issue #767). An `ontologies:` entry names a Turtle or RDF/XML
file inside the project and an allow-list of IRIs; the project loader parses
the file (`stel.ontology_import`) and merges what it finds into the same
`vocabularies:`, `classes:` and `relations:` a project could have written
inline, so every consumer downstream sees an ordinary declaration.

The allow-list is required and is the whole import: FIBO alone is thousands
of classes, and importing all of it is never what a project wants. Its order
is the order terms and classes take, so reformatting or re-serializing the
file -- RDF has no statement order -- never changes identity.

This module is the declaration only; it does no I/O and needs no RDF library.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .identifiers import validate_node_name

OntologyFormat = Literal["turtle", "xml"]

# Suffix -> rdflib parser name. `.owl` files are RDF/XML by convention
# (Protege's default save format); a Turtle file named `.owl` declares
# `format: turtle`.
_FORMAT_BY_SUFFIX: dict[str, OntologyFormat] = {
    ".ttl": "turtle",
    ".rdf": "xml",
    ".owl": "xml",
    ".xml": "xml",
}

# An absolute IRI: a scheme, a colon, and no whitespace. Deliberately loose:
# the check that matters is that the IRI is in the file, which resolution
# makes, and its error says prefixes are not expanded.
_ABSOLUTE_IRI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:\S+$")


def validate_iri(value: str, *, surface: str) -> str:
    """Strip and check an absolute IRI; `surface` names the field in the error."""
    value = value.strip()
    if not _ABSOLUTE_IRI.match(value):
        raise ValueError(
            f"{surface} {value!r} is not an absolute IRI; write the full IRI "
            "(prefixes are not expanded)"
        )
    return value


class OntologyInclude(BaseModel):
    """One allow-listed IRI, with an optional stel-side name.

    `name` overrides the derived name: a class or relation name otherwise comes
    from the IRI's local name, and a term label from its preferred label. Two
    allow-listed IRIs whose local names collide, or a local name that is not a
    valid class name, need one.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    iri: str
    name: str | None = None

    @field_validator("iri")
    @classmethod
    def _iri(cls, value: str) -> str:
        return validate_iri(value, surface="ontology `include:` entry")

    @field_validator("name")
    @classmethod
    def _name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("ontology `include:` entry `name:` must not be empty")
        return value


class OntologyImport(BaseModel):
    """One ontology file and the IRIs a project imports from it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    format: OntologyFormat | None = None
    include: tuple[OntologyInclude, ...] = Field(min_length=1)
    # The vocabulary that receives the allow-listed skos:Concept and
    # owl:NamedIndividual IRIs. Required when the allow-list names any; a
    # file that contributes only classes and relations leaves it unset.
    vocabulary: str | None = None
    # Language tag a label, alternative label or definition is read in. A
    # literal with no tag is the fallback when none carries this one.
    lang: str = "en"

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("ontology `path:` must not be empty")
        return value

    @field_validator("include", mode="before")
    @classmethod
    def _include_entries(cls, value: object) -> object:
        # A bare IRI string is the common case; normalize it here so the rest
        # of stel sees one shape.
        if isinstance(value, list | tuple):
            return tuple({"iri": item} if isinstance(item, str) else item for item in value)
        return value

    @field_validator("vocabulary")
    @classmethod
    def _vocabulary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_node_name(value.strip(), kind="Vocabulary")

    @field_validator("lang")
    @classmethod
    def _lang(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ontology `lang:` must not be empty")
        return value

    @model_validator(mode="after")
    def _validate(self) -> OntologyImport:
        if self.format is None and self.inferred_format() is None:
            raise ValueError(
                f"ontology `path: {self.path}` has no recognized suffix "
                f"({', '.join(sorted(_FORMAT_BY_SUFFIX))}); declare `format: turtle` "
                "or `format: xml`"
            )
        seen: set[str] = set()
        for entry in self.include:
            if entry.iri in seen:
                raise ValueError(f"ontology `include:` lists {entry.iri} twice")
            seen.add(entry.iri)
        return self

    def inferred_format(self) -> OntologyFormat | None:
        return _FORMAT_BY_SUFFIX.get(PurePosixPath(self.path).suffix.lower())

    def parser_format(self) -> OntologyFormat:
        resolved = self.format or self.inferred_format()
        assert resolved is not None, "validated at construction"
        return resolved
