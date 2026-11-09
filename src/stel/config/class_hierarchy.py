"""Validated single-parent entity classes and their ancestry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)

from .identifiers import validate_node_name
from .ontology import validate_iri


class ClassDef(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str
    # Optional so a root class can still carry an `iri:`; a ClassDef with
    # neither behaves exactly like the bare name.
    subclass_of: str | None = None
    # The owl:Class IRI when the class was imported from an ontology (#767).
    iri: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        value = value.strip()
        validate_node_name(value, kind="Class")
        return value

    @field_validator("subclass_of")
    @classmethod
    def _parent(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        validate_node_name(value, kind="Class")
        return value

    @field_validator("iri")
    @classmethod
    def _iri(cls, value: str | None) -> str | None:
        return None if value is None else validate_iri(value, surface="class `iri:`")

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Dumps only what was declared: `{name, subclass_of}` as before IRIs
        # existed, `{name, iri}` for an imported root class.
        return {key: value for key, value in handler(self).items() if value is not None}


ClassDeclaration = str | ClassDef


def class_name(value: ClassDeclaration) -> str:
    return value if isinstance(value, str) else value.name


def class_names(classes: Sequence[ClassDeclaration]) -> tuple[str, ...]:
    return tuple(class_name(value) for value in classes)


def class_parents(classes: Sequence[ClassDeclaration]) -> dict[str, str]:
    return {
        value.name: value.subclass_of
        for value in classes
        if isinstance(value, ClassDef) and value.subclass_of is not None
    }


def validate_classes(classes: Sequence[ClassDeclaration]) -> None:
    names = class_names(classes)
    for name in names:
        validate_node_name(name, kind="Class")
    if len(names) != len(set(names)):
        raise ValueError("classes must be unique")
    parents = class_parents(classes)
    for name, parent in parents.items():
        if parent not in names:
            raise ValueError(f"class '{name}' has undeclared parent '{parent}'")
    for start in names:
        path: list[str] = []
        current = start
        while current in parents:
            if current in path:
                cycle = [*path[path.index(current):], current]
                raise ValueError(f"class hierarchy cycle: {' -> '.join(cycle)}")
            path.append(current)
            current = parents[current]


def class_ancestors(name: str, parents: Mapping[str, str]) -> tuple[str, ...]:
    """Nearest parent first; a guard keeps even invalid input bounded."""
    result: list[str] = []
    seen = {name}
    while name in parents and parents[name] not in seen:
        name = parents[name]
        result.append(name)
        seen.add(name)
    return tuple(result)
