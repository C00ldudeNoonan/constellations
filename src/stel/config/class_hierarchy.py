"""Validated single-parent entity classes and their ancestry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from .identifiers import validate_node_name


class ClassDef(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str
    subclass_of: str

    @field_validator("name", "subclass_of")
    @classmethod
    def _name(cls, value: str) -> str:
        value = value.strip()
        validate_node_name(value, kind="Class")
        return value


ClassDeclaration = str | ClassDef


def class_name(value: ClassDeclaration) -> str:
    return value if isinstance(value, str) else value.name


def class_names(classes: Sequence[ClassDeclaration]) -> tuple[str, ...]:
    return tuple(class_name(value) for value in classes)


def class_parents(classes: Sequence[ClassDeclaration]) -> dict[str, str]:
    return {value.name: value.subclass_of for value in classes if isinstance(value, ClassDef)}


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
