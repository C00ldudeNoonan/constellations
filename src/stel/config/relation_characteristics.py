"""Relation characteristics: `inverse_of`, `symmetric` and `transitive` (issue #775).

A declared relation has a name and class pairings; nothing said that `owns` and
`owned_by` state one fact, so two models asserting each produced two row types
for it. A characteristic is a property of the relation *name*, like
`functional` (ADR-0056), so every pairing of one name must agree on it.

This module is pure: it checks the declaration and derives each relation's
canonical orientation. The extraction driver applies it to rows.

* `inverse_of: B` on `A` says `A(x, y)` holds exactly when `B(y, x)` does. One
  side may declare it; the other is inferred. The first-declared name of the
  pair is canonical, so reformatting never changes which type a row carries.
* `symmetric: true` says `A(x, y)` holds exactly when `A(y, x)` does.
* `transitive: true` is metadata only: no closure is computed (a downstream
  graph or traversal can use it).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .vocabulary import RelationTypeDef

__all__ = [
    "RelationOrientation",
    "canonical_orientations",
    "check_relation_characteristics",
    "has_characteristics",
    "orientation_identity",
    "symmetric_names",
]


@dataclass(frozen=True)
class RelationOrientation:
    """Where an asserted relation type lands: its canonical name and whether
    subject and object swap to get there."""

    canonical: str
    swap: bool


def has_characteristics(relations: Sequence[RelationTypeDef]) -> bool:
    """True when any declared relation is an inverse or symmetric, i.e. when
    rows are re-oriented and so record the type that was asserted."""
    return any(relation.inverse_of is not None or relation.symmetric for relation in relations)


def symmetric_names(relations: Sequence[RelationTypeDef]) -> frozenset[str]:
    return frozenset(relation.name for relation in relations if relation.symmetric)


def canonical_orientations(
    relations: Sequence[RelationTypeDef],
) -> dict[str, RelationOrientation]:
    """Each inverse relation's orientation, keyed by asserted name.

    Both halves of a pair appear: the canonical one maps to itself with no
    swap, so a lookup answers for any name in a pair. A relation with no
    inverse is absent. Assumes `check_relation_characteristics` passed.
    """
    partner = _partners(relations)
    first_seen = _declaration_order(relations)
    orientations: dict[str, RelationOrientation] = {}
    for name, other in partner.items():
        canonical = min(name, other, key=first_seen.__getitem__)
        orientations[name] = RelationOrientation(canonical=canonical, swap=name != canonical)
    return orientations


def orientation_identity(
    relations: Sequence[RelationTypeDef],
) -> dict[str, object] | None:
    """What decides how rows are oriented, for a model's code identity.

    `None` when nothing is re-oriented, so a project that declares no inverse or
    symmetric relation keeps the identity it had. `transitive` is metadata that
    changes no row, so it is not part of it.
    """
    if not has_characteristics(relations):
        return None
    return {
        "inverse": {
            name: [orientation.canonical, orientation.swap]
            for name, orientation in sorted(canonical_orientations(relations).items())
        },
        "symmetric": sorted(symmetric_names(relations)),
    }


def check_relation_characteristics(relations: Sequence[RelationTypeDef]) -> None:
    """Raise ValueError when the characteristics are inconsistent."""
    by_name: dict[str, list[RelationTypeDef]] = {}
    for relation in relations:
        by_name.setdefault(relation.name, []).append(relation)
    for name, pairings in by_name.items():
        _require_agreement(name, pairings)
    partner = _partners(relations)
    symmetric = symmetric_names(relations)
    for name in sorted(symmetric | partner.keys()):
        if any(pairing.functional for pairing in by_name[name]):
            # Supersession keys a fact on its subject; re-orienting the row would
            # key it on the other end, so the two cannot both hold.
            raise ValueError(
                f"relation '{name}' is `functional` and also symmetric or an inverse; "
                "canonical orientation can swap its subject and object, so "
                "supersession could not key it. Declare `functional` on a relation "
                "with no inverse"
            )
    for name, other in partner.items():
        if name in symmetric:
            raise ValueError(
                f"relation '{name}' is both `symmetric` and an inverse of '{other}'; "
                "a symmetric relation is its own inverse, so declare one or the other"
            )
        if _flag(by_name[name], "transitive") != _flag(by_name[other], "transitive"):
            raise ValueError(
                f"relation '{name}' and its inverse '{other}' disagree on `transitive`; "
                "an inverse of a transitive relation is transitive, so declare both"
            )
        _require_mirrored(name, other, by_name)
    for name in symmetric:
        _require_mirrored(name, name, by_name)


def _flag(pairings: list[RelationTypeDef], field: str) -> bool:
    return bool(getattr(pairings[0], field))


def _require_agreement(name: str, pairings: list[RelationTypeDef]) -> None:
    for field in ("inverse_of", "symmetric", "transitive"):
        if len({getattr(pairing, field) for pairing in pairings}) > 1:
            raise ValueError(
                f"relation '{name}' declares `{field}` on one pairing and not on "
                f"another; `{field}` belongs to the relation type, so every pairing "
                "must agree"
            )


def _partners(relations: Sequence[RelationTypeDef]) -> dict[str, str]:
    """name -> its inverse's name, both directions, declared on either side.

    Raises when an inverse names no declared relation, names itself, or two
    declarations claim different partners.
    """
    declared = {relation.name for relation in relations}
    partner: dict[str, str] = {}
    for relation in relations:
        other = relation.inverse_of
        if other is None:
            continue
        if other == relation.name:
            raise ValueError(
                f"relation '{relation.name}' declares itself as its `inverse_of`; "
                "use `symmetric: true` for a relation that is its own inverse"
            )
        if other not in declared:
            raise ValueError(
                f"relation '{relation.name}' declares `inverse_of: {other}`, which is "
                f"not declared under `relations:`. Declared: {sorted(declared)}"
            )
        for left, right in ((relation.name, other), (other, relation.name)):
            if partner.setdefault(left, right) != right:
                raise ValueError(
                    f"relation '{left}' is an inverse of both '{partner[left]}' and "
                    f"'{right}'; a relation has at most one inverse"
                )
    return partner


def _require_mirrored(
    name: str, other: str, by_name: dict[str, list[RelationTypeDef]]
) -> None:
    """Every (subject, object) pairing of `name` needs (object, subject) under
    `other`: the mirror is what makes the two assertions the same fact."""
    mirrored = {(pairing.object_class, pairing.subject_class) for pairing in by_name[other]}
    declared = {(pairing.subject_class, pairing.object_class) for pairing in by_name[name]}
    missing = sorted(declared - mirrored)
    if missing:
        kind = "symmetric relation" if name == other else f"inverse of '{other}'"
        pairs = ", ".join(f"{subject} -> {obj}" for subject, obj in missing)
        raise ValueError(
            f"relation '{name}' ({kind}) declares pairing(s) {pairs} whose mirror "
            f"(object -> subject) is not declared under '{other}'; the two pairings "
            "must be mirrors of each other"
        )


def _declaration_order(relations: Sequence[RelationTypeDef]) -> dict[str, int]:
    order: dict[str, int] = {}
    for relation in relations:
        order.setdefault(relation.name, len(order))
    return order
