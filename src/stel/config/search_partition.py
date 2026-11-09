"""A segmented search group and the partition each member publishes (ADR-0037).

A member declares the rows it owns on its own `search:` block, so `for_each` can
generate members and the partition rides in the same config that every
fingerprint, descriptor and manifest already hashes. The group lives in the
project file and only names its members: a tag or shared name never makes one.

Predicates are conjunctive, so a partition is one of two shapes, each a plain
AND the read seam can express: a half-open range `[lower, upper)`, either end
open, or the null member `nulls: true`. The nulls a column holds and the values
beyond the finite bounds are accounted for by the null member and by the open
ends, which is how a group covers its whole domain without a disjunction.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .identifiers import validate_node_name

PartitionType = Literal["date", "integer", "string"]
PartitionBound = date | int | str


class SearchPartitionConfig(BaseModel):
    """The slice of the upstream relation one group member publishes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str = Field(min_length=1)
    type: PartitionType
    # Inclusive lower and exclusive upper bound. Omitting one leaves that end
    # open, which is what makes the first and last members absorb every value
    # beyond the finite bounds.
    lower: PartitionBound | None = None
    upper: PartitionBound | None = None
    # The member that owns rows whose partition field is null. A comparison
    # against null is never true, so no range member can claim them.
    nulls: bool = False

    @model_validator(mode="before")
    @classmethod
    def _coerce_bounds(cls, data: Any) -> Any:
        # A `for_each` template can only hold a bound as a string, and YAML may
        # hand a date as either a date or a string; the declared type decides.
        if not isinstance(data, dict):
            return data
        kind = data.get("type")
        coerced = dict(data)
        for key in ("lower", "upper"):
            value = coerced.get(key)
            if value is not None:
                coerced[key] = coerce_bound(value, kind=kind, key=key)
        return coerced

    @model_validator(mode="after")
    def _validate_shape(self) -> SearchPartitionConfig:
        if self.nulls:
            if self.lower is not None or self.upper is not None:
                raise ValueError(
                    "search.partition `nulls: true` owns the rows whose field is null and "
                    "takes no bounds; declare a range member for the rest"
                )
            return self
        if (
            self.lower is not None
            and self.upper is not None
            and not bound_sort_key(self.lower) < bound_sort_key(self.upper)
        ):
            raise ValueError(
                f"search.partition lower ({self.lower}) must be below upper ({self.upper}); "
                "the range is [lower, upper)"
            )
        return self

    def identity(self) -> dict[str, Any]:
        """The partition as JSON, the shape that goes into a descriptor."""
        return {
            "field": self.field,
            "type": self.type,
            "lower": _bound_json(self.lower),
            "upper": _bound_json(self.upper),
            "nulls": self.nulls,
        }


def _bound_json(value: PartitionBound | None) -> str | int | None:
    return value.isoformat() if isinstance(value, date) else value


def bound_sort_key(value: PartitionBound) -> tuple[int, int, str]:
    """An orderable key for a bound; one group never mixes types, so the type
    rank only keeps the comparison total for a mixed (and rejected) pair."""
    if isinstance(value, date):
        return (0, value.toordinal(), "")
    if isinstance(value, int):
        return (1, value, "")
    return (2, 0, value)


def coerce_bound(value: Any, *, kind: Any, key: str) -> PartitionBound:
    if kind == "date":
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                pass
        raise ValueError(f"search.partition {key} must be an ISO date for `type: date`")
    if kind == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
        raise ValueError(f"search.partition {key} must be an integer for `type: integer`")
    if kind == "string":
        if isinstance(value, str):
            return value
        raise ValueError(f"search.partition {key} must be a string for `type: string`")
    return value


class SearchGroupConfig(BaseModel):
    """One logical serving name over independently published members."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    members: tuple[str, ...] = Field(min_length=2)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return validate_node_name(value, kind="search group", reserve_internal=True)

    @field_validator("members")
    @classmethod
    def _validate_members(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("search group members must not repeat a model")
        return value
