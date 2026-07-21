from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from ..search import SearchFilter, SearchFilterOperator, SearchScalar


class AuthorizationError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class PolicyAttribute:
    name: str
    data_type: str


@dataclass(frozen=True, slots=True, repr=False)
class Principal:
    subject_id: str
    tenant_id: str | None = None
    access_groups: tuple[str, ...] = ()
    policy_claims: Mapping[str, SearchScalar | tuple[SearchScalar, ...]] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise ValueError("principal subject_id must not be empty")
        groups = tuple(sorted(set(self.access_groups)))
        if any(not group for group in groups):
            raise ValueError("principal access groups must not be empty")
        object.__setattr__(self, "access_groups", groups)
        object.__setattr__(self, "policy_claims", MappingProxyType(dict(self.policy_claims)))

    def __repr__(self) -> str:
        return "Principal(<redacted>)"


class PrincipalResolver(Protocol):
    def resolve(self) -> Principal | None: ...


@dataclass(frozen=True, slots=True)
class StaticPrincipalResolver:
    principal: Principal | None

    def resolve(self) -> Principal | None:
        return self.principal


class EnvironmentPrincipalResolver:
    """Resolve one local stdio principal from operator-owned environment state."""

    def resolve(self) -> Principal | None:
        subject_id = os.environ.get("DBT_ML_MCP_PRINCIPAL_ID", "").strip()
        if not subject_id:
            return None
        tenant_id = os.environ.get("DBT_ML_MCP_TENANT_ID")
        groups = tuple(
            value.strip()
            for value in os.environ.get("DBT_ML_MCP_ACCESS_GROUPS", "").split(",")
            if value.strip()
        )
        claims = _parse_policy_claims(os.environ.get("DBT_ML_MCP_POLICY_CLAIMS"))
        return Principal(
            subject_id=subject_id,
            tenant_id=tenant_id.strip() if tenant_id and tenant_id.strip() else None,
            access_groups=groups,
            policy_claims=claims,
        )


class AuthorizationProvider(Protocol):
    def search_policy_filters(
        self,
        principal: Principal,
        *,
        access: str,
        attributes: Sequence[PolicyAttribute],
    ) -> tuple[SearchFilter, ...]: ...

    def can_read(
        self,
        principal: Principal,
        row: Mapping[str, Any],
        *,
        attributes: Sequence[PolicyAttribute] = (),
    ) -> bool: ...


class ClaimAuthorizationProvider:
    """Minimal deterministic policy adapter for local stdio and tests.

    Production transports can inject a policy decision service without changing
    any MCP request contract. This implementation only matches trusted claims;
    it never treats a tool argument as authorization state.
    """

    def search_policy_filters(
        self,
        principal: Principal,
        *,
        access: str,
        attributes: Sequence[PolicyAttribute],
    ) -> tuple[SearchFilter, ...]:
        if access == "public":
            return ()
        filters: list[SearchFilter] = []
        for attribute in attributes:
            if attribute.data_type == "array[string]":
                raise AuthorizationError(
                    "The active retrieval store cannot compile array-valued policy claims"
                )
            value = self._claim_value(principal, attribute.name)
            if value is None or value == ():
                raise AuthorizationError(
                    "The caller has no trusted value for every required policy attribute"
                )
            operator = (
                SearchFilterOperator.IN
                if isinstance(value, tuple)
                else SearchFilterOperator.EQUAL
            )
            filters.append(SearchFilter(attribute.name, operator, value))
        if not filters:
            raise AuthorizationError(
                "The governed context model has no enforceable policy attributes"
            )
        return tuple(filters)

    def can_read(
        self,
        principal: Principal,
        row: Mapping[str, Any],
        *,
        attributes: Sequence[PolicyAttribute] = (),
    ) -> bool:
        if row.get("authorization_resolved") is not True:
            return False
        if row.get("is_public") is True:
            return True
        matched = False
        tenant_id = row.get("tenant_id")
        if isinstance(tenant_id, str) and tenant_id:
            if principal.tenant_id != tenant_id:
                return False
            matched = True
        row_groups = row.get("access_groups")
        groups = {
            value
            for value in row_groups
            if isinstance(value, str) and value
        } if isinstance(row_groups, list | tuple) else set()
        if groups:
            if not groups.intersection(principal.access_groups):
                return False
            matched = True
        for attribute in attributes:
            claim = self._claim_value(principal, attribute.name)
            row_value = row.get(attribute.name)
            if claim is None or row_value is None:
                return False
            if isinstance(claim, tuple):
                if row_value not in claim:
                    return False
            elif row_value != claim:
                return False
            matched = True
        for field, claim in principal.policy_claims.items():
            row_value = row.get(field)
            if row_value is None:
                continue
            if isinstance(claim, tuple):
                if row_value not in claim:
                    return False
            elif row_value != claim:
                return False
            matched = True
        return matched

    @staticmethod
    def _claim_value(
        principal: Principal,
        field: str,
    ) -> SearchScalar | tuple[SearchScalar, ...] | None:
        if field in principal.policy_claims:
            return principal.policy_claims[field]
        if field in {"tenant", "tenant_id"}:
            return principal.tenant_id
        if field in {"access_group", "group"}:
            return principal.access_groups
        return None


def _parse_policy_claims(
    raw: str | None,
) -> Mapping[str, SearchScalar | tuple[SearchScalar, ...]]:
    if raw is None or not raw.strip():
        return MappingProxyType({})
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        raise AuthorizationError("DBT_ML_MCP_POLICY_CLAIMS must be a JSON object") from None
    if not isinstance(decoded, dict):
        raise AuthorizationError("DBT_ML_MCP_POLICY_CLAIMS must be a JSON object")
    claims: dict[str, SearchScalar | tuple[SearchScalar, ...]] = {}
    for name, value in decoded.items():
        if not isinstance(name, str) or not name:
            raise AuthorizationError("Policy claim names must be non-empty strings")
        if isinstance(value, list):
            if not value or any(not _is_claim_scalar(item) for item in value):
                raise AuthorizationError(
                    "Policy claim arrays must contain one or more scalar values"
                )
            claims[name] = tuple(value)
        elif _is_claim_scalar(value):
            claims[name] = value
        else:
            raise AuthorizationError("Policy claims must contain only scalar values")
    return MappingProxyType(claims)


def _is_claim_scalar(value: Any) -> bool:
    return isinstance(value, str | int | float | bool) and not (
        isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")})
    )
