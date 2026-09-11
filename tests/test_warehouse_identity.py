"""The warehouse identity a governed read executes as (issue #395, ADR-0010).

stel is the enforcement point for governed context: it compiles a policy filter
and the warehouse runs whatever it is sent, so one bug in the filter path
reaches every caller's data. Executing a caller's reads as a narrower warehouse
principal is the layer that makes the warehouse refuse instead.

Refusal and pooling semantics were pinned here against a test adapter before
either real implementation landed, which is why they are exercised abstractly
below rather than only through BigQuery. BigQuery impersonation is #568;
MotherDuck per-caller tokens are the open design question in #569.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from stel.adapters import create_adapter, parse_warehouse_config
from stel.adapters.base import (
    OPERATOR_IDENTITY,
    AdapterCapabilityError,
    ReadPredicate,
    WarehouseAdapter,
    WarehouseIdentity,
)
from stel.adapters.registry import (
    _REGISTRY,
    adapter_supports_identity_scoped_connection,
)
from stel.config.profile import WarehouseConfig
from stel.mcp_server.authorization import AuthorizationError, PolicyAttribute, Principal
from stel.mcp_server.grants import (
    WAREHOUSE_IDENTITY_ATTRIBUTE,
    Grant,
    GrantAuthorizationProvider,
    GrantConfigurationError,
    GrantWarehouseIdentityResolver,
    OperatorWarehouseIdentityResolver,
    StaticGrantStore,
    WarehouseGrantStore,
)
from stel.mcp_server.service import ContextService

_PRINCIPAL = Principal(subject_id="alice")


# ─── the identity value ─────────────────────────────────────────────────────


def test_a_blank_principal_is_refused_rather_than_read_as_the_operator() -> None:
    """`None` means the operator; `""` would mean it too, silently.

    A blank cell in a grants relation reaching this constructor must not
    produce an identity that quietly connects with full credentials.
    """
    assert WarehouseIdentity().is_operator is True
    assert WarehouseIdentity("svc@example.com").is_operator is False
    with pytest.raises(ValueError, match="must not be blank"):
        WarehouseIdentity("   ")


# ─── the adapter capability ─────────────────────────────────────────────────


def test_only_the_adapters_that_implement_the_capability_claim_it() -> None:
    """The default is inert; a claim exists only where #568/#569 landed it.

    BigQuery narrows via impersonation with no new secret (#568) and claims
    the capability. DuckDB/MotherDuck cannot yet -- #569 is the open design
    question for a per-caller token -- so it still inherits the safe default.
    """
    assert adapter_supports_identity_scoped_connection("duckdb") is False
    assert adapter_supports_identity_scoped_connection("bigquery") is True


def test_an_adapter_that_cannot_scope_refuses_rather_than_connecting_as_operator(
    tmp_path: Path,
) -> None:
    """The failure mode this whole layer exists to prevent is silent success.

    Returning the operator's connection for a caller-scoped request would
    serve every caller full credentials while the deployment believed
    otherwise, so the default `config_for_identity` refuses.
    """
    config = parse_warehouse_config(
        {"type": "duckdb", "path": str(tmp_path / "w.duckdb")}
    )
    # Positive control: the operator identity is what every path uses today.
    with create_adapter(config, identity=OPERATOR_IDENTITY) as adapter:
        assert adapter is not None

    with pytest.raises(AdapterCapabilityError, match="cannot execute reads"):
        create_adapter(config, identity=WarehouseIdentity("svc@example.com"))


def test_an_adapter_claiming_the_capability_receives_the_narrowed_config() -> None:
    """The write half of the capability: an adapter that says True must be
    handed a config carrying the principal, not the identity object."""
    seen: list[str | None] = []

    class _IdentityAwareConfig(WarehouseConfig):
        principal: str | None = None

    class _IdentityAwareAdapter(_StubAdapter):
        @classmethod
        def adapter_type(cls) -> str:
            return "identity-test"

        @classmethod
        def config_model(cls) -> type[WarehouseConfig]:
            return _IdentityAwareConfig

        @classmethod
        def supports_identity_scoped_connection(cls) -> bool:
            return True

        @classmethod
        def config_for_identity(
            cls, config: WarehouseConfig, identity: WarehouseIdentity
        ) -> WarehouseConfig:
            return config.model_copy(update={"principal": identity.principal})

        def _connect(self) -> None:
            assert isinstance(self.config, _IdentityAwareConfig)
            seen.append(self.config.principal)

    # Registered directly rather than through `register`: what is under test
    # is `create_adapter`'s identity plumbing, and satisfying all 22 abstract
    # methods of `WarehouseAdapter` would say nothing more about it. The
    # registry is process-global; `_restore_adapter_registry` in conftest
    # snapshots and restores it, which a local `finally` could not do -- a
    # pop only removes, so it would delete an entry this test replaced.
    _REGISTRY["identity-test"] = cast(type[WarehouseAdapter], _IdentityAwareAdapter)
    config = _IdentityAwareConfig(type="identity-test")
    with create_adapter(config, identity=WarehouseIdentity("svc@example.com")):
        pass
    with create_adapter(config, identity=OPERATOR_IDENTITY):
        pass

    assert adapter_supports_identity_scoped_connection("identity-test") is True
    assert seen == ["svc@example.com", None]


# `_restore_adapter_registry` (conftest) has to *restore*, not just delete what
# a test added -- a `finally` that pops its own key would leave a real adapter
# missing if the test had replaced one. These two pin that across tests, which
# is the only place the property is observable (PR #570 review).


def test_replacing_a_registered_adapter_is_contained() -> None:
    _REGISTRY["duckdb"] = cast(type[WarehouseAdapter], _StubAdapter)
    assert _REGISTRY["duckdb"] is _StubAdapter


def test_the_replaced_adapter_is_back_for_the_next_test() -> None:
    assert _REGISTRY["duckdb"] is not _StubAdapter
    assert adapter_supports_identity_scoped_connection("duckdb") is False


# ─── resolving the identity from grants ─────────────────────────────────────


def _resolver(*values: str) -> GrantWarehouseIdentityResolver:
    return GrantWarehouseIdentityResolver(
        StaticGrantStore(
            [
                Grant("alice", WAREHOUSE_IDENTITY_ATTRIBUTE, value)
                for value in values
            ]
        )
    )


def test_the_granted_identity_is_what_the_caller_reads_as() -> None:
    assert _resolver("svc-acme@example.com").identity_for(_PRINCIPAL) == (
        WarehouseIdentity("svc-acme@example.com")
    )


def test_no_granted_identity_is_a_denial_not_a_fallback() -> None:
    """There is no fallback to the operator's credentials.

    A subject that was never provisioned must not read as "unprotected"; that
    is the silent-success failure this layer removes.
    """
    with pytest.raises(AuthorizationError, match="no warehouse identity"):
        _resolver().identity_for(_PRINCIPAL)


def test_two_granted_identities_is_a_configuration_error_not_a_denial() -> None:
    """A subject may legitimately hold several `tenant_id` grants; it cannot
    legitimately execute as two principals. Reporting that as a denial would
    leave the operator with an empty catalog and no reason for it."""
    with pytest.raises(GrantConfigurationError, match="only one principal"):
        _resolver("svc-a@example.com", "svc-b@example.com").identity_for(_PRINCIPAL)


def test_the_same_identity_granted_twice_is_not_ambiguous() -> None:
    """Two rows saying the same thing is duplication, not contradiction, and
    refusing it would make a harmless relation edit break serving."""
    assert _resolver("svc@example.com", "svc@example.com").identity_for(
        _PRINCIPAL
    ) == WarehouseIdentity("svc@example.com")


def test_the_operator_resolver_is_what_an_unconfigured_deployment_gets() -> None:
    assert OperatorWarehouseIdentityResolver().identity_for(_PRINCIPAL) == (
        OPERATOR_IDENTITY
    )


def test_the_reserved_attribute_cannot_double_as_a_policy_attribute() -> None:
    """One grant with two meanings is how an operator revokes a connection
    identity while believing they revoked a row filter."""
    provider = GrantAuthorizationProvider(_resolver("svc@example.com")._store)
    attributes = (PolicyAttribute(WAREHOUSE_IDENTITY_ATTRIBUTE, "string"),)
    with pytest.raises(GrantConfigurationError, match="reserved"):
        provider.search_policy_filters(
            _PRINCIPAL, access="governed", attributes=attributes
        )
    with pytest.raises(GrantConfigurationError, match="reserved"):
        provider.can_read(
            _PRINCIPAL,
            {"authorization_resolved": True},
            attributes=attributes,
        )


def test_the_grants_relation_itself_is_read_as_the_operator() -> None:
    """Reading grants is what resolves an identity, so it cannot be read under
    one. This is the ordering the whole seam rests on (ADR-0010)."""
    reader = _RecordingReader()
    WarehouseGrantStore(reader, relation="ops.grants").grants_for("alice")
    assert reader.identities == [OPERATOR_IDENTITY]


# ─── the startup refusals ───────────────────────────────────────────────────


def test_enforcement_without_a_grants_relation_is_refused(tmp_path: Path) -> None:
    """The identity is a grant, so there is nowhere to look one up without the
    relation. Refused before any project or warehouse work."""
    with pytest.raises(ValueError, match="needs a grants_relation"):
        ContextService.from_project(
            tmp_path / "no-such-project",
            enforce_warehouse_identity=True,
        )


def test_enforcement_on_an_adapter_that_cannot_do_it_fails_at_startup(
    tmp_path: Path,
) -> None:
    """Not at the first request, and never by downgrading.

    A deployment that believes it has warehouse-level enforcement and does not
    is worse than one that refuses to boot.
    """
    (tmp_path / "stel_project.yml").write_text(
        "name: p\nversion: '0.1.0'\nprofile: p\n", encoding="utf-8"
    )
    (tmp_path / "profiles.yml").write_text(
        "p:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        f"        path: {tmp_path / 'w.duckdb'}\n"
        "        schema: main\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot execute reads as a named"):
        ContextService.from_project(
            tmp_path,
            grants_relation="ops.grants",
            enforce_warehouse_identity=True,
        )


class _RecordingReader:
    """A `GrantRowReader` that remembers which principal each read ran as."""

    def __init__(self) -> None:
        self.identities: list[WarehouseIdentity] = []

    def read_rows(
        self,
        relation: str,
        *,
        identity: WarehouseIdentity,
        predicates: Sequence[ReadPredicate],
        max_rows: int,
        columns: Sequence[str] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        self.identities.append(identity)
        return ()


class _StubAdapter:
    """Enough of `WarehouseAdapter` to be constructed and entered."""

    def __init__(self, config: WarehouseConfig, *, project_dir: Path | None = None) -> None:
        self.config = config
        self.project_dir = project_dir

    def __enter__(self) -> _StubAdapter:
        self._connect()
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def _connect(self) -> None:
        return None
