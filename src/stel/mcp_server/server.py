from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ..optional_dependencies import import_optional_dependency
from .contracts import (
    MCP_CONTEXT_SCHEMA_VERSION,
    BusinessFilter,
    GetContextLineageRequest,
    GetContextLineageResponse,
    GetDocumentRequest,
    GetDocumentResponse,
    LineageReferenceType,
    ListContextModelsRequest,
    ListContextModelsResponse,
    SearchContextRequest,
    SearchContextResponse,
)
from .grants import DEFAULT_GRANT_TTL_SECONDS
from .service import ContextServerSettings, ContextService


def create_mcp_server(
    service: ContextService, token_verifier: Any = None
) -> Any:
    fastmcp = import_optional_dependency(
        "mcp.server.fastmcp",
        extra="mcp",
        feature="stel MCP serving",
    )
    app = fastmcp.FastMCP(
        "stel",
        instructions=(
            "Read-only governed document context. Use dbt MCP, not this server, "
            "for semantic-layer metrics."
        ),
        json_response=True,
        # When present, the SDK verifies the bearer token before any tool runs
        # and exposes the result to `AccessTokenPrincipalResolver` (issue
        # #392). Absent, the server is unauthenticated and only a trusted
        # proxy in front can supply identity.
        token_verifier=token_verifier,
    )

    @app.tool()  # type: ignore[untyped-decorator]
    def list_context_models(
        limit: int = 20,
        cursor: str | None = None,
        schema_version: Literal["mcp_context/v1"] = MCP_CONTEXT_SCHEMA_VERSION,
    ) -> ListContextModelsResponse:
        """List only the governed context models available to the caller."""
        return service.list_context_models(
            ListContextModelsRequest(
                schema_version=schema_version,
                limit=limit,
                cursor=cursor,
            )
        )

    @app.tool()  # type: ignore[untyped-decorator]
    def search_context(
        model: str,
        query: str,
        mode: Literal["vector", "text", "hybrid"] = "hybrid",
        limit: int = 10,
        filters: list[BusinessFilter] | None = None,
        schema_version: Literal["mcp_context/v1"] = MCP_CONTEXT_SCHEMA_VERSION,
    ) -> SearchContextResponse:
        """Search an available context model with caller-derived authorization."""
        return service.search_context(
            SearchContextRequest(
                schema_version=schema_version,
                model=model,
                query=query,
                mode=mode,
                limit=limit,
                filters=tuple(filters or ()),
            )
        )

    @app.tool()  # type: ignore[untyped-decorator]
    def get_document(
        model: str,
        document_id: str,
        document_version_id: str,
        limit: int = 20,
        cursor: str | None = None,
        schema_version: Literal["mcp_context/v1"] = MCP_CONTEXT_SCHEMA_VERSION,
    ) -> GetDocumentResponse:
        """Fetch one authorized document version as bounded, paginated chunks."""
        return service.get_document(
            GetDocumentRequest(
                schema_version=schema_version,
                model=model,
                document_id=document_id,
                document_version_id=document_version_id,
                limit=limit,
                cursor=cursor,
            )
        )

    @app.tool()  # type: ignore[untyped-decorator]
    def get_context_lineage(
        model: str,
        reference_type: LineageReferenceType,
        reference_id: str,
        schema_version: Literal["mcp_context/v1"] = MCP_CONTEXT_SCHEMA_VERSION,
    ) -> GetContextLineageResponse:
        """Trace one authorized document, context, chunk, or search result."""
        return service.get_context_lineage(
            GetContextLineageRequest(
                schema_version=schema_version,
                model=model,
                reference_type=reference_type,
                reference_id=reference_id,
            )
        )

    return app


NETWORK_TRANSPORTS = ("streamable-http", "sse")


def serve_stdio(
    project_dir: Path,
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
    grants_relation: str | None = None,
    grant_ttl_seconds: float | None = None,
    settings: ContextServerSettings | None = None,
) -> None:
    service = ContextService.from_project(
        project_dir,
        target=target,
        profiles_dir=profiles_dir,
        grants_relation=grants_relation,
        grant_ttl_seconds=grant_ttl_seconds or DEFAULT_GRANT_TTL_SECONDS,
        settings=settings,
    )
    _run(service, transport="stdio")


def serve_network(
    project_dir: Path,
    *,
    transport: str,
    host: str,
    port: int,
    principal_resolver: Any,
    token_verifier: Any = None,
    target: str | None = None,
    profiles_dir: Path | None = None,
    grants_relation: str | None = None,
    grant_ttl_seconds: float | None = None,
    settings: ContextServerSettings | None = None,
) -> None:
    """Serve over a network transport with a per-request principal resolver.

    `principal_resolver` is required and has no default, deliberately. The
    environment resolver is correct for stdio — the operator running the
    process *is* the principal — and inverted over a network, where it would
    collapse every caller into whichever identity the process started with.
    The failure is silent: policy filters still apply, just the wrong ones, so
    the server answers confidently out of someone else's tenant. Making the
    parameter required means that mistake cannot be made by omission
    (issue #392).
    """
    if transport not in NETWORK_TRANSPORTS:
        raise ValueError(
            f"Unknown network transport {transport!r}; expected one of "
            f"{', '.join(NETWORK_TRANSPORTS)}"
        )
    _reject_process_wide_identity(principal_resolver, transport)
    _reject_unverified_token_identity(principal_resolver, token_verifier)
    service = ContextService.from_project(
        project_dir,
        target=target,
        profiles_dir=profiles_dir,
        grants_relation=grants_relation,
        grant_ttl_seconds=grant_ttl_seconds or DEFAULT_GRANT_TTL_SECONDS,
        settings=settings,
        principal_resolver=principal_resolver,
    )
    _run(
        service,
        transport=transport,
        token_verifier=token_verifier,
        host=host,
        port=port,
    )


def _reject_process_wide_identity(principal_resolver: Any, transport: str) -> None:
    """Refuse a network transport whose identity is process-wide.

    A refusal rather than a warning: the symptom of getting this wrong is
    correct-looking answers scoped to the wrong tenant, which no operator
    would notice from the outside.
    """
    from .authorization import EnvironmentPrincipalResolver, StaticPrincipalResolver

    if isinstance(
        principal_resolver, EnvironmentPrincipalResolver | StaticPrincipalResolver
    ):
        raise ValueError(
            f"Transport {transport!r} serves many callers, but "
            f"{type(principal_resolver).__name__} resolves one identity for the "
            "whole process — every caller would be served as that principal, "
            "with that principal's tenant filters. Use a per-request resolver."
        )


def _reject_unverified_token_identity(
    principal_resolver: Any, token_verifier: Any
) -> None:
    """Refuse a token-derived identity with nothing verifying the tokens.

    `AccessTokenPrincipalResolver` reads whatever the SDK put in the auth
    context. With no verifier configured the SDK puts nothing there, so every
    call resolves to None and the server refuses everything — noisy, and not a
    security hole. The hole is the shape this guards against instead: a
    verifier that exists but was never handed to the transport, where the
    resolver silently sees nothing while the operator believes tokens are
    being checked.

    Both directions are refused, because both mean the deployment is not the
    one the operator configured.
    """
    from .authorization import AccessTokenPrincipalResolver

    resolves_from_token = isinstance(principal_resolver, AccessTokenPrincipalResolver)
    if resolves_from_token and token_verifier is None:
        raise ValueError(
            "The principal is taken from a verified bearer token, but no token "
            "verifier is configured — nothing would verify anything, and every "
            "call would arrive unauthenticated."
        )
    if token_verifier is not None and not resolves_from_token:
        raise ValueError(
            "A token verifier is configured, but the principal is resolved "
            f"by {type(principal_resolver).__name__} rather than from the "
            "verified token — the tokens would be checked and then ignored."
        )


def _run(
    service: ContextService,
    *,
    transport: str,
    token_verifier: Any = None,
    **settings: Any,
) -> None:
    try:
        # Resolve credentials and open the warehouse once before the transport
        # starts, so auth problems fail loudly at boot rather than as per-call
        # "timeout" errors mid-session (issue #365).
        service.warm_up()
        app = create_mcp_server(service, token_verifier=token_verifier)
        for name, value in settings.items():
            setattr(app.settings, name, value)
        app.run(transport=transport)
    finally:
        service.close()
