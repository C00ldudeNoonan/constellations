# No `from __future__ import annotations`: the SDK reads a tool's type hints to
# find its injected `Context`, and `Context` is imported inside
# `create_mcp_server` (the `mcp` extra is optional), so the annotations have to
# be evaluated where it is in scope rather than looked up later by name.
from pathlib import Path
from typing import Any, Literal

from ..optional_dependencies import import_optional_dependency
from .authorization import serving_request
from .contracts import (
    MCP_CONTEXT_SCHEMA_VERSION,
    BusinessFilter,
    EntityScope,
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
from .service import CallerInfo, ContextServerSettings, ContextService


def _caller_info(context: Any, transport: str | None) -> CallerInfo:
    """Identify the caller of the request being served (issue #528).

    Read from the `Context` the SDK injects into the tool: in SDK v2 that is
    the only route to the request, the ambient contextvar v1 offered being
    gone. `Context` parameters are excluded from a tool's input schema, so the
    tool contract is untouched.

    Best-effort by construction, and absent stays absent. The client's identity
    is optional in the protocol -- the handshake's `clientInfo` on older
    protocol versions, nothing at all from a client that does not send it --
    and every field here is nullable. A log row that cannot say who called is
    still worth more than a failed query, and inventing a name for a client
    that gave none would poison the one column #528 exists to measure.
    """
    request_context = context.request_context
    params = request_context.session.client_params
    info = None if params is None else params.client_info
    request_id = request_context.request_id
    return CallerInfo(
        request_id=None if request_id is None else str(request_id),
        client_name=None if info is None else info.name,
        client_version=None if info is None else info.version,
        transport=transport,
    )


def create_mcp_server(
    service: ContextService,
    token_verifier: Any = None,
    *,
    issuer_url: str | None = None,
    public_url: str | None = None,
    transport: str | None = None,
) -> Any:
    mcpserver = import_optional_dependency(
        "mcp.server.mcpserver",
        extra="mcp",
        feature="stel MCP serving",
    )
    # Local on purpose, and named `Context` so the tool signatures below read
    # as the SDK documents them; see the note at the top of this module.
    Context = mcpserver.Context
    auth = _auth_settings(
        token_verifier, issuer_url=issuer_url, public_url=public_url
    )
    app = mcpserver.MCPServer(
        "stel",
        instructions=(
            "Read-only governed document context. Use dbt MCP, not this server, "
            "for semantic-layer metrics."
        ),
        # When present, the SDK verifies the bearer token before any tool
        # runs and exposes the result to `AccessTokenPrincipalResolver`
        # (issue #392). Absent, the server is unauthenticated and only a
        # trusted proxy in front can supply identity.
        token_verifier=token_verifier,
        # Required by the SDK whenever a verifier is present, and the
        # source of the discovery metadata (issue #464). See
        # `_auth_settings`.
        auth=auth,
    )

    @app.tool()
    def list_context_models(
        ctx: Context,
        limit: int = 20,
        cursor: str | None = None,
        schema_version: Literal["mcp_context/v2"] = MCP_CONTEXT_SCHEMA_VERSION,
    ) -> ListContextModelsResponse:
        """List only the governed context models available to the caller."""
        with serving_request(ctx.request_context):
            return service.list_context_models(
                ListContextModelsRequest(
                    schema_version=schema_version,
                    limit=limit,
                    cursor=cursor,
                ),
                caller=_caller_info(ctx, transport),
            )

    @app.tool()
    def search_context(
        ctx: Context,
        model: str,
        query: str,
        mode: Literal["vector", "text", "hybrid"] = "hybrid",
        limit: int = 10,
        candidate_limit: int | None = None,
        filters: list[BusinessFilter] | None = None,
        entity_scope: EntityScope | None = None,
        detail: Literal["lean", "full"] = "lean",
        schema_version: Literal["mcp_context/v2"] = MCP_CONTEXT_SCHEMA_VERSION,
    ) -> SearchContextResponse:
        """Search an available context model with caller-derived authorization.

        `candidate_limit` is how many candidates each retrieval mode fetches
        before fusion, not how many results come back — that stays `limit`.
        Unset, it scales with `limit`.

        `filters` use the operators advertised by `list_context_models`.
        An `array[string]` field accepts only `array_contains_any`, with a
        nonempty array of values; a row matches if it shares any value.

        `entity_scope` restricts results to chunks linked to declared
        vocabulary terms: `{"class": "central_bank"}` for every term of a
        class or its subclasses, or `{"term": "Federal Reserve", "expand": "narrower"}` for one
        term and the terms beneath it. A model's declared terms are published
        as its `entity_types`. Expansion is off unless asked for, and every
        scoped response reports what the scope resolved to in
        `entity_scope_applied`, with each result's `matched_terms` naming the
        term it actually matched — so a hit reached through the hierarchy is
        never indistinguishable from an exact one.

        `detail` is `lean` (the default) or `full`. Lean results carry the
        passage, its ids, `citation`, `entities` and `attributes`; `full` adds
        each result's `interval`, `freshness` and `lineage`. For one chunk's
        lineage without paying for it on every result, call
        `get_context_lineage`.
        """
        with serving_request(ctx.request_context):
            return service.search_context(
                SearchContextRequest(
                    schema_version=schema_version,
                    model=model,
                    query=query,
                    mode=mode,
                    limit=limit,
                    candidate_limit=candidate_limit,
                    filters=tuple(filters or ()),
                    entity_scope=entity_scope,
                    detail=detail,
                ),
                caller=_caller_info(ctx, transport),
            )

    @app.tool()
    def get_document(
        ctx: Context,
        model: str,
        document_id: str,
        document_version_id: str,
        limit: int = 20,
        cursor: str | None = None,
        schema_version: Literal["mcp_context/v2"] = MCP_CONTEXT_SCHEMA_VERSION,
    ) -> GetDocumentResponse:
        """Fetch one authorized document version as bounded, paginated chunks."""
        with serving_request(ctx.request_context):
            return service.get_document(
                GetDocumentRequest(
                    schema_version=schema_version,
                    model=model,
                    document_id=document_id,
                    document_version_id=document_version_id,
                    limit=limit,
                    cursor=cursor,
                ),
                caller=_caller_info(ctx, transport),
            )

    @app.tool()
    def get_context_lineage(
        ctx: Context,
        model: str,
        reference_type: LineageReferenceType,
        reference_id: str,
        schema_version: Literal["mcp_context/v2"] = MCP_CONTEXT_SCHEMA_VERSION,
    ) -> GetContextLineageResponse:
        """Trace one authorized document, context, chunk, or search result."""
        with serving_request(ctx.request_context):
            return service.get_context_lineage(
                GetContextLineageRequest(
                    schema_version=schema_version,
                    model=model,
                    reference_type=reference_type,
                    reference_id=reference_id,
                ),
                caller=_caller_info(ctx, transport),
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
    enforce_warehouse_identity: bool = False,
    settings: ContextServerSettings | None = None,
) -> None:
    service = ContextService.from_project(
        project_dir,
        target=target,
        profiles_dir=profiles_dir,
        grants_relation=grants_relation,
        grant_ttl_seconds=grant_ttl_seconds or DEFAULT_GRANT_TTL_SECONDS,
        enforce_warehouse_identity=enforce_warehouse_identity,
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
    issuer_url: str | None = None,
    public_url: str | None = None,
    target: str | None = None,
    profiles_dir: Path | None = None,
    grants_relation: str | None = None,
    grant_ttl_seconds: float | None = None,
    enforce_warehouse_identity: bool = False,
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
        enforce_warehouse_identity=enforce_warehouse_identity,
        settings=settings,
        principal_resolver=principal_resolver,
    )
    _run(
        service,
        transport=transport,
        token_verifier=token_verifier,
        issuer_url=issuer_url,
        public_url=public_url,
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


def _auth_settings(
    token_verifier: Any, *, issuer_url: str | None, public_url: str | None
) -> Any:
    """The SDK's `AuthSettings`, or None when nothing is being verified.

    stel is a **resource server**: it validates tokens somebody else issued.
    It is not an authorization server, so `auth_server_provider` — the other
    parameter, and the one #464 named — is the wrong seam twice over. It would
    make stel issue tokens, and the SDK refuses it alongside a `token_verifier`
    anyway. What a resource server declares instead is `resource_server_url`,
    which the SDK turns into the RFC 9728 protected-resource metadata document
    and the `resource_metadata` parameter of its `WWW-Authenticate` challenges.

    These settings are also **not optional**: the SDK raises
    `Cannot specify auth_server_provider or token_verifier without auth
    settings` when a verifier arrives without them. Every stel release with
    token verification passed one without, so `--jwt-*` never got past startup
    on the SDK versions the `mcp` extra allows. Nothing caught it because the
    only test that built a server built the unauthenticated one.
    """
    if token_verifier is None:
        # Unauthenticated, or identity from a trusted proxy. Passing settings
        # here would make the SDK demand a verifier to go with them.
        return None
    if not issuer_url or not public_url:
        raise ValueError(
            "A token verifier needs both an issuer URL and this deployment's "
            "own public URL: the first names who may issue tokens, the second "
            "is what the discovery metadata publishes and what a client is "
            "told to ask about in a WWW-Authenticate challenge."
        )
    settings_module = import_optional_dependency(
        "mcp.server.auth.settings",
        extra="mcp",
        feature="stel MCP serving",
    )
    return settings_module.AuthSettings(
        issuer_url=issuer_url,
        resource_server_url=public_url,
        # Stated rather than defaulted: the SDK warns when it is unset and
        # will make True the default in 3.0. True makes the SDK refuse a token
        # whose `AccessToken.resource` is not `resource_server_url`, and stel's
        # verifiers do not set that field -- they check the token's audience
        # against the operator's `--*-audience` themselves, and the audience
        # is configured separately from the public URL, so the two need not
        # be equal. Both verifiers fail closed on a missing or mismatched
        # audience (`tokens.py`, `introspection.py`).
        validate_token_resource=False,
    )


def loopback_host_security() -> Any:
    """Refuse a request whose `Host` or `Origin` is not a loopback name.

    SDK 1.x built its server with the default host `127.0.0.1`, which turned
    this protection on whatever `--host` later said, so every bind refused a
    foreign `Host` (421). SDK 2 turns it on only for a loopback bind. Passing
    it explicitly keeps the 1.x behaviour on `0.0.0.0` and LAN binds, which
    DNS rebinding would otherwise reach through a name that resolves to
    loopback. A proxy in front must forward a loopback `Host`.
    """
    settings = import_optional_dependency(
        "mcp.server.transport_security", extra="mcp", feature="The MCP server"
    )
    names = ("127.0.0.1", "localhost", "[::1]")
    return settings.TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[f"{name}:*" for name in names],
        allowed_origins=[f"http://{name}:*" for name in names],
    )


def _run(
    service: ContextService,
    *,
    transport: str,
    token_verifier: Any = None,
    issuer_url: str | None = None,
    public_url: str | None = None,
    **network: Any,
) -> None:
    """Warm the service, build the server and run its transport.

    `network` is `host` and `port` for a network transport and empty for stdio.
    SDK v2 takes them as arguments to `run`, not as settings on the server
    (v1 assigned `app.settings.host`), and a transport rejects a keyword it
    does not know, so they are passed only where they exist.
    """
    try:
        # Resolve credentials and open the warehouse once before the transport
        # starts, so auth problems fail loudly at boot rather than as per-call
        # "timeout" errors mid-session (issue #365).
        service.warm_up()
        app = create_mcp_server(
            service,
            token_verifier=token_verifier,
            issuer_url=issuer_url,
            public_url=public_url,
            transport=transport,
        )
        if network:
            network["transport_security"] = loopback_host_security()
        if transport == "streamable-http":
            # Plain JSON replies rather than an SSE stream per call: every
            # tool returns one result, and clients that do not read SSE
            # still work. This was a constructor argument on v1.
            network["json_response"] = True
        app.run(transport=transport, **network)
    finally:
        service.close()
