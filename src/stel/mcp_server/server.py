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
from .service import ContextServerSettings, ContextService


def create_mcp_server(service: ContextService) -> Any:
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


def serve_stdio(
    project_dir: Path,
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
    settings: ContextServerSettings | None = None,
) -> None:
    service = ContextService.from_project(
        project_dir,
        target=target,
        profiles_dir=profiles_dir,
        settings=settings,
    )
    try:
        # Resolve credentials and open the warehouse once before the stdio
        # transport starts, so auth problems fail loudly at boot rather than
        # as per-call "timeout" errors mid-session (issue #365).
        service.warm_up()
        create_mcp_server(service).run(transport="stdio")
    finally:
        service.close()
