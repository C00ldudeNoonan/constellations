from pathlib import Path

from ..config.profile import WarehouseConfig
from ..config.source import WAREHOUSE_SOURCE_SCHEME
from .base import DocumentRef, DocumentSource, SourceError, SourceScan
from .gcs import GCSDocumentSource
from .gdrive import GDRIVE_SCHEME, GoogleDriveDocumentSource
from .local import LocalDocumentSource
from .warehouse import WarehouseDocumentSource

__all__ = [
    "DocumentRef",
    "DocumentSource",
    "GCSDocumentSource",
    "GoogleDriveDocumentSource",
    "LocalDocumentSource",
    "SourceError",
    "SourceScan",
    "WarehouseDocumentSource",
    "get_document_source",
]


def get_document_source(
    path: str,
    *,
    warehouse: WarehouseConfig | None = None,
    project_dir: Path | None = None,
) -> DocumentSource:
    """Pick the source implementation from the path's URI scheme.

    A `warehouse://` source reads through the active adapter, so callers that
    can serve one pass the resolved warehouse config; the object sources
    ignore it. Every current caller resolves a profile before discovery, so
    the error below marks a new call site wired up incompletely, not a user
    mistake.
    """
    if path.startswith(WAREHOUSE_SOURCE_SCHEME):
        if warehouse is None or project_dir is None:
            raise SourceError(
                "warehouse:// sources need the resolved warehouse config; "
                "this caller did not provide one"
            )
        return WarehouseDocumentSource(warehouse, project_dir)
    if path.startswith("gs://"):
        return GCSDocumentSource()
    if path.startswith(GDRIVE_SCHEME):
        return GoogleDriveDocumentSource()
    return LocalDocumentSource()
