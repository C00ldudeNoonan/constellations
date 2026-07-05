from .base import DocumentRef, DocumentSource, SourceError, SourceScan
from .gcs import GCSDocumentSource
from .local import LocalDocumentSource

__all__ = [
    "DocumentRef",
    "DocumentSource",
    "GCSDocumentSource",
    "LocalDocumentSource",
    "SourceError",
    "SourceScan",
    "get_document_source",
]


def get_document_source(path: str) -> DocumentSource:
    """Pick the source implementation from the path's URI scheme."""
    if path.startswith("gs://"):
        return GCSDocumentSource()
    return LocalDocumentSource()
