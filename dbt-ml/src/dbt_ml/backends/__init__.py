from . import (
    email_backend,  # noqa: F401  # side-effect: registers EmailBackend
    html_backend,  # noqa: F401  # side-effect: registers HtmlBackend
    json_backend,  # noqa: F401  # side-effect: registers JsonBackend
    llm_backend,  # noqa: F401  # side-effect: registers LLMBackend
    markdown_backend,  # noqa: F401  # side-effect: registers MarkdownBackend
    pdf_backend,  # noqa: F401  # side-effect: registers PdfBackend
)
from .base import BaseBackend, ExtractionResult
from .options import (
    BackendOptionContract,
    BackendOptionsError,
    get_backend_option_contract,
    list_backend_option_contracts,
    register_backend_option_contract,
    validate_backend_options,
)
from .registry import BackendNotFoundError, get_backend, list_backends, register

__all__ = [
    "BackendNotFoundError",
    "BackendOptionContract",
    "BackendOptionsError",
    "BaseBackend",
    "ExtractionResult",
    "get_backend",
    "get_backend_option_contract",
    "list_backend_option_contracts",
    "list_backends",
    "register",
    "register_backend_option_contract",
    "validate_backend_options",
]
