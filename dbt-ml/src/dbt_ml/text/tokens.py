from __future__ import annotations

import functools
from typing import TYPE_CHECKING, cast

from ..optional_dependencies import import_optional_dependency

if TYPE_CHECKING:
    import tiktoken as tiktoken_types

# Map family-style names to tiktoken encodings. Users can also pass a
# tiktoken encoding name directly (e.g. "cl100k_base").
_FAMILY_TO_ENCODING = {
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-3.5": "cl100k_base",
    "openai": "cl100k_base",
    # Anthropic doesn't publish a public tokenizer; cl100k is a close-enough
    # proxy for cost/length estimation. Document the approximation.
    "claude": "cl100k_base",
    "anthropic": "cl100k_base",
}


@functools.lru_cache(maxsize=8)
def _get_encoding(name: str) -> tiktoken_types.Encoding:
    tiktoken = import_optional_dependency(
        "tiktoken", extra="text", feature="Token counting"
    )

    encoding_name = _FAMILY_TO_ENCODING.get(name.lower(), name)
    try:
        return cast(
            "tiktoken_types.Encoding", tiktoken.get_encoding(encoding_name)
        )
    except (KeyError, ValueError):
        # Fallback: try as a model name
        return cast(
            "tiktoken_types.Encoding", tiktoken.encoding_for_model(name)
        )


def count_tokens(text: str, *, model: str = "cl100k_base") -> int:
    """Count tokens for `text` under `model`.

    `model` accepts:
        - tiktoken encoding names: "cl100k_base", "o200k_base", "p50k_base"
        - family aliases: "gpt-4", "gpt-4o", "openai", "claude", "anthropic"
        - OpenAI model ids: "gpt-4o-mini", "text-embedding-3-small", ...

    For Anthropic Claude, this is an approximation using cl100k_base
    (within ~5-10% of actual). For accurate Claude token counts use the
    Anthropic Messages API's `count_tokens` endpoint.
    """
    if not text:
        return 0
    return len(_get_encoding(model).encode(text))
