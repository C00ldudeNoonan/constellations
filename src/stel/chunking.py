"""Text chunking for the `chunk:` model kind (issue #86).

Splitters are pure functions over a string so they're trivially testable and
deterministic; chunk IDs are content-addressed so re-running unchanged input
yields identical IDs (a hard requirement for incremental MERGE downstream).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .agent_context import make_chunk_id
from .config.model import ChunkConfig

# Separator hierarchy for the recursive splitter: try to break on the largest
# semantic boundary that keeps a chunk under the size limit, falling back to
# finer ones, and finally to a hard character cut.
_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Divides the rendered metadata block from the chunk text it introduces
# (issue #308). A visible rule reads as a header to a human and to a model,
# and it is what separates the block's lines from the document's own.
METADATA_SEPARATOR = "---"


class ChunkingError(ValueError):
    """A chunk model's sizes leave no room to split text (issue #308)."""


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_id(document_id: str, index: int, text: str) -> str:
    """Deterministic, content-addressed: identical (document, position, text)
    always yields the same id, so unchanged re-runs are stable and any text
    change produces a new id."""
    return make_chunk_id(document_id, index, text)


def render_metadata_block(record: Mapping[str, Any], fields: list[str]) -> str:
    """Render `fields` from `record` as a block to prepend to the chunk text.

    Declared order, not sorted: the order is the author's, and a stable
    rendering is what keeps chunk ids stable. Null values are skipped rather
    than rendered as "None", so a sparse column does not teach the embedder a
    word that never appears in the document.
    """
    lines = [
        f"{field}: {record[field]}"
        for field in fields
        if record.get(field) is not None
    ]
    if not lines:
        return ""
    return "\n".join(lines) + f"\n{METADATA_SEPARATOR}\n"


def measure(text: str, config: ChunkConfig) -> int:
    """Length of `text` in the unit `config.strategy` splits by."""
    if config.strategy == "tokens":
        return len(_encoding(config.encoding).encode(text))
    return len(text)


def split_text(text: str, config: ChunkConfig, *, reserved: int = 0) -> list[Chunk]:
    """Split `text`, leaving `reserved` units of room in every chunk.

    `reserved` is what an in-text metadata block occupies once prepended
    (issue #308). Charging it against `chunk_size` rather than adding it on top
    is the difference between a chunk that still fits the embedder's input
    limit and one that silently does not — and a provider configured without
    truncation rejects the oversized request rather than quietly shortening it.
    """
    if not text or not text.strip():
        return []
    size = config.chunk_size - reserved
    if size <= 0:
        raise ChunkingError(
            f"in-text metadata needs {reserved} of the {config.chunk_size} "
            f"{_unit(config)} chunk_size allows, leaving no room for text. "
            "Raise chunk_size or shorten the metadata fields."
        )
    if config.chunk_overlap >= size:
        raise ChunkingError(
            f"chunk_overlap ({config.chunk_overlap}) must stay below the "
            f"{size} {_unit(config)} left for text once in-text metadata takes "
            f"{reserved} of chunk_size {config.chunk_size}."
        )
    if config.strategy == "tokens":
        pieces = _split_tokens(text, size, config.chunk_overlap, config.encoding)
    else:
        pieces = _split_recursive(text, size, config.chunk_overlap)
    return [Chunk(index=i, text=piece) for i, piece in enumerate(pieces)]


def _unit(config: ChunkConfig) -> str:
    return "tokens" if config.strategy == "tokens" else "characters"


def _split_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    splits = _recurse(text, chunk_size, _RECURSIVE_SEPARATORS)
    return _merge_with_overlap(splits, chunk_size, overlap)


def _recurse(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []
    separator = separators[-1]
    remaining = separators
    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            remaining = separators[i + 1 :]
            break
        if sep in text:
            separator = sep
            remaining = separators[i + 1 :]
            break

    if separator == "":
        # Hard cut when no separator helps (e.g. a single very long token).
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces: list[str] = []
    parts = text.split(separator)
    for i, part in enumerate(parts):
        # Re-attach the separator that split() consumed, but only *between*
        # parts — the final part had no trailing separator in the source, so
        # appending one would inject characters that were never there.
        # Chunk text is what gets embedded/retrieved, so it must match the
        # source exactly apart from window-boundary whitespace.
        segment = part + separator if i < len(parts) - 1 else part
        if not segment:
            continue
        if len(segment) <= chunk_size:
            pieces.append(segment)
        else:
            pieces.extend(_recurse(segment, chunk_size, remaining))
    return pieces


def _merge_with_overlap(
    splits: list[str], chunk_size: int, overlap: int
) -> list[str]:
    """Greedily pack fine-grained splits into chunks under `chunk_size`,
    carrying `overlap` trailing characters from each chunk into the next."""
    chunks: list[str] = []
    current = ""
    for split in splits:
        if current and len(current) + len(split) > chunk_size:
            chunks.append(current.strip())
            current = (current[-overlap:] if overlap else "") + split
        else:
            current += split
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def _encoding(name: str) -> Any:
    from .optional_dependencies import import_optional_dependency

    tiktoken = import_optional_dependency(
        "tiktoken", extra="text", feature="The `tokens` chunk strategy"
    )
    return tiktoken.get_encoding(name)


def _split_tokens(
    text: str, chunk_size: int, overlap: int, encoding: str
) -> list[str]:
    enc = _encoding(encoding)
    tokens = enc.encode(text)
    step = chunk_size - overlap
    pieces: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        pieces.append(enc.decode(window).strip())
        if start + chunk_size >= len(tokens):
            break
    return [p for p in pieces if p]
