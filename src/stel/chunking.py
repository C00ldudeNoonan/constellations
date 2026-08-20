"""Text chunking for the `chunk:` model kind (issue #86).

Splitters are pure functions over a string so they're trivially testable and
deterministic; chunk IDs are content-addressed so re-running unchanged input
yields identical IDs (a hard requirement for incremental MERGE downstream).
"""
from __future__ import annotations

from dataclasses import dataclass

from .agent_context import make_chunk_id
from .config.model import ChunkConfig

# Separator hierarchy for the recursive splitter: try to break on the largest
# semantic boundary that keeps a chunk under the size limit, falling back to
# finer ones, and finally to a hard character cut.
_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_id(document_id: str, index: int, text: str) -> str:
    """Deterministic, content-addressed: identical (document, position, text)
    always yields the same id, so unchanged re-runs are stable and any text
    change produces a new id."""
    return make_chunk_id(document_id, index, text)


def split_text(text: str, config: ChunkConfig) -> list[Chunk]:
    if not text or not text.strip():
        return []
    if config.strategy == "tokens":
        pieces = _split_tokens(
            text, config.chunk_size, config.chunk_overlap, config.encoding
        )
    else:
        pieces = _split_recursive(text, config.chunk_size, config.chunk_overlap)
    return [Chunk(index=i, text=piece) for i, piece in enumerate(pieces)]


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


def _split_tokens(
    text: str, chunk_size: int, overlap: int, encoding: str
) -> list[str]:
    from .optional_dependencies import import_optional_dependency

    tiktoken = import_optional_dependency(
        "tiktoken", extra="text", feature="The `tokens` chunk strategy"
    )
    enc = tiktoken.get_encoding(encoding)
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
