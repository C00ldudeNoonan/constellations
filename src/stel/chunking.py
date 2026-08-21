"""Text chunking for the `chunk:` model kind (issue #86).

Splitters are pure functions over a string so they're trivially testable and
deterministic; chunk IDs are content-addressed so re-running unchanged input
yields identical IDs (a hard requirement for incremental MERGE downstream).
"""
from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .agent_context import make_chunk_id
from .config.model import ChunkConfig, HeadingConfig

# Separator hierarchy for the recursive splitter: try to break on the largest
# semantic boundary that keeps a chunk under the size limit, falling back to
# finer ones, and finally to a hard character cut.
_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Divides the rendered metadata block from the chunk text it introduces
# (issue #308). A visible rule reads as a header to a human and to a model,
# and it is what separates the block's lines from the document's own.
METADATA_SEPARATOR = "---"

# Sorts after any real heading name at the same offset, so a heading starting
# exactly at a chunk's first character counts as covering that chunk.
_AFTER_EVERY_NAME = "￿"


class ChunkingError(ValueError):
    """A chunk model's sizes leave no room to split text (issue #308)."""


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    # The heading this chunk falls under, when the model declares detection
    # (issue #332). None when no heading precedes it, or none is configured.
    section: str | None = None


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
        # No source offsets, so no heading attribution — rejected at config
        # load rather than silently emitting nulls (issue #332).
        return [
            Chunk(index=index, text=piece)
            for index, piece in enumerate(
                _split_tokens(text, size, config.chunk_overlap, config.encoding)
            )
        ]

    placed = _split_recursive(text, size, config.chunk_overlap)
    headings = _find_headings(text, config.headings)
    return [
        Chunk(index=index, text=piece, section=_section_at(headings, start))
        for index, (start, piece) in enumerate(placed)
    ]


def _find_headings(
    text: str, config: HeadingConfig | None
) -> list[tuple[int, str]]:
    """Every heading in the document, as (offset, name), in order.

    Detected once against the full source rather than per chunk: the splitter
    is the only place that sees both the whole document and every boundary,
    so a heading sitting in a chunk's tail — or a chunk starting mid-heading —
    resolves by position instead of by guessing from fragments (issue #332).
    """
    if config is None:
        return []
    pattern = re.compile(config.pattern, re.MULTILINE)
    found: list[tuple[int, str]] = []
    for match in pattern.finditer(text):
        # A capture group names the section; without one the whole match is
        # the name. That is what lets `^(Item\s+\d+[A-C]?)[.:]` yield
        # "Item 1A" rather than "Item 1A." without stel guessing at
        # punctuation.
        name = match.group(1) if pattern.groups else match.group(0)
        if name and name.strip():
            found.append((match.start(), name.strip()))
    return found


def _section_at(headings: list[tuple[int, str]], start: int) -> str | None:
    """The last heading at or before `start`. None before the first heading."""
    if not headings:
        return None
    index = bisect_right(headings, (start, _AFTER_EVERY_NAME)) - 1
    return headings[index][1] if index >= 0 else None


def _unit(config: ChunkConfig) -> str:
    return "tokens" if config.strategy == "tokens" else "characters"


def _split_recursive(
    text: str, chunk_size: int, overlap: int
) -> list[tuple[int, str]]:
    """Split, returning each chunk with its start offset in `text`.

    The offset is what makes exact heading attribution possible (issue #332):
    `_recurse` produces pieces that concatenate back to the source, so their
    cumulative lengths give every boundary position, and the merge carries
    those through.
    """
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
) -> list[tuple[int, str]]:
    """Greedily pack fine-grained splits into chunks under `chunk_size`,
    carrying roughly `overlap` trailing characters from each chunk into the
    next — snapped to a separator boundary (issue #331).

    Each chunk comes back with its start offset in the source. `_recurse`
    guarantees the splits concatenate back to the document, so tracking a
    running position through the pack is exact rather than a search.
    """
    chunks: list[tuple[int, str]] = []
    current = ""
    current_start = 0
    position = 0
    for split in splits:
        if current and len(current) + len(split) > chunk_size:
            _append_chunk(chunks, current_start, current)
            tail = _overlap_tail(current, overlap)
            # The carried tail sits immediately before this split in the
            # source, so the new chunk starts that many characters earlier.
            current_start = position - len(tail)
            current = tail + split
        else:
            if not current:
                current_start = position
            current += split
        position += len(split)
    _append_chunk(chunks, current_start, current)
    return chunks


def _append_chunk(chunks: list[tuple[int, str]], start: int, text: str) -> None:
    stripped = text.strip()
    if not stripped:
        return
    # `strip()` moves the chunk's first character, so the offset moves with it
    # — otherwise a chunk beginning after a paragraph break would be
    # attributed to whatever heading preceded that whitespace.
    chunks.append((start + (len(text) - len(text.lstrip())), stripped))


def _overlap_tail(text: str, overlap: int) -> str:
    """The trailing text carried into the next chunk, starting on a boundary.

    A fixed `text[-overlap:]` slice starts wherever the count lands, which is
    mid-word almost every time — measured at 81.5% of chunks on a real 10-K,
    against 4.2% with overlap disabled. That single term dominated everything
    else: work upstream to emit real paragraph structure showed almost no
    boundary improvement because the overlap was reintroducing arbitrary
    offsets on its own (issue #331).

    So step back *approximately* `overlap` and snap to the nearest boundary
    from the same separator hierarchy the splitter already walks. The overlap
    then carries whole sentences or paragraphs, and the next chunk starts on a
    real boundary. `chunk_size` is already a target rather than an exact cap,
    so treating `chunk_overlap` the same way matches the existing contract.
    """
    if overlap <= 0 or not text:
        return ""
    if overlap >= len(text):
        return text

    target = len(text) - overlap
    # Bound how far snapping may move the boundary: at most twice the
    # requested overlap, at least half of it. Outside that band
    # "approximately `overlap`" stops meaning anything, and a plain cut is the
    # more honest answer.
    earliest = max(0, len(text) - 2 * overlap)
    latest = len(text) - max(1, overlap // 2)

    for separator in _RECURSIVE_SEPARATORS:
        if not separator:
            continue
        # A chunk can start just *after* a separator, so that is the candidate
        # position rather than the separator's own index.
        candidates: list[int] = []
        position = text.find(separator, earliest)
        while position != -1 and position < latest:
            candidates.append(position + len(separator))
            position = text.find(separator, position + 1)
        if candidates:
            # Highest-priority separator with any candidate wins, and within
            # it the position closest to the requested overlap — so a
            # paragraph break is preferred to a sentence break, which is
            # preferred to a word break, exactly as when splitting.
            return text[min(candidates, key=lambda p: abs(p - target)) :]
    # No boundary in the band (one very long token, say): the fixed slice is
    # what it always was.
    return text[-overlap:]


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
