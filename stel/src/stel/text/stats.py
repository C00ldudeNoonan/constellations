from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_WORD_RE = re.compile(r"\w+", re.UNICODE)
# Rough sentence boundary: punctuation followed by whitespace or EOS. Won't
# beat spaCy for accuracy but works for the "fast stats" use case.
_SENT_RE = re.compile(r"[.!?]+(?=\s+|$)")


@dataclass(frozen=True)
class TextStats:
    char_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def text_stats(text: str) -> TextStats:
    """Quick text statistics. Returns 0s for empty input."""
    if not text:
        return TextStats(0, 0, 0, 0)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    words = _WORD_RE.findall(text)
    return TextStats(
        char_count=len(text),
        word_count=len(words),
        sentence_count=max(len(sentences), 1) if text.strip() else 0,
        paragraph_count=max(len(paragraphs), 1) if text.strip() else 0,
    )
