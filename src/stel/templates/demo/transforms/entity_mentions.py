"""Find every mention of a declared system in every passage.

A dictionary match against the project's `systems` vocabulary: each term's
label and aliases, case-insensitively, on word boundaries. It needs no
language model, so it runs offline and gives the same answer every time.
For open-ended entities -- people, companies, places nobody listed --
swap in the built-in `stel.text.transforms.nlp_entities` (spaCy).

The output has the shape `nlp_entities` produces, so `link_entities` and
`extract_relations` downstream do not know the difference.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

import polars as pl

from stel.transforms import TransformContext

# A sentence ends at terminal punctuation or a line break; a co-occurrence
# relation is two mentions in one sentence.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")

_SCHEMA: dict[str, pl.DataType] = {
    "entity_id": pl.String(),
    "context_id": pl.String(),
    "sentence_index": pl.Int64(),
    "start": pl.Int64(),
    "end": pl.Int64(),
    "label": pl.String(),
    "entity_text": pl.String(),
}


def validate_options(options: dict[str, Any]) -> None:
    if not isinstance(options.get("vocabulary"), str):
        raise ValueError("entity_mentions needs `vocabulary:`, the name of a declared vocabulary")


def _patterns(ctx: TransformContext) -> list[tuple[re.Pattern[str], str]]:
    name = ctx.options["vocabulary"]
    if name not in ctx.vocabularies:
        raise ValueError(f"vocabulary '{name}' is not declared in stel_project.yml")
    surfaces = [
        (surface, term.entity_class or "concept")
        for term in ctx.vocabularies[name].terms
        for surface in (term.label, *term.aliases)
    ]
    # Longest surface form first, so "Webhook Dispatcher" claims its span
    # before the shorter alias "dispatcher" can.
    surfaces.sort(key=lambda item: len(item[0]), reverse=True)
    return [
        (re.compile(rf"(?<![\w-]){re.escape(surface)}(?![\w-])", re.IGNORECASE), label)
        for surface, label in surfaces
    ]


def _sentence_starts(text: str) -> list[int]:
    return [0, *(match.end() for match in _SENTENCE_END.finditer(text))]


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    patterns = _patterns(ctx)
    rows: list[dict[str, Any]] = []
    for chunk in deps["document_chunks"].iter_rows(named=True):
        text = str(chunk["text"])
        starts = _sentence_starts(text)
        taken: list[tuple[int, int]] = []
        for pattern, label in patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                if any(start < other_end and other_start < end for other_start, other_end in taken):
                    continue
                taken.append((start, end))
                sentence = sum(1 for boundary in starts if boundary <= start) - 1
                key = f"{chunk['context_id']}:{start}:{end}"
                rows.append(
                    {
                        "entity_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:32],
                        "context_id": str(chunk["context_id"]),
                        "sentence_index": sentence,
                        "start": start,
                        "end": end,
                        "label": label,
                        "entity_text": match.group(0),
                    }
                )
    return pl.DataFrame(rows, schema=_SCHEMA)
