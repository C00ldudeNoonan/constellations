"""Redact PII from a text column and (optionally) emit detected spans.

YAML:

    transform:
      type: python
      module: stel.text.transforms.redact_pii
      options:
        text_field: body                  # source column (required)
        output_field: body_redacted       # redacted text (default: text_field, i.e. in-place)
        entities_field: pii_entities      # optional: JSON array of detected spans
        include_raw_text: false            # opt in to raw matches in entities_field
        retain_input_text: false           # opt in to keeping body beside body_redacted
        keep_fields: [ticket_id, body_redacted, pii_entities]  # exact final projection
        # drop_fields: [body]               # alternative to keep_fields
        entities: [PHONE_NUMBER, EMAIL_ADDRESS, US_SSN]  # optional allow-list
        replacement: "[{type}]"           # default; "{type}" substituted at runtime
        score_threshold: 0.4              # default Presidio threshold
        spacy_model: en_core_web_sm
        language: en

First-time setup:
    python -m spacy download en_core_web_sm
"""
from __future__ import annotations

import json

import polars as pl

from ...transforms import TransformContext
from ..pii import redact_pii
from ._helpers import require_text_column, upstream_df


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    df = upstream_df(deps)
    text_field = ctx.options.get("text_field", "text")
    out_field = ctx.options.get("output_field", text_field)
    entities_field = ctx.options.get("entities_field")
    include_raw_text = ctx.options.get("include_raw_text", False)
    if not isinstance(include_raw_text, bool):
        raise ValueError("'include_raw_text' must be a boolean")
    retain_input_text = ctx.options.get("retain_input_text", False)
    if not isinstance(retain_input_text, bool):
        raise ValueError("'retain_input_text' must be a boolean")
    keep_fields = _field_list_option(ctx.options.get("keep_fields"), "keep_fields")
    drop_fields = _field_list_option(ctx.options.get("drop_fields"), "drop_fields")
    if keep_fields is not None and drop_fields is not None:
        raise ValueError("'keep_fields' and 'drop_fields' are mutually exclusive")
    entities = ctx.options.get("entities")
    replacement = ctx.options.get("replacement", "[{type}]")
    score_threshold = float(ctx.options.get("score_threshold", 0.4))
    spacy_model = ctx.options.get("spacy_model", "en_core_web_sm")
    language = ctx.options.get("language", "en")

    require_text_column(df, text_field)

    redacted_texts: list[str] = []
    detected_entities: list[str] = []
    for t in df[text_field].to_list():
        redacted, entities_found = redact_pii(
            t or "",
            entities=entities,
            language=language,
            score_threshold=score_threshold,
            replacement=replacement,
            spacy_model=spacy_model,
        )
        redacted_texts.append(redacted)
        detected_entities.append(
            json.dumps(
                [e.to_dict(include_text=include_raw_text) for e in entities_found]
            )
        )

    out = df.with_columns(pl.Series(out_field, redacted_texts))
    if entities_field:
        out = out.with_columns(pl.Series(entities_field, detected_entities))
    if out_field != text_field and not retain_input_text:
        out = out.drop(text_field)
    if keep_fields is not None:
        _require_known_fields(out, keep_fields, "keep_fields")
        out = out.select(keep_fields)
    elif drop_fields is not None:
        automatically_removed = (
            {text_field} if out_field != text_field and not retain_input_text else set()
        )
        _require_known_fields(
            out, drop_fields, "drop_fields", allow_absent=automatically_removed
        )
        out = out.drop([field for field in drop_fields if field in out.columns])
    return out


def _field_list_option(value: object, name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(field, str) for field in value):
        raise ValueError(f"'{name}' must be a list of column names")
    return [field for field in value if isinstance(field, str)]


def _require_known_fields(
    df: pl.DataFrame,
    fields: list[str],
    option: str,
    *,
    allow_absent: set[str] | None = None,
) -> None:
    missing = sorted(set(fields) - set(df.columns) - (allow_absent or set()))
    if missing:
        raise ValueError(
            f"'{option}' contains columns not present in the transform output: {missing}"
        )
