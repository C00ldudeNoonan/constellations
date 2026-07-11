"""LLM-extract structured invoice fields from raw PDF text.

Reads `raw_pdf_text` (one row per PDF), calls Claude per row with a JSON
schema, returns a Polars DataFrame. Cache path + model id come from the
active profile's `llm:` block — no credentials in this file.
"""
from __future__ import annotations

import polars as pl

from dbt_ml.backends.llm_backend import extract_fields_from_text
from dbt_ml.transforms import TransformContext

SCHEMA = [
    {"name": "invoice_id", "type": "string",
     "description": "The invoice identifier (e.g. INV-00042) — appears after 'Invoice number:'"},
    {"name": "vendor", "type": "string",
     "description": "The supplier company name — appears after 'From:'"},
    {"name": "issue_date", "type": "string",
     "description": "ISO date YYYY-MM-DD — appears after 'Issue date:'"},
    {"name": "currency", "type": "string",
     "description": "Three-letter currency code: USD, EUR, GBP, or CAD"},
    {"name": "total", "type": "number",
     "description": "The total due as a number, e.g. 1234.56"},
]


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    raw = deps["raw_pdf_text"]
    llm_cfg = ctx.llm
    model = llm_cfg.model if llm_cfg else "claude-haiku-4-5"
    cache_path = str(llm_cfg.cache_path) if llm_cfg and llm_cfg.cache_path else None
    api_key_env = llm_cfg.api_key_env if llm_cfg else "ANTHROPIC_API_KEY"

    rows = []
    for row in raw.iter_rows(named=True):
        fields = extract_fields_from_text(
            row["text"],
            fields_spec=SCHEMA,
            model=model,
            cache_path=cache_path,
            api_key_env=api_key_env,
        )
        rows.append({"document_id": row["document_id"], **fields})

    return pl.DataFrame(rows)
