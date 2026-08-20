# pdf_invoice_pipeline — the stel thesis demo

This pipeline takes **actual PDFs** all the way to **structured DuckDB tables**:

```
invoice_pdfs (source: .pdf files)
    └── raw_pdf_text         (pdf backend: pypdf → text + page_count)
            └── extracted_invoices  (transform: per-row LLM call → structured fields)
```

The `raw_pdf_text` model uses stel's `pdf` backend (pypdf, deterministic, no
API). The `extracted_invoices` transform reads each row's text and asks Claude
to fill in a JSON schema. LLM responses are cached in `target/llm_cache.duckdb`
keyed on `(model, content_hash, schema_hash)`, so re-runs over the same PDFs
don't re-pay for tokens.

The transform uses the second-arg `TransformContext` to read its LLM
configuration (model id, cache path) from the active profile — no credentials
inline.

## Run

```bash
export ANTHROPIC_API_KEY=...    # required for the LLM extraction step

uv run stel --project-dir examples/pdf_invoice_pipeline seed --count 5
uv run stel --project-dir examples/pdf_invoice_pipeline run
uv run stel --project-dir examples/pdf_invoice_pipeline test
uv run stel --project-dir examples/pdf_invoice_pipeline show extracted_invoices
```

Re-run `run` immediately — the second pass should report 5 docs skipped on the
PDF extraction (incremental) AND every LLM call cached, so the transform is
near-instant.
