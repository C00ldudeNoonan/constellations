# Summary page experiment

Run from the repository root in the locked environment with the `text` extra:

```bash
uv run python examples/summary_pages/compare.py
```

This prints a deterministic `summary_page_compare/v1` report for three
synthetic cases (broad, multi-section, exact-identifier) with two arms,
passages and pages plus passages, plus the maintenance work one chunk edit
causes. No warehouse, inference provider or credentials are involved; the
summarizer is the offline `LeadSentenceSummarizer`. The tokenizer may download
its public encoding vocabulary on first use.

The fixture exercises mechanics. Its coverage numbers are not ALE-281 answer
golds and the report's `decision` stays `defer`. See
[the contract](../../docs/architecture/summary-pages.md).
