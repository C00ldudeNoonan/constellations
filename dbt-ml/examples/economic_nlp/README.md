# Economic NLP Example

This project turns a small committed economic-policy corpus into normalized
token and named-entity child tables. It uses the optional spaCy integration;
CI validates the same provider contract with a deterministic fake and does not
download a language model.

```bash
uv sync --extra nlp
uv run python -m spacy download en_core_web_sm
uv run dbt-ml --project-dir examples/economic_nlp build
```

`document_tokens` emits one row per non-space token. `document_entities` emits
one row per named entity, but deliberately omits the matched entity text.
Both tables carry stable child IDs, model identity, and the explicitly allowed
`publisher` and `published_at` fields. Neither table retains the source
document's raw `text` column.
