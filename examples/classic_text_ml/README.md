# Classic Text ML Example

This example is the first executable slice of stel's classic text and
document ML lane. It flows from JSON extraction into a built-in TF-IDF feature
model.

The ML model writes a long-form feature table and persists metadata under
`target/artifacts/ticket_tfidf/`.

```bash
uv run stel --project-dir examples/classic_text_ml compile
uv run stel --project-dir examples/classic_text_ml seed --type tickets --count 20
uv run stel --project-dir examples/classic_text_ml run
```
