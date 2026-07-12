# Classic ML Text and Document Models

dbt-ml should treat classic text and document ML as a primary lane, not just a
fallback for LLM or RAG work. Count vectors, TF-IDF, hashing features,
classifiers, clustering, topic models, named-entity enrichment, deduplication,
and deterministic metrics are cheaper, easier to reproduce, and often enough
for production document pipelines.

This page defines the v0.2 design contract. The executable slice is
`task: features` with built-in count, TF-IDF, and hashing vectorizers plus
`task: classifier` with built-in multinomial Naive Bayes. Roadmap task names
remain part of the configuration vocabulary, but fail compile until an
executable provider exists.

## Model Shape

Classic ML models use the same model list and DAG rules as extraction and
Python transform models. They declare dependencies with `depends_on` and use an
`ml:` block for task, mode, provider, artifact, metrics, and provider options.

```yaml
models:
  - name: ticket_tfidf
    depends_on: [ref('raw_tickets')]
    ml:
      task: features
      mode: fit_transform
      provider: builtin.tfidf
      text_field: body
      artifact:
        path: target/artifacts/ticket_tfidf
      metrics: [row_count, vocabulary_size]
      options:
        ngram_range: [1, 2]
        min_df: 3
        max_features: 50000
```

## Feature Providers

The built-in feature providers are pure Python and do not require
scikit-learn:

| provider | Output | Artifact behavior |
| --- | --- | --- |
| `builtin.count` | Long-form sparse count features. `binary: true` stores presence/absence counts. | Persists `metadata.json` and `vocabulary.json`. |
| `builtin.tfidf` | Long-form sparse TF-IDF features with `tf`, `idf`, `tfidf`, and `value`. | Persists `metadata.json` and `vocabulary.json`. |
| `builtin.hashing` | Stateless long-form hashed features with `hash_bucket` and `value`. | Persists `metadata.json` only. |

Common vectorizer options:

| option | Meaning |
| --- | --- |
| `analyzer` | `word`, `char`, or `char_wb`; defaults to `word`. |
| `ngram_range` | Two-item range such as `[1, 2]`. |
| `token_pattern` | Regex used by the word analyzer. |
| `stop_words` | List of terms or `english` for a small built-in English stop-word set. |
| `min_df` / `max_df` | Document-frequency filters. Integers are document counts; floats from 0 to 1 are proportions. |
| `max_features` | Keep the highest-frequency vocabulary terms before sorting the vocabulary. |
| `binary` | For count/TF-IDF, collapse repeated terms in a document to a count of 1. |
| `n_features` | Hash bucket count for `builtin.hashing`; defaults to `1048576`. |
| `alternate_sign` | Hashing option that alternates signs by hash value; defaults to `true`. |

Provider options are strict and provider-specific. Unknown keys, quoted
booleans, invalid regular expressions, non-positive sizes, and options that a
provider would ignore fail compile instead of being coerced or silently
dropped.

All feature providers materialize sparse rows with stable identifiers and
feature metadata. Vocabulary-based providers expose terms through
`vocabulary.json`; hashing is useful when users need fixed-width features
without fitting or storing a vocabulary.

## Tasks

The first task set is intentionally small and maps to stable production
patterns:

| task | Purpose | Likely first providers |
| --- | --- | --- |
| `features` | Count, TF-IDF, hashing, dense/sparse document features | `builtin.count`, `builtin.tfidf`, `builtin.hashing` |
| `classifier` | Supervised document or row classification | `builtin.naive_bayes`, later `sklearn.logistic_regression`, `sklearn.linear_svc`, `sklearn.sgd_classifier` |
| `regressor` | Supervised numeric prediction | `sklearn.linear_model`, `sklearn.ensemble` |
| `cluster` | Document clustering and nearest-neighbor grouping | `sklearn.kmeans`, later HDBSCAN-style providers |
| `topic_model` | Topic discovery and document-topic tables | `sklearn.nmf`, `sklearn.lda` |
| `nlp` | Tokenization, POS, NER, sentiment, key phrases | `spacy`, lightweight built-ins |

Providers should remain optional dependencies. The base package should keep
working for extraction and pure-Python transforms without installing
scikit-learn or spaCy.

Only `features` and `classifier` are executable today. `regressor`, `cluster`,
`topic_model`, and `nlp` are roadmap contracts and are rejected during
preflight rather than failing after warehouse work begins.

## Supervised Classification

The first executable classifier is `builtin.naive_bayes`, a pure-Python
multinomial Naive Bayes provider for deterministic text classification. It
uses the same analyzer, n-gram, stop-word, and document-frequency options as
feature providers, plus `alpha` for smoothing.

```yaml
models:
  - name: ticket_priority_nb
    depends_on: [ref('raw_tickets')]
    ml:
      task: classifier
      mode: fit_transform
      provider: builtin.naive_bayes
      text_field: summary
      label_field: priority
      metrics: [accuracy, class_count, vocabulary_size]
      options:
        ngram_range: [1, 2]
        min_df: 1
        alpha: 1.0
```

`fit_transform` trains the classifier, persists `model.json`, and materializes
one prediction row per input row with `prediction`, `score`, JSON
`probabilities`, and `correct` when labels are present. `fit` writes artifact
metadata only, while `predict` and `load_pretrained` reuse a persisted artifact
against new rows.

## Modes

The same grammar needs to support training and applying artifacts:

| mode | Meaning |
| --- | --- |
| `fit_transform` | Fit from upstream rows and materialize transformed output in one run. |
| `fit` | Fit and persist an artifact, optionally emitting metadata/metrics only. |
| `predict` | Load an artifact and materialize predictions/features for incoming rows. |
| `load_pretrained` | Apply a compatible dbt-ml-native artifact produced elsewhere, without fitting it. |

For `predict` and `load_pretrained`, provider options come from the persisted
artifact. Supplying `ml.options` on those modes is an error because the current
runtime would otherwise ignore them. Generic scikit-learn or third-party model
files are not accepted yet; pretrained artifacts must use the dbt-ml schema and
payload contract described below.

## Artifact Contract

Artifacts should live under `target/artifacts/<model_name>/` by default unless
the user provides `ml.artifact.path`. A provider may write multiple files
there, but it must expose a stable metadata record and register the latest
artifact in `target/artifacts/registry.json`.

Artifact paths are dedicated directories: they cannot overlap project source,
model, transform, target metadata, or another artifact directory. When a
prediction model shares a path with an in-project fit model, there may be only
one writer and the reader declares both dependencies in this order:

```yaml
depends_on: [ref('prediction_input'), ref('artifact_writer')]
```

The first dependency is always the data relation; the second is an ordering
edge that guarantees the artifact writer runs first. Unrelated extra ML
dependencies fail compile.

```json
{
  "artifact_schema_version": 2,
  "artifact_type": "classic_ml",
  "model_name": "ticket_tfidf",
  "task": "features",
  "provider": "builtin.tfidf",
  "mode": "fit_transform",
  "artifact_version": "16-char-hash",
  "artifact_files_hash": "16-char-hash",
  "code_version": "16-char-hash",
  "config_hash": "16-char-hash",
  "runtime": {
    "python": "3.12.4",
    "dbt_ml": "0.1.0",
    "polars": "1.x",
    "provider": "builtin.tfidf"
  },
  "training_input": {
    "refs": ["raw_tickets"],
    "row_count": 1000,
    "content_hash": "16-char-hash"
  },
  "metrics": {
    "vocabulary_size": 12000
  },
  "files": ["metadata.json", "vocabulary.json"]
}
```

`manifest.json` contains the static `ml:` config and `code_version`.
`run_results.json` includes artifact path, artifact version, training input,
metrics, and the full artifact metadata for executed ML models. Generated docs
render the same metadata on the model page when a run result is available.

An empty `ml.metrics` list reports every metric supported by the provider;
otherwise it selects the named subset. `ml.artifact.include_metrics: false`
keeps run-result metrics but omits them from persisted artifact metadata and
the artifact registry.

Prediction modes validate artifacts before materializing output:

- missing metadata or payload files fail with a missing-artifact error;
- unsupported schema versions or provider/task mismatches fail with an
  incompatible-artifact error;
- changed payload bytes or edited metadata fail with a stale-artifact error.

## Versioning

Classic ML versioning should account for:

- the `ml:` config block;
- provider name and provider options;
- project-local code if a custom provider is used;
- upstream training data hash or declared artifact hash;
- dependency/provider version when it affects output.

The `ml:` block is included in `code_version`, which covers static config. The
built-in feature executors also record a training input hash, config hash,
runtime versions, payload hash, and artifact version in `run_results.json`.

## Initial Examples

`examples/classic_text_ml/` is a runnable support-ticket feature and
classification pipeline using `builtin.tfidf`, `builtin.count`,
`builtin.hashing`, and `builtin.naive_bayes`.
