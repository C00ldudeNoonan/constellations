# document_clustering

Unsupervised Classic ML over a small committed document corpus (issue #42):
TF-IDF features → k-means `cluster` and NMF `topic_model`.

Requires the `ml` extra (scikit-learn):

```
uv sync --extra ml           # or: pip install 'dbt-ml[ml]'
uv run dbt-ml run
```

No `seed` step — the corpus under `corpus/` is committed. `dbt-ml run` builds:

- `doc_clusters` — one row per document (`cluster`, `distance`), plus
  `doc_clusters__representative_docs` (documents nearest each centroid).
- `doc_topics` — one row per document × topic (`weight`), plus
  `doc_topics__topics` (top terms per topic).

The corpus has three obvious themes (sports, finance, cooking): the three NMF
topics line up with them cleanly, and the k-means clusters approximate them.
Swap the cluster provider to `builtin.hdbscan` or `builtin.dbscan` for
density-based grouping.
