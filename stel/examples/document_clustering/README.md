# document_clustering

Unsupervised Classic ML over a small committed document corpus (issue #42):
TF-IDF features → k-means `cluster` and NMF `topic_model`.

Requires the `ml` extra (scikit-learn):

```
uv sync --extra ml           # or: pip install 'stel[ml]'
uv run stel run
```

No `seed` step — the corpus under `corpus/` is committed. `stel run` builds:

- `doc_clusters` — one row per document (`cluster`, `distance`), plus
  `doc_clusters__topics` (c-TF-IDF top terms per cluster),
  `doc_clusters__representative_docs` (documents nearest each centroid), and
  `doc_clusters__neighbors` (each document's nearest neighbors).
- `doc_topics` — one row per document × topic (`weight`), plus
  `doc_topics__topics` (top terms per topic).

`builtin.kmeans` and `builtin.nmf` also support `mode: load_pretrained` to
assign new documents to an already-fitted model.

The corpus has three obvious themes (sports, finance, cooking): the three NMF
topics line up with them cleanly, and the k-means clusters approximate them.
Swap the cluster provider to `builtin.hdbscan` or `builtin.dbscan` for
density-based grouping.
