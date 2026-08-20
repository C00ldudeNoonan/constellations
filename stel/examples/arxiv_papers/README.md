# arxiv_papers — traditional ML data-quality checks on document metadata

This pipeline is the demonstration vehicle for **deterministic, traditional ML
data-quality checks** ([issue #10](https://github.com/C00ldudeNoonan/dbt-ml/issues/10)) —
no LLM judge, no sampling, fully reproducible.

```
arxiv_papers (source: *.json)
    └── raw_papers          (json backend → typed table + quality checks)
            └── papers_by_category  (transform: per-category counts)
```

The `raw_papers` model runs the full deterministic quality battery:

| Check | What it catches |
|-------|-----------------|
| `not_null` / `unique` | missing or duplicate ids |
| `matches_regex` | malformed arXiv ids (must be `YYMM.NNNNN`) |
| `accepted_values` | a `primary_category` outside the known arXiv CS/stats set |
| `accepted_range` | implausible author counts |
| `null_rate` | silent extraction failures (the #1 LLM failure mode) |
| **`grounded_in`** | **a title that doesn't appear in its abstract — a deterministic faithfulness proxy that catches hallucinated values with zero LLM calls** |

## Run it (synthetic, offline)

```bash
uv run stel --project-dir examples/arxiv_papers seed --count 50 --type arxiv
uv run stel --project-dir examples/arxiv_papers run
uv run stel --project-dir examples/arxiv_papers test
uv run stel --project-dir examples/arxiv_papers show papers_by_category
```

## Run it on real arXiv data

```bash
# pull real papers from the arXiv API into ./data/papers/
uv run python examples/arxiv_papers/scripts/fetch_arxiv.py --category cs.LG --count 50
uv run stel --project-dir examples/arxiv_papers run
uv run stel --project-dir examples/arxiv_papers test
```

On real data the quality checks become genuinely informative — e.g. if a paper's
extracted title doesn't appear in its abstract, `grounded_in` flags it.

## Why this dataset

arXiv records come with **ground truth** (the API's structured metadata), which
is exactly what makes data-quality checks meaningful — you have something to
check against. It's the text-world analog of why people reach for MNIST to demo
quality: labels. (MNIST itself is deferred until stel grows image backends.)
