# HyDE, re-run with current OpenAI models

A small, self-contained re-run of **HyDE** — Gao et al., *Precise Zero-Shot Dense
Retrieval without Relevance Labels*, ACL 2023 — on the SciFact benchmark, using
only OpenAI models.

```
plain:  query ──► embedding ──────────────────────────────────────► nearest documents
HyDE:   query ──► LLM writes 8 hypothetical documents ──► embeddings
              ──► average them with the query embedding (Eq. 8) ──► nearest documents
```

The idea: a short query and a long scientific abstract look different to an
embedding model. HyDE asks an LLM to *write* a passage that would answer the query —
facts may be wrong — and searches with the embedding of that passage instead. The
fake document lands near real documents that look like it.

## Setup as in the paper

| | |
|---|---|
| Dataset | BEIR SciFact test set: 300 claims, 5,183 abstracts |
| Prompt | Appendix A.1: *"Please write a scientific paper passage to support or refute the claim"* |
| Hypothetical documents | 8 per query (as in the official code), temperature 0.7, max 512 tokens |
| Generator | `gpt-6-luna`, reasoning off |
| Combination | mean of the query vector and the 8 document vectors (Eq. 8) |
| Search | exact: every document scored by dot product (OpenAI embeddings are unit length) |
| Metrics | nDCG@10 and Recall@100 |

The hypothetical documents are generated once and cached
(`cache/scifact_gpt-6-luna_hypothetical_docs.json`, included), so every embedding
model is tested on exactly the same text.

## Results

| Embedding model | nDCG@10 plain | nDCG@10 HyDE | Diff | 95% CI of diff | Recall@100 plain → HyDE |
|---|---|---|---|---|---|
| `text-embedding-3-large` | 77.7 | 77.9 | +0.1 | [−1.6, +2.0] | 97.7 → 98.0 |
| `text-embedding-3-small` | 73.1 | 75.1 | +2.1 | [−0.1, +4.2] | 97.3 → 97.0 |
| `text-embedding-ada-002` | 72.8 | 74.9 | +2.1 | [−0.1, +4.3] | 96.3 → 97.3 |

The confidence interval comes from a paired bootstrap over the 300 queries
(10,000 resamples). Per-query scores are in `results_scifact_*.csv`.

**Reading it:**
- HyDE helps the older and smaller embedding models by about two nDCG points and
  does nothing measurable for `text-embedding-3-large`.
- None of the three differences is clearly separated from noise on 300 queries:
  every interval touches or crosses zero.
- For most queries the ranking does not change at all (227, 208 and 199 of 300 have
  identical nDCG@10). The effect comes from a minority of queries.

The paper's gain on SciFact was over the *unsupervised* Contriever encoder: 64.9 →
69.1 nDCG@10 (Table 2). Strong modern embedding models already close most of the gap
between a short claim and a long abstract, which leaves HyDE little to add — at the
cost of 8 LLM calls per query.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "OPENAI_API_KEY=..." > .env
.venv/bin/python main.py                                                 # text-embedding-3-large
.venv/bin/python main.py text-embedding-3-small text-embedding-ada-002   # other models
```

The dataset downloads on the first run (`data/`, not committed). Document
embeddings are cached in `cache/*.npy` (not committed, ~120 MB); the hypothetical
documents are already in `cache/`, so a re-run only pays for embeddings.
Set `DATASET = "nfcorpus"` in `main.py` to try a corpus with very short queries.

SciFact (Wadden et al., 2020) is licensed CC BY-NC 4.0; the result CSVs quote its
300 test claims.
