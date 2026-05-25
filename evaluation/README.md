# evaluation

Two-stage evaluation of the trained cross-encoder on Russian-language IR
benchmarks. A BM25 first stage retrieves the top-100 passages per query
per benchmark; the trained reranker scores those 100 candidates; the
final metrics are averaged across the benchmark suite.

The two stages are decoupled so that the expensive first stage can be
computed once and cached on disk. Hyperparameters live in
`configs/thesis/eval.toml`; paths come from `.env`.

## Entrypoints

- `generate_bm25_top100.py` — builds a BM25 index for each benchmark in
  the suite and writes the top-100 candidate passages per query to
  `${EVAL_BM25_TOP100_PATH}`. Run once, cache forever.
- `evaluate_reranker_top100.py` — loads the cached top-100, scores each
  candidate with the trained cross-encoder, computes NDCG@10 (primary)
  and MAP@10, MRR@10, P@1, P@3, P@5 (secondary), and writes the
  per-benchmark and averaged results to `${EVAL_RESULTS_PATH}`.

Both scripts accept `--config configs/thesis/eval.toml` and `--help`.
The runner `evaluation/run.sh` chains both in order.

## Configuration

`configs/thesis/eval.toml` carries the suite-level knobs:

| Key                  | Default        | Description                                            |
|----------------------|----------------|--------------------------------------------------------|
| `benchmark`          | `RusBEIR`      | Name of the benchmark suite to evaluate against.       |
| `first_stage`        | `bm25`         | First-stage retriever; currently only `bm25`.          |
| `first_stage_top_k`  | `100`          | Candidates per query passed to the reranker.           |
| `primary_metric`     | `ndcg@10`      | Metric reported as the headline result.                |
| `secondary_metrics`  | `[map@10, mrr@10, p@1, p@3, p@5]` | Metrics also reported per benchmark.    |

## Data isolation

Benchmarks are screened for overlap with the training corpus before
inclusion in the suite. Two regimes apply:

- **Direct overlap** — any benchmark whose passage corpus contains
  passages that appear (verbatim or near-duplicate) in the labelled
  training pairs is excluded entirely. There is no safe way to evaluate
  a reranker on passages it has trained against.
- **Corpus-only overlap** — benchmarks whose passage corpus shares
  documents with the training corpus but whose relevance judgements are
  disjoint from any training pair are retained, with a documented risk
  note: the model has seen the documents but never with the benchmark's
  labels.

The accepted benchmark list and the per-benchmark overlap classification
are part of the suite definition picked up via the `benchmark` key.
