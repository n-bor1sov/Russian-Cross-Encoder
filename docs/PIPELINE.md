# Pipeline

This document is a narrative walkthrough of the six pipeline stages that turn a
labelled query–passage corpus into a trained Russian cross-encoder reranker.
Each stage is self-contained and ships its own `README.md` inside its folder
under `pipeline/`; this document gives the high-level picture and the
inter-stage contracts. For a per-stage parameter reference, follow the
`Code:` pointers at the end of each section.

## Input data requirements

The pipeline expects a single unified corpus of labelled query–passage pairs.
Every input shard is a Parquet file with the following columns:

| Column        | Type   | Description                                                    |
|---------------|--------|----------------------------------------------------------------|
| `query`       | string | The query text                                                 |
| `passage`     | string | The labelled positive passage for the query                    |
| `query_id`    | string | Stable identifier for the query                                |
| `passage_id`  | string | Stable identifier for the passage                              |
| `lang`        | string | ISO 639-1 language code                                        |
| `dataset`     | string | Source dataset name                                            |

Additional constraints, all enforced by downstream stages:

- All strings are UTF-8 and NFC-normalised.
- No null or empty values in any column.
- Identifiers are stable: the same query instance always carries the same
  `query_id`, the same passage instance always carries the same `passage_id`.
- Cross-lingual coupling: if a source dataset ships the same conceptual
  query–passage pair in multiple languages, all language variants share a
  single `query_id` (and likewise for `passage_id`). The splitter and the
  hard-negative miner rely on this to keep language variants aligned and to
  avoid treating one variant as a negative for another.
- Parquet with Snappy compression is the recommended on-disk format.

Helpers for normalising raw sources to this schema live in `data_prep/`.

## Pipeline overview

The pipeline is six sequential stages: a 95:5 train/validation split, dense
embedding of every query and passage, consistency filtering against the
dataset-local dense index, hard-negative mining, compilation into a
fixed-column training Parquet, and cross-encoder fine-tuning. Stages 2, 3,
and 4 share the same frozen dense retriever as their backbone — the same
embedding model and the same per-dataset FAISS indices — so the labelled
data, the filtering signal, and the mined negatives all come from a single,
consistent dense view of the corpus.

## Splitting

The splitter produces a 95:5 train/validation partition over the unified
corpus. It does not split at the row level. Instead, it builds the bipartite
graph of queries and passages and assigns every connected component to one
partition, so that a query found in one partition cannot share any passage
(positive or otherwise) with a query in the other partition. This prevents
trivial leakage of the labelled positives across the train/val boundary.

Cross-lingual variants of the same conceptual pair share their `query_id`
and `passage_id`, which means the bipartite graph already glues them into a
single component. As a result, all language variants of a given pair end up
in the same partition together, and language-specific evaluation remains
meaningful.

The split is seeded; rerunning with the same seed and the same input
produces the same partition. Component sizes vary across source datasets,
so the realised per-dataset split ratio fluctuates around 95:5 and is not
forced to that ratio exactly.

Code: `pipeline/01_split/`

## Embedding

The embedding stage runs every query and every passage in both partitions
through a frozen dense retriever (an OpenAI-compatible embedding endpoint;
the thesis run uses Qwen3-Embedding at 1024 dimensions). Local tokenisation
is used to truncate inputs to a fixed token budget before the API call.

The output of this stage is a set of per-dataset FAISS indices, one
index per source dataset, alongside the raw embedding tensors keyed by
identifier. The indices are flat (exact search). Building one index per
dataset rather than a single global index is a deliberate choice: it keeps
memory tractable, and it scopes consistency filtering and hard-negative
mining to the dataset whose annotation guidelines produced the labelled
pair, which avoids contradictory signals across heterogeneous sources.

Embedding is the only stage that requires an external API. The remaining
stages operate on the saved embeddings and FAISS indices.

Code: `pipeline/02_embed/`

## Consistency Filtering

Consistency filtering removes labelled pairs whose positive passage is not
recoverable from the dense index. For each pair `(q, p+)`, the stage
searches the dataset-local FAISS index with the embedding of `q` and looks
for `p+` in the top-`k` results. A pair survives if its positive is found
within the cutoff.

To avoid penalising pairs whose positive is just one of several valid
passages for the same query, the stage also computes a virtual rank: it
re-runs the search after temporarily removing other labelled positives of
the same query from the candidate pool. If the positive shows up within the
virtual cutoff, the pair survives even when its raw retrieval rank exceeded
`k`.

Filtering produces both a filtered Parquet (for downstream stages) and a
restricted set of FAISS indices that contain only embeddings of surviving
queries and passages. The filtering decision is dataset-relative: a pair
is judged consistent or inconsistent against the index of its own dataset.
As a result, attrition rates differ across datasets and are not directly
comparable.

Code: `pipeline/03_filter_consistency/`

## Hard Negative Mining

Hard-negative mining selects, for each surviving query, a small set of
hard negative passages — passages that are similar to the query but
distinguishable from the labelled positive by the dense retriever. The
selection rule is margin-based: a candidate is kept only if its similarity
to the query falls below the mean same-language positive similarity for
that query by at least a fraction `δ`.

Search depth is adaptive. The miner starts at `k0` results and grows the
search depth up to `k_max` until at least `n_min` valid negatives are
found, or the depth ceiling is reached. Queries with fewer than `n_min`
valid negatives are dropped.

Cross-lingual positives are explicitly excluded from the candidate pool.
Because cross-lingual variants of the same conceptual pair share a
`query_id`, the miner can identify them and refuses to treat one variant
as a negative for another. Without this exclusion the training signal
would contradict itself across language variants of the same pair.

The output is a per-query record of mined hard negatives, ready for the
compilation stage.

Code: `pipeline/04_hard_negative_mining/`

## Compilation

The compilation stage turns the filtered pairs and the mined hard
negatives into a single training Parquet. For each surviving query it
materialises one row with the query, its labelled positive, and a fixed
number of hard negatives.

For multilingual sources that ship the same conceptual pair in multiple
languages, the stage performs cross-lingual augmentation: every base
`(q, p+)` row is expanded into the four cross-product variants over the
two languages of the query and the positive. Hard negatives are paired by
matching language to the query side, so the model never sees a row whose
query and negative are in different languages.

The resulting on-disk schema is fixed at `(query, positive, negative_1,
…, negative_N)` with `N = negatives_per_query`. A separate `analyze.py`
helper reports per-dataset statistics for the compiled Parquet but is not
part of the training path.

Code: `pipeline/05_compile/`

## Training

Training is a single-epoch cross-encoder fine-tune over the compiled
Parquet, distributed across multiple GPUs via `torchrun`. The loss is
Multiple Negatives Ranking Loss (MNRL) with a temperature scale; per
query the loss sees `k_h` hard negatives drawn from the row and `k_r`
random in-batch negatives drawn from other queries in the same dataset.

Batches are dataset-homogeneous: a custom sampler groups queries of the
same source dataset into the same step on every GPU. This keeps the
in-batch random-negative pool drawn from a single annotation regime and
avoids cross-dataset contamination of the contrastive signal. Bucketing
is performed once up front by `prepare_buckets.py`; the runner invokes
this idempotently before launching `torchrun`.

Two memory tricks are enabled by default: gradient caching, which lets
MNRL operate on the full per-step batch without holding all activations
in memory at once, and gradient checkpointing, which trades recomputation
for activation memory. Validation runs about twenty times per epoch, and
checkpoint selection uses MAP@10 on the held-out validation parquet.

Code: `pipeline/06_train/`

## Evaluation

Evaluation is two-stage retrieval-then-rerank. A BM25 first stage produces
the top-100 passages per query on each Russian-language IR benchmark in
the suite; the trained cross-encoder then reranks those 100 candidates and
the suite-level metrics are averaged. NDCG@10 is the primary metric; MAP@10,
MRR@10, P@1, P@3, and P@5 are reported as secondary metrics.

The two-stage entrypoints are decoupled so that the expensive first stage
can be computed once and cached. `generate_bm25_top100.py` writes the
cached top-100 candidates to `EVAL_BM25_TOP100_PATH`; `evaluate_reranker_top100.py`
reranks the cache and writes a metrics JSON to `EVAL_RESULTS_PATH`. Both
read their hyperparameters from `configs/thesis/eval.toml`. The benchmarks
included in the suite are screened for data overlap with the training
corpus: any benchmark whose passage corpus has direct overlap with the
labelled training pairs is excluded; benchmarks with corpus-only overlap
(shared documents but disjoint relevance judgements) are retained with a
documented risk note.

Code: `evaluation/`
