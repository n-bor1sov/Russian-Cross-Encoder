# Stage 3 — Consistency Filtering

Drop labelled pairs whose positive passage is not recoverable from the
dataset-local dense index, using a virtual-rank fallback that tolerates
pairs whose positive is one of several equally-valid passages.

## What it does

For each labelled pair `(q, p+)` the stage searches the dataset-local
passage index with the embedding of `q` and reads back the top results.
The pair is consistent if `p+` appears within the top `k`. Pairs that
fail this raw test are not immediately dropped: they are re-tested under
a virtual rank, computed by temporarily removing other labelled positives
of the same query from the candidate pool. A pair survives if its
positive is found within `k` results under either the raw or the virtual
rank.

The two-step test handles the common case where a single query has
several legitimate positives and the dense retriever simply ranks a
different valid positive first. Without the virtual-rank fallback these
pairs would be filtered out even though they are correct.

The stage writes both a filtered Parquet and a restricted set of FAISS
indices that retain only embeddings of surviving queries and passages.
Downstream stages (hard-negative mining, compilation) consume these
filtered artefacts.

## Inputs

- `${SPLIT_DATA_DIR}/{train,val}/*.parquet` — split shards from stage 1.
- `${EMBEDDINGS_DIR}/<dataset>/*` — per-dataset embeddings and FAISS
  indices from stage 2.

## Outputs

- `${FILTERING_DATA_ROOT}/filtered/*.parquet` — surviving labelled
  pairs, partition-preserving.
- `${EMBEDDINGS_DIR}/filtered/<dataset>/*` — per-dataset embeddings and
  FAISS indices restricted to surviving identifiers.

## Key parameters

| Key                 | Default | Description                                                                            |
|---------------------|---------|----------------------------------------------------------------------------------------|
| `k`                 | `30`    | Top-`k` cutoff for the raw retrieval rank and virtual rank tests.                       |
| `k_max`             | `100`   | Upper search depth used while computing virtual ranks.                                 |
| `top_k_for_restore` | `30`    | Depth retained when restoring per-dataset embedding files for surviving identifiers.   |

All keys live in `configs/thesis/03_filter_consistency.toml`. CLI flags
override TOML.

## Files

- `score.py` — scores each pair with raw retrieval rank and virtual
  rank; produces an intermediate scored Parquet.
- `filter_by_rank.py` — applies the `k` cutoff to the scored Parquet
  and writes the surviving rows.
- `restore_embeddings.py` — rebuilds per-dataset embedding files and
  FAISS indices restricted to surviving identifiers.
- `restore_only.py` — variant entrypoint that only re-runs the restore
  step against an already-scored Parquet; imports from `score.py`.
- `run.sh` — shell wrapper that chains score → filter → restore.

## Running

    bash pipeline/03_filter_consistency/run.sh

To restore the embedding view without re-scoring (e.g. after tweaking
`top_k_for_restore`):

    python pipeline/03_filter_consistency/restore_only.py \
      --config configs/thesis/03_filter_consistency.toml

## Notes

- Filtering is dataset-relative: a pair is judged against the index of
  its own dataset. Attrition rates therefore differ across datasets and
  are not directly comparable.
- A common diagnostic: if a dataset shows near-zero survival, its
  positive passages are probably absent from the dataset's own passage
  pool — check that `data_prep/` produced the right `passage_id`
  population for that source.
- `k=30` is the documented default; per-dataset deviations used in the
  thesis run are listed in `docs/REPRODUCING_THESIS.md`.
