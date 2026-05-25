# Stage 4 — Hard Negative Mining

For each surviving query, pick passages that are similar to the query
but fall below the mean same-language positive similarity by at least a
fraction `δ`. Search depth is adaptive.

## What it does

The miner walks the surviving queries one at a time. For each query it
computes the mean cosine similarity between the query and its labelled
same-language positives; this becomes the per-query reference. A
candidate passage is accepted as a hard negative iff its similarity to
the query is below the reference by at least a fraction `δ` of the
reference, i.e. similarity strictly less than `(1 − δ) · reference`.
Candidates that fail this margin test are treated as too close to the
positive and are skipped.

Search depth starts at `k0` and grows by doubling up to `k_max` until
at least `n_min` valid hard negatives are found. Queries that still
have fewer than `n_min` valid negatives at depth `k_max` are dropped
entirely from the training set.

Cross-lingual positives of the query are removed from the candidate
pool before the margin test. Because cross-lingual variants of the
same conceptual pair share their `query_id`, the miner can identify
them by ID rather than by content. Without this exclusion a Russian
positive could be mined as a hard negative for an English variant of
the same pair, producing a contradictory training signal.

## Inputs

- `${FILTERING_DATA_ROOT}/filtered/*.parquet` — surviving labelled
  pairs from stage 3.
- `${EMBEDDINGS_DIR}/filtered/<dataset>/*` — per-dataset filtered
  embeddings and FAISS indices.

## Outputs

- `${HNM_ROOT}/<dataset>/*.parquet` — per-query records of mined hard
  negatives. Each row carries the `query_id`, the list of mined
  `passage_id`s, and their dense similarities.

## Key parameters

| Key                                | Default | Description                                                              |
|------------------------------------|---------|--------------------------------------------------------------------------|
| `seed`                             | `42`    | Random seed for tie-breaking and any sub-sampling.                       |
| `k0`                               | `50`    | Initial FAISS search depth.                                              |
| `k_max`                            | `1000`  | Maximum adaptive search depth before giving up.                          |
| `n_min`                            | `8`     | Minimum number of valid hard negatives required; under this, drop query. |
| `delta`                            | `0.05`  | Relative margin parameter; candidate must be `(1 − δ)` below reference.  |
| `exclude_cross_lingual_positives`  | `true`  | If true, removes cross-lingual variants of positives from the pool.      |

All keys live in `configs/thesis/04_hnm.toml`. CLI flags override TOML.

## Files

- `mine.py` — Python entrypoint; runs the adaptive search and margin
  test.
- `embedding_lookup.py` — helper that joins mined `passage_id`s back to
  their texts and metadata.
- `slurm/HNM.sh`, `slurm/HNM_live.sh`, `slurm/kill_hnm.sh`,
  `slurm/status_hnm.sh` — cluster launchers for the distributed
  variant.
- `run.sh` — single-node shell wrapper.

## Running

    bash pipeline/04_hard_negative_mining/run.sh

To mine without the cross-lingual exclusion (for ablation work):

    bash pipeline/04_hard_negative_mining/run.sh --exclude-cross-lingual-positives false

For multi-node cluster runs, see the `slurm/` launchers; they invoke
`mine.py` with the same TOML config.

## Notes

- Cross-lingual exclusion is on by default and should stay on for any
  multilingual training run; turning it off introduces contradictory
  signal across language variants.
- Queries with too few negatives are dropped silently; check the
  per-dataset survival counts in the stage log to spot datasets whose
  passage pool is too small for the configured `n_min`.
- `δ` is relative, not absolute. A very high mean positive similarity
  (e.g. for tight annotation schemes) tightens the margin in absolute
  terms; loosen `δ` if the miner consistently drops too many queries.
- For ablation runs that skip stage 3, point this stage at the
  unfiltered embeddings via `HNM_INPUT_DIR=${EMBEDDINGS_DIR}`.
