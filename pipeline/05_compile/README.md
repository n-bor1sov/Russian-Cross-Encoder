# Stage 5 — Compilation

Assemble the filtered pairs and the mined hard negatives into a single
training Parquet with a fixed column layout, expanding multilingual
sources by cross-lingual augmentation.

## What it does

The compiler reads the surviving labelled pairs from stage 3 and the
mined hard-negative records from stage 4. For each query it emits one
training row carrying the query text, the labelled positive text, and
a fixed number of hard negatives (`negatives_per_query`). Rows whose
mined-negative pool is shorter than `negatives_per_query` are dropped.

For multilingual sources where the same conceptual pair appears in
multiple languages, cross-lingual augmentation produces all four
cross-product variants per base pair: (query in language A, positive
in language A), (query in language A, positive in language B), (query
in language B, positive in language A), (query in language B,
positive in language B). The compiler identifies these variants
through the shared `query_id` and `passage_id` and emits one
materialised row per variant.

Hard negatives are paired by language: the negative text written into
each row matches the language of the query side of that row. This
keeps every training row internally language-consistent and avoids
exposing the model to query–negative pairs where the two sides differ
in language.

A separate `analyze.py` helper computes per-dataset statistics on the
compiled Parquet (row counts, average sequence lengths, language
distribution). It is informational only and not part of the training
path.

## Inputs

- `${FILTERING_DATA_ROOT}/filtered/*.parquet` — surviving labelled
  pairs from stage 3.
- `${HNM_ROOT}/<dataset>/*.parquet` — per-query hard-negative records
  from stage 4.

## Outputs

- `${COMPILED_DATA_DIR}/train.parquet`,
  `${COMPILED_DATA_DIR}/val.parquet` — training-ready Parquets with the
  schema `(query, positive, negative_1, …, negative_N)` where
  `N = negatives_per_query`.

## Key parameters

| Key                     | Default | Description                                                                        |
|-------------------------|---------|------------------------------------------------------------------------------------|
| `cross_lingual_augment` | `true`  | If true, expands multilingual pairs into all four query/positive language variants.|
| `negatives_per_query`   | `8`     | Number of hard negatives written per row; rows with fewer are dropped.             |

All keys live in `configs/thesis/05_compile.toml`. CLI flags override TOML.

## Files

- `compile.py` — Python entrypoint; performs the join, augmentation,
  and materialisation.
- `analyze.py` — informational helper for per-dataset statistics on
  the compiled Parquet.
- `run.sh` — shell wrapper.

## Running

    bash pipeline/05_compile/run.sh

To compile without cross-lingual augmentation:

    bash pipeline/05_compile/run.sh --cross-lingual-augment false

To inspect the compiled corpus:

    python pipeline/05_compile/analyze.py --input "${COMPILED_DATA_DIR}/train.parquet"

## Notes

- If a downstream training run reports many rows being dropped at
  load time, check that `negatives_per_query` here matches the
  combined `k_h + k_r` the trainer expects.
- For the `k_h = 15` ablation, raise `negatives_per_query` to at
  least 15 here and re-run from this stage onwards.
- `analyze.py` is safe to run repeatedly; it produces only a stats
  JSON next to its `--input`.
