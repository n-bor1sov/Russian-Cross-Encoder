# configs

Per-stage TOML configuration for the pipeline. The canonical thesis
configuration lives in `configs/thesis/`, one file per pipeline stage
plus one for evaluation.

## Override rules

For every Python entrypoint, configuration values are resolved in this
order, highest precedence first:

1. **CLI flags** (e.g. `--batch-size-per-gpu 8`).
2. **TOML values** loaded from the file passed via `--config`.
3. **Argparse defaults** baked into the script.

This means a one-off run can override any TOML value from the command
line without editing the file, and a fresh checkout with no `--config`
flag still produces sensible (script-default) behaviour.

## Path convention

Paths and credentials do not live in TOML. They live in `.env` at the
repo root (template: `.env.example`). The shell runners under
`pipeline/<stage>/run.sh` `source .env` before launching the Python
entrypoint, so every script receives its paths via environment
variables. Configuration files therefore carry only algorithmic
parameters; they remain portable across machines.

## Files

`configs/thesis/` mirrors the pipeline structure:

- `01_split.toml` — train/validation splitter.
- `02_embed.toml` — embedding stage.
- `03_filter_consistency.toml` — consistency filtering.
- `04_hnm.toml` — hard negative mining.
- `05_compile.toml` — compilation.
- `06_train.toml` — training procedure.
- `eval.toml` — two-stage evaluation.

## Per-stage key reference

### `01_split.toml`

| Key            | Type  | Default | Description                                |
|----------------|-------|---------|--------------------------------------------|
| `seed`         | int   | `42`    | Random seed for component assignment.      |
| `val_fraction` | float | `0.05`  | Target fraction assigned to validation.    |

### `02_embed.toml`

| Key                | Type   | Default            | Description                                                |
|--------------------|--------|--------------------|------------------------------------------------------------|
| `model`            | string | `Qwen3-Embedding`  | Embedding model name passed to the API.                    |
| `dimension`        | int    | `1024`             | Expected embedding dimension.                              |
| `batch_size`       | int    | `128`              | Texts per embedding API call.                              |
| `max_tokens`       | int    | `512`              | Local tokenizer truncation budget.                         |
| `faiss_index_type` | string | `Flat`             | FAISS index factory string.                                |
| `gpu`              | bool   | `true`             | Build and search FAISS on GPU if true.                     |

### `03_filter_consistency.toml`

| Key                 | Type | Default | Description                                          |
|---------------------|------|---------|------------------------------------------------------|
| `k`                 | int  | `30`    | Cutoff for raw and virtual retrieval rank tests.     |
| `k_max`             | int  | `100`   | Upper search depth for virtual rank.                 |
| `top_k_for_restore` | int  | `30`    | Depth retained when restoring embedding files.       |

### `04_hnm.toml`

| Key                               | Type  | Default | Description                                                          |
|-----------------------------------|-------|---------|----------------------------------------------------------------------|
| `seed`                            | int   | `42`    | Random seed for tie-breaking.                                        |
| `k0`                              | int   | `50`    | Initial FAISS search depth.                                          |
| `k_max`                           | int   | `1000`  | Maximum adaptive search depth.                                       |
| `n_min`                           | int   | `8`     | Minimum valid negatives per query; under this, drop the query.       |
| `delta`                           | float | `0.05`  | Relative margin parameter.                                           |
| `exclude_cross_lingual_positives` | bool  | `true`  | Removes cross-lingual variants of positives from the candidate pool. |

### `05_compile.toml`

| Key                     | Type | Default | Description                                                                 |
|-------------------------|------|---------|-----------------------------------------------------------------------------|
| `cross_lingual_augment` | bool | `true`  | Expand multilingual pairs into all four query/positive language variants.   |
| `negatives_per_query`   | int  | `8`     | Hard negatives per row; rows with fewer mined negatives are dropped.        |

### `06_train.toml`

| Key                       | Type   | Default              | Description                                                                |
|---------------------------|--------|----------------------|----------------------------------------------------------------------------|
| `seed`                    | int    | `42`                 | Random seed.                                                               |
| `base_model`              | string | `RuModernBERT-base`  | Base encoder identifier (path resolved via `${TRAIN_MODEL_PATH}`).         |
| `max_seq_length`          | int    | `4096`               | Maximum tokenized sequence length.                                         |
| `loss`                    | string | `MNRL`               | Loss name.                                                                 |
| `scale`                   | float  | `10.0`               | Temperature scale factor for MNRL.                                         |
| `k_h`                     | int    | `8`                  | Hard negatives per query.                                                  |
| `k_r`                     | int    | `2`                  | Random in-batch negatives per query.                                       |
| `batch_size_per_gpu`      | int    | `16`                 | Per-GPU batch size.                                                        |
| `num_buckets`             | int    | `64`                 | Buckets per dataset for the dataset-homogeneous sampler.                   |
| `optimizer`               | string | `adamw_fused`        | Optimizer identifier.                                                      |
| `learning_rate`           | float  | `2e-5`               | Peak learning rate.                                                        |
| `lr_scheduler`            | string | `cosine`             | Learning-rate schedule.                                                    |
| `warmup_ratio`            | float  | `0.1`                | Fraction of steps used for linear warmup.                                  |
| `weight_decay`            | float  | `0.01`               | AdamW weight decay.                                                        |
| `num_epochs`              | int    | `1`                  | Number of epochs.                                                          |
| `gradient_caching`        | bool   | `true`               | Enable Gradient Cache for MNRL.                                            |
| `gradient_checkpointing`  | bool   | `true`               | Enable activation checkpointing.                                           |
| `eval_steps_per_epoch`    | int    | `20`                 | Number of validation passes per epoch.                                     |
| `checkpoint_metric`       | string | `map@10`             | Metric used to select the best checkpoint.                                 |

### `eval.toml`

| Key                  | Type   | Default                                | Description                                |
|----------------------|--------|----------------------------------------|--------------------------------------------|
| `benchmark`          | string | `RusBEIR`                              | Benchmark suite to evaluate against.       |
| `first_stage`        | string | `bm25`                                 | First-stage retriever.                     |
| `first_stage_top_k`  | int    | `100`                                  | Candidates per query passed to reranker.   |
| `primary_metric`     | string | `ndcg@10`                              | Headline metric.                           |
| `secondary_metrics`  | list   | `[map@10, mrr@10, p@1, p@3, p@5]`      | Additional metrics reported per benchmark. |
