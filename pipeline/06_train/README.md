# Stage 6 — Training

Fine-tune a cross-encoder on the compiled training Parquet with MNRL,
dataset-aware batch sampling, gradient caching, and MAP@10 checkpoint
selection. Distributed across multiple GPUs via `torchrun`.

## What it does

Training is a single-epoch fine-tune of a cross-encoder over the
compiled Parquet. The loss is Multiple Negatives Ranking Loss (MNRL)
with a temperature scale: per query the loss sees `k_h` hard negatives
drawn from the row plus `k_r` random in-batch negatives drawn from
other queries in the same step.

The sampler is dataset-homogeneous. Before training, the helper
`prepare_buckets.py` hashes each query into one of `num_buckets`
buckets, partitioned per source dataset. At each step, all GPUs draw
queries from the same bucket of the same dataset, so the in-batch
random-negative pool is confined to one annotation regime. This keeps
the contrastive signal clean and avoids cross-dataset contamination.
`prepare_buckets.py` is idempotent — the runner calls it on every
launch but it skips work if the bucket index is already on disk.

Two memory tricks are on by default. Gradient caching lets MNRL see
the full per-step contrastive pool without holding all activations in
memory at once. Gradient checkpointing trades recomputation for
activation memory. Combined, they make 4096-token sequences tractable
on a 16-per-GPU batch at 80GB VRAM.

Validation runs about twenty times per epoch on the held-out Parquet
and reports MAP@10, which is the checkpoint-selection metric. The
trainer writes the best-scoring checkpoint under `${TRAIN_OUTPUT_DIR}`
alongside per-rank training logs in `${TRAIN_LOG_DIR}`.

## Inputs

- `${TRAIN_DATASET_PATH}` — bucketed training Parquet produced by
  `prepare_buckets.py` from the stage 5 compiled corpus.
- `${EVAL_DATASET_PATH}` — bucketed validation Parquet.
- `${TRAIN_MODEL_PATH}` — base encoder checkpoint to fine-tune.

## Outputs

- `${TRAIN_OUTPUT_DIR}/checkpoint-*/` — saved checkpoints, one per
  validation step.
- `${TRAIN_OUTPUT_DIR}/best/` — symlink (or copy) to the
  highest-MAP@10 checkpoint.
- `${TRAIN_LOG_DIR}/rank-*.log` — per-rank training logs.

## Key parameters

| Key                       | Default            | Description                                                                |
|---------------------------|--------------------|----------------------------------------------------------------------------|
| `seed`                    | `42`               | Random seed.                                                               |
| `base_model`              | `RuModernBERT-base`| Base encoder identifier (loaded from `${TRAIN_MODEL_PATH}` in practice).   |
| `max_seq_length`          | `4096`             | Maximum tokenized sequence length.                                         |
| `loss`                    | `MNRL`             | Loss name.                                                                 |
| `scale`                   | `10.0`             | Temperature scale factor for MNRL.                                         |
| `k_h`                     | `8`                | Hard negatives per query.                                                  |
| `k_r`                     | `2`                | Random in-batch negatives per query.                                       |
| `batch_size_per_gpu`      | `16`               | Per-GPU batch size; effective batch is multiplied by world size.           |
| `num_buckets`             | `64`               | Number of buckets per dataset for the dataset-homogeneous sampler.         |
| `optimizer`               | `adamw_fused`      | Optimizer identifier.                                                      |
| `learning_rate`           | `2e-5`             | Peak learning rate.                                                        |
| `lr_scheduler`            | `cosine`           | Learning-rate schedule.                                                    |
| `warmup_ratio`            | `0.1`              | Fraction of steps used for linear warmup.                                  |
| `weight_decay`            | `0.01`             | AdamW weight decay.                                                        |
| `num_epochs`              | `1`                | Number of epochs over the training Parquet.                                |
| `gradient_caching`        | `true`             | Enables Gradient Cache for MNRL.                                           |
| `gradient_checkpointing`  | `true`             | Enables activation checkpointing.                                          |
| `eval_steps_per_epoch`    | `20`               | Number of validation passes per epoch.                                     |
| `checkpoint_metric`       | `map@10`           | Metric used to select the best checkpoint.                                 |

All keys live in `configs/thesis/06_train.toml`. CLI flags override
TOML. Paths come from `.env`; the trainer reads `${NPROC_PER_NODE}`
to size the `torchrun` launch.

## Files

- `prepare_buckets.py` — one-shot bucketing helper, called
  idempotently by the runner.
- `train.py` — distributed training entrypoint.
- `sampler.py` — custom dataset-homogeneous batch sampler.
- `run.sh` — shell wrapper that loads `.env` and launches `torchrun`.

## Running

    bash pipeline/06_train/run.sh

To run on a different number of GPUs without editing the runner:

    NPROC_PER_NODE=4 bash pipeline/06_train/run.sh

To override a hyperparameter without editing the TOML:

    bash pipeline/06_train/run.sh --batch-size-per-gpu 8 --num-epochs 2

## Notes

- The sampler relies on Accelerate's batch-sampler resharding being
  disabled (`even_batches=False`, `dispatch_batches=False`); the
  trainer sets these explicitly. If a downstream Accelerate upgrade
  changes the defaults, the sampler will start producing
  cross-dataset batches and the run will silently degrade.
- `prepare_buckets.py` runs on every launch but only does work the
  first time; subsequent launches are no-ops.
- For very small VRAM budgets, drop `batch_size_per_gpu` and rely on
  gradient accumulation; the effective contrastive pool size matters
  more than the per-step pool size with gradient caching on.
- Validation evaluates MAP@10 on the held-out Parquet; final
  benchmark numbers come from the two-stage evaluation under
  `evaluation/`, not from this in-training validation.
