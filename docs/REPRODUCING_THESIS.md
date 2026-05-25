# Reproducing the thesis run

This document records the configuration used to produce the trained
checkpoint described in `thesis.pdf`. The narrative description of each
pipeline stage lives in `docs/PIPELINE.md`; this document covers
hardware, the dataset roster, per-dataset parameters, the end-to-end
command sequence, expected runtimes, and where to look when something
fails.

## Hardware notes

The thesis training configuration uses 8 × NVIDIA H200 GPUs (141 GB
HBM3e per device) connected by InfiniBand within a single node. The
data-prep and embedding stages also assume one node with at least one
recent CUDA-capable GPU; they do not require eight.

Equivalent setups expected to work without code changes:

- 8 × NVIDIA H100 80GB SXM with NVLink/InfiniBand.
- 8 × NVIDIA A100 80GB SXM with NVLink/InfiniBand.

For smaller-VRAM nodes (≤ 40 GB per device), the training stage needs
two adjustments in `configs/thesis/06_train.toml`:

- Lower `batch_size_per_gpu` until a step fits in memory.
- Compensate for the smaller effective batch via gradient accumulation
  (the trainer accepts `--gradient-accumulation-steps`).

The `NPROC_PER_NODE` environment variable controls the `torchrun`
process count. Set it explicitly when the node has a different GPU
count than the default (8):

    NPROC_PER_NODE=4 bash pipeline/06_train/run.sh

Distributed training assumes one process per GPU; multi-node setups
require the usual `torchrun` rendezvous flags (`--nnodes`, `--node_rank`,
`--rdzv_*`), passed through as extra args.

## Source dataset roster

The thesis training run combines several Russian-language and
multilingual query–passage corpora. The exact roster is not encoded in
the repository — the pipeline depends only on the resulting Parquet
shards in `${UNIFIED_DATA_DIR}` matching the input schema documented in
`docs/PIPELINE.md`. Fill the table below with the sources you actually
use:

| Dataset key | HuggingFace ID | Notes |
|-------------|----------------|-------|
| TBD         | TBD            | Brief one-line description. |
| TBD         | TBD            | …                            |

When extending or replacing this list, the only requirement is that
each source yields rows with the documented six-column schema. Source
selection has no effect on the pipeline code itself.

## Per-dataset `k` thresholds (consistency filtering)

`k` controls the retrieval-rank cutoff in stage 3. The default is `30`
everywhere. Per-dataset deviations, if any, are recorded below; the
default applies to any dataset not listed.

| Dataset key     | `k`  | Reason                                                              |
|-----------------|------|---------------------------------------------------------------------|
| (default)       | `30` | Used for every dataset unless explicitly overridden below.          |

If the engineer reproducing the run finds that a specific dataset has
extremely low survival at `k = 30`, raise `k` for that dataset before
re-running stage 3. Record the change here. As shipped, no deviations
are documented.

## Per-dataset HNM parameters

Stage 4 selects hard negatives under the parameters in
`configs/thesis/04_hnm.toml`. The defaults below apply to every dataset
unless explicitly overridden.

| Dataset key     | `k0` | `k_max` | `n_min` | `delta` | Notes                          |
|-----------------|------|---------|---------|---------|--------------------------------|
| (default)       | `50` | `1000`  | `8`     | `0.05`  | Used for every dataset.        |

As with the `k` thresholds, no per-dataset deviations are documented for
the shipped run. Per-dataset overrides can be applied via CLI flags on
`pipeline/04_hard_negative_mining/run.sh` (loop over the dataset names
and rerun the miner with adjusted `--delta` or `--n-min`).

## End-to-end commands

Place the unified input data under `${UNIFIED_DATA_DIR}` (one or more
Parquet shards matching the input schema in `docs/PIPELINE.md`). Then,
from a fresh checkout:

    cp .env.example .env
    # edit .env: paths, embedding API credentials, base model path

    uv sync                # or: pip install -e .

    ./run_pipeline.sh      # chains stages 1–6 + evaluation
    bash evaluation/run.sh # rerun evaluation against a different checkpoint

`./run_pipeline.sh` invokes each stage's `run.sh` in order. Each stage
can also be re-run individually, which is the recommended workflow when
iterating on a single stage:

    bash pipeline/01_split/run.sh
    bash pipeline/02_embed/run.sh
    bash pipeline/03_filter_consistency/run.sh
    bash pipeline/04_hard_negative_mining/run.sh
    bash pipeline/05_compile/run.sh
    bash pipeline/06_train/run.sh
    bash evaluation/run.sh

CLI flags pass through every runner (everything after the last
documented flag is forwarded to the Python entrypoint), so one-off
overrides do not require editing the TOML configs.

## Expected runtimes

Order-of-magnitude estimates on the reference hardware. Wall-clock
varies with corpus size, network throughput to the embedding endpoint,
and concurrent jobs on the node.

| Stage                          | Reference time             | Sensitivity                                                                |
|--------------------------------|----------------------------|----------------------------------------------------------------------------|
| 1. Splitting                   | ~10 minutes                | Single-pass CPU work; scales linearly in number of unique IDs.             |
| 2. Embedding                   | several hours              | Dominated by embedding-API throughput, not local compute.                  |
| 3. Consistency filtering       | ~1–2 hours                 | FAISS-GPU search over each per-dataset index.                              |
| 4. Hard negative mining        | ~2–4 hours                 | Adaptive search depth; clusters that hit `k_max` dominate the wall time.   |
| 5. Compilation                 | ~30 minutes                | Single-pass Parquet I/O plus the cross-lingual augmentation expansion.     |
| 6. Training                    | ~24 hours on 8 × H200      | One epoch over the compiled Parquet; scales sub-linearly with GPU count.   |
| Evaluation (BM25 + reranker)   | ~1–2 hours                 | Dominated by the reranker pass over 100 candidates per benchmark query.    |

These numbers are deliberately rounded; treat them as planning
estimates, not contracts.

## Where to look when something fails

- **Training logs:** `${TRAIN_LOG_DIR}/rank-*.log`. One file per rank;
  per-step loss, MAP@10 validation results, and any backward-pass
  errors land here. Rank 0 carries the canonical view.
- **ClearML dashboard:** if `USE_CLEARML=1`, every training run
  registers under `${CLEARML_PROJECT_NAME}` / `${CLEARML_TASK_NAME}`
  with live loss curves, GPU utilisation, and the resolved config. This
  is the fastest way to diagnose throughput regressions across runs.
- **Profiler:** `tools/profile_training.py` wraps a short training run
  with PyTorch profiler hooks. Use it when MAP@10 looks fine but
  throughput is below the reference numbers above.
- **FAISS-GPU OOM:** drop `batch_size` in `configs/thesis/02_embed.toml`
  and disable GPU FAISS (`gpu = false`) as a fallback; the indices are
  small enough for CPU search at a small speed cost.
- **Embedding-API stalls:** the embedder retries on transient errors;
  sustained 429s mean the endpoint quota is exhausted. Pause the run
  rather than burn retries.
