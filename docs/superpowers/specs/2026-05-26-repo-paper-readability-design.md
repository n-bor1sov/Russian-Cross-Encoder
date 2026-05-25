# Repository Restructure for Paper-Reader Clarity — Design Spec

**Date:** 2026-05-26
**Status:** Approved (pending writing-plans)

## Context

The repository currently exists as a set of working research scripts grouped by ad-hoc folder (`dataset_scripts/`, `train_scripts/`, `benchmarking/`). It will be published as the primary implementation artifact of a thesis on a Russian cross-encoder reranker. A reader of the paper needs to be able to land in the repo and (a) understand which code implements which method, (b) reproduce the trained model end-to-end on their own data, and (c) adapt the pipeline to a different corpus or language with reasonable effort.

This spec defines a reorganization of the repository to meet those needs, without rewriting the underlying algorithms.

## Goals

1. Reorganize files so the pipeline's structure is visible in the file tree.
2. Lift hyperparameters out of script bodies into per-stage TOML configuration.
3. Provide shell runners that capture the exact command used in the thesis run.
4. Ship a self-contained example that verifies the install end-to-end without an external embedding API or a cluster.
5. Provide narrative and per-stage documentation that stands on its own, without requiring the reader to cross-reference the thesis PDF.

## Non-goals

- No checkpoint or dataset hosting. The trained model and compiled dataset are not mirrored.
- No automated source-data acquisition. README documents sources; reader downloads them.
- No conversion to an installable Python package. Stays a folder of standalone scripts.
- No algorithm changes. Pipeline behavior is preserved. The single exception is fixing `restore_only.py` after the deletion of the old filtering script it imports from.
- No tests. `examples/run.sh` is a smoke test only, not a correctness test.
- No new lifted parameters beyond those listed in the per-stage TOML configs.
- No translation of docs or in-code comments.
- No linter, formatter, or pre-commit setup.
- No GitHub Actions or release automation.
- No reorganization of `thesis.pdf` content; the PDF stays as a single artifact at the repo root.

## Target folder structure

```
CE_repo/
├── README.md
├── LICENSE                            # MIT
├── pyproject.toml                     # declares every top-level dependency explicitly
├── uv.lock
├── .env.example                       # paths and API keys
├── .gitignore                         # adds examples/work/
├── thesis.pdf
├── run_pipeline.sh                    # chains all six stages + evaluation
│
├── data_prep/                         # OFF-PIPELINE — helpers for normalizing raw sources
│   ├── README.md
│   ├── merge_to_shards.py
│   └── add_dataset_column.py
│
├── pipeline/
│   ├── 01_split/
│   │   ├── README.md
│   │   ├── split.py
│   │   └── run.sh
│   ├── 02_embed/
│   │   ├── README.md
│   │   ├── embed_shards.py
│   │   └── run.sh
│   ├── 03_filter_consistency/
│   │   ├── README.md
│   │   ├── score.py
│   │   ├── filter_by_rank.py
│   │   ├── restore_embeddings.py
│   │   ├── restore_only.py
│   │   └── run.sh
│   ├── 04_hard_negative_mining/
│   │   ├── README.md
│   │   ├── mine.py
│   │   ├── embedding_lookup.py
│   │   ├── slurm/
│   │   │   ├── HNM.sh
│   │   │   ├── HNM_live.sh
│   │   │   ├── kill_hnm.sh
│   │   │   └── status_hnm.sh
│   │   └── run.sh
│   ├── 05_compile/
│   │   ├── README.md
│   │   ├── compile.py
│   │   ├── analyze.py
│   │   └── run.sh
│   └── 06_train/
│       ├── README.md
│       ├── prepare_buckets.py
│       ├── train.py
│       ├── sampler.py
│       └── run.sh
│
├── evaluation/
│   ├── README.md
│   ├── generate_bm25_top100.py
│   ├── evaluate_reranker_top100.py
│   └── run.sh
│
├── tools/
│   ├── README.md
│   ├── profile_training.py
│   └── slerp_merge.py                 # already exists somewhere; will be migrated
│
├── configs/
│   ├── README.md                      # config schema reference
│   └── thesis/
│       ├── 01_split.toml
│       ├── 02_embed.toml
│       ├── 03_filter_consistency.toml
│       ├── 04_hnm.toml
│       ├── 05_compile.toml
│       ├── 06_train.toml
│       └── eval.toml
│
├── docs/
│   ├── PIPELINE.md                    # narrative walkthrough of the six stages
│   ├── REPRODUCING_THESIS.md          # end-to-end commands, dataset roster, hardware notes
│   └── REPRODUCING_ABLATIONS.md       # three ablations: filtering, K_H sweep, SLERP
│
└── examples/
    ├── README.md
    ├── data/
    │   ├── mini_dataset_a.parquet     # ~50 synthetic query-passage pairs, two languages
    │   ├── mini_dataset_b.parquet     # ~50 pairs, single language
    │   └── mini_compiled.parquet      # pre-compiled training parquet so the example
    │                                  # doesn't need an embedding API
    ├── configs/                       # mirror of configs/thesis/ with tiny parameters
    │   ├── 01_split.toml
    │   ├── 02_embed.toml
    │   ├── 03_filter_consistency.toml
    │   ├── 04_hnm.toml
    │   ├── 05_compile.toml
    │   ├── 06_train.toml
    │   └── eval.toml
    ├── .env.example                   # points all paths under examples/work/
    ├── run.sh                         # default: training + eval starting from mini_compiled.parquet
    ├── run_full.sh                    # full pipeline; requires an embedding API
    └── expected_outputs/              # schema-only sample of each stage's output
```

## Configuration model

Each stage has its own TOML file under `configs/thesis/`. A script reads only its own config. Paths and API keys remain in `.env`. CLI flags override TOML values for ad-hoc runs.

The full canonical configuration is split across seven files. Two examples illustrate the schema:

```toml
# configs/thesis/04_hnm.toml
# Hard Negative Mining — margin-based selection.
seed  = 42
k0    = 50          # initial search depth
k_max = 1000        # max adaptive depth
n_min = 8           # min valid negatives per query (else drop query)
delta = 0.05        # relative margin parameter

exclude_cross_lingual_positives = true   # prevents false negatives across language variants
```

```toml
# configs/thesis/06_train.toml
# Training Procedure
seed = 42
base_model     = "RuModernBERT-base"
max_seq_length = 4096

loss  = "MNRL"
scale = 10.0                            # temperature scale factor

k_h = 8                                 # hard negatives per query (fixed columns)
k_r = 2                                 # random in-batch negatives

batch_size_per_gpu = 16
num_buckets        = 64

optimizer       = "adamw_fused"
learning_rate   = 2e-5
lr_scheduler    = "cosine"
warmup_ratio    = 0.1
weight_decay    = 0.01
num_epochs      = 1
gradient_caching       = true
gradient_checkpointing = true

eval_steps_per_epoch = 20
checkpoint_metric    = "map@10"
```

Every TOML key is documented in `configs/README.md`. The per-dataset `k` threshold for consistency filtering uses a single default value; per-dataset values are not lifted into config (they live in `docs/REPRODUCING_THESIS.md` for reference).

## Shell runners

Each pipeline stage folder ships a `run.sh` wrapper that loads `.env`, picks the matching TOML config, and invokes the Python entrypoint. Two representative examples:

```bash
# pipeline/04_hard_negative_mining/run.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

python pipeline/04_hard_negative_mining/mine.py \
  --config configs/thesis/04_hnm.toml \
  --input-shards "${SHARDS_DIR}/filtered" \
  --embeddings-dir "${EMBEDDINGS_DIR}" \
  --output "${HNM_ROOT}" \
  "$@"
```

```bash
# pipeline/06_train/run.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a

torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" \
  pipeline/06_train/train.py \
  --config configs/thesis/06_train.toml \
  --train-dataset "${TRAIN_DATASET_PATH}" \
  --eval-dataset "${EVAL_DATASET_PATH}" \
  --output-dir "${TRAIN_OUTPUT_DIR}" \
  "$@"
```

The top-level `run_pipeline.sh` chains the six pipeline stages and evaluation in order.

Stage-skipping for ablations uses env-var overrides on the runners (e.g., `HNM_INPUT_DIR=…` to point Stage 4 at unfiltered embeddings for Ablation A).

## Examples directory

`examples/` is a self-contained miniature run that verifies the install without requiring the real corpora, an embedding API, or a multi-GPU cluster.

- `examples/data/` ships synthetic mini-parquets (about 100 query–passage pairs across two fake datasets, two languages).
- `examples/data/mini_compiled.parquet` is a pre-compiled training parquet (Stage 5 output for the mini data), so a reader without an embedding endpoint can still test stages 6 and evaluation.
- `examples/configs/` mirrors `configs/thesis/` with tiny parameters (small `k`, small batch size, two epochs at low LR).
- `examples/.env.example` points all paths under `examples/work/`.
- `examples/run.sh` runs the no-external-dependency path by default: it starts from `mini_compiled.parquet` and exercises only the training stage and the evaluation step. Stages 2–5 (embedding, filtering, hard-negative mining, compilation) are skipped because they require an embedding API. Running them on the synthetic data is supported via `examples/run_full.sh`, which expects `EMBEDDING_API_BASE_URL` and friends to be set.
- `examples/expected_outputs/` contains schema-only samples of each stage's output, so a reader can diff their own outputs against the expected column shape.

Target runtime for the default `run.sh`: about 10 minutes on a single GPU.

## Documentation

No document references thesis chapter or section numbers in user-facing prose. Equation numbers inside code comments are allowed; section references in user-facing markdown are not.

### `README.md`

Short. Orients the reader, points at deeper docs, gives install and quickstart commands.

Skeleton:

```markdown
# Russian Cross-Encoder Reranker

End-to-end training pipeline and trained-checkpoint recipe for a Russian-language
cross-encoder reranker. The repository contains the full data-processing,
training, and evaluation code that produced the results in the accompanying
thesis (`thesis.pdf`).

## What's in the repository

| Folder           | What it contains                                                    |
|------------------|---------------------------------------------------------------------|
| `data_prep/`     | Off-pipeline helpers for normalizing raw sources to the input schema |
| `pipeline/`      | The six-stage training pipeline (splitting → training)              |
| `evaluation/`    | Two-stage BM25 + reranker evaluation on Russian IR benchmarks       |
| `configs/`       | Per-stage TOML configurations; `configs/thesis/` reproduces the paper |
| `docs/`          | `PIPELINE.md`, `REPRODUCING_THESIS.md`, `REPRODUCING_ABLATIONS.md`  |
| `examples/`      | Self-contained miniature run to verify the install                  |
| `tools/`         | Off-pipeline utilities                                              |

For a narrative walkthrough of the pipeline, read `docs/PIPELINE.md`.

## Requirements

- Python 3.13+
- CUDA 12+ (for FAISS-GPU and distributed training)
- An OpenAI-compatible embedding endpoint
- A single GPU is enough to run `examples/`; the thesis training configuration
  uses 8 GPUs. Detailed hardware notes are in `docs/REPRODUCING_THESIS.md`.

The full dependency list is in `pyproject.toml`.

## Install

    uv sync
    # or: pip install -e .

## Configure

    cp .env.example .env
    # Edit .env to point at your data directories, embedding API, and base model

Required environment variables are documented in `.env.example`. Hyperparameters
used in the thesis are in `configs/thesis/`; see `configs/README.md` for the
config schema and override rules.

## Quickstart

Verify the install end-to-end on the bundled miniature corpus:

    bash examples/run.sh

## Reproducing the thesis run

See `docs/REPRODUCING_THESIS.md`. The short version:

    ./run_pipeline.sh
    bash evaluation/run.sh

Each stage can also be run individually:

    bash pipeline/01_split/run.sh
    bash pipeline/02_embed/run.sh
    # ...

For the ablation studies, see `docs/REPRODUCING_ABLATIONS.md`.

## Input data

The pipeline expects a corpus of query–passage pairs already in a common
schema. Expected columns and constraints are documented in `docs/PIPELINE.md`
under "Input data requirements". Helpers for normalizing common source
formats are in `data_prep/`.

## Citation

If you use this code, please cite:

    @thesis{<key>,
      author = {Borisov, Nikita Mikhailovich},
      title  = {A Contextual Cross-Encoder for High-Precision Textual
                Reranking in Russian Language Information Retrieval},
      school = {Innopolis University},
      year   = {2026},
    }

## License

MIT. See `LICENSE`.
```

### `docs/PIPELINE.md`

Narrative walkthrough. Opens with input data requirements, then describes the six pipeline stages and the evaluation step in 2–4 paragraphs each. Plain language, no equation references in prose, no thesis cross-references.

The input data requirements section specifies:

| Column        | Type   | Description                                                    |
|---------------|--------|----------------------------------------------------------------|
| `query`       | string | The query text                                                 |
| `passage`     | string | The labeled positive passage for the query                     |
| `query_id`    | string | Stable identifier for the query                                |
| `passage_id`  | string | Stable identifier for the passage                              |
| `lang`        | string | ISO 639-1 language code                                        |
| `dataset`     | string | Source dataset name                                            |

Constraints: UTF-8 NFC-normalized strings, no nulls or empty values, stable identifiers (same instance always has the same ID), cross-lingual variants of the same conceptual pair share a `query_id`, recommended format is Parquet with Snappy compression.

The six stages are described in this order:

1. **Splitting** (bipartite-component split at 95:5 with cross-lingual coupling).
2. **Embedding** (frozen dense model, per-dataset FAISS index).
3. **Consistency Filtering** (retrieval-rank and virtual-rank gate against the dataset-local index).
4. **Hard Negative Mining** (adaptive-depth retrieval, margin-based selection at fraction δ below mean positive similarity, cross-lingual positive exclusion).
5. **Compilation** (cross-lingual augmentation for aligned multilingual sources, language-consistent negative pairing, materialization to fixed-column Parquet).
6. **Training** (cross-encoder fine-tuning with MNRL, dataset-aware batch sampling, gradient caching, MAP@10 for checkpoint selection).

Plus the evaluation step (BM25 top-100 → reranker → NDCG@10 averaged across the benchmark suite).

### Per-stage `pipeline/XX_*/README.md`

Each stage's README is self-contained and follows the same skeleton:

```markdown
# <Stage name>

<One-sentence statement of what this stage does.>

## What it does

<2–4 short paragraphs. Inputs, outputs, key idea, relevant constraints.
State rules informally in plain language. No external section references.>

## Inputs

- `<path or pattern>` — what it is, where it comes from

## Outputs

- `<path or pattern>` — what it is, what the downstream stage expects

## Key parameters

| Key | Default | Description |
|-----|---------|-------------|
| ... | ...     | ...         |

All keys live in `configs/thesis/<stage>.toml`. CLI flags override TOML.

## Files

- `<file>.py` — what it does

## Running

    bash pipeline/XX_*/run.sh

## Notes

<Subtle items: known limitations, what happens if a parameter is set
poorly, useful diagnostic logs.>
```

### `data_prep/README.md`

Short. Re-states the input-schema table (so a reader landing here doesn't have to bounce). Frames the helpers as off-pipeline: "used during the thesis work, but their choices are not prescribed by the pipeline."

### `configs/README.md`

Documents the TOML schema for each stage's config, the override rules (CLI flags > TOML > defaults), and clarifies that paths live in `.env`, not in TOML.

### `docs/REPRODUCING_THESIS.md`

End-to-end reproduction recipe. Contents:

- Hardware notes (8 × H200 for the thesis training configuration; equivalent setups; `NPROC_PER_NODE` override).
- Source dataset roster with HF dataset IDs and any preprocessing notes.
- Per-dataset `k` thresholds used during consistency filtering.
- Per-dataset HNM parameters.
- End-to-end command sequence with expected outputs at each step.

### `docs/REPRODUCING_ABLATIONS.md`

Three ablations, each in self-contained subsections:

- **Filtering ablation:** V1 unfiltered (skip Stage 3, point Stage 4 at unfiltered embeddings via `HNM_INPUT_DIR` env override) vs V2 filtered (baseline).
- **K_H sweep:** vary `k_h` ∈ {0, 8, 15} in `06_train.toml` with `k_r = 2`. For `k_h = 15`, update `negatives_per_query` in `05_compile.toml` and re-run from Stage 5.
- **SLERP checkpoint merging:** post-training only; uses `tools/slerp_merge.py`. Two regimes documented: merging across data-composition runs, merging across consecutive checkpoints from a single run.

## Cleanup actions

### Deletions

- `dataset_scripts/03_filter_and_restore_datasets.py` — superseded by the `_fixed` variant.
- `dataset_scripts/debug_faiss_gpu.py` if present — verify with maintainer before deletion; otherwise move to `tools/`.
- Any Jupyter notebook referenced by the current stale README but absent from the working tree (e.g., `test/visualize_results_v2.ipynb`) — no action; already gone.

### Moves and renames

All moves use `git mv` to preserve history.

| From                                                       | To                                                  |
|-----------------------------------------------------------|-----------------------------------------------------|
| `dataset_scripts/01_merge_to_shards.py`                   | `data_prep/merge_to_shards.py`                      |
| `dataset_scripts/add_parquet_dataset_column.py`           | `data_prep/add_dataset_column.py`                   |
| `dataset_scripts/split_datasets.py`                       | `pipeline/01_split/split.py`                        |
| `dataset_scripts/02_embed_shards.py`                      | `pipeline/02_embed/embed_shards.py`                 |
| `dataset_scripts/03_filter_and_restore_datasets_fixed.py` | `pipeline/03_filter_consistency/score.py`           |
| `dataset_scripts/04_restore_embeddings_by_dataset.py`     | `pipeline/03_filter_consistency/restore_embeddings.py` |
| `dataset_scripts/05_filter_by_rank.py`                    | `pipeline/03_filter_consistency/filter_by_rank.py`  |
| `dataset_scripts/restore_only_datasets.py`                | `pipeline/03_filter_consistency/restore_only.py`    |
| `dataset_scripts/hnm/HNM.py`                              | `pipeline/04_hard_negative_mining/mine.py`          |
| `dataset_scripts/hnm/embedding_lookup.py`                 | `pipeline/04_hard_negative_mining/embedding_lookup.py` |
| `dataset_scripts/hnm/HNM.sh`                              | `pipeline/04_hard_negative_mining/slurm/HNM.sh`     |
| `dataset_scripts/hnm/HNM_live.sh`                         | `pipeline/04_hard_negative_mining/slurm/HNM_live.sh` |
| `dataset_scripts/hnm/kill_hnm.sh`                         | `pipeline/04_hard_negative_mining/slurm/kill_hnm.sh` |
| `dataset_scripts/hnm/status_hnm.sh`                       | `pipeline/04_hard_negative_mining/slurm/status_hnm.sh` |
| `dataset_scripts/finale/final_dataset_compilation.py`     | `pipeline/05_compile/compile.py`                    |
| `dataset_scripts/finale/analyze_final_dataset.py`         | `pipeline/05_compile/analyze.py`                    |
| `train_scripts/prepare_query_buckets.py`                  | `pipeline/06_train/prepare_buckets.py`              |
| `train_scripts/train_bucketed.py`                         | `pipeline/06_train/train.py`                        |
| `train_scripts/sampler_bucketed.py`                       | `pipeline/06_train/sampler.py`                      |
| `train_scripts/profile_training.py`                       | `tools/profile_training.py`                         |
| `benchmarking/generate_bm25_top100.py`                    | `evaluation/generate_bm25_top100.py`                |
| `benchmarking/evaluate_reranker_top100.py`                | `evaluation/evaluate_reranker_top100.py`            |

Wherever `multi_slerp` (or the equivalent SLERP merging code) currently lives in the repo, migrate it to `tools/slerp_merge.py` and add a thin argparse CLI.

After the moves, the following directories become empty and are removed: `dataset_scripts/`, `dataset_scripts/hnm/`, `dataset_scripts/finale/`, `train_scripts/`, `benchmarking/`.

### Code-level ripple fixes

1. `pipeline/03_filter_consistency/restore_only.py` currently uses `runpy.run_path` to load functions from `03_filter_and_restore_datasets.py`. Replace the `runpy` indirection with a normal import from sibling `score.py` (the new home of the `_fixed` variant). Use a `sys.path` shim if needed for script-style invocation.
2. Slurm launchers in `pipeline/04_hard_negative_mining/slurm/` hardcode the path to `HNM.py`. Update each `.sh` to point at `pipeline/04_hard_negative_mining/mine.py` (relative to repo root). Update any env-var defaults inside those scripts that point at old data paths.
3. Sweep the repository for every old filename stem (`merge_to_shards`, `split_datasets`, `02_embed_shards`, `03_filter_and_restore_datasets`, `04_restore_embeddings_by_dataset`, `05_filter_by_rank`, `restore_only_datasets`, `HNM`, `embedding_lookup`, `final_dataset_compilation`, `analyze_final_dataset`, `prepare_query_buckets`, `train_bucketed`, `sampler_bucketed`, `profile_training`, `generate_bm25_top100`, `evaluate_reranker_top100`) and fix every reference to the new path. Includes cross-script imports, log messages, and any docstrings that name sibling files.

### Config and metadata updates

1. `.env.example` — update path comments to reference the new folders. Variable names themselves stay the same.
2. `.gitignore` — add `examples/work/`.
3. `pyproject.toml` — declare every top-level runtime dependency explicitly: `torch`, `sentence-transformers`, `transformers`, `datasets`, `faiss-gpu`, `openai`, `networkx`, `pyarrow`, `accelerate`. Optional dependency group `tracking` for `clearml`. Keep existing `ipykernel`, `ipywidgets`, `matplotlib`, `pandas`, `tqdm`.
4. Add `LICENSE` file containing MIT license text. No `CITATION.cff`.
5. `README.md` — full rewrite per the skeleton above. Stale references to `test/test_reranker.py`, `hnm0.py`, `PositiveOnly_train_bucketed.py` disappear.

### Pre-move verification

Before deletion, the implementation phase must verify:

1. `restore_only.py` works correctly against the `_fixed` variant after the import rewrite. The two `03_…` variants may have differing function signatures; if so, fix the call sites.
2. Whether `dataset_scripts/debug_faiss_gpu.py` exists and whether it should be deleted or moved to `tools/`. Confirm with maintainer.
3. Where `multi_slerp` (or equivalent SLERP code) currently lives. Migration target is `tools/slerp_merge.py`.

## Acceptance criteria

The implementation is complete when all of the following hold.

**Structural**

- Every file in the moves table exists at its new path; no file remains under `dataset_scripts/`, `train_scripts/`, or `benchmarking/`.
- The old `03_filter_and_restore_datasets.py` is deleted; only `score.py` (from the `_fixed` variant) remains.
- All target folders exist: `data_prep/`, `pipeline/01_split/`–`pipeline/06_train/`, `evaluation/`, `tools/`, `configs/thesis/`, `docs/`, `examples/`.

**Configs and runners**

- `configs/thesis/01_split.toml` through `06_train.toml` and `eval.toml` exist with the documented keys.
- Each `pipeline/XX_*/` folder contains a `run.sh` that loads `.env` and invokes its Python entrypoint with the matching TOML config.
- `pipeline/06_train/run.sh` uses `torchrun` and respects `NPROC_PER_NODE`.
- `evaluation/run.sh` exists.
- Top-level `run_pipeline.sh` chains stages 1–6 plus evaluation.
- Every Python entrypoint accepts `--config <path>` and `--help`; CLI flags override TOML values.

**Docs**

- `README.md` matches the documented skeleton; hardware notes live in `docs/REPRODUCING_THESIS.md`, not in README.
- `docs/PIPELINE.md` exists; opens with input data requirements and covers six pipeline stages plus evaluation; no thesis section references.
- `docs/REPRODUCING_THESIS.md` exists and covers source dataset roster, per-dataset `k` and HNM parameters, hardware notes, and end-to-end commands.
- `docs/REPRODUCING_ABLATIONS.md` exists and documents all three ablations as described.
- Every `pipeline/XX_*/` folder has a `README.md` following the per-stage template.
- `data_prep/README.md`, `evaluation/README.md`, `tools/README.md`, `configs/README.md`, `examples/README.md` all exist.
- No user-facing document contains `§3.X`, "Chapter 3", "Methodology chapter", or equivalent thesis section references. Equation numbers inside code comments are permitted.

**Example**

- `examples/data/` ships synthetic mini-parquets plus `mini_compiled.parquet`.
- `examples/configs/` mirrors `configs/thesis/` with small parameters.
- `examples/run.sh` runs from a fresh checkout to completion on a single GPU within about 10 minutes.

**Metadata**

- `LICENSE` file contains MIT license text.
- `pyproject.toml` declares all top-level runtime dependencies explicitly.
- `.env.example` path comments reflect the new folder names.
- `.gitignore` excludes `examples/work/`.

**No regressions**

- Every migrated script still runs (verified via `examples/run.sh` reaching the training stage).
- `pipeline/03_filter_consistency/restore_only.py` imports correctly from `score.py`.
- Slurm launchers in `pipeline/04_hard_negative_mining/slurm/` point at the new `mine.py` path.
- The strings `dataset_scripts/`, `train_scripts/`, and `benchmarking/` do not appear anywhere in the working tree.
