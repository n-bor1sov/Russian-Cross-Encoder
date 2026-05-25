# Russian Cross-Encoder Reranker

Training pipeline for a cross-encoder reranker specialized for Russian-language retrieval. The project covers the full lifecycle from raw datasets to a trained model: data sharding, consistency filtering, hard negative mining, bucketed distributed training, and evaluation.

## Overview

```
dataset_scripts/     # 7-stage data preparation pipeline
train_scripts/       # Distributed training with DDP on 8+ GPUs
test/                # Evaluation on reranking benchmarks
hnm0.py              # Standalone hard negative mining entry point
```

### Data pipeline

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `dataset_scripts/01_merge_to_shards.py` | Merge raw parquets into balanced shards preserving query–passage graph connectivity |
| 2 | `dataset_scripts/02_embed_shards.py` | Embed shards with an OpenAI-compatible API; build per-shard FAISS indices |
| 3 | `dataset_scripts/03_filter_and_restore_datasets.py` | Consistency filtering: keep only pairs where the positive is retrievable |
| 4 | `dataset_scripts/04_restore_embeddings_by_dataset.py` | Restore per-dataset embedding files after filtering |
| 5 | `dataset_scripts/05_filter_by_rank.py` | Additional rank-based filtering |
| — | `dataset_scripts/finale/final_dataset_compilation.py` | Compile filtered datasets and mined hard negatives into a single training dataset |
| — | `dataset_scripts/hnm/HNM.py` | Shard-level hard negative mining (distributed version of `hnm0.py`) |

### Training

`train_scripts/train_bucketed.py` trains a `sentence-transformers` `CrossEncoder` with:

- **Loss**: `CachedMultipleNegativesRankingLoss` (scale = 10.0)
- **Negatives**: 8 hard + 2 random per query
- **Batching**: dataset-homogeneous bucketed batches, DDP-safe via custom `SameDatasetBatchSampler`
- **Distributed**: 8-GPU DDP via `torchrun`
- **Optimizer**: AdamW fused, cosine LR schedule, gradient checkpointing
- **Max sequence length**: 4096 tokens
- **Tracking**: ClearML (optional)

Run with:
```bash
torchrun --nproc_per_node=8 train_scripts/train_bucketed.py
```

### Evaluation

`test/test_reranker.py` evaluates models on reranking benchmarks (e.g. `mteb/RuBQReranking`). Supports:

- Cross-encoder (`sentence-transformers`)
- Embedding-based (SentenceTransformer + cosine similarity)
- Qwen3-Reranker (causal LM, yes/no logit scoring)
- mxbai-rerank, jina-reranker-v3, FlagReranker (BGE)

Metrics reported: NDCG@10, Accuracy@1/3/5, MAP@10, MRR@10.

```bash
python test/test_reranker.py --model_path ./checkpoint-2432 --data_dir ./data/RuBQReranking
```

## Setup

### Requirements

- Python 3.13+
- CUDA 12+ (for GPU FAISS and distributed training)
- `uv` (recommended) or `pip`

```bash
uv sync
# or: pip install -e .
```

Key dependencies (see `pyproject.toml`):
- `sentence-transformers` >= 5.2
- `transformers`, `datasets`, `torch`
- `faiss-gpu` (for embedding and hard negative mining on GPU)
- `openai` (embedding API client)
- `clearml` (optional experiment tracking)

### Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
# Edit .env with your paths and credentials
```

Required variables:

| Variable | Description |
|----------|-------------|
| `HF_HOME` | HuggingFace cache root |
| `HF_DATASETS_CACHE` | HuggingFace datasets cache |
| `EMBEDDING_API_BASE_URL` | OpenAI-compatible embedding endpoint |
| `EMBEDDING_API_KEY` | API key for the embedding endpoint |
| `EMBEDDING_MODEL_NAME` | Model name passed to the embedding API |
| `QWEN_MODEL_PATH` | Path to local Qwen tokenizer (used for text truncation before embedding) |
| `TRAIN_MODEL_PATH` | Path to the base encoder to fine-tune |
| `TRAIN_DATASET_PATH` | Path to bucketed training parquets |
| `EVAL_DATASET_PATH` | Path to bucketed validation parquets |
| `TRAIN_OUTPUT_DIR` | Directory to save checkpoints |

Optional:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAIN_LOG_DIR` | `./logs` | Per-rank training log directory |
| `USE_CLEARML` | `1` | Set to `0` to disable ClearML tracking |
| `CLEARML_PROJECT_NAME` | `CrossEncoders` | ClearML project name |
| `CLEARML_TASK_NAME` | `reranker-training` | ClearML task name |

Load the variables before running any script:
```bash
export $(grep -v '^#' .env | xargs)
```

## Hard negative mining

`hnm0.py` mines hard negatives for a single dataset parquet using FAISS.

```bash
# Step 1: compute and save embeddings
python hnm0.py --embeddings-only \
    -d /path/to/dataset.parquet \
    -o /path/to/output \
    -n dataset_name

# Step 2: mine from pre-computed embeddings
python hnm0.py \
    -d /path/to/dataset.parquet \
    -o /path/to/output \
    -n dataset_name \
    --load-embeddings /path/to/output/embeddings_dataset_name
```

For multi-shard distributed mining, use `dataset_scripts/hnm/HNM.py` with the shell scripts provided alongside it.

## Bucketed dataset preparation

Group training rows by stable query hash into per-rank buckets before distributed training. This ensures each GPU always processes one dataset at a time, which is required by the `SameDatasetBatchSampler`.

```bash
python train_scripts/prepare_query_buckets.py \
    --input-dir /path/to/final/dataset \
    --output-dir /path/to/bucketed/dataset \
    --num-buckets 64
```

## Repository structure

```
.
├── dataset_scripts/
│   ├── 01_merge_to_shards.py                   # Graph-safe shard creation
│   ├── 02_embed_shards.py                       # API embedding + FAISS index build
│   ├── 03_filter_and_restore_datasets.py        # Consistency filtering
│   ├── 03_filter_and_restore_datasets_fixed.py  # Bug-fixed variant
│   ├── 04_restore_embeddings_by_dataset.py      # Restore embeddings post-filter
│   ├── 05_filter_by_rank.py                     # Rank-based filtering
│   ├── add_parquet_dataset_column.py
│   ├── debug_faiss_gpu.py
│   ├── split_datasets.py
│   ├── restore_only_datasets.py
│   ├── finale/
│   │   ├── final_dataset_compilation.py         # Assemble final training set
│   │   └── analyze_final_dataset.py
│   └── hnm/
│       ├── HNM.py                               # Distributed shard-level HNM
│       ├── HNM.sh / HNM_live.sh                # Launch scripts
│       └── embedding_lookup.py
├── train_scripts/
│   ├── train_bucketed.py                        # Main DDP training script
│   ├── sampler_bucketed.py                      # Dataset-homogeneous batch sampler
│   ├── prepare_query_buckets.py                 # Pre-training data bucketing
│   └── PositiveOnly_train_bucketed.py           # Variant without hard negatives
├── test/
│   ├── test_reranker.py                         # Multi-model evaluation script
│   └── visualize_results_v2.ipynb
├── hnm0.py                                      # Standalone HNM (single dataset)
├── .env.example                                 # Environment variable template
├── pyproject.toml
└── uv.lock
```
