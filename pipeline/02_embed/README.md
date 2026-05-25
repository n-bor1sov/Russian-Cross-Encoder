# Stage 2 — Embedding

Embed every query and passage in both partitions with a frozen dense
retriever and build one FAISS index per source dataset.

## What it does

The embedder reads each Parquet shard, batches the unique queries and
passages, and calls an OpenAI-compatible `/v1/embeddings` endpoint to
produce a fixed-dimensional dense vector for every text. A local
tokenizer (Qwen) is used beforehand to truncate inputs to `max_tokens`
so the API never refuses oversized requests.

The vectors are written to disk keyed by `query_id` and `passage_id`,
which makes them reusable for the filtering and mining stages without
re-calling the API. In parallel, the embedder builds one flat FAISS
index per source dataset over the passages of that dataset, and a
corresponding index over the queries. Building one index per dataset
rather than a single global one keeps the index small enough to fit on
a single GPU and confines retrieval to the annotation regime that
produced the labelled pair.

This is the only stage that requires an external service. Once the
embeddings and indices are on disk, the rest of the pipeline is fully
offline.

## Inputs

- `${SPLIT_DATA_DIR}/{train,val}/*.parquet` — split shards from stage 1.
- `${EMBEDDING_API_BASE_URL}`, `${EMBEDDING_API_KEY}`,
  `${EMBEDDING_MODEL_NAME}` — endpoint credentials.
- `${QWEN_MODEL_PATH}` — local tokenizer used for length truncation.

## Outputs

- `${EMBEDDINGS_DIR}/<dataset>/queries.npy` and `passages.npy` — raw
  embedding tensors keyed by identifier.
- `${EMBEDDINGS_DIR}/<dataset>/passages.faiss` and `queries.faiss` —
  per-dataset FAISS indices (flat, exact search).

Stages 3 and 4 read from `${EMBEDDINGS_DIR}` directly.

## Key parameters

| Key                | Default          | Description                                                                   |
|--------------------|------------------|-------------------------------------------------------------------------------|
| `model`            | `Qwen3-Embedding`| Embedding model name passed to the API.                                       |
| `dimension`        | `1024`           | Expected output dimension; an assertion fails the run if the API disagrees.   |
| `batch_size`       | `128`            | Number of texts per embedding API call.                                       |
| `max_tokens`       | `512`            | Local truncation budget before calling the API.                               |
| `faiss_index_type` | `Flat`           | FAISS index factory string; flat means exact search.                          |
| `gpu`              | `true`           | If true, builds and searches FAISS indices on GPU.                            |

All keys live in `configs/thesis/02_embed.toml`. CLI flags override TOML.

## Files

- `embed_shards.py` — Python entrypoint.
- `run.sh` — shell wrapper.

## Running

    bash pipeline/02_embed/run.sh

To embed only the validation partition for a smoke test:

    bash pipeline/02_embed/run.sh --partition val

## Notes

- Per-dataset indexing is intentional: a single global index would
  silently fold heterogeneous annotation regimes into one similarity
  space and would not fit on a single GPU at corpus scale.
- The embedder is idempotent at the identifier level — rerunning skips
  texts whose embeddings already exist on disk — so partial failures
  are cheap to retry.
- The API client retries on transient errors; sustained 429s mean the
  endpoint quota is exhausted and the run should pause rather than
  burn retries.
