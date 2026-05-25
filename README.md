# Russian Cross-Encoder Reranker

End-to-end training pipeline and trained-checkpoint recipe for a
Russian-language cross-encoder reranker. The repository contains the
full data-processing, training, and evaluation code that produced the
results in the accompanying thesis (`thesis.pdf`).

## What's in the repository

| Folder         | What it contains                                                       |
|----------------|------------------------------------------------------------------------|
| `data_prep/`   | Off-pipeline helpers for normalising raw sources to the input schema   |
| `pipeline/`    | The six-stage training pipeline (splitting → training)                 |
| `evaluation/`  | Two-stage BM25 + reranker evaluation on Russian IR benchmarks          |
| `configs/`     | Per-stage TOML configurations; `configs/thesis/` reproduces the paper  |
| `docs/`        | `PIPELINE.md`, `REPRODUCING_THESIS.md`, `REPRODUCING_ABLATIONS.md`     |
| `examples/`    | Self-contained miniature run to verify the install                     |
| `tools/`       | Off-pipeline utilities                                                 |

For a narrative walkthrough of the pipeline, read `docs/PIPELINE.md`.

## Requirements

- Python 3.13+
- CUDA 12+ (for FAISS-GPU and distributed training)
- An OpenAI-compatible embedding endpoint
- A single GPU is enough to run `examples/`; the thesis training
  configuration uses 8 GPUs. Detailed hardware notes are in
  `docs/REPRODUCING_THESIS.md`.

The full dependency list is in `pyproject.toml`.

## Install

    uv sync
    # or: pip install -e .

## Configure

    cp .env.example .env
    # Edit .env to point at your data directories, embedding API,
    # and base model

Required environment variables are documented in `.env.example`.
Hyperparameters used for the thesis are in `configs/thesis/`; see
`configs/README.md` for the config schema and override rules.

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

The pipeline expects a corpus of query–passage pairs already in a
common schema. Expected columns and constraints are documented in
`docs/PIPELINE.md` under "Input data requirements". Helpers for
normalising common source formats are in `data_prep/`.

## License

MIT. See `LICENSE`.
