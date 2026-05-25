# Stage 1 — Splitting

Partition the unified corpus into a 95:5 train/validation split that keeps
the bipartite query–passage graph intact and keeps cross-lingual variants
of the same conceptual pair together.

## What it does

The splitter operates on the bipartite graph of queries and passages
implied by the labelled pairs: a query and a passage are linked iff they
appear together as a labelled pair somewhere in the corpus. The splitter
finds the connected components of that graph and assigns each component
to a single partition. As a result, no query in the train partition can
share any passage with a query in the validation partition.

Because cross-lingual variants of the same conceptual pair share their
`query_id` and `passage_id`, all language variants of a given pair fall
into the same connected component and therefore the same partition. The
splitter does not need any extra language-aware logic to enforce this.

The choice of which components go to validation is randomised under a
fixed seed and stops when the realised validation fraction is closest to
the target. Rerunning with the same input and seed reproduces the same
partition.

## Inputs

- `${UNIFIED_DATA_DIR}/*.parquet` — unified input shards with the
  six-column input schema (see `docs/PIPELINE.md`).

## Outputs

- `${SPLIT_DATA_DIR}/train/*.parquet` — training partition shards.
- `${SPLIT_DATA_DIR}/val/*.parquet` — validation partition shards.

Downstream stages (embedding, filtering, mining) consume both
partitions independently and keep their outputs separated by partition.

## Key parameters

| Key            | Default | Description                                                  |
|----------------|---------|--------------------------------------------------------------|
| `seed`         | `42`    | Random seed for component assignment.                        |
| `val_fraction` | `0.05`  | Target fraction of pairs assigned to the validation set.     |

All keys live in `configs/thesis/01_split.toml`. CLI flags override TOML.

## Files

- `split.py` — Python entrypoint; builds the bipartite graph and writes
  the partitioned Parquets.
- `run.sh` — shell wrapper that loads `.env` and invokes `split.py`
  against `configs/thesis/01_split.toml`.

## Running

    bash pipeline/01_split/run.sh

To override `val_fraction` for an ad-hoc run:

    bash pipeline/01_split/run.sh --val-fraction 0.1

## Notes

- The realised per-dataset split ratio is not forced to 95:5 because
  component sizes vary across source datasets; small datasets with a few
  large components may end up entirely in one partition.
- Components glue every cross-lingual variant of a pair together, so
  monolingual evaluation on the validation partition still sees a
  consistent set of pairs.
- If a downstream stage reports a query appearing in both partitions,
  the input shards almost certainly violate the stable-identifier
  contract; rerun `data_prep/` to renormalise.
