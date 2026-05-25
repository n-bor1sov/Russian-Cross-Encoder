# data_prep

Off-pipeline helpers for normalising raw source datasets into the input
schema that `pipeline/` consumes. These scripts capture the choices made
during the thesis work; their behaviour is not prescribed by the
pipeline and a reader applying this code to a new corpus is expected to
adapt or replace them.

## Expected output schema

Every script in this folder produces (or moves toward) the same input
schema that `pipeline/01_split/` expects:

| Column        | Type   | Description                                                    |
|---------------|--------|----------------------------------------------------------------|
| `query`       | string | The query text                                                 |
| `passage`     | string | The labelled positive passage for the query                    |
| `query_id`    | string | Stable identifier for the query                                |
| `passage_id`  | string | Stable identifier for the passage                              |
| `lang`        | string | ISO 639-1 language code                                        |
| `dataset`     | string | Source dataset name                                            |

Constraints (also documented in `docs/PIPELINE.md`): UTF-8 NFC strings,
no nulls, stable identifiers, cross-lingual variants of the same
conceptual pair share `query_id` and `passage_id`, Parquet+Snappy on
disk.

## Helpers

- `merge_to_shards.py` — concatenates per-source Parquet files into
  balanced shards, preserving the bipartite query–passage graph so that
  the splitter in stage 1 can find connected components without
  re-reading every source.
- `add_dataset_column.py` — annotates an existing Parquet with a
  `dataset` column. Useful when a source file has every other required
  column but lacks the dataset tag.

Both scripts accept `--help`. Neither is on the critical path: the
pipeline only requires a directory of Parquets matching the schema at
`${UNIFIED_DATA_DIR}`, regardless of how that directory was produced.
