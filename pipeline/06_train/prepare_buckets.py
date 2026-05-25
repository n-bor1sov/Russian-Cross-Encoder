
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("prepare_buckets")


def stable_hash63(value: Any) -> int:
    """Stable non-negative 63-bit hash suitable for HF/Arrow int64 columns."""
    text = "" if value is None else str(value)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)


def first_non_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            text = first_non_empty(item)
            if text:
                return text
        return ""
    text = str(value).strip()
    return text


def collect_input_files(input_root: Path, split: str) -> list[Path]:
    split_dir = input_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    files = sorted(split_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {split_dir}")
    return files


def infer_dataset_name(table: pa.Table, source_file: Path, dataset_col: str | None) -> str:
    if dataset_col and dataset_col in table.column_names and table.num_rows > 0:
        values = table[dataset_col].combine_chunks().to_pylist()
        for value in values:
            if value is not None and str(value).strip():
                return str(value).strip()
    return source_file.stem


def ensure_columns(table: pa.Table, columns: dict[str, pa.Array]) -> pa.Table:
    result = table
    for name, array in columns.items():
        if name in result.column_names:
            result = result.set_column(result.column_names.index(name), name, array)
        else:
            result = result.append_column(name, array)
    return result


class BucketWriter:
    """Writes bucketed Parquet part files and records manifest entries.

    To keep the implementation robust on shared filesystems, every batch group is
    written as a separate part file. This avoids long-lived hundreds of open
    ParquetWriter handles. The training loader can consume many part files from
    the manifest.
    """

    def __init__(
        self,
        output_root: Path,
        compression: str,
    ) -> None:
        self.output_root = output_root
        self.compression = compression
        self.part_counters: dict[tuple[str, str, int], int] = defaultdict(int)
        self.entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.total_rows: dict[str, int] = defaultdict(int)
        self.text_length_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "max_chars": 0})

    def write(self, split: str, dataset_name: str, bucket_id: int, table: pa.Table, query_id_col: str) -> None:
        if table.num_rows == 0:
            return

        key = (split, dataset_name, int(bucket_id))
        part_id = self.part_counters[key]
        self.part_counters[key] += 1

        rel_path = Path(split) / dataset_name / f"bucket_{bucket_id:06d}" / f"part_{part_id:06d}.parquet"
        out_path = self.output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        pq.write_table(table, out_path, compression=self.compression)

        unique_qids = 0
        if query_id_col in table.column_names:
            unique_qids = len(set(x for x in table[query_id_col].combine_chunks().to_pylist() if x is not None))

        entry = {
            "split": split,
            "dataset": dataset_name,
            "bucket_id": int(bucket_id),
            "path": rel_path.as_posix(),
            "rows": int(table.num_rows),
            "unique_query_ids": int(unique_qids),
            "part": int(part_id),
        }
        self.entries[split].append(entry)
        self.total_rows[split] += int(table.num_rows)


def process_file(
    *,
    split: str,
    input_file: Path,
    writer: BucketWriter,
    num_buckets: int,
    query_id_col: str,
    dataset_col: str | None,
    positive_col: str,
    lang_col: str | None,
    num_negatives: int,
    batch_size: int,
) -> tuple[int, int]:
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    if total_rows == 0:
        logger.warning("Skipping empty parquet file: %s", input_file)
        return 0, 0

    processed_rows = 0
    written_rows = 0

    batch_iter = parquet_file.iter_batches(batch_size=batch_size)
    progress = tqdm(
        total=total_rows,
        desc=f"{split}/{input_file.name}",
        unit="row",
        leave=False,
    )

    try:
        for record_batch in batch_iter:
            table = pa.Table.from_batches([record_batch])
            n = table.num_rows
            processed_rows += n
            progress.update(n)

            if query_id_col not in table.column_names:
                raise ValueError(f"{input_file} is missing required query_id column: {query_id_col}")

            if dataset_col and dataset_col in table.column_names:
                dataset_values = [
                    str(x).strip() if x is not None and str(x).strip() else input_file.stem
                    for x in table[dataset_col].combine_chunks().to_pylist()
                ]
            else:
                dataset_values = [input_file.stem] * n

            query_ids = table[query_id_col].combine_chunks().to_pylist()

            query_key64_values: list[int] = []
            bucket_id_values: list[int] = []
            bad_qids = 0
            for qid in query_ids:
                qid_text = "" if qid is None else str(qid).strip()
                if not qid_text:
                    bad_qids += 1
                    # Keep deterministic but unique-ish fallback to avoid crashing long jobs.
                    qid_text = f"__missing_query_id__:{input_file}:{processed_rows}:{len(query_key64_values)}"
                qh = stable_hash63(qid_text)
                query_key64_values.append(qh)
                bucket_id_values.append(qh % num_buckets)

            if bad_qids:
                logger.warning("%s: %d row(s) with empty/missing query_id; used deterministic fallback", input_file, bad_qids)

            positive_key64_values: list[int] = []
            if positive_col in table.column_names:
                positives = table[positive_col].combine_chunks().to_pylist()
                for pos in positives:
                    text = first_non_empty(pos)
                    positive_key64_values.append(stable_hash63(text) if text else stable_hash63("__empty_positive__"))
            else:
                # Missing positive column is not ideal, but keeping a unique row key is safer than over-collapsing.
                positive_key64_values = [
                    stable_hash63(f"__missing_positive__:{input_file}:{processed_rows}:{i}") for i in range(n)
                ]

            extra_columns: dict[str, pa.Array] = {
                "query_key64": pa.array(query_key64_values, type=pa.int64()),
                "bucket_id": pa.array(bucket_id_values, type=pa.int32()),
                "positive_key64": pa.array(positive_key64_values, type=pa.int64()),
            }
            for neg_idx in range(1, num_negatives + 1):
                neg_col = None
                for candidate in (f"negative{neg_idx}", f"negative_{neg_idx}"):
                    if candidate in table.column_names:
                        neg_col = candidate
                        break
                if neg_col is None:
                    continue
                neg_values = table[neg_col].combine_chunks().to_pylist()
                neg_key_values = [
                    stable_hash63(text) if (text := first_non_empty(neg)) else stable_hash63(f"__empty_negative__:{neg_idx}")
                    for neg in neg_values
                ]
                extra_columns[f"negative_key64_{neg_idx}"] = pa.array(neg_key_values, type=pa.int64())

            if lang_col and lang_col in table.column_names and lang_col != "lang":
                lang_values = [
                    first_non_empty(value)
                    for value in table[lang_col].combine_chunks().to_pylist()
                ]
                extra_columns["lang"] = pa.array(lang_values, type=pa.string())
            elif "lang" not in table.column_names:
                extra_columns["lang"] = pa.array([""] * n, type=pa.string())

            table = ensure_columns(
                table,
                extra_columns,
            )

            # Split current record batch by (dataset, bucket_id). This keeps same query_id
            # in the same bucket and also supports input files that unexpectedly contain
            # several dataset values.
            group_to_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
            for i, bucket_id in enumerate(bucket_id_values):
                group_to_indices[(dataset_values[i], int(bucket_id))].append(i)

            group_iter = group_to_indices.items()
            if len(group_to_indices) > 32:
                group_iter = tqdm(
                    group_iter,
                    total=len(group_to_indices),
                    desc=f"{split}/{input_file.name}: write dataset/buckets",
                    unit="group",
                    leave=False,
                )

            for (dataset_name, bucket_id), indices in group_iter:
                subset = table.take(pa.array(indices, type=pa.int64()))
                stats = writer.text_length_stats[dataset_name]
                stats["rows"] += subset.num_rows
                text_cols = [name for name in ("query", "anchor", positive_col) if name in subset.column_names]
                text_cols.extend(name for name in subset.column_names if name.startswith("negative") and not name.startswith("negative_key64"))
                for col in text_cols:
                    values = subset[col].combine_chunks().to_pylist()
                    max_chars = max((len(first_non_empty(value)) for value in values), default=0)
                    stats["max_chars"] = max(stats["max_chars"], max_chars)
                writer.write(split, dataset_name, bucket_id, subset, query_id_col=query_id_col)
                written_rows += subset.num_rows
    finally:
        progress.close()

    return processed_rows, written_rows


def simulate_balance(entries: list[dict[str, Any]], world_sizes: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    # For each dataset assign buckets by modulo and collect rows per rank.
    for world_size in world_sizes:
        per_rank = [0 for _ in range(world_size)]
        by_dataset: dict[str, list[int]] = defaultdict(lambda: [0 for _ in range(world_size)])
        for entry in entries:
            rank = int(entry["bucket_id"]) % world_size
            rows = int(entry["rows"])
            per_rank[rank] += rows
            by_dataset[str(entry["dataset"])][rank] += rows
        mean_rows = sum(per_rank) / max(world_size, 1)
        result[str(world_size)] = {
            "rows_per_rank": per_rank,
            "max_over_mean": (max(per_rank) / mean_rows) if mean_rows else 0.0,
            "datasets_with_zero_rank": {
                ds: counts for ds, counts in by_dataset.items() if any(c == 0 for c in counts)
            },
        }
    return result


def write_manifest(
    *,
    output_root: Path,
    writer: BucketWriter,
    input_root: Path,
    num_buckets: int,
    query_id_col: str,
    dataset_col: str | None,
    positive_col: str,
    lang_col: str | None,
    num_negatives: int,
    default_max_length: int,
    compression: str,
    balance_world_sizes: list[int],
) -> None:
    splits = {split: sorted(entries, key=lambda e: (e["dataset"], e["bucket_id"], e["path"])) for split, entries in writer.entries.items()}

    totals = {
        split: {
            "rows": sum(int(e["rows"]) for e in entries),
            "files": len(entries),
            "datasets": sorted(set(str(e["dataset"]) for e in entries)),
            "non_empty_buckets": len(set((str(e["dataset"]), int(e["bucket_id"])) for e in entries)),
            "balance": simulate_balance(entries, balance_world_sizes),
        }
        for split, entries in splits.items()
    }

    manifest = {
        "version": 1,
        "created_at_unix": int(time.time()),
        "input_root": str(input_root),
        "num_buckets": int(num_buckets),
        "hash": "blake2b_63_little",
        "query_id_col": query_id_col,
        "dataset_col": dataset_col,
        "positive_col": positive_col,
        "lang_col": lang_col,
        "num_negatives": int(num_negatives),
        "per_dataset_max_length": {
            dataset: min(int(default_max_length), max(128, int(math.ceil(stats["max_chars"] / 4))))
            for dataset, stats in sorted(writer.text_length_stats.items())
        },
        "per_dataset_text_stats": dict(sorted(writer.text_length_stats.items())),
        "compression": compression,
        "splits": splits,
        "totals": totals,
    }

    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("Wrote manifest: %s", manifest_path)
    for split, info in totals.items():
        logger.info(
            "Split %s: rows=%d files=%d datasets=%d non_empty_dataset_buckets=%d",
            split,
            info["rows"],
            info["files"],
            len(info["datasets"]),
            info["non_empty_buckets"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket query-passage parquet datasets by stable query_id hash.")
    parser.add_argument("--input_root", type=Path, required=True, help="Root containing split directories, e.g. train/ and validation/.")
    parser.add_argument("--output_root", type=Path, required=True, help="Output root for bucketed parquet files and manifest.json.")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], help="Split directory names to process.")
    parser.add_argument("--num_buckets", type=int, default=1024, help="Number of stable query buckets.")
    parser.add_argument("--query_id_col", default="query_id", help="Column containing stable semantic query id.")
    parser.add_argument("--dataset_col", default="dataset", help="Dataset-name column. If absent, source parquet stem is used.")
    parser.add_argument("--positive_col", default="positive", help="Positive passage column for positive_key64.")
    parser.add_argument("--lang_col", default="lang", help="Optional source language column to persist as lang for eval stratification.")
    parser.add_argument("--num_negatives", type=int, default=8, help="Number of negativeN / negative_N columns to hash.")
    parser.add_argument("--default_max_length", type=int, default=4096, help="Upper bound for per-dataset max_length hints in manifest.")
    parser.add_argument("--batch_size", type=int, default=50_000, help="PyArrow record batch size.")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument(
        "--balance_world_sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 5, 8],
        help="World sizes to simulate in manifest balance stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_buckets <= 0:
        raise ValueError("--num_buckets must be positive")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_root}. Pass --overwrite to continue.")
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Input root: %s", input_root)
    logger.info("Output root: %s", output_root)
    logger.info("Splits: %s", args.splits)
    logger.info("num_buckets=%d query_id_col=%s dataset_col=%s", args.num_buckets, args.query_id_col, args.dataset_col)

    writer = BucketWriter(output_root=output_root, compression=args.compression)

    grand_processed = 0
    grand_written = 0

    for split in args.splits:
        files = collect_input_files(input_root, split)
        logger.info("Processing split=%s with %d parquet file(s)", split, len(files))
        file_iter = tqdm(files, desc=f"Split {split}: files", unit="file", leave=True)

        for input_file in file_iter:
            processed, written = process_file(
                split=split,
                input_file=input_file,
                writer=writer,
                num_buckets=args.num_buckets,
                query_id_col=args.query_id_col,
                dataset_col=args.dataset_col,
                positive_col=args.positive_col,
                lang_col=args.lang_col,
                num_negatives=args.num_negatives,
                batch_size=args.batch_size,
            )
            grand_processed += processed
            grand_written += written
            logger.info("%s: processed=%d written=%d", input_file, processed, written)

    if grand_processed != grand_written:
        raise RuntimeError(f"Row count mismatch: processed={grand_processed}, written={grand_written}")

    write_manifest(
        output_root=output_root,
        writer=writer,
        input_root=input_root,
        num_buckets=args.num_buckets,
        query_id_col=args.query_id_col,
        dataset_col=args.dataset_col,
        positive_col=args.positive_col,
        lang_col=args.lang_col,
        num_negatives=args.num_negatives,
        default_max_length=args.default_max_length,
        compression=args.compression,
        balance_world_sizes=args.balance_world_sizes,
    )

    logger.info("Done. Total rows processed/written: %d", grand_written)


if __name__ == "__main__":
    main()
