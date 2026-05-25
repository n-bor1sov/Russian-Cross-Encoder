from __future__ import annotations

import argparse
import gc
import json
import tomllib
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm.auto import tqdm


REQUIRED_STATS_COLUMNS = ("consistency", "retrieval_rank", "virtual_rank")
REQUIRED_FILTER_COLUMNS = ("retrieval_rank", "virtual_rank")


def parquet_paths(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")
    return paths


def require_columns(schema: pa.Schema, columns: tuple[str, ...], path: Path) -> None:
    missing = [column for column in columns if schema.get_field_index(column) == -1]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def count_true(mask: pa.Array | pa.ChunkedArray) -> int:
    return int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)


def percent(part: int, total: int) -> float:
    return (part / total * 100.0) if total else 0.0


def inspect_dataset(path: Path, batch_size: int, top_k: int) -> dict[str, int | str]:
    parquet_file = pq.ParquetFile(path)
    require_columns(parquet_file.schema_arrow, REQUIRED_STATS_COLUMNS, path)

    rows = 0
    consistency_false = 0
    retrieval_rank_found = 0
    retrieval_rank_below_top_k = 0
    virtual_rank_missing = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(REQUIRED_STATS_COLUMNS)):
        table = pa.Table.from_batches([batch])
        retrieval_rank = table["retrieval_rank"]
        rows += table.num_rows
        consistency_false += count_true(pc.equal(table["consistency"], False))
        retrieval_rank_found += count_true(pc.not_equal(retrieval_rank, -1))
        retrieval_rank_below_top_k += count_true(
            pc.and_(
                pc.not_equal(retrieval_rank, -1),
                pc.less(retrieval_rank, top_k),
            )
        )
        virtual_rank_missing += count_true(pc.equal(table["virtual_rank"], -1))
        del table
        gc.collect()

    return {
        "dataset": path.stem,
        "path": str(path),
        "rows": rows,
        "consistency_false": consistency_false,
        "retrieval_rank_not_minus_one": retrieval_rank_found,
        "retrieval_rank_below_top_k": retrieval_rank_below_top_k,
        "top_k": top_k,
        "virtual_rank_minus_one": virtual_rank_missing,
    }


def print_stats(stats: dict[str, int | str]) -> None:
    rows = int(stats["rows"])
    consistency_false = int(stats["consistency_false"])
    retrieval_rank_found = int(stats["retrieval_rank_not_minus_one"])
    retrieval_rank_below_top_k = int(stats["retrieval_rank_below_top_k"])
    top_k = int(stats["top_k"])
    virtual_rank_missing = int(stats["virtual_rank_minus_one"])

    print(f"\n{stats['dataset']}")
    print(f"  rows: {rows:,}")
    print(
        "  consistency == False: "
        f"{consistency_false:,} ({percent(consistency_false, rows):.2f}%)"
    )
    print(
        "  retrieval_rank != -1: "
        f"{retrieval_rank_found:,} ({percent(retrieval_rank_found, rows):.2f}%)"
    )
    print(
        f"  retrieval_rank < {top_k} and != -1: "
        f"{retrieval_rank_below_top_k:,} ({percent(retrieval_rank_below_top_k, rows):.2f}%)"
    )
    print(
        "  virtual_rank == -1: "
        f"{virtual_rank_missing:,} ({percent(virtual_rank_missing, rows):.2f}%)"
    )


def filter_dataset(
    input_path: Path,
    output_path: Path,
    *,
    top_k: int,
    batch_size: int,
    compression: str,
) -> dict[str, int | str]:
    parquet_file = pq.ParquetFile(input_path)
    require_columns(parquet_file.schema_arrow, REQUIRED_FILTER_COLUMNS, input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = pq.ParquetWriter(
        str(output_path),
        parquet_file.schema_arrow,
        compression=compression,
    )

    input_rows = 0
    output_rows = 0
    kept_by_retrieval_rank = 0
    kept_by_virtual_rank = 0
    try:
        batches = parquet_file.iter_batches(batch_size=batch_size)
        for batch in tqdm(batches, desc=f"Filtering {input_path.name}", leave=False):
            table = pa.Table.from_batches([batch])
            retrieval_rank = table["retrieval_rank"]
            virtual_rank = table["virtual_rank"]
            retrieval_keep = pc.and_(
                pc.not_equal(retrieval_rank, -1),
                pc.less(retrieval_rank, top_k),
            )
            retrieval_outside_top_k = pc.or_(
                pc.equal(retrieval_rank, -1),
                pc.greater_equal(retrieval_rank, top_k),
            )
            virtual_keep = pc.and_(
                retrieval_outside_top_k,
                pc.and_(
                    pc.not_equal(virtual_rank, -1),
                    pc.less(virtual_rank, top_k),
                ),
            )
            mask = pc.or_(retrieval_keep, virtual_keep)
            filtered = table.filter(mask)
            input_rows += table.num_rows
            output_rows += filtered.num_rows
            kept_by_retrieval_rank += count_true(retrieval_keep)
            kept_by_virtual_rank += count_true(virtual_keep)
            if filtered.num_rows:
                writer.write_table(filtered)
            del table, filtered
            gc.collect()
    finally:
        writer.close()

    return {
        "dataset": input_path.stem,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "kept_by_retrieval_rank": kept_by_retrieval_rank,
        "kept_by_virtual_rank": kept_by_virtual_rank,
        "kept_percent": percent(output_rows, input_rows),
    }


def run_stats(input_dir: Path, batch_size: int, top_k: int) -> None:
    paths = parquet_paths(input_dir)
    totals = {
        "rows": 0,
        "consistency_false": 0,
        "retrieval_rank_not_minus_one": 0,
        "retrieval_rank_below_top_k": 0,
        "top_k": top_k,
        "virtual_rank_minus_one": 0,
    }

    for path in tqdm(paths, desc="Inspecting datasets"):
        stats = inspect_dataset(path, batch_size, top_k)
        print_stats(stats)
        for key in (
            "rows",
            "consistency_false",
            "retrieval_rank_not_minus_one",
            "retrieval_rank_below_top_k",
            "virtual_rank_minus_one",
        ):
            totals[key] += int(stats[key])

    print_stats(
        {
            "dataset": "CUMULATIVE ALL DATASETS",
            "path": str(input_dir),
            **totals,
        }
    )


def run_filter(
    input_dir: Path,
    output_dir: Path,
    *,
    top_k: int,
    batch_size: int,
    compression: str,
) -> None:
    if input_dir == output_dir:
        raise ValueError("--output-dir must be different from --input-dir")

    paths = parquet_paths(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for input_path in tqdm(paths, desc="Filtering datasets"):
        output_path = output_dir / input_path.name
        result = filter_dataset(
            input_path,
            output_path,
            top_k=top_k,
            batch_size=batch_size,
            compression=compression,
        )
        results.append(result)
        print(
            f"{input_path.stem}: kept {result['output_rows']:,}/{result['input_rows']:,} "
            f"({result['kept_percent']:.2f}%; retrieval={result['kept_by_retrieval_rank']:,}, "
            f"virtual={result['kept_by_virtual_rank']:,})"
        )

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "filter_mode": "hybrid_retrieval_then_virtual",
        "top_k": top_k,
        "batch_size": batch_size,
        "compression": compression,
        "datasets": results,
    }
    with (output_dir / "filter_by_rank_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    total_input = sum(int(result["input_rows"]) for result in results)
    total_output = sum(int(result["output_rows"]) for result in results)
    print(
        f"\nDone. Kept {total_output:,}/{total_input:,} rows "
        f"({percent(total_output, total_input):.2f}%) in {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter restored dataset-wise parquets by retrieval rank with virtual-rank fallback."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--top-k",
        "--k",
        dest="k",
        type=int,
        default=None,
        help="Top-k threshold used for filtering and stats reporting.",
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Only inspect input parquets and print per-dataset statistics.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML config file (configs/thesis/03_filter_consistency.toml). CLI flags override TOML values.",
    )
    args = parser.parse_args()

    if args.config is not None:
        _cfg = tomllib.loads(args.config.read_text())
        for _k, _v in _cfg.items():
            if getattr(args, _k, "_missing_sentinel") is None:
                setattr(args, _k, _v)

    if args.k is None:
        args.k = 30

    return args


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.k <= 0:
        raise ValueError("--top-k must be positive")

    if args.stats:
        run_stats(input_dir, args.batch_size, args.k)
        return

    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --stats is set")

    run_filter(
        input_dir,
        args.output_dir.resolve(),
        top_k=args.k,
        batch_size=args.batch_size,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
