from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("03_filter_and_restore_datasets.py")
MODULE = runpy.run_path(str(MODULE_PATH))
restore_dataset_outputs = MODULE["restore_dataset_outputs"]
validate_dataset_outputs = MODULE["validate_dataset_outputs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore dataset-wise outputs from already scored shard parquets.")
    parser.add_argument("--shards-root", required=True, type=Path)
    parser.add_argument("--final-output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--restore-batch-size", type=int, default=100_000)
    parser.add_argument("--compression", default="snappy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards_root = args.shards_root.resolve()
    scored_paths = sorted(shards_root.glob(f"group_*/filtered_data_top_{args.top_k}.parquet"))
    if not scored_paths:
        raise FileNotFoundError(
            f"No scored shard parquets found under {shards_root} for top_k={args.top_k}. "
            "Expected paths like group_*/filtered_data_top_<top_k>.parquet"
        )

    output_dir = args.final_output_dir.resolve()
    dataset_counts = restore_dataset_outputs(
        scored_paths=scored_paths,
        output_dir=output_dir,
        batch_size=args.restore_batch_size,
        compression=args.compression,
    )
    validate_dataset_outputs(scored_paths, dataset_counts)

    manifest = {
        "top_k": args.top_k,
        "restore_batch_size": args.restore_batch_size,
        "compression": args.compression,
        "scored_paths": [str(path) for path in scored_paths],
        "dataset_output_counts": dataset_counts,
        "final_output_dir": str(output_dir),
    }
    with (shards_root / f"restore_only_manifest_top_{args.top_k}.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"Restored {len(scored_paths)} scored shard files into {output_dir}")


if __name__ == "__main__":
    main()
