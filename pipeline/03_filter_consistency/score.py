from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
import tomllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def normalize(value: Any) -> str:
    return "" if value is None else str(value)


def scoped_key(value: Any, lang: Any, dataset: Any) -> tuple[str, str, str]:
    return normalize(value), normalize(lang), normalize(dataset)


def safe_dataset_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return clean or "unknown_dataset"


def selected_shards(shards_root: Path, shard: int | None, num_shards: int | None) -> list[int]:
    if shard is not None:
        return [shard]
    if num_shards is not None:
        return list(range(num_shards))
    ids: list[int] = []
    for path in sorted(shards_root.glob("group_*/shard_*.parquet")):
        try:
            ids.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    if not ids:
        raise FileNotFoundError(f"No group_*/shard_*.parquet files found under {shards_root}")
    return ids


def group_dir(shards_root: Path, shard_id: int) -> Path:
    return shards_root / f"group_{shard_id}"


def shard_path(shards_root: Path, shard_id: int) -> Path:
    return group_dir(shards_root, shard_id) / f"shard_{shard_id:02d}.parquet"


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_numpy_dict(path: Path) -> dict:
    raw = np.load(str(path), allow_pickle=True)
    if raw.ndim == 0:
        return raw.item()
    return dict(raw)


def move_index_to_gpu(cpu_index, gpu_id: int = 0):
    if faiss.get_num_gpus() == 0:
        raise RuntimeError("No GPUs detected. Install faiss-gpu or check CUDA installation.")
    resources = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(resources, gpu_id, cpu_index)
    return gpu_index, resources


def load_passage_index(shard_base: Path, use_gpu: bool, gpu_id: int):
    passages_index = faiss.read_index(str(shard_base / "faiss_passages_index.bin"))
    gpu_resources = None
    if use_gpu:
        passages_index, gpu_resources = move_index_to_gpu(passages_index, gpu_id)
    return passages_index, gpu_resources


def load_query_embeddings(shard_base: Path) -> dict[Any, np.ndarray]:
    scoped_path = shard_base / "query_embedding.pkl"
    legacy_path = shard_base / "query_embedding_by_text.pkl"
    if scoped_path.exists():
        return load_pickle(scoped_path)
    if legacy_path.exists():
        return load_pickle(legacy_path)
    raise FileNotFoundError(f"No query embedding map found in {shard_base}")


def load_passage_key_to_faiss(shard_base: Path) -> dict[tuple[str, str, str], int]:
    key_map_path = shard_base / "passage_key_to_faiss.pkl"
    if key_map_path.exists():
        return load_pickle(key_map_path)

    id_map = load_numpy_dict(shard_base / "passages_id_map.npy")
    key_map: dict[tuple[str, str, str], int] = {}
    for faiss_id, entry in id_map.items():
        passage_id, _text, dataset, lang = entry
        key_map[(normalize(passage_id), normalize(lang), normalize(dataset))] = int(faiss_id)
    return key_map


def embedding_for_row(
    query_embeddings: dict[Any, np.ndarray],
    query_id: Any,
    query_text: Any,
    lang: Any,
    dataset: Any,
) -> Optional[np.ndarray]:
    key = scoped_key(query_id, lang, dataset)
    embedding = query_embeddings.get(key)
    if embedding is not None:
        return embedding
    return query_embeddings.get(normalize(query_text).strip())


def build_query_positives(dataset_path: Path, batch_size: int) -> dict[tuple[str, str, str], list[tuple[str, str, str]]]:
    positives: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    seen: dict[tuple[str, str, str], set[tuple[str, str, str]]] = {}
    parquet_file = pq.ParquetFile(dataset_path)
    total_rows = parquet_file.metadata.num_rows
    columns = ["query_id", "passage_id", "lang", "dataset"]

    with tqdm(total=total_rows, desc="Building positives map") as pbar:
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            rows = batch.to_pydict()
            for query_id, passage_id, lang, dataset in zip(
                rows["query_id"], rows["passage_id"], rows["lang"], rows["dataset"]
            ):
                query_key = scoped_key(query_id, lang, dataset)
                passage_key = scoped_key(passage_id, lang, dataset)
                if query_key not in positives:
                    positives[query_key] = []
                    seen[query_key] = set()
                if passage_key not in seen[query_key]:
                    positives[query_key].append(passage_key)
                    seen[query_key].add(passage_key)
            pbar.update(len(rows["query_id"]))
    return positives


def apply_hole_fitting(
    retrieved_faiss_ids: list[int],
    all_positive_faiss_ids: set[int],
    current_positive_faiss_id: int | None,
    top_k: int,
) -> tuple[bool, int]:
    if current_positive_faiss_id is None:
        return False, -1
    filtered_ids = [
        pid for pid in retrieved_faiss_ids if pid not in all_positive_faiss_ids or pid == current_positive_faiss_id
    ]
    try:
        virtual_rank = filtered_ids.index(current_positive_faiss_id) + 1
    except ValueError:
        return False, -1
    return virtual_rank <= top_k, virtual_rank


def search_faiss(
    passages_index,
    embeddings: np.ndarray,
    top_k: int,
    faiss_search_batch_size: int,
) -> np.ndarray:
    if faiss_search_batch_size <= 0 or len(embeddings) <= faiss_search_batch_size:
        _, retrieved_ids = passages_index.search(embeddings, top_k)
        return retrieved_ids

    retrieved_chunks: list[np.ndarray] = []
    for start in range(0, len(embeddings), faiss_search_batch_size):
        batch = embeddings[start : start + faiss_search_batch_size]
        _, retrieved_ids = passages_index.search(batch, top_k)
        retrieved_chunks.append(retrieved_ids)
    return np.concatenate(retrieved_chunks, axis=0)


def process_rows(
    rows: list[dict[str, Any]],
    passages_index,
    passage_key_to_faiss: dict[tuple[str, str, str], int],
    query_embeddings: dict[Any, np.ndarray],
    query_positives: dict[tuple[str, str, str], list[tuple[str, str, str]]],
    top_k: int,
    hole_fitting_top_k: int,
    faiss_search_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(rows)
    consistency_flags = np.zeros(n_rows, dtype=bool)
    retrieval_ranks = np.full(n_rows, -1, dtype=np.int32)
    virtual_ranks = np.full(n_rows, -1, dtype=np.int32)

    embeddings_by_row: list[np.ndarray | None] = [None] * n_rows
    positive_count_by_row = np.zeros(n_rows, dtype=np.int32)

    for idx, row in enumerate(rows):
        query_key = scoped_key(row.get("query_id"), row.get("lang"), row.get("dataset"))
        positives = query_positives.get(query_key, [])
        positive_count_by_row[idx] = len(positives)
        embedding = embedding_for_row(
            query_embeddings,
            row.get("query_id"),
            row.get("query"),
            row.get("lang"),
            row.get("dataset"),
        )
        if embedding is not None:
            embeddings_by_row[idx] = embedding.astype(np.float32, copy=False)

    retrieved_by_row: list[list[int] | None] = [None] * n_rows
    normal_rows = [idx for idx in range(n_rows) if embeddings_by_row[idx] is not None and positive_count_by_row[idx] <= 1]
    hole_rows = [idx for idx in range(n_rows) if embeddings_by_row[idx] is not None and positive_count_by_row[idx] > 1]

    if normal_rows:
        emb_matrix = np.stack([embeddings_by_row[idx] for idx in normal_rows], axis=0)
        retrieved_ids = search_faiss(passages_index, emb_matrix, top_k, faiss_search_batch_size)
        for result_idx, row_idx in enumerate(normal_rows):
            retrieved_by_row[row_idx] = retrieved_ids[result_idx].tolist()

    if hole_rows:
        emb_matrix = np.stack([embeddings_by_row[idx] for idx in hole_rows], axis=0)
        retrieved_ids = search_faiss(passages_index, emb_matrix, hole_fitting_top_k, faiss_search_batch_size)
        for result_idx, row_idx in enumerate(hole_rows):
            retrieved_by_row[row_idx] = retrieved_ids[result_idx].tolist()

    for idx, row in enumerate(rows):
        retrieved_ids = retrieved_by_row[idx]
        if retrieved_ids is None:
            continue

        query_key = scoped_key(row.get("query_id"), row.get("lang"), row.get("dataset"))
        passage_key = scoped_key(row.get("passage_id"), row.get("lang"), row.get("dataset"))
        current_fid = passage_key_to_faiss.get(passage_key)
        positives = query_positives.get(query_key, [])
        num_positives = len(positives)

        keep = False
        retrieval_rank = -1
        virtual_rank = -1

        if current_fid is not None and current_fid in retrieved_ids:
            retrieval_rank = retrieved_ids.index(current_fid) + 1
            if retrieval_rank <= top_k:
                keep = True
            elif num_positives > 1:
                all_positive_fids = {passage_key_to_faiss.get(positive_key) for positive_key in positives}
                all_positive_fids.discard(None)
                if all_positive_fids:
                    keep, virtual_rank = apply_hole_fitting(
                        retrieved_ids,
                        all_positive_fids,
                        current_fid,
                        top_k,
                    )

        consistency_flags[idx] = keep
        retrieval_ranks[idx] = retrieval_rank
        virtual_ranks[idx] = virtual_rank

    return consistency_flags, retrieval_ranks, virtual_ranks


def score_shard(
    dataset_path: Path,
    output_path: Path,
    passages_index,
    passage_key_to_faiss: dict[tuple[str, str, str], int],
    query_embeddings: dict[Any, np.ndarray],
    query_positives: dict[tuple[str, str, str], list[tuple[str, str, str]]],
    top_k: int,
    hole_fitting_top_k: int,
    large_chunk_size: int,
    small_chunk_size: int,
    faiss_search_batch_size: int,
    num_workers: int,
    compression: str,
) -> dict[str, Any]:
    if num_workers > 1 and hasattr(passages_index, "getDevice"):
        print("GPU index detected; forcing num_workers=1.")
        num_workers = 1

    parquet_file = pq.ParquetFile(dataset_path)
    total_rows = parquet_file.metadata.num_rows
    writer: pq.ParquetWriter | None = None
    consistent_count = 0
    processed_rows = 0

    try:
        with tqdm(total=total_rows, desc=f"Scoring {dataset_path.name}") as pbar:
            for batch in parquet_file.iter_batches(batch_size=large_chunk_size):
                table = pa.Table.from_batches([batch])
                small_tables = [
                    table.slice(start, min(small_chunk_size, table.num_rows - start))
                    for start in range(0, table.num_rows, small_chunk_size)
                ]
                row_chunks = [small_table.to_pylist() for small_table in small_tables]
                chunk_results: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None] * len(row_chunks)

                if num_workers == 1:
                    for idx, rows in enumerate(row_chunks):
                        chunk_results[idx] = process_rows(
                            rows,
                            passages_index,
                            passage_key_to_faiss,
                            query_embeddings,
                            query_positives,
                            top_k,
                            hole_fitting_top_k,
                            faiss_search_batch_size,
                        )
                else:
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        futures = {
                            executor.submit(
                                process_rows,
                                rows,
                                passages_index,
                                passage_key_to_faiss,
                                query_embeddings,
                                query_positives,
                                top_k,
                                hole_fitting_top_k,
                                faiss_search_batch_size,
                            ): idx
                            for idx, rows in enumerate(row_chunks)
                        }
                        for future in as_completed(futures):
                            chunk_results[futures[future]] = future.result()

                consistency = np.concatenate([result[0] for result in chunk_results if result is not None])
                retrieval = np.concatenate([result[1] for result in chunk_results if result is not None])
                virtual = np.concatenate([result[2] for result in chunk_results if result is not None])
                scored_table = table.append_column("consistency", pa.array(consistency))
                scored_table = scored_table.append_column("retrieval_rank", pa.array(retrieval, type=pa.int32()))
                scored_table = scored_table.append_column("virtual_rank", pa.array(virtual, type=pa.int32()))

                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(str(output_path), scored_table.schema, compression=compression)
                writer.write_table(scored_table)

                consistent_count += int(consistency.sum())
                processed_rows += table.num_rows
                pbar.update(table.num_rows)
                del table, scored_table, small_tables, row_chunks, chunk_results
                gc.collect()
    finally:
        if writer is not None:
            writer.close()

    return {
        "input_path": str(dataset_path),
        "output_path": str(output_path),
        "row_count": processed_rows,
        "consistent_count": consistent_count,
        "consistent_ratio": consistent_count / processed_rows if processed_rows else 0.0,
    }


def restore_dataset_outputs(
    scored_paths: list[Path],
    output_dir: Path,
    batch_size: int,
    compression: str,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    output_counts: dict[str, int] = defaultdict(int)

    try:
        for scored_path in scored_paths:
            parquet_file = pq.ParquetFile(scored_path)
            batches = parquet_file.iter_batches(batch_size=batch_size)
            for batch in tqdm(batches, desc=f"Restoring {scored_path.name}"):
                table = pa.Table.from_batches([batch])
                datasets = sorted(set(normalize(item) for item in table.column("dataset").to_pylist()))
                for dataset in datasets:
                    mask = pc.equal(table["dataset"], dataset)
                    dataset_table = table.filter(mask)
                    if dataset_table.num_rows == 0:
                        continue
                    if dataset not in writers:
                        output_path = output_dir / f"{safe_dataset_name(dataset)}.parquet"
                        writers[dataset] = pq.ParquetWriter(
                            str(output_path),
                            dataset_table.schema,
                            compression=compression,
                        )
                    writers[dataset].write_table(dataset_table)
                    output_counts[dataset] += dataset_table.num_rows
                del table
                gc.collect()
    finally:
        for writer in writers.values():
            writer.close()

    return dict(output_counts)


def validate_scored_shards(shard_stats: list[dict[str, Any]]) -> None:
    for stats in shard_stats:
        input_rows = pq.ParquetFile(stats["input_path"]).metadata.num_rows
        output_rows = pq.ParquetFile(stats["output_path"]).metadata.num_rows
        if input_rows != output_rows:
            raise AssertionError(
                f"Scored row count mismatch for {stats['input_path']}: {input_rows} input vs {output_rows} output"
            )
        if output_rows != stats["row_count"]:
            raise AssertionError(
                f"Manifest row count mismatch for {stats['output_path']}: {stats['row_count']} vs {output_rows}"
            )


def validate_dataset_outputs(scored_paths: list[Path], dataset_counts: dict[str, int]) -> None:
    scored_total = sum(pq.ParquetFile(path).metadata.num_rows for path in scored_paths)
    restored_total = sum(dataset_counts.values())
    if scored_total != restored_total:
        raise AssertionError(f"Restored {restored_total} rows, expected {scored_total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score consistency-filtering shards and restore dataset-wise outputs.")
    parser.add_argument("--shards-root", required=True, type=Path)
    parser.add_argument("--final-output-dir", required=True, type=Path)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--top-k", "--k", dest="k", type=int, default=None)
    parser.add_argument(
        "--hole-fitting-top-k",
        "--k-max",
        dest="k_max",
        type=int,
        default=None,
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--large-chunk-size", type=int, default=2_000)
    parser.add_argument("--small-chunk-size", type=int, default=1_000)
    parser.add_argument(
        "--faiss-search-batch-size",
        type=int,
        default=16,
        help="Maximum number of query vectors per FAISS search call. Use 0 to disable micro-batching.",
    )
    parser.add_argument("--positives-batch-size", type=int, default=100_000)
    parser.add_argument("--restore-batch-size", type=int, default=100_000)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument(
        "--skip-restore",
        action="store_true",
        help="Only write scored shard parquets; do not write dataset-wise outputs.",
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
    if args.k_max is None:
        args.k_max = 50

    return args


def main() -> None:
    args = parse_args()
    if args.use_gpu and args.num_workers > 1:
        print("GPU mode is not safe with multiple threads; forcing --num-workers=1.")
        args.num_workers = 1

    shards_root = args.shards_root.resolve()
    shard_ids = selected_shards(shards_root, args.shard, args.num_shards)
    scored_paths: list[Path] = []
    shard_stats: list[dict[str, Any]] = []

    for shard_id in shard_ids:
        base = group_dir(shards_root, shard_id)
        dataset_path = shard_path(shards_root, shard_id)
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)

        print(f"Filtering shard {shard_id}: {dataset_path}")
        passages_index, gpu_resources = load_passage_index(base, args.use_gpu, args.gpu_id)
        try:
            query_embeddings = load_query_embeddings(base)
            passage_key_to_faiss = load_passage_key_to_faiss(base)
            query_positives = build_query_positives(dataset_path, args.positives_batch_size)
            scored_path = base / f"filtered_data_top_{args.k}.parquet"
            shard_stats.append(
                score_shard(
                    dataset_path=dataset_path,
                    output_path=scored_path,
                    passages_index=passages_index,
                    passage_key_to_faiss=passage_key_to_faiss,
                    query_embeddings=query_embeddings,
                    query_positives=query_positives,
                    top_k=args.k,
                    hole_fitting_top_k=args.k_max,
                    large_chunk_size=args.large_chunk_size,
                    small_chunk_size=args.small_chunk_size,
                    faiss_search_batch_size=args.faiss_search_batch_size,
                    num_workers=args.num_workers,
                    compression=args.compression,
                )
            )
            scored_paths.append(scored_path)
        finally:
            if gpu_resources is not None:
                del gpu_resources
            del passages_index
            gc.collect()

    dataset_counts: dict[str, int] = {}
    if not args.skip_restore:
        dataset_counts = restore_dataset_outputs(
            scored_paths=scored_paths,
            output_dir=args.final_output_dir.resolve(),
            batch_size=args.restore_batch_size,
            compression=args.compression,
        )
        validate_dataset_outputs(scored_paths, dataset_counts)
    validate_scored_shards(shard_stats)

    manifest = {
        "top_k": args.k,
        "hole_fitting_top_k": args.k_max,
        "faiss_search_batch_size": args.faiss_search_batch_size,
        "shards": shard_stats,
        "dataset_output_counts": dataset_counts,
        "final_output_dir": str(args.final_output_dir.resolve()),
    }
    with (shards_root / f"filter_manifest_top_{args.k}.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
