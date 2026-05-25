from __future__ import annotations

import os
import argparse
import gc
import json
import logging
import math
import random
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

IDENTITY_COLUMNS = ("query_id", "passage_id", "lang", "dataset")

# pa.string() contiguous utf8 buffer: cast fails at exact 2**31-1; truncate and splits stay below with slack.
_ARROW_STRING_CAST_SLACK_CODEUNITS = 262_144
ARROW_STRING_UTF8_MAX_CODEUNITS = (2**31) - 1 - _ARROW_STRING_CAST_SLACK_CODEUNITS


def normalize(value: Any) -> str:
    return "" if value is None else str(value)


def scoped_key(value: Any, lang: Any, dataset: Any) -> tuple[str, str, str]:
    return normalize(value), normalize(lang), normalize(dataset)


def discover_parquets(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    return files


def ensure_columns(schema: pa.Schema) -> None:
    missing = [name for name in IDENTITY_COLUMNS if name not in schema.names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _as_large_utf8(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """Приводит колонку к large_string (числа/другие типы → текст, словарь раскрывается)."""
    t = col.type
    if pa.types.is_large_string(t):
        return col
    if pa.types.is_string(t):
        return pc.cast(col, pa.large_string())
    return pc.cast(col, pa.large_string())


def _truncate_utf8_to_fit_arrow_string(col: pa.ChunkedArray, max_codeunits: int) -> pa.ChunkedArray:
    """
    Усекает каждое utf8-значение по длине в байтах UTF-8 (code units).
    Это защищает финальный cast в pa.string() от переполнения лимита Arrow string.
    """
    large = _as_large_utf8(col)
    lengths_bytes = pc.binary_length(large)
    need = pc.greater(lengths_bytes, pa.scalar(max_codeunits, type=pa.int64()))
    sliced = pc.utf8_slice_codeunits(large, start=0, stop=max_codeunits)
    return pc.if_else(need, sliced, large)


def _trimmed_large_string_table(table: pa.Table, column_names: list[str]) -> pa.Table:
    selected = table.select(column_names)
    cols = [
        _truncate_utf8_to_fit_arrow_string(
            selected.column(name),
            ARROW_STRING_UTF8_MAX_CODEUNITS,
        ).combine_chunks()
        for name in column_names
    ]
    return pa.Table.from_arrays(cols, names=column_names)


def cast_table_columns_to_string(table: pa.Table, column_names: list[str]) -> pa.Table:
    """column order as column_names; all columns pa.string(), cell truncation + recursive row split if cast fails."""
    trimmed = _trimmed_large_string_table(table, column_names)
    n = trimmed.num_rows
    if n == 0:
        return pa.Table.from_arrays(
            [pa.array([], type=pa.string()) for _ in column_names],
            names=column_names,
        )

    string_chunks: list[list[pa.Array]] = [[] for _ in column_names]

    def append_cast_range(row_start: int, row_end: int) -> None:
        """Row slice [row_start, row_end) of trimmed; on ArrowInvalid split half (same bounds for all columns)."""
        if row_start >= row_end:
            return
        sub = trimmed.slice(row_start, row_end - row_start)
        try:
            # Atomic: only append after every column casts; partial appends break row alignment.
            pieces: list[pa.Array] = [
                pc.cast(sub.column(name).combine_chunks(), pa.string()) for name in column_names
            ]
        except ArrowInvalid:
            if row_end - row_start <= 1:
                raise RuntimeError(
                    "Одна строка не проходит cast в pa.string(); увеличьте _ARROW_STRING_CAST_SLACK_CODEUNITS "
                    "или пишите шарды в large_string."
                ) from None
            mid = row_start + (row_end - row_start) // 2
            append_cast_range(row_start, mid)
            append_cast_range(mid, row_end)
            return
        for i, arr in enumerate(pieces):
            string_chunks[i].append(arr)

    append_cast_range(0, n)
    out_cols = [pa.chunked_array(chunks, type=pa.string()) for chunks in string_chunks]
    return pa.Table.from_arrays(out_cols, names=column_names)


class SqliteComponentState:
    """SQLite-backed union-find state for query/passage connected components."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA cache_size=-200000")
        self._setup()
        row = self.conn.execute("SELECT COALESCE(MAX(id), -1) + 1 FROM nodes").fetchone()
        self.next_id = int(row[0])

    def _setup(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                parent INTEGER NOT NULL,
                size INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS keys (
                kind TEXT NOT NULL,
                value TEXT NOT NULL,
                lang TEXT NOT NULL,
                dataset TEXT NOT NULL,
                node_id INTEGER NOT NULL,
                PRIMARY KEY (kind, value, lang, dataset)
            );
            CREATE TABLE IF NOT EXISTS component_counts (
                root INTEGER NOT NULL,
                dataset TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                PRIMARY KEY (root, dataset)
            );
            CREATE TABLE IF NOT EXISTS component_assignments (
                root INTEGER PRIMARY KEY,
                dataset TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                shard_id INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_keys_node ON keys(node_id);
            CREATE INDEX IF NOT EXISTS idx_assignments_shard ON component_assignments(shard_id);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    def get_or_add(self, kind: str, key: tuple[str, str, str]) -> int:
        value, lang, dataset = key
        row = self.conn.execute(
            "SELECT node_id FROM keys WHERE kind = ? AND value = ? AND lang = ? AND dataset = ?",
            (kind, value, lang, dataset),
        ).fetchone()
        if row is not None:
            return int(row[0])

        node_id = self.next_id
        self.next_id += 1
        self.conn.execute(
            "INSERT INTO nodes(id, parent, size) VALUES (?, ?, ?)",
            (node_id, node_id, 1),
        )
        self.conn.execute(
            "INSERT INTO keys(kind, value, lang, dataset, node_id) VALUES (?, ?, ?, ?, ?)",
            (kind, value, lang, dataset, node_id),
        )
        return node_id

    def find(self, node_id: int) -> int:
        path: list[int] = []
        current = int(node_id)
        while True:
            row = self.conn.execute("SELECT parent FROM nodes WHERE id = ?", (current,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown node id {current}")
            parent = int(row[0])
            if parent == current:
                root = current
                break
            path.append(current)
            current = parent
        if path:
            self.conn.executemany(
                "UPDATE nodes SET parent = ? WHERE id = ?",
                [(root, item) for item in path],
            )
        return root

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root

        left_size = int(
            self.conn.execute("SELECT size FROM nodes WHERE id = ?", (left_root,)).fetchone()[0]
        )
        right_size = int(
            self.conn.execute("SELECT size FROM nodes WHERE id = ?", (right_root,)).fetchone()[0]
        )
        if left_size < right_size:
            left_root, right_root = right_root, left_root
            left_size, right_size = right_size, left_size

        self.conn.execute("UPDATE nodes SET parent = ? WHERE id = ?", (left_root, right_root))
        self.conn.execute("UPDATE nodes SET size = ? WHERE id = ?", (left_size + right_size, left_root))
        return left_root

    def get_existing(self, kind: str, key: tuple[str, str, str]) -> int:
        value, lang, dataset = key
        row = self.conn.execute(
            "SELECT node_id FROM keys WHERE kind = ? AND value = ? AND lang = ? AND dataset = ?",
            (kind, value, lang, dataset),
        ).fetchone()
        if row is None:
            raise KeyError((kind, value, lang, dataset))
        return int(row[0])

    def add_component_count(self, root: int, dataset: str, count: int) -> None:
        self.conn.execute(
            """
            INSERT INTO component_counts(root, dataset, row_count)
            VALUES (?, ?, ?)
            ON CONFLICT(root, dataset) DO UPDATE SET
                row_count = row_count + excluded.row_count
            """,
            (int(root), dataset, int(count)),
        )

    def iter_component_counts(self) -> Iterable[tuple[int, str, int]]:
        cursor = self.conn.execute(
            "SELECT root, dataset, row_count FROM component_counts ORDER BY dataset, root"
        )
        yield from ((int(root), dataset, int(row_count)) for root, dataset, row_count in cursor)

    def assign_component(self, root: int, dataset: str, row_count: int, shard_id: int) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO component_assignments(root, dataset, row_count, shard_id)
            VALUES (?, ?, ?, ?)
            """,
            (int(root), dataset, int(row_count), int(shard_id)),
        )

    def shard_for_root(self, root: int) -> int:
        row = self.conn.execute(
            "SELECT shard_id FROM component_assignments WHERE root = ?",
            (int(root),),
        ).fetchone()
        if row is None:
            raise KeyError(f"No shard assignment for component {root}")
        return int(row[0])

    def component_assignment_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM component_assignments").fetchone()[0])


def count_rows(files: list[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def iter_identity_batches(files: list[Path], batch_size: int):
    for path in files:
        parquet_file = pq.ParquetFile(path)
        ensure_columns(parquet_file.schema_arrow)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(IDENTITY_COLUMNS)):
            yield path, batch.to_pydict()


def build_components(files: list[Path], state: SqliteComponentState, batch_size: int, total_rows: int) -> None:
    logger.info(
        "Схлопывание связных компонент: файлов=%d, строк=%d, batch_size=%d",
        len(files),
        total_rows,
        batch_size,
    )
    with tqdm(total=total_rows, desc="Building connected components") as pbar:
        for _, batch in iter_identity_batches(files, batch_size):
            qids = batch["query_id"]
            pids = batch["passage_id"]
            langs = batch["lang"]
            datasets = batch["dataset"]
            for qid, pid, lang, dataset in zip(qids, pids, langs, datasets):
                query_node = state.get_or_add("query", scoped_key(qid, lang, dataset))
                passage_node = state.get_or_add("passage", scoped_key(pid, lang, dataset))
                state.union(query_node, passage_node)
            state.commit()
            pbar.update(len(qids))
    logger.info("Связные компоненты построены")


def count_components(
    files: list[Path],
    state: SqliteComponentState,
    batch_size: int,
    total_rows: int,
) -> dict[str, int]:
    logger.info(
        "Подсчёт строк по компонентам: файлов=%d, строк=%d",
        len(files),
        total_rows,
    )
    dataset_totals: dict[str, int] = defaultdict(int)
    with tqdm(total=total_rows, desc="Counting component rows") as pbar:
        for _, batch in iter_identity_batches(files, batch_size):
            partial: dict[tuple[int, str], int] = defaultdict(int)
            qids = batch["query_id"]
            langs = batch["lang"]
            datasets = batch["dataset"]
            for qid, lang, dataset in zip(qids, langs, datasets):
                dataset_name = normalize(dataset)
                node = state.get_existing("query", scoped_key(qid, lang, dataset))
                root = state.find(node)
                partial[(root, dataset_name)] += 1
                dataset_totals[dataset_name] += 1
            for (root, dataset_name), count in partial.items():
                state.add_component_count(root, dataset_name, count)
            state.commit()
            pbar.update(len(qids))
    out = dict(dataset_totals)
    logger.info("Итого по датасетам (строки): %s", dict(sorted(out.items())))
    return out


def balanced_assign_components(
    state: SqliteComponentState,
    total_rows: int,
    target_shard_size: int,
    dataset_totals: dict[str, int],
    seed: int,
) -> tuple[list[int], list[dict[str, int]]]:
    num_shards = max(1, math.ceil(total_rows / target_shard_size))
    shard_capacities = [target_shard_size] * num_shards
    shard_capacities[-1] = total_rows - target_shard_size * (num_shards - 1)
    if shard_capacities[-1] <= 0:
        shard_capacities[-1] = target_shard_size

    logger.info(
        "Назначение шардов: num_shards=%d target_shard_size=%d total_rows=%d датасетов=%d seed=%d",
        num_shards,
        target_shard_size,
        total_rows,
        len(dataset_totals),
        seed,
    )
    ratios = {dataset: count / total_rows for dataset, count in dataset_totals.items()}
    shard_dataset_counts: list[dict[str, int]] = [defaultdict(int) for _ in range(num_shards)]
    shard_counts = [0] * num_shards

    by_dataset: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for root, dataset, row_count in state.iter_component_counts():
        by_dataset[dataset].append((root, row_count))

    rng = random.Random(seed)
    for dataset, components in sorted(by_dataset.items()):
        rng.shuffle(components)
        components.sort(key=lambda item: item[1], reverse=True)
        for root, row_count in tqdm(components, desc=f"Assigning {dataset}", leave=False):
            best_shard = min(
                range(num_shards),
                key=lambda sid: (
                    max(0, shard_counts[sid] + row_count - shard_capacities[sid]),
                    abs(
                        shard_dataset_counts[sid][dataset]
                        + row_count
                        - shard_capacities[sid] * ratios[dataset]
                    ),
                    abs(shard_counts[sid] + row_count - shard_capacities[sid]),
                    shard_counts[sid],
                    sid,
                ),
            )
            state.assign_component(root, dataset, row_count, best_shard)
            shard_dataset_counts[best_shard][dataset] += row_count
            shard_counts[best_shard] += row_count
        state.commit()

    logger.info("Назначение завершено: строк по шардам=%s", shard_counts)
    return shard_counts, [dict(counts) for counts in shard_dataset_counts]


def build_shard_mask(
    batch: dict[str, list[Any]],
    state: SqliteComponentState,
    target_shard_id: int,
) -> np.ndarray:
    qids = batch["query_id"]
    langs = batch["lang"]
    datasets = batch["dataset"]
    mask = np.zeros(len(qids), dtype=bool)
    for idx, (qid, lang, dataset) in enumerate(zip(qids, langs, datasets)):
        node = state.get_existing("query", scoped_key(qid, lang, dataset))
        root = state.find(node)
        mask[idx] = state.shard_for_root(root) == target_shard_id
    return mask


def write_shards_one_at_a_time(
    files: list[Path],
    output_dir: Path,
    state: SqliteComponentState,
    num_shards: int,
    batch_size: int,
    compression: str,
    row_group_size: int,
) -> list[Path]:
    shard_paths: list[Path] = []
    canonical_columns = list(pq.read_schema(files[0]).names)
    output_schema = pa.schema([(name, pa.string()) for name in canonical_columns])
    logger.info(
        "Запись parquet: шардов=%d, входных файлов=%d, каталог=%s; колонки — utf8 string (при необходимости несколько chunk)",
        num_shards,
        len(files),
        output_dir,
    )

    for shard_id in range(num_shards):
        group_dir = output_dir / f"group_{shard_id}"
        group_dir.mkdir(parents=True, exist_ok=True)
        shard_path = group_dir / f"shard_{shard_id:02d}.parquet"
        shard_paths.append(shard_path)
        writer: pq.ParquetWriter | None = None
        rows_written = 0

        try:
            for path in tqdm(files, desc=f"Writing shard {shard_id:02d}"):
                parquet_file = pq.ParquetFile(path)
                for record_batch in parquet_file.iter_batches(batch_size=batch_size):
                    table = pa.Table.from_batches([record_batch])
                    table = cast_table_columns_to_string(table, canonical_columns)
                    identity_batch = table.select(list(IDENTITY_COLUMNS)).to_pydict()
                    mask = build_shard_mask(identity_batch, state, shard_id)
                    if not mask.any():
                        del table, mask
                        continue
                    shard_table = table.filter(pa.array(mask))
                    if writer is None:
                        writer = pq.ParquetWriter(
                            shard_path,
                            shard_table.schema,
                            compression=compression,
                            use_dictionary=True,
                        )
                    writer.write_table(shard_table, row_group_size=row_group_size)
                    rows_written += shard_table.num_rows
                    del table, shard_table, mask
                gc.collect()
        finally:
            if writer is not None:
                writer.close()

        if rows_written == 0:
            pq.write_table(
                pa.Table.from_batches([], schema=output_schema),
                shard_path,
                compression=compression,
            )
        logger.info("Шард %02d: записано строк=%d -> %s", shard_id, rows_written, shard_path)

    return shard_paths


def validate_no_split(
    files: list[Path],
    state: SqliteComponentState,
    batch_size: int,
    total_rows: int,
) -> None:
    logger.info("Проверка: scoped id не разрезаны между шардами (%d строк)", total_rows)
    seen: dict[tuple[str, str, str, str], int] = {}
    with tqdm(total=total_rows, desc="Validating scoped ID placement") as pbar:
        for _, batch in iter_identity_batches(files, batch_size):
            for kind, values in (("query", batch["query_id"]), ("passage", batch["passage_id"])):
                for value, lang, dataset in zip(values, batch["lang"], batch["dataset"]):
                    key = (kind, *scoped_key(value, lang, dataset))
                    node = state.get_existing(kind, key[1:])
                    shard_id = state.shard_for_root(state.find(node))
                    previous = seen.get(key)
                    if previous is not None and previous != shard_id:
                        raise AssertionError(f"Scoped {kind} id {key[1:]} is split across shards")
                    seen[key] = shard_id
            pbar.update(len(batch["query_id"]))
    logger.info("Проверка пройдена: разрезов scoped id нет")


def write_manifest(
    output_dir: Path,
    files: list[Path],
    total_rows: int,
    target_shard_size: int,
    batch_size: int,
    seed: int,
    shard_counts: list[int],
    shard_dataset_counts: list[dict[str, int]],
    dataset_totals: dict[str, int],
    state_path: Path,
    shard_paths: list[Path],
) -> None:
    manifest = {
        "input_files": [str(path) for path in files],
        "total_rows": total_rows,
        "target_shard_size": target_shard_size,
        "batch_size": batch_size,
        "seed": seed,
        "identity_scope": ["id", "lang", "dataset"],
        "state_path": str(state_path),
        "dataset_totals": dataset_totals,
        "dataset_ratios": {
            dataset: count / total_rows for dataset, count in sorted(dataset_totals.items())
        },
        "shards": [
            {
                "shard_id": shard_id,
                "path": str(shard_paths[shard_id]),
                "row_count": shard_counts[shard_id],
                "dataset_counts": shard_dataset_counts[shard_id],
                "dataset_ratios": {
                    dataset: count / shard_counts[shard_id]
                    for dataset, count in sorted(shard_dataset_counts[shard_id].items())
                    if shard_counts[shard_id] > 0
                },
            }
            for shard_id in range(len(shard_counts))
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge parquet files into ratio-preserving connected-component shards."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-shard-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--row-group-size", type=int, default=100_000)
    parser.add_argument(
        "--state-db",
        type=Path,
        default=None,
        help="SQLite state path. Defaults to <output-dir>/component_state.sqlite.",
    )
    parser.add_argument(
        "--keep-state",
        action="store_true",
        help="Keep the SQLite state database after a successful run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-dir before writing.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the final scoped-ID no-split validation pass.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень логирования (по умолчанию INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if input_dir == output_dir:
        raise ValueError("--input-dir and --output-dir must be different")
    if args.target_shard_size <= 0:
        raise ValueError("--target-shard-size must be positive")

    logger.info("Вход: %s", input_dir)
    logger.info("Выход: %s", output_dir)

    if output_dir.exists() and args.overwrite:
        logger.info("Удаляю существующий output-dir (--overwrite)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_parquets(input_dir)
    total_rows = count_rows(files)
    logger.info("Найдено %d parquet, всего строк=%d", len(files), total_rows)
    if total_rows == 0:
        raise ValueError("Input parquets contain no rows")

    state_path = (args.state_db or output_dir / "component_state.sqlite").resolve()
    if state_path.exists():
        state_path.unlink()

    logger.info("SQLite состояние: %s", state_path)
    state = SqliteComponentState(state_path)
    shard_paths: list[Path] = []
    try:
        build_components(files, state, args.batch_size, total_rows)
        dataset_totals = count_components(files, state, args.batch_size, total_rows)
        shard_counts, shard_dataset_counts = balanced_assign_components(
            state=state,
            total_rows=total_rows,
            target_shard_size=args.target_shard_size,
            dataset_totals=dataset_totals,
            seed=args.seed,
        )
        if state.component_assignment_count() == 0:
            raise RuntimeError("No component assignments were created")
        if not args.skip_validation:
            validate_no_split(files, state, args.batch_size, total_rows)
        else:
            logger.warning("Пропуск финальной валидации (--skip-validation)")
        shard_paths = write_shards_one_at_a_time(
            files=files,
            output_dir=output_dir,
            state=state,
            num_shards=len(shard_counts),
            batch_size=args.batch_size,
            compression=args.compression,
            row_group_size=args.row_group_size,
        )
        write_manifest(
            output_dir=output_dir,
            files=files,
            total_rows=total_rows,
            target_shard_size=args.target_shard_size,
            batch_size=args.batch_size,
            seed=args.seed,
            shard_counts=shard_counts,
            shard_dataset_counts=shard_dataset_counts,
            dataset_totals=dataset_totals,
            state_path=state_path,
            shard_paths=shard_paths,
        )
        logger.info("Манифест: %s", output_dir / "manifest.json")
    finally:
        state.close()
        if not args.keep_state and state_path.exists():
            state_path.unlink()
            for wal_path in state_path.parent.glob(f"{state_path.name}-*"):
                wal_path.unlink(missing_ok=True)
            logger.info("Временный SQLite удалён (используйте --keep-state, чтобы оставить)")
        elif args.keep_state:
            logger.info("SQLite состояние сохранено: %s", state_path)

    logger.info("Готово: записано шардов=%d в %s", len(shard_paths), output_dir)


if __name__ == "__main__":
    main()


# Ex: nohup python 01_merge_to_shards.py --input-dir /path/to/cleaned_datasets/splits/train --output-dir /path/to/consistency_filtering_data > shardsoutput.log 2>&1 &