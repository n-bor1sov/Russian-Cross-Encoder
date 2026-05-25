from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from tqdm.auto import tqdm


EmbeddingKind = Literal["query", "passage"]


def normalize(value: Any) -> str:
    return "" if value is None else str(value)


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


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def parse_embedding_key(key: Any) -> tuple[str, str, str]:
    """Return (id, lang, dataset) from current or legacy key layouts."""
    if not isinstance(key, tuple):
        raise ValueError(f"Expected tuple embedding key, got {type(key).__name__}: {key!r}")
    if len(key) == 3:
        item_id, lang, dataset = key
        return normalize(item_id), normalize(lang), normalize(dataset)
    if len(key) == 4:
        item_id, _text, dataset, lang = key
        return normalize(item_id), normalize(lang), normalize(dataset)
    raise ValueError(f"Unsupported embedding key shape: {key!r}")


def embedding_to_blob(embedding: Any) -> tuple[bytes, str, int]:
    array = np.asarray(embedding, dtype=np.float32)
    if array.ndim != 1:
        array = array.reshape(-1)
    contiguous = np.ascontiguousarray(array)
    return contiguous.tobytes(), str(contiguous.dtype), int(contiguous.shape[0])


def blobs_equal(left: bytes, right: bytes, rtol: float, atol: float) -> bool:
    left_arr = np.frombuffer(left, dtype=np.float32)
    right_arr = np.frombuffer(right, dtype=np.float32)
    if left_arr.shape != right_arr.shape:
        return False
    return bool(np.allclose(left_arr, right_arr, rtol=rtol, atol=atol))


class ConflictLogger:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "conflicts.jsonl"
        self.handle = self.path.open("a", encoding="utf-8")
        self.count = 0

    def write(
        self,
        *,
        dataset: str,
        kind: EmbeddingKind,
        item_id: str,
        lang: str,
        source_shard: int,
        existing_source_shard: int,
        reason: str,
    ) -> None:
        self.count += 1
        self.handle.write(
            json.dumps(
                {
                    "dataset": dataset,
                    "kind": kind,
                    "id": item_id,
                    "lang": lang,
                    "source_shard": source_shard,
                    "existing_source_shard": existing_source_shard,
                    "reason": reason,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    def close(self) -> None:
        self.handle.close()


class SqliteEmbeddingWriter:
    def __init__(
        self,
        output_dir: Path,
        conflict_logger: ConflictLogger,
        duplicate_rtol: float,
        duplicate_atol: float,
    ) -> None:
        self.output_dir = output_dir
        self.conflict_logger = conflict_logger
        self.duplicate_rtol = duplicate_rtol
        self.duplicate_atol = duplicate_atol
        self.connections: dict[tuple[str, EmbeddingKind], sqlite3.Connection] = {}
        self.stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: {
                "query": {"written": 0, "duplicates": 0, "conflicts": 0},
                "passage": {"written": 0, "duplicates": 0, "conflicts": 0},
            }
        )

    def _path(self, dataset: str, kind: EmbeddingKind) -> Path:
        dataset_dir = self.output_dir / safe_dataset_name(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir / f"{kind}_embeddings.sqlite"

    def _connection(self, dataset: str, kind: EmbeddingKind) -> sqlite3.Connection:
        key = (dataset, kind)
        if key in self.connections:
            return self.connections[key]

        conn = sqlite3.connect(str(self._path(dataset, kind)))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT NOT NULL,
                lang TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dtype TEXT NOT NULL,
                dim INTEGER NOT NULL,
                source_shard INTEGER NOT NULL,
                PRIMARY KEY (id, lang)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_lang ON embeddings(lang)")
        self.connections[key] = conn
        return conn

    def add(
        self,
        *,
        dataset: str,
        kind: EmbeddingKind,
        item_id: str,
        lang: str,
        embedding: Any,
        source_shard: int,
    ) -> None:
        blob, dtype, dim = embedding_to_blob(embedding)
        conn = self._connection(dataset, kind)
        row = conn.execute(
            "SELECT embedding, source_shard FROM embeddings WHERE id = ? AND lang = ?",
            (item_id, lang),
        ).fetchone()

        if row is None:
            conn.execute(
                """
                INSERT INTO embeddings(id, lang, embedding, dtype, dim, source_shard)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (item_id, lang, sqlite3.Binary(blob), dtype, dim, source_shard),
            )
            self.stats[dataset][kind]["written"] += 1
            return

        existing_blob, existing_source_shard = row
        if blobs_equal(existing_blob, blob, self.duplicate_rtol, self.duplicate_atol):
            self.stats[dataset][kind]["duplicates"] += 1
            return

        self.stats[dataset][kind]["conflicts"] += 1
        self.conflict_logger.write(
            dataset=dataset,
            kind=kind,
            item_id=item_id,
            lang=lang,
            source_shard=source_shard,
            existing_source_shard=int(existing_source_shard),
            reason="same dataset/id/lang has different embedding",
        )

    def commit(self) -> None:
        for conn in self.connections.values():
            conn.commit()

    def close(self) -> None:
        for conn in self.connections.values():
            conn.commit()
            conn.close()


class PklEmbeddingWriter:
    def __init__(
        self,
        output_dir: Path,
        conflict_logger: ConflictLogger,
        duplicate_rtol: float,
        duplicate_atol: float,
    ) -> None:
        self.output_dir = output_dir
        self.conflict_logger = conflict_logger
        self.duplicate_rtol = duplicate_rtol
        self.duplicate_atol = duplicate_atol
        self.data: dict[str, dict[EmbeddingKind, dict[tuple[str, str], np.ndarray]]] = defaultdict(
            lambda: {"query": {}, "passage": {}}
        )
        self.source_shards: dict[tuple[str, EmbeddingKind, str, str], int] = {}
        self.stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: {
                "query": {"written": 0, "duplicates": 0, "conflicts": 0},
                "passage": {"written": 0, "duplicates": 0, "conflicts": 0},
            }
        )

    def add(
        self,
        *,
        dataset: str,
        kind: EmbeddingKind,
        item_id: str,
        lang: str,
        embedding: Any,
        source_shard: int,
    ) -> None:
        key = (item_id, lang)
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim != 1:
            array = array.reshape(-1)

        existing = self.data[dataset][kind].get(key)
        if existing is None:
            self.data[dataset][kind][key] = array
            self.source_shards[(dataset, kind, item_id, lang)] = source_shard
            self.stats[dataset][kind]["written"] += 1
            return

        if np.allclose(existing, array, rtol=self.duplicate_rtol, atol=self.duplicate_atol):
            self.stats[dataset][kind]["duplicates"] += 1
            return

        self.stats[dataset][kind]["conflicts"] += 1
        self.conflict_logger.write(
            dataset=dataset,
            kind=kind,
            item_id=item_id,
            lang=lang,
            source_shard=source_shard,
            existing_source_shard=self.source_shards[(dataset, kind, item_id, lang)],
            reason="same dataset/id/lang has different embedding",
        )

    def commit(self) -> None:
        return

    def close(self) -> None:
        for dataset, by_kind in self.data.items():
            dataset_dir = self.output_dir / safe_dataset_name(dataset)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            for kind, embeddings in by_kind.items():
                with (dataset_dir / f"{kind}_embeddings.pkl").open("wb") as handle:
                    pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


def add_embedding_map(
    *,
    writer: SqliteEmbeddingWriter | PklEmbeddingWriter,
    embedding_map: dict[Any, Any],
    kind: EmbeddingKind,
    source_shard: int,
) -> int:
    processed = 0
    for raw_key, embedding in tqdm(
        embedding_map.items(),
        desc=f"Restoring {kind} embeddings from shard {source_shard}",
        leave=False,
    ):
        item_id, lang, dataset = parse_embedding_key(raw_key)
        writer.add(
            dataset=dataset,
            kind=kind,
            item_id=item_id,
            lang=lang,
            embedding=embedding,
            source_shard=source_shard,
        )
        processed += 1
    return processed


def write_manifests(
    output_dir: Path,
    writer: SqliteEmbeddingWriter | PklEmbeddingWriter,
    *,
    output_format: str,
    source_shards: list[int],
    processed: dict[str, int],
    conflict_count: int,
) -> None:
    run_manifest = {
        "format": output_format,
        "source_shards": source_shards,
        "processed": processed,
        "conflicts": conflict_count,
        "datasets": writer.stats,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, ensure_ascii=False, indent=2)

    for dataset, by_kind in writer.stats.items():
        dataset_dir = output_dir / safe_dataset_name(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with (dataset_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset": dataset,
                    "format": output_format,
                    "stats": by_kind,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore shard-level query/passage embedding PKLs into dataset-level stores."
    )
    parser.add_argument("--shards-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--format", choices=["sqlite", "pkl"], default="sqlite")
    parser.add_argument("--duplicate-rtol", type=float, default=1e-5)
    parser.add_argument("--duplicate-atol", type=float, default=1e-8)
    parser.add_argument("--commit-every-shards", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output-dir before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards_root = args.shards_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.commit_every_shards <= 0:
        raise ValueError("--commit-every-shards must be positive")
    if output_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_ids = selected_shards(shards_root, args.shard, args.num_shards)
    conflict_logger = ConflictLogger(output_dir)
    writer: SqliteEmbeddingWriter | PklEmbeddingWriter
    if args.format == "sqlite":
        writer = SqliteEmbeddingWriter(output_dir, conflict_logger, args.duplicate_rtol, args.duplicate_atol)
    else:
        writer = PklEmbeddingWriter(output_dir, conflict_logger, args.duplicate_rtol, args.duplicate_atol)

    processed = {"query": 0, "passage": 0}
    try:
        for idx, shard_id in enumerate(shard_ids, start=1):
            base = group_dir(shards_root, shard_id)
            query_path = base / "query_embedding.pkl"
            passage_path = base / "passage_embedding.pkl"
            if not query_path.exists():
                raise FileNotFoundError(query_path)
            if not passage_path.exists():
                raise FileNotFoundError(passage_path)

            query_embeddings = load_pickle(query_path)
            processed["query"] += add_embedding_map(
                writer=writer,
                embedding_map=query_embeddings,
                kind="query",
                source_shard=shard_id,
            )
            del query_embeddings
            gc.collect()

            passage_embeddings = load_pickle(passage_path)
            processed["passage"] += add_embedding_map(
                writer=writer,
                embedding_map=passage_embeddings,
                kind="passage",
                source_shard=shard_id,
            )
            del passage_embeddings
            gc.collect()

            if idx % args.commit_every_shards == 0:
                writer.commit()

        writer.commit()
    finally:
        writer.close()
        conflict_logger.close()

    write_manifests(
        output_dir,
        writer,
        output_format=args.format,
        source_shards=shard_ids,
        processed=processed,
        conflict_count=conflict_logger.count,
    )
    print(f"Done. Restored embeddings to {output_dir}")
    print(f"Processed query={processed['query']:,}, passage={processed['passage']:,}")
    print(f"Conflicts: {conflict_logger.count:,}")


if __name__ == "__main__":
    main()
