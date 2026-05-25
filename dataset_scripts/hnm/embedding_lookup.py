from __future__ import annotations

import argparse
import pickle
import re
import sqlite3
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


EmbeddingKind = Literal["query", "passage"]


def safe_dataset_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return clean or "unknown_dataset"


class DatasetEmbeddingLookup:
    """
    Lightweight lookup hook for embeddings restored by 04_restore_embeddings_by_dataset.py.

    SQLite mode keeps only one small DB connection per requested dataset/kind open.
    PKL mode is supported for small stores, but loads the selected dataset/kind file into RAM.
    """

    def __init__(self, embeddings_root: str | Path, storage_format: Literal["sqlite", "pkl"] = "sqlite") -> None:
        self.embeddings_root = Path(embeddings_root)
        self.storage_format = storage_format
        self._sqlite_connections: dict[tuple[str, EmbeddingKind], sqlite3.Connection] = {}
        self._pkl_cache: dict[tuple[str, EmbeddingKind], dict[tuple[str, str], np.ndarray]] = {}

    def close(self) -> None:
        for conn in self._sqlite_connections.values():
            conn.close()
        self._sqlite_connections.clear()
        self._pkl_cache.clear()

    def __enter__(self) -> "DatasetEmbeddingLookup":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_query(self, dataset: str, query_id: str, lang: str = "") -> np.ndarray | None:
        return self.get("query", dataset=dataset, item_id=query_id, lang=lang)

    def get_passage(self, dataset: str, passage_id: str, lang: str = "") -> np.ndarray | None:
        return self.get("passage", dataset=dataset, item_id=passage_id, lang=lang)

    def get(self, kind: EmbeddingKind, dataset: str, item_id: str, lang: str = "") -> np.ndarray | None:
        if self.storage_format == "sqlite":
            return self._get_sqlite(kind, dataset, item_id, lang)
        if self.storage_format == "pkl":
            return self._get_pkl(kind, dataset, item_id, lang)
        raise ValueError(f"Unsupported storage format: {self.storage_format}")

    def batch_get(
        self,
        kind: EmbeddingKind,
        dataset: str,
        item_ids: Iterable[str],
        lang: str = "",
    ) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        if self.storage_format == "sqlite":
            conn = self._sqlite_connection(dataset, kind)
            for item_id in item_ids:
                row = conn.execute(
                    "SELECT embedding FROM embeddings WHERE id = ? AND lang = ?",
                    (str(item_id), str(lang)),
                ).fetchone()
                if row is not None:
                    found[str(item_id)] = np.frombuffer(row[0], dtype=np.float32).copy()
            return found

        store = self._pkl_store(dataset, kind)
        for item_id in item_ids:
            embedding = store.get((str(item_id), str(lang)))
            if embedding is not None:
                found[str(item_id)] = embedding
        return found

    def _dataset_dir(self, dataset: str) -> Path:
        return self.embeddings_root / safe_dataset_name(dataset)

    def _sqlite_connection(self, dataset: str, kind: EmbeddingKind) -> sqlite3.Connection:
        key = (dataset, kind)
        conn = self._sqlite_connections.get(key)
        if conn is not None:
            return conn

        path = self._dataset_dir(dataset) / f"{kind}_embeddings.sqlite"
        if not path.exists():
            raise FileNotFoundError(path)
        conn = sqlite3.connect(str(path))
        self._sqlite_connections[key] = conn
        return conn

    def _get_sqlite(self, kind: EmbeddingKind, dataset: str, item_id: str, lang: str) -> np.ndarray | None:
        conn = self._sqlite_connection(dataset, kind)
        row = conn.execute(
            "SELECT embedding FROM embeddings WHERE id = ? AND lang = ?",
            (str(item_id), str(lang)),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def _pkl_store(self, dataset: str, kind: EmbeddingKind) -> dict[tuple[str, str], np.ndarray]:
        key = (dataset, kind)
        store = self._pkl_cache.get(key)
        if store is not None:
            return store

        path = self._dataset_dir(dataset) / f"{kind}_embeddings.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            store = pickle.load(handle)
        self._pkl_cache[key] = store
        return store

    def _get_pkl(self, kind: EmbeddingKind, dataset: str, item_id: str, lang: str) -> np.ndarray | None:
        store = self._pkl_store(dataset, kind)
        embedding = store.get((str(item_id), str(lang)))
        if embedding is None:
            return None
        return np.asarray(embedding, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lookup one restored query/passage embedding.")
    parser.add_argument("--embeddings-root", required=True, type=Path)
    parser.add_argument("--format", choices=["sqlite", "pkl"], default="sqlite")
    parser.add_argument("--kind", choices=["query", "passage"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--lang", default="")
    parser.add_argument("--print-vector", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with DatasetEmbeddingLookup(args.embeddings_root, args.format) as lookup:
        embedding = lookup.get(args.kind, dataset=args.dataset, item_id=args.id, lang=args.lang)
    if embedding is None:
        raise SystemExit("Embedding not found")

    print(f"Found {args.kind} embedding: dim={embedding.shape[0]}, dtype={embedding.dtype}")
    if args.print_vector:
        print(embedding.tolist())


if __name__ == "__main__":
    main()
