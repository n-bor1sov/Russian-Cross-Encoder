from __future__ import annotations  # noqa: I001

import os

import torch  # noqa: F401
import argparse
import gc
import json
import logging
import os
import pickle
import random
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import openai
import pyarrow.parquet as pq
from httpx import Client
from tqdm.auto import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

_EMBED_RETRY_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _embedding_request_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.InternalServerError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return getattr(exc, "status_code", None) in _EMBED_RETRY_HTTP_STATUS
    return False


def normalize(value: Any) -> str:
    return "" if value is None else str(value)


def scoped_key(value: Any, lang: Any, dataset: Any) -> tuple[str, str, str]:
    return normalize(value), normalize(lang), normalize(dataset)


def create_http_client(timeout: int) -> Client:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return Client(verify=ssl_context, timeout=timeout)


def resolve_tokenizer_source(model_path: str | Path) -> tuple[str, bool]:
    model_path_str = str(model_path)
    local_candidate = Path(model_path_str).expanduser()
    path_separators = tuple(sep for sep in (os.sep, os.altsep) if sep)
    is_local_path = (
        local_candidate.is_absolute()
        or model_path_str.startswith(".")
        or any(sep in model_path_str for sep in path_separators)
    )

    if not is_local_path:
        return model_path_str, False

    if not local_candidate.exists():
        raise FileNotFoundError(
            f"Tokenizer model path does not exist: {local_candidate}. "
            "Check that --model-path points to a mounted directory with tokenizer files."
        )
    if not local_candidate.is_dir():
        raise NotADirectoryError(f"Tokenizer model path is not a directory: {local_candidate}")

    return str(local_candidate.resolve()), True


class BasicEmbedder:
    def __init__(
        self,
        model_path: str | Path,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int,
        n_parallel: int,
        max_retries: int,
        retry_backoff_base: float,
    ) -> None:
        self.model_name = model_name
        self.n_parallel = max(1, n_parallel)
        self.max_retries = max(0, max_retries)
        self.retry_backoff_base = max(0.0, retry_backoff_base)
        self.http_client = create_http_client(timeout)
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=self.http_client,
        )
        tokenizer_source, local_files_only = resolve_tokenizer_source(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            padding_side="left",
            local_files_only=local_files_only,
        )

    def _truncate(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer(
            text,
            return_tensors=None,
            padding=False,
            truncation=False,
        )["input_ids"]
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

    def _retry_sleep(self, backoff_attempt: int) -> float:
        base = self.retry_backoff_base * (2**backoff_attempt)
        delay = min(120.0, base * random.uniform(0.8, 1.2))
        time.sleep(delay)
        return delay

    def _embed_batch(self, texts: list[str], instruction: str, max_tokens: int) -> list[list[float]]:
        if not texts:
            return []
        return self._embed_chunk_with_retry_shrink(texts, instruction, max_tokens)

    def _embed_chunk_with_retry_shrink(
        self,
        texts: list[str],
        instruction: str,
        max_tokens: int,
        *,
        backoff_attempt: int = 0,
    ) -> list[list[float]]:
        batch_texts = [self._truncate(text, max_tokens) for text in texts]
        inputs = [instruction.format(text=text) for text in batch_texts]
        try:
            response = self.client.embeddings.create(model=self.model_name, input=inputs)
            embeddings = [item.embedding for item in response.data]
        except Exception as exc:
            if not _embedding_request_retryable(exc):
                raise
            if len(texts) == 1:
                logger.warning(
                    "Embedding batch failed for a single segment (attempt %s/%s): %s",
                    backoff_attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if backoff_attempt >= self.max_retries:
                    raise
                delay = self._retry_sleep(backoff_attempt)
                logger.warning(
                    "Retry embedding single segment after %.2fs (attempt %s/%s)",
                    delay,
                    backoff_attempt + 2,
                    self.max_retries + 1,
                )
                return self._embed_chunk_with_retry_shrink(
                    texts, instruction, max_tokens, backoff_attempt=backoff_attempt + 1
                )

            delay = self._retry_sleep(backoff_attempt)
            left_n = len(texts) // 2
            logger.warning(
                "Embedding batch failed for size %s, retry in %.2fs then split into %s + %s: %s",
                len(texts),
                delay,
                left_n,
                len(texts) - left_n,
                exc,
            )
            left = self._embed_chunk_with_retry_shrink(
                texts[:left_n], instruction, max_tokens, backoff_attempt=backoff_attempt + 1
            )
            right = self._embed_chunk_with_retry_shrink(
                texts[left_n:], instruction, max_tokens, backoff_attempt=backoff_attempt + 1
            )
            return left + right

        expected = len(inputs)
        if len(embeddings) != expected:
            raise RuntimeError(
                f"Embedding API returned {len(embeddings)} vectors for {expected} inputs"
            )
        return embeddings

    def _embed(self, texts: list[str], instruction: str, batch_size: int, max_tokens: int, desc: str) -> np.ndarray:
        if not texts:
            return np.asarray([], dtype=np.float32)

        starts = list(range(0, len(texts), batch_size))
        embeddings_by_batch: dict[int, list[list[float]]] = {}

        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            future_to_batch: dict[Any, int] = {}
            for batch_idx, start in enumerate(starts):
                batch_texts = texts[start : start + batch_size]
                future = executor.submit(self._embed_batch, batch_texts, instruction, max_tokens)
                future_to_batch[future] = batch_idx

            for future in tqdm(
                as_completed(future_to_batch),
                total=len(future_to_batch),
                desc=desc,
                leave=False,
            ):
                batch_idx = future_to_batch[future]
                embeddings_by_batch[batch_idx] = future.result()

        all_embeddings: list[list[float]] = []
        for batch_idx in range(len(starts)):
            all_embeddings.extend(embeddings_by_batch[batch_idx])
        return np.asarray(all_embeddings, dtype=np.float32)

    def embed_query(self, texts: list[str], batch_size: int, max_tokens: int) -> np.ndarray:
        instruction = (
            "Instruct: Represent the semantic intent of the query for retrieving relevant documents.\n"
            "Query: {text}"
        )
        return self._embed(texts, instruction, batch_size, max_tokens, "Embedding queries")

    def embed_passage(self, texts: list[str], batch_size: int, max_tokens: int) -> np.ndarray:
        instruction = "Represent the semantic meaning of the document for retrieval.\nDocument: {text}"
        return self._embed(texts, instruction, batch_size, max_tokens, "Embedding passages")


def shard_dir(shards_root: Path, shard_id: int) -> Path:
    return shards_root / f"group_{shard_id}"


def shard_path(shards_root: Path, shard_id: int) -> Path:
    return shard_dir(shards_root, shard_id) / f"shard_{shard_id:02d}.parquet"


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


def add_embeddings(
    index: faiss.IndexIDMap,
    embeddings: np.ndarray,
    next_faiss_id: int,
) -> np.ndarray:
    faiss_ids = np.arange(next_faiss_id, next_faiss_id + len(embeddings), dtype=np.int64)
    if len(embeddings) > 0:
        index.add_with_ids(embeddings.astype(np.float32, copy=False), faiss_ids)
    return faiss_ids


def build_shard_embeddings(
    dataset_path: Path,
    output_dir: Path,
    embedder: BasicEmbedder,
    embedding_dimension: int,
    embedding_batch_size_query: int,
    embedding_batch_size_passage: int,
    parquet_batch_size: int,
    max_tokens: int,
    save_legacy_text_maps: bool,
) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(dataset_path)
    total_rows = parquet_file.metadata.num_rows

    queries_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dimension))
    passages_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dimension))

    queries_id_map: dict[int, tuple[str, str, str, str]] = {}
    passages_id_map: dict[int, tuple[str, str, str, str]] = {}
    query_key_to_faiss: dict[tuple[str, str, str], int] = {}
    passage_key_to_faiss: dict[tuple[str, str, str], int] = {}
    query_embeddings_by_key: dict[tuple[str, str, str], np.ndarray] = {}
    passage_embeddings_by_key: dict[tuple[str, str, str], np.ndarray] = {}
    query_embeddings_by_text: dict[str, np.ndarray] = {}
    passage_embeddings_by_text: dict[str, np.ndarray] = {}
    duplicate_query_text_mismatch: dict[tuple[str, str, str], list[str]] = {}
    duplicate_passage_text_mismatch: dict[tuple[str, str, str], list[str]] = {}

    next_query_faiss_id = 0
    next_passage_faiss_id = 0
    processed_rows = 0

    with tqdm(total=total_rows, desc=f"Scanning {dataset_path.name}") as pbar:
        for record_batch in parquet_file.iter_batches(batch_size=parquet_batch_size):
            rows = record_batch.to_pydict()
            queries_to_embed: list[str] = []
            query_keys_to_embed: list[tuple[str, str, str]] = []
            passages_to_embed: list[str] = []
            passage_keys_to_embed: list[tuple[str, str, str]] = []

            for query_id, query, passage_id, passage, lang, dataset in zip(
                rows["query_id"],
                rows["query"],
                rows["passage_id"],
                rows["passage"],
                rows["lang"],
                rows["dataset"],
                strict=True,
            ):
                query_key = scoped_key(query_id, lang, dataset)
                passage_key = scoped_key(passage_id, lang, dataset)
                query_text = normalize(query).strip()
                passage_text = normalize(passage).strip()

                if query_text:
                    if query_key not in query_key_to_faiss:
                        query_key_to_faiss[query_key] = -1
                        queries_to_embed.append(query_text)
                        query_keys_to_embed.append(query_key)
                    elif query_key in query_embeddings_by_key:
                        previous = queries_id_map[query_key_to_faiss[query_key]][1]
                        if previous != query_text:
                            duplicate_query_text_mismatch.setdefault(query_key, []).append(query_text)

                if passage_text:
                    if passage_key not in passage_key_to_faiss:
                        passage_key_to_faiss[passage_key] = -1
                        passages_to_embed.append(passage_text)
                        passage_keys_to_embed.append(passage_key)
                    elif passage_key in passage_embeddings_by_key:
                        previous = passages_id_map[passage_key_to_faiss[passage_key]][1]
                        if previous != passage_text:
                            duplicate_passage_text_mismatch.setdefault(passage_key, []).append(passage_text)

            if queries_to_embed:
                query_embeddings = embedder.embed_query(
                    queries_to_embed,
                    batch_size=embedding_batch_size_query,
                    max_tokens=max_tokens,
                )
                query_faiss_ids = add_embeddings(queries_index, query_embeddings, next_query_faiss_id)
                for faiss_id, key, text, embedding in zip(
                    query_faiss_ids, query_keys_to_embed, queries_to_embed, query_embeddings, strict=True
                ):
                    fid = int(faiss_id)
                    queries_id_map[fid] = (key[0], text, key[2], key[1])
                    query_key_to_faiss[key] = fid
                    query_embeddings_by_key[key] = embedding
                    if save_legacy_text_maps and text not in query_embeddings_by_text:
                        query_embeddings_by_text[text] = embedding
                next_query_faiss_id += len(query_embeddings)

            if passages_to_embed:
                passage_embeddings = embedder.embed_passage(
                    passages_to_embed,
                    batch_size=embedding_batch_size_passage,
                    max_tokens=max_tokens,
                )
                passage_faiss_ids = add_embeddings(passages_index, passage_embeddings, next_passage_faiss_id)
                for faiss_id, key, text, embedding in zip(
                    passage_faiss_ids,
                    passage_keys_to_embed,
                    passages_to_embed,
                    passage_embeddings,
                    strict=True,
                ):
                    fid = int(faiss_id)
                    passages_id_map[fid] = (key[0], text, key[2], key[1])
                    passage_key_to_faiss[key] = fid
                    passage_embeddings_by_key[key] = embedding
                    if save_legacy_text_maps and text not in passage_embeddings_by_text:
                        passage_embeddings_by_text[text] = embedding
                next_passage_faiss_id += len(passage_embeddings)

            processed_rows += len(rows["query_id"])
            pbar.update(len(rows["query_id"]))
            gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(queries_index, str(output_dir / "faiss_queries_index.bin"))
    faiss.write_index(passages_index, str(output_dir / "faiss_passages_index.bin"))
    np.save(str(output_dir / "queries_id_map.npy"), queries_id_map)
    np.save(str(output_dir / "passages_id_map.npy"), passages_id_map)

    with (output_dir / "query_key_to_faiss.pkl").open("wb") as handle:
        pickle.dump(query_key_to_faiss, handle)
    with (output_dir / "passage_key_to_faiss.pkl").open("wb") as handle:
        pickle.dump(passage_key_to_faiss, handle)
    with (output_dir / "query_embedding.pkl").open("wb") as handle:
        pickle.dump(query_embeddings_by_key, handle)
    with (output_dir / "passage_embedding.pkl").open("wb") as handle:
        pickle.dump(passage_embeddings_by_key, handle)

    if save_legacy_text_maps:
        with (output_dir / "query_embedding_by_text.pkl").open("wb") as handle:
            pickle.dump(query_embeddings_by_text, handle)
        with (output_dir / "passage_embedding_by_text.pkl").open("wb") as handle:
            pickle.dump(passage_embeddings_by_text, handle)

    if processed_rows != total_rows:
        raise AssertionError(f"Processed {processed_rows} rows, expected {total_rows}")
    if queries_index.ntotal != len(queries_id_map) or queries_index.ntotal != len(query_embeddings_by_key):
        raise AssertionError("Query FAISS index, ID map, and embedding map sizes differ")
    if passages_index.ntotal != len(passages_id_map) or passages_index.ntotal != len(passage_embeddings_by_key):
        raise AssertionError("Passage FAISS index, ID map, and embedding map sizes differ")
    if any(faiss_id < 0 for faiss_id in query_key_to_faiss.values()):
        raise AssertionError("Some query keys were not assigned to FAISS IDs")
    if any(faiss_id < 0 for faiss_id in passage_key_to_faiss.values()):
        raise AssertionError("Some passage keys were not assigned to FAISS IDs")

    manifest = {
        "dataset_path": str(dataset_path),
        "processed_rows": processed_rows,
        "embedding_dimension": embedding_dimension,
        "query_vectors": int(queries_index.ntotal),
        "passage_vectors": int(passages_index.ntotal),
        "query_key_count": len(query_key_to_faiss),
        "passage_key_count": len(passage_key_to_faiss),
        "duplicate_query_text_mismatch_count": sum(len(items) for items in duplicate_query_text_mismatch.values()),
        "duplicate_passage_text_mismatch_count": sum(len(items) for items in duplicate_passage_text_mismatch.values()),
        "artifacts": {
            "faiss_queries_index": "faiss_queries_index.bin",
            "faiss_passages_index": "faiss_passages_index.bin",
            "queries_id_map": "queries_id_map.npy",
            "passages_id_map": "passages_id_map.npy",
            "query_key_to_faiss": "query_key_to_faiss.pkl",
            "passage_key_to_faiss": "passage_key_to_faiss.pkl",
            "query_embedding": "query_embedding.pkl",
            "passage_embedding": "passage_embedding.pkl",
        },
    }
    with (output_dir / "embedding_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS indices and scoped embedding maps for shards.")
    parser.add_argument("--shards-root", required=True, type=Path)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--model-path", default=os.environ.get("QWEN_MODEL_PATH", ""))
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--embedding-batch-size-query", type=int, default=4096)
    parser.add_argument("--embedding-batch-size-passage", type=int, default=512)
    parser.add_argument("--parquet-batch-size", type=int, default=200_000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--api-base-url", default=os.environ.get("EMBEDDING_API_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("EMBEDDING_API_KEY", ""))
    parser.add_argument("--api-model", default=os.environ.get("EMBEDDING_MODEL_NAME", ""))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--n-parallel", type=int, default=2)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retry count per embedding batch after a transient HTTP/API failure.",
    )
    parser.add_argument(
        "--retry-backoff-base",
        type=float,
        default=1.0,
        help="Base delay in seconds for exponential backoff between retries.",
    )
    parser.add_argument(
        "--save-legacy-text-maps",
        action="store_true",
        help="Also save query/passsage embedding maps keyed by text for old notebooks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards_root = args.shards_root.resolve()
    embedder = BasicEmbedder(
        model_path=args.model_path,
        base_url=args.api_base_url,
        api_key=args.api_key,
        model_name=args.api_model,
        timeout=args.timeout,
        n_parallel=args.n_parallel,
        max_retries=args.max_retries,
        retry_backoff_base=args.retry_backoff_base,
    )

    manifests: list[dict[str, Any]] = []
    for shard_id in selected_shards(shards_root, args.shard, args.num_shards):
        path = shard_path(shards_root, shard_id)
        if not path.exists():
            raise FileNotFoundError(path)
        logger.info("Embedding shard %s: %s", shard_id, path)
        manifests.append(
            build_shard_embeddings(
                dataset_path=path,
                output_dir=path.parent,
                embedder=embedder,
                embedding_dimension=args.embedding_dimension,
                embedding_batch_size_query=args.embedding_batch_size_query,
                embedding_batch_size_passage=args.embedding_batch_size_passage,
                parquet_batch_size=args.parquet_batch_size,
                max_tokens=args.max_tokens,
                save_legacy_text_maps=args.save_legacy_text_maps,
            )
        )
        gc.collect()

    with (shards_root / "embedding_run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"shards": manifests}, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    main()

# Ex: nohup python embed_shards.py --shards-root /path/to/shards --model-path /path/to/qwen-tokenizer > embed_output.log 2>&1 &
