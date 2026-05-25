# ruff: noqa: I001
import os

import argparse
import gc
import json
import logging
import pickle
import ssl
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal

import numpy as np
import openai
from datasets import load_dataset
from httpx import Client
from tqdm import tqdm
from transformers import AutoTokenizer

from embedding_lookup import DatasetEmbeddingLookup

EMBEDDING_DIM = 1024
logger = logging.getLogger(__name__)


def normalize(value) -> str:
    return "" if value is None else str(value)


def scoped_item_key(item_id, lang) -> str:
    return f"{normalize(item_id)}_{normalize(lang)}"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hard Negative Mining for retrieval datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Path to the output directory",
    )
    parser.add_argument("-d", "--dataset-path", type=str, required=True, help="Path to the dataset parquet file")
    parser.add_argument(
        "-n",
        "--dataset-name",
        type=str,
        default="mmarco",
        help="Fallback dataset name for rows without a dataset column",
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=None,
        help="Root produced by 01_restore_embeddings_by_dataset.py",
    )
    parser.add_argument(
        "--embedding-format",
        choices=["sqlite", "pkl"],
        default="sqlite",
        help="Storage format produced by 01_restore_embeddings_by_dataset.py",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=EMBEDDING_DIM,
        help="Embedding vector dimension used to initialise FAISS",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID to use for FAISS index (default: 0)",
    )
    parser.add_argument(
        "--load-embeddings",
        type=str,
        default=None,
        metavar="PATH",
        help="Legacy path to HNM's old npy/pkl embedding dump",
    )
    parser.add_argument("--model-path", default=os.environ.get("QWEN_MODEL_PATH", ""))
    parser.add_argument("--api-base-url", default=os.environ.get("EMBEDDING_API_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("EMBEDDING_API_KEY", ""))
    parser.add_argument("--api-model", default=os.environ.get("EMBEDDING_MODEL_NAME", ""))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--embedding-batch-size-query", type=int, default=4096)
    parser.add_argument("--embedding-batch-size-passage", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--n-parallel", type=int, default=2)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


# -------- Embedder --------


def create_http_client(timeout: int) -> Client:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return Client(verify=ssl_context, timeout=timeout)


class BasicEmbedder:
    """Lightweight wrapper around the embedding API used when restored embeddings are unavailable."""

    QUERY_INSTRUCTION = (
        "Instruct: Represent the semantic intent of the query for retrieving relevant documents.\nQuery: {text}"
    )
    PASSAGE_INSTRUCTION = "Represent the semantic meaning of the document for retrieval.\nDocument: {text}"

    def __init__(
        self,
        model_path: str | Path,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int,
        n_parallel: int,
        query_batch_size: int,
        passage_batch_size: int,
        max_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.n_parallel = max(1, n_parallel)
        self.query_batch_size = query_batch_size
        self.passage_batch_size = passage_batch_size
        self.max_tokens = max_tokens
        self.http_client = create_http_client(timeout)
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, http_client=self.http_client)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), padding_side="left")

    def _truncate_texts(self, texts: list[str], tokenize_batch_size: int = 4096) -> list[str]:
        truncated: list[str] = []
        for start in range(0, len(texts), tokenize_batch_size):
            batch = texts[start : start + tokenize_batch_size]
            encoded = self.tokenizer(batch, padding=False, truncation=False)["input_ids"]
            for text, tokens in zip(batch, encoded, strict=True):
                if len(tokens) > self.max_tokens:
                    text = self.tokenizer.decode(tokens[: self.max_tokens], skip_special_tokens=True)
                truncated.append(text)
            del encoded
        return truncated

    def _embed(self, texts: list[str], instruction: str, batch_size: int, desc: str) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        logger.info("%s: preparing %s texts, batch_size=%s", desc, len(texts), batch_size)
        truncated = self._truncate_texts(texts)
        n_batches = (len(truncated) + batch_size - 1) // batch_size

        def sub_batches():
            for i in range(0, len(truncated), batch_size):
                chunk = truncated[i : i + batch_size]
                yield [instruction.format(text=t) for t in chunk]

        def api_call(inputs: list[str]) -> np.ndarray:
            response = self.client.embeddings.create(model=self.model_name, input=inputs)
            return np.array([item.embedding for item in response.data], dtype=np.float32)

        result_chunks: list[np.ndarray] = []
        n_workers = min(self.n_parallel, n_batches)
        if n_workers <= 1:
            for batch_inputs in tqdm(sub_batches(), desc=desc, total=n_batches):
                result_chunks.append(api_call(batch_inputs))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for chunk_arr in tqdm(pool.map(api_call, sub_batches()), total=n_batches, desc=desc):
                    result_chunks.append(chunk_arr)

        embeddings = np.vstack(result_chunks) if result_chunks else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        logger.info("%s: built embeddings with shape %s", desc, embeddings.shape)
        return embeddings

    def embed_query(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, self.QUERY_INSTRUCTION, self.query_batch_size, "Embedding queries")

    def embed_passage(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, self.PASSAGE_INSTRUCTION, self.passage_batch_size, "Embedding passages")


# -------- Mining Configuration & Miner --------


@dataclass
class MiningConfig:
    """Configuration for hard negative mining."""

    top_k: int = 20
    margin_type: Literal["absolute", "relative"] = "relative"
    margin_value: float = 0.05
    n_negatives: int | None = None
    max_search_k: int = 2048

    # FAISS index parameters
    index_type: Literal["flat", "ivfpq"] = "flat"

    # IVFPQ parameters (only used when index_type="ivfpq")
    nlist: int = 100
    m: int = 64
    nbits: int = 8
    nprobe: int = 10


class HardNegativeMiner:
    """
    Mine hard negatives for retrieval training by indexing passages in FAISS
    and finding near-miss candidates for each query.
    """

    def __init__(
        self,
        config: MiningConfig,
        query_embed_fn: Callable[[list[str]], np.ndarray],
        passage_embed_fn: Callable[[list[str]], np.ndarray],
        embedding_dimension: int | None = None,
        gpu_id: int = 0,
        use_gpu: bool = True,
        verbose: bool = True,
    ):
        self.config = config
        self.query_embed_fn = query_embed_fn
        self.passage_embed_fn = passage_embed_fn
        self.verbose = verbose
        self.gpu_id = gpu_id

        if embedding_dimension is None:
            embedding_dimension = self._infer_dimension()
        self.embedding_dimension = embedding_dimension

        self.gpu_resources = None
        self.faiss_index = self._build_faiss_index(embedding_dimension, config, gpu_id, use_gpu)

        # Passage storage — embeddings kept in RAM for direct positive-score computation
        self.passage_embeddings: np.ndarray | None = None
        self.passage_langs: dict[str, str | None] = {}
        self.p_keys: list[str] = []
        self.idx_by_pid: dict[str, int] = {}

        # Query storage
        self.query_embeddings: np.ndarray | None = None
        self.queries: dict[str, str] = {}
        self.query_langs: dict[str, str | None] = {}
        self.q_keys: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_dimension(self) -> int:
        """Infer embedding dimension from a single sample."""
        sample_vec = self.passage_embed_fn(["sample"])
        return int(np.asarray(sample_vec).shape[-1])

    def _build_faiss_index(
        self, dimension: int, config: MiningConfig, gpu_id: int, use_gpu: bool = True
    ):
        import faiss

        if config.index_type == "ivfpq":
            if dimension % config.m != 0:
                raise ValueError(f"Embedding dimension ({dimension}) must be divisible by m ({config.m})")
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, config.nlist, config.m, config.nbits)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
            logger.info(
                "Created IVFPQ index: nlist=%s, m=%s, nbits=%s, nprobe=%s",
                config.nlist,
                config.m,
                config.nbits,
                config.nprobe,
            )
        else:
            index = faiss.IndexFlatIP(dimension)
            logger.info("Created FlatIP index (exact search)")

        if not use_gpu:
            return index

        try:
            num_gpus = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
            logger.info("FAISS GPUs available: %s", num_gpus)
            if num_gpus == 0:
                logger.warning("No FAISS-compatible GPUs found, using CPU")
                return index
            if gpu_id >= num_gpus:
                logger.warning("gpu_id=%s >= num_gpus=%s, falling back to CPU", gpu_id, num_gpus)
                return index

            self.gpu_resources = faiss.StandardGpuResources()
            self.gpu_resources.setTempMemory(512 * 1024 * 1024)

            if config.index_type == "ivfpq":
                cloner_options = faiss.GpuClonerOptions()
                cloner_options.useFloat16LookupTables = True
                index = faiss.index_cpu_to_gpu(self.gpu_resources, gpu_id, index, cloner_options)
                logger.info("Using GPU IVFPQ index on GPU %s with float16 lookup tables", gpu_id)
            else:
                index = faiss.index_cpu_to_gpu(self.gpu_resources, gpu_id, index)
                logger.info("Using GPU FAISS index on GPU %s", gpu_id)
        except Exception as e:
            logger.warning("GPU FAISS initialisation failed (%s), falling back to CPU", e)
            self.gpu_resources = None

        return index

    def _ensure_faiss_ready(self, vecs: np.ndarray) -> np.ndarray:
        """Ensure *vecs* is float32, C-contiguous — required by FAISS (esp. GPU)."""
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32)
        if not vecs.flags["C_CONTIGUOUS"]:
            vecs = np.ascontiguousarray(vecs)
        return vecs

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def load_precomputed_embeddings(self, embeddings_dir: str) -> None:
        """Load pre-computed embeddings from disk and populate the FAISS index."""
        if self.verbose:
            logger.info("Loading legacy pre-computed embeddings from %s", embeddings_dir)

        passage_embeddings = np.load(os.path.join(embeddings_dir, "passage_embeddings.npy"))
        query_embeddings = np.load(os.path.join(embeddings_dir, "query_embeddings.npy"))

        with open(os.path.join(embeddings_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        self.p_keys = metadata["p_keys"]
        self.passage_langs = metadata["passage_langs"]
        self.idx_by_pid = metadata["idx_by_pid"]
        self.queries = metadata["queries"]
        self.query_langs = metadata["query_langs"]
        self.q_keys = metadata["q_keys"]

        self.passage_embeddings = passage_embeddings.astype(np.float32)
        self.query_embeddings = query_embeddings.astype(np.float32)

        vecs = self._ensure_faiss_ready(self.passage_embeddings)

        if self.config.index_type == "ivfpq" and not self.faiss_index.is_trained:
            training_size = min(len(vecs), self.config.nlist * 100)
            logger.info("Training IVFPQ index with %s vectors", training_size)
            self.faiss_index.train(vecs[:training_size])
            if self.verbose:
                logger.info("IVFPQ index trained with %s vectors", training_size)

        logger.info("Adding %s passage vectors to FAISS index", len(vecs))
        self.faiss_index.add(vecs)

        if self.verbose:
            logger.info(
                "Loaded %s passages, %s queries; FAISS index has %s vectors",
                len(self.p_keys),
                len(self.q_keys),
                self.faiss_index.ntotal,
            )

    def load_restored_embeddings(
        self,
        dataset,
        embeddings_root: str | Path,
        storage_format: Literal["sqlite", "pkl"],
        default_dataset_name: str,
        query_fallback_embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        passage_fallback_embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        """Load query/passage embeddings restored into per-dataset stores."""
        if self.verbose:
            logger.info("Loading restored embeddings from %s (%s)", embeddings_root, storage_format)

        passage_specs: dict[str, tuple[str, str, str]] = {}
        query_specs: dict[str, tuple[str, str, str]] = {}
        passage_texts: dict[str, str] = {}
        passage_langs: dict[str, str | None] = {}
        query_langs: dict[str, str | None] = {}
        queries: dict[str, str] = {}
        p_keys: list[str] = []
        q_keys: list[str] = []
        fallback_dataset = normalize(default_dataset_name)

        if self.verbose:
            logger.info("Collecting unique queries and passages from dataset")

        for r in dataset:
            lang = normalize(r.get("lang"))
            row_dataset = normalize(r.get("dataset")) or fallback_dataset

            qid = normalize(r["query_id"])
            qkey = scoped_item_key(qid, lang)
            if qkey not in query_specs:
                query_specs[qkey] = (row_dataset, qid, lang)
                query_langs[qkey] = lang
                queries[qkey] = normalize(r.get("query"))
                q_keys.append(qkey)

            pid = normalize(r["passage_id"])
            pkey = scoped_item_key(pid, lang)
            if pkey not in passage_specs:
                passage_specs[pkey] = (row_dataset, pid, lang)
                passage_texts[pkey] = normalize(r.get("passage"))
                passage_langs[pkey] = lang
                p_keys.append(pkey)

        if not p_keys:
            raise ValueError("No passages found in dataset")
        if not q_keys:
            raise ValueError("No queries found in dataset")

        logger.info("Collected %s unique passages and %s unique queries", len(p_keys), len(q_keys))
        with DatasetEmbeddingLookup(embeddings_root, storage_format) as lookup:
            logger.info("Loading passage embeddings from restored store")
            passage_embeddings = self._load_ordered_lookup_embeddings(
                lookup=lookup,
                kind="passage",
                specs=passage_specs,
                ordered_keys=p_keys,
                texts_by_key=passage_texts,
                fallback_embed_fn=passage_fallback_embed_fn,
            )
            logger.info("Loading query embeddings from restored store")
            query_embeddings = self._load_ordered_lookup_embeddings(
                lookup=lookup,
                kind="query",
                specs=query_specs,
                ordered_keys=q_keys,
                texts_by_key=queries,
                fallback_embed_fn=query_fallback_embed_fn,
            )

        self.p_keys = p_keys
        self.passage_langs = passage_langs
        self.idx_by_pid = {pid: i for i, pid in enumerate(p_keys)}
        self.queries = queries
        self.query_langs = query_langs
        self.q_keys = q_keys
        self.passage_embeddings = passage_embeddings
        self.query_embeddings = query_embeddings

        vecs = self._ensure_faiss_ready(self.passage_embeddings)
        if self.config.index_type == "ivfpq" and not self.faiss_index.is_trained:
            training_size = min(len(vecs), self.config.nlist * 100)
            logger.info("Training IVFPQ index with %s vectors", training_size)
            self.faiss_index.train(vecs[:training_size])
            if self.verbose:
                logger.info("IVFPQ index trained with %s vectors", training_size)
        logger.info("Adding %s passage vectors to FAISS index", len(vecs))
        self.faiss_index.add(vecs)

        if self.verbose:
            logger.info(
                "Loaded %s passages, %s queries; FAISS index has %s vectors",
                len(self.p_keys),
                len(self.q_keys),
                self.faiss_index.ntotal,
            )

    def _load_ordered_lookup_embeddings(
        self,
        *,
        lookup: DatasetEmbeddingLookup,
        kind: Literal["query", "passage"],
        specs: dict[str, tuple[str, str, str]],
        ordered_keys: list[str],
        texts_by_key: dict[str, str],
        fallback_embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ) -> np.ndarray:
        embeddings_by_spec: dict[tuple[str, str, str], np.ndarray] = {}
        ids_by_dataset_lang: dict[tuple[str, str], list[str]] = defaultdict(list)

        for key in ordered_keys:
            dataset_name, item_id, lang = specs[key]
            ids_by_dataset_lang[(dataset_name, lang)].append(item_id)

        for (dataset_name, lang), item_ids in tqdm(
            ids_by_dataset_lang.items(),
            desc=f"Loading {kind} embeddings",
            disable=not self.verbose,
        ):
            found = lookup.batch_get(kind, dataset=dataset_name, item_ids=item_ids, lang=lang)
            for item_id, embedding in found.items():
                array = np.asarray(embedding, dtype=np.float32).reshape(-1)
                if array.shape[0] != self.embedding_dimension:
                    raise ValueError(
                        f"{kind} embedding dimension mismatch for dataset={dataset_name!r}, "
                        f"id={item_id!r}, lang={lang!r}: expected {self.embedding_dimension}, "
                        f"got {array.shape[0]}"
                    )
                embeddings_by_spec[(dataset_name, item_id, lang)] = array

        missing_keys: list[str] = []
        arrays: list[np.ndarray] = []
        for key in ordered_keys:
            spec = specs[key]
            embedding = embeddings_by_spec.get(spec)
            if embedding is None:
                missing_keys.append(key)
                continue
            arrays.append(embedding)

        if missing_keys:
            examples = [
                f"dataset={specs[key][0]!r}, id={specs[key][1]!r}, lang={specs[key][2]!r}"
                for key in missing_keys[:10]
            ]
            sample = "; ".join(examples)
            if fallback_embed_fn is None:
                raise ValueError(f"Missing {len(missing_keys)} {kind} embeddings. Examples: {sample}")

            logger.warning(
                "Missing %s %s embeddings in restored store; computing them with live embedder. Examples: %s",
                len(missing_keys),
                kind,
                sample,
            )
            missing_texts = [texts_by_key[key] for key in missing_keys]
            fallback_embeddings = np.asarray(fallback_embed_fn(missing_texts), dtype=np.float32)
            if fallback_embeddings.shape != (len(missing_keys), self.embedding_dimension):
                raise ValueError(
                    f"Fallback {kind} embeddings have shape {fallback_embeddings.shape}, "
                    f"expected ({len(missing_keys)}, {self.embedding_dimension})"
                )
            for key, embedding in zip(missing_keys, fallback_embeddings, strict=True):
                embeddings_by_spec[specs[key]] = np.asarray(embedding, dtype=np.float32).reshape(-1)

            arrays = [embeddings_by_spec[specs[key]] for key in ordered_keys]

        embeddings = np.vstack(arrays).astype(np.float32, copy=False)
        logger.info("Loaded %s %s embeddings with shape %s", len(arrays), kind, embeddings.shape)
        return embeddings

    def index_passages(self, dataset, batch_size: int = 4096) -> None:
        """
        Index passages from *dataset* into the FAISS index and store their
        embeddings for later direct positive-score computation.
        """
        seen_pids: set[str] = set(self.p_keys)
        new_p_keys: list[str] = []
        new_p_texts: list = []
        new_p_langs: list = []

        if self.verbose:
            logger.info("Collecting unique passages from dataset")

        for r in dataset:
            pid = scoped_item_key(r["passage_id"], r.get("lang"))
            if pid not in seen_pids:
                seen_pids.add(pid)
                new_p_keys.append(pid)
                new_p_texts.append(r["passage"])
                new_p_langs.append(normalize(r.get("lang")))

        del seen_pids

        if not new_p_keys:
            if self.verbose:
                logger.info("No new passages to index")
            return

        if self.verbose:
            logger.info("Found %s unique passages to index", len(new_p_keys))

        new_passage_embeddings = np.empty(
            (len(new_p_keys), self.embedding_dimension), dtype=np.float32
        )

        start_offset = 0

        # Train IVFPQ if needed; reuse training embeddings to avoid double computation
        if self.config.index_type == "ivfpq" and not self.faiss_index.is_trained:
            if self.verbose:
                logger.info("Training IVFPQ index")
            training_size = min(len(new_p_keys), self.config.nlist * 100)
            training_texts = new_p_texts[:training_size]
            training_vecs = self._ensure_faiss_ready(
                np.asarray(self.passage_embed_fn(training_texts), dtype=np.float32)
            )
            self.faiss_index.train(training_vecs)
            if self.verbose:
                logger.info("IVFPQ index trained with %s vectors", len(training_vecs))

            self.faiss_index.add(training_vecs)
            new_passage_embeddings[:training_size] = training_vecs

            current_idx = len(self.p_keys)
            for k in range(training_size):
                pid = new_p_keys[k]
                self.passage_langs[pid] = new_p_langs[k]
                self.p_keys.append(pid)
                self.idx_by_pid[pid] = current_idx
                current_idx += 1
                new_p_texts[k] = None

            start_offset = training_size
            del training_vecs, training_texts
            gc.collect()

        # Embed and add remaining passages
        current_idx = len(self.p_keys)
        n_remaining = len(new_p_keys) - start_offset
        n_batches = (n_remaining + batch_size - 1) // batch_size if n_remaining > 0 else 0

        for i in tqdm(
            range(start_offset, len(new_p_keys), batch_size),
            desc="Indexing passages",
            total=n_batches,
        ):
            batch_end = min(i + batch_size, len(new_p_keys))
            batch_pids = new_p_keys[i:batch_end]
            batch_texts = new_p_texts[i:batch_end]

            batch_vecs = self._ensure_faiss_ready(
                np.asarray(self.passage_embed_fn(batch_texts), dtype=np.float32)
            )
            self.faiss_index.add(batch_vecs)
            new_passage_embeddings[i:batch_end] = batch_vecs

            for k, pid in enumerate(batch_pids):
                self.passage_langs[pid] = new_p_langs[i + k]
                self.p_keys.append(pid)
                self.idx_by_pid[pid] = current_idx
                current_idx += 1

            del batch_vecs

            for j in range(i, batch_end):
                new_p_texts[j] = None

            try:
                import torch
                if torch.cuda.is_available() and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        if self.passage_embeddings is not None:
            self.passage_embeddings = np.vstack([self.passage_embeddings, new_passage_embeddings])
        else:
            self.passage_embeddings = new_passage_embeddings

        del new_p_texts, new_p_keys, new_p_langs
        gc.collect()

        if self.verbose:
            logger.info("Indexed %s total passages", len(self.p_keys))

    def index_queries(self, dataset) -> None:
        """Index queries from *dataset*."""
        queries: dict[str, str] = {}
        query_langs: dict[str, str | None] = {}
        q_keys: list[str] = []

        if self.verbose:
            logger.info("Collecting unique queries from dataset")

        for r in dataset:
            qid = scoped_item_key(r["query_id"], r.get("lang"))
            if qid not in queries:
                queries[qid] = r["query"]
                query_langs[qid] = normalize(r.get("lang"))
                q_keys.append(qid)

        if not q_keys:
            raise ValueError("No queries found in dataset")

        q_texts = [queries[k] for k in q_keys]
        if self.verbose:
            logger.info("Found %s unique queries, computing embeddings", len(q_texts))

        query_embeddings = np.asarray(self.query_embed_fn(q_texts), dtype=np.float32)
        if self.verbose:
            logger.info("Built query embeddings with shape %s", query_embeddings.shape)

        self.queries = queries
        self.query_langs = query_langs
        self.q_keys = q_keys
        self.query_embeddings = query_embeddings

        del q_texts
        gc.collect()

        if self.verbose:
            logger.info("Indexed %s queries", len(q_keys))

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------

    def _compute_margin_threshold(self, avg_pos_score: float) -> float:
        """Return the score threshold below which a candidate counts as a hard negative."""
        if self.config.margin_type == "relative":
            return avg_pos_score - abs(avg_pos_score) * self.config.margin_value
        else:  # absolute
            return avg_pos_score - self.config.margin_value

    def _recursive_mine(
        self,
        batch_q_keys: list[str],
        batch_q_embs: np.ndarray,
        qpairs: dict[str, list[str]],
        all_pos_indices_by_q: dict[str, np.ndarray],
        same_lang_pos_indices_by_q: dict[str, np.ndarray],
        search_k: int,
    ) -> list[dict]:
        """
        Mine hard negatives for a batch of queries.

        Positive scores are computed directly (dot product with stored passage
        embeddings) so positives no longer need to appear in the FAISS top-k.
        The average score of same-language positives is used as the reference
        for the margin threshold, while ALL positives (across languages) are
        excluded from negative candidates.
        """
        if not batch_q_keys:
            return []

        if self.verbose:
            logger.info("Searching %s queries with search_k=%s", len(batch_q_keys), search_k)

        batch_q_embs = self._ensure_faiss_ready(batch_q_embs)

        t0 = time()
        batch_scores, batch_indices = self.faiss_index.search(batch_q_embs, search_k)
        if self.verbose:
            logger.info("FAISS search took %.3fs", time() - t0)

        batch_results: list[dict] = []
        retry_batch_indices: list[int] = []
        existing_hard_negs: dict[str, list[dict]] = {}

        p_keys = self.p_keys
        passage_embs = self.passage_embeddings

        for batch_idx, qkey in enumerate(batch_q_keys):
            all_pos_indices = all_pos_indices_by_q[qkey]
            same_lang_pos_indices = same_lang_pos_indices_by_q[qkey]
            pos_pids = list(set(qpairs.get(qkey, [])))

            query_emb = batch_q_embs[batch_idx]

            # Compute positive reference score directly (no dependence on top-k)
            if len(same_lang_pos_indices) > 0:
                sl_scores = query_emb @ passage_embs[same_lang_pos_indices].T
                avg_pos_score = float(sl_scores.mean())
            elif len(all_pos_indices) > 0:
                all_scores = query_emb @ passage_embs[all_pos_indices].T
                avg_pos_score = float(all_scores.mean())
            else:
                batch_results.append({
                    "query_id": qkey,
                    "query": self.queries[qkey],
                    "positive_ids": pos_pids,
                    "hard_negatives": [],
                })
                continue

            margin_threshold = self._compute_margin_threshold(avg_pos_score)

            scores = batch_scores[batch_idx]
            indices_arr = batch_indices[batch_idx]

            valid_mask = indices_arr >= 0
            scores = scores[valid_mask]
            indices_arr = indices_arr[valid_mask]

            # Exclude ALL positives (all languages) from negative candidates
            if len(all_pos_indices) > 0:
                all_pos_mask = np.isin(indices_arr, all_pos_indices)
            else:
                all_pos_mask = np.zeros(len(indices_arr), dtype=bool)

            neg_mask = ~all_pos_mask & (scores <= margin_threshold)
            neg_scores = scores[neg_mask]
            neg_indices = indices_arr[neg_mask]
            neg_ranks = np.where(neg_mask)[0]

            hard_negs = [
                {
                    "passage_id": p_keys[int(idx)],
                    "score": float(sc),
                    "avg_pos_score": avg_pos_score,
                    "rank": int(rk),
                    "lang": self.passage_langs.get(p_keys[int(idx)]),
                }
                for rk, idx, sc in zip(neg_ranks, neg_indices, neg_scores, strict=True)
            ]

            n_needed = self.config.n_negatives if self.config.n_negatives is not None else len(hard_negs)

            if self.config.n_negatives is not None and len(hard_negs) < n_needed:
                retry_batch_indices.append(batch_idx)
                existing_hard_negs[qkey] = hard_negs
            else:
                batch_results.append({
                    "query_id": qkey,
                    "query": self.queries[qkey],
                    "positive_ids": pos_pids,
                    "hard_negatives": hard_negs[:n_needed],
                })

        # ------ Recursive widening ------
        if retry_batch_indices:
            new_search_k = search_k * 2
            max_search_k = min(len(self.p_keys), self.config.max_search_k)

            queries_needing_more = [batch_q_keys[j] for j in retry_batch_indices]

            if new_search_k <= max_search_k:
                if self.verbose:
                    logger.info(
                        "Retrying %s queries with search_k=%s",
                        len(queries_needing_more),
                        new_search_k,
                    )
                retry_embs = batch_q_embs[retry_batch_indices]

                recursive_results = self._recursive_mine(
                    queries_needing_more,
                    retry_embs,
                    qpairs,
                    all_pos_indices_by_q,
                    same_lang_pos_indices_by_q,
                    new_search_k,
                )
                for rec in recursive_results:
                    qkey = rec["query_id"]
                    merged = self._merge_hard_negatives(existing_hard_negs.get(qkey, []), rec["hard_negatives"])
                    if self.config.n_negatives is not None:
                        merged = merged[: self.config.n_negatives]
                    rec["hard_negatives"] = merged
                    batch_results.append(rec)
            else:
                # Reached the search_k cap — return whatever we have
                if self.verbose:
                    logger.info(
                        "Reached max search_k (%s); %s queries have insufficient negatives",
                        max_search_k,
                        len(queries_needing_more),
                    )
                for qkey in queries_needing_more:
                    batch_results.append({
                        "query_id": qkey,
                        "query": self.queries[qkey],
                        "positive_ids": list(set(qpairs.get(qkey, []))),
                        "hard_negatives": existing_hard_negs.get(qkey, []),
                    })

        return batch_results

    @staticmethod
    def _merge_hard_negatives(existing: list[dict], new: list[dict]) -> list[dict]:
        """Merge two lists of hard negatives, de-duplicating by ``passage_id``."""
        merged: dict[str, dict] = {}
        for neg in existing:
            merged[neg["passage_id"]] = dict(neg)

        for neg in new:
            pid = neg["passage_id"]
            if pid in merged:
                prev = merged[pid]
                prev["rank"] = max(prev["rank"], neg["rank"])
                if neg["score"] > prev["score"]:
                    prev["score"] = neg["score"]
                    prev["avg_pos_score"] = neg["avg_pos_score"]
            else:
                merged[pid] = dict(neg)

        return sorted(merged.values(), key=lambda x: (x["score"], x["passage_id"]), reverse=True)

    def mine(
        self,
        dataset,
        mining_batch_size: int = 256,
    ) -> list[dict]:
        """
        Mine hard negatives using pre-indexed passages and queries.

        Returns:
            List of dicts ``{query_id, query, positive_ids, hard_negatives}``.
            Queries with no or insufficient negatives are included as well.
        """
        if self.verbose:
            logger.info("Collecting query-passage pairs from dataset")

        base_qpairs: dict[str, set[str]] = defaultdict(set)
        qid_to_base: dict[str, str] = {}

        for r in dataset:
            base_qid = str(r["query_id"])
            qid = scoped_item_key(r["query_id"], r.get("lang"))
            pid = scoped_item_key(r["passage_id"], r.get("lang"))
            base_qpairs[base_qid].add(pid)
            qid_to_base[qid] = base_qid

        # Each language-specific query gets *all* positives from every language
        # variant of the same base query_id (avoids cross-language false negatives).
        qpairs: dict[str, list[str]] = {
            qid: list(base_qpairs[base_qid]) for qid, base_qid in qid_to_base.items()
        }

        del base_qpairs, qid_to_base

        # Only process queries that actually have positive pairs
        active_q_indices: list[int] = []
        active_q_keys: list[str] = []
        for i, qkey in enumerate(self.q_keys):
            if qpairs.get(qkey):
                active_q_indices.append(i)
                active_q_keys.append(qkey)

        if self.verbose:
            logger.info(
                "Found %s language-specific queries, %s with positive pairs",
                len(qpairs),
                len(active_q_keys),
            )

        q_embs = self.query_embeddings[active_q_indices]

        if self.config.index_type == "ivfpq" and hasattr(self.faiss_index, "nprobe"):
            self.faiss_index.nprobe = self.config.nprobe

        # Pre-compute positive FAISS indices per query (all-language and same-language)
        idx_by_pid = self.idx_by_pid
        all_pos_indices_by_q: dict[str, np.ndarray] = {}
        same_lang_pos_indices_by_q: dict[str, np.ndarray] = {}

        for qkey in active_q_keys:
            q_lang = self.query_langs.get(qkey)
            all_pids = qpairs.get(qkey, [])

            all_indices = [idx_by_pid[pid] for pid in all_pids if pid in idx_by_pid]
            same_lang_indices = [
                idx_by_pid[pid]
                for pid in all_pids
                if pid in idx_by_pid and self.passage_langs.get(pid) == q_lang
            ]

            all_pos_indices_by_q[qkey] = np.array(all_indices, dtype=np.int64)
            same_lang_pos_indices_by_q[qkey] = np.array(same_lang_indices, dtype=np.int64)

        initial_search_k = self.config.top_k

        # Mine in batches
        mined: list[dict] = []
        max_rank = -1

        for batch_start in tqdm(range(0, len(active_q_keys), mining_batch_size), desc="Mining batches"):
            batch_end = min(batch_start + mining_batch_size, len(active_q_keys))
            batch_q_keys = active_q_keys[batch_start:batch_end]
            batch_q_embs = q_embs[batch_start:batch_end]

            batch_results = self._recursive_mine(
                batch_q_keys,
                batch_q_embs,
                qpairs,
                all_pos_indices_by_q,
                same_lang_pos_indices_by_q,
                initial_search_k,
            )

            for result in batch_results:
                if result["hard_negatives"]:
                    batch_max = max(hn["rank"] for hn in result["hard_negatives"])
                    if batch_max > max_rank:
                        max_rank = batch_max

            mined.extend(batch_results)

        if self.verbose:
            if max_rank >= 0:
                logger.info("Greatest negative rank in retrieved list: %s", max_rank)
            else:
                logger.info("No hard negatives were mined")
            n_with_negs = sum(1 for r in mined if r["hard_negatives"])
            n_full = sum(
                1 for r in mined
                if self.config.n_negatives is None or len(r["hard_negatives"]) >= self.config.n_negatives
            )
            logger.info(
                "Mining complete: %s results (%s with negatives, %s with full quota)",
                len(mined),
                n_with_negs,
                n_full,
            )

        return mined


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    logger.info("Loading dataset from %s", args.dataset_path)
    dataset = load_dataset("parquet", data_files=args.dataset_path, split="train")
    logger.info("Loaded dataset with %s rows", len(dataset))

    # ------------------------------------------------------------------
    # Mining with restored embeddings when available, otherwise via live API.
    # ------------------------------------------------------------------
    cfg = MiningConfig(
        top_k=128,
        margin_type="relative",
        margin_value=0.05,
        n_negatives=15,
        max_search_k=2048,
        index_type="flat",
    )

    def _noop_embed(_texts: list[str]) -> np.ndarray:
        raise RuntimeError("Embedder should not be called when loading pre-computed embeddings")

    if args.load_embeddings:
        logger.info("Loading legacy pre-computed embeddings")
        miner = HardNegativeMiner(
            config=cfg,
            query_embed_fn=_noop_embed,
            passage_embed_fn=_noop_embed,
            gpu_id=args.gpu_id,
            use_gpu=True,
            verbose=True,
            embedding_dimension=args.embedding_dimension,
        )
        miner.load_precomputed_embeddings(args.load_embeddings)
        gc.collect()
    elif args.embeddings_root is not None:
        logger.info("Using restored embeddings from %s", args.embeddings_root)
        embedder = BasicEmbedder(
            model_path=args.model_path,
            base_url=args.api_base_url,
            api_key=args.api_key,
            model_name=args.api_model,
            timeout=args.timeout,
            n_parallel=args.n_parallel,
            query_batch_size=args.embedding_batch_size_query,
            passage_batch_size=args.embedding_batch_size_passage,
            max_tokens=args.max_tokens,
        )
        miner = HardNegativeMiner(
            config=cfg,
            query_embed_fn=_noop_embed,
            passage_embed_fn=_noop_embed,
            gpu_id=args.gpu_id,
            use_gpu=True,
            verbose=True,
            embedding_dimension=args.embedding_dimension,
        )
        miner.load_restored_embeddings(
            dataset=dataset,
            embeddings_root=args.embeddings_root,
            storage_format=args.embedding_format,
            default_dataset_name=args.dataset_name,
            query_fallback_embed_fn=embedder.embed_query,
            passage_fallback_embed_fn=embedder.embed_passage,
        )
        del embedder
        gc.collect()
    else:
        logger.info("No pre-computed embeddings provided; using live embedding API")
        embedder = BasicEmbedder(
            model_path=args.model_path,
            base_url=args.api_base_url,
            api_key=args.api_key,
            model_name=args.api_model,
            timeout=args.timeout,
            n_parallel=args.n_parallel,
            query_batch_size=args.embedding_batch_size_query,
            passage_batch_size=args.embedding_batch_size_passage,
            max_tokens=args.max_tokens,
        )
        miner = HardNegativeMiner(
            config=cfg,
            query_embed_fn=embedder.embed_query,
            passage_embed_fn=embedder.embed_passage,
            gpu_id=args.gpu_id,
            use_gpu=True,
            verbose=True,
            embedding_dimension=args.embedding_dimension,
        )
        logger.info("Indexing passages with live embeddings")
        miner.index_passages(dataset, batch_size=4096)
        gc.collect()
        logger.info("Indexing queries with live embeddings")
        miner.index_queries(dataset)
        del embedder
        gc.collect()

    logger.info("Mining hard negatives")
    mined = miner.mine(
        dataset,
        mining_batch_size=16,
    )

    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"hard_negatives_{args.dataset_name}.json")
    pickle_path = os.path.join(output_dir, f"hard_negatives_{args.dataset_name}.pkl")

    logger.info("Saving mined hard negatives to JSON: %s", json_path)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(mined, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s results to JSON", len(mined))
    except Exception as e:
        logger.exception("Error saving to JSON file %s: %s", json_path, e)

    logger.info("Saving mined hard negatives to pickle: %s", pickle_path)
    try:
        with open(pickle_path, "wb") as f:
            pickle.dump(mined, f)
        logger.info("Saved %s results to pickle", len(mined))
    except Exception as e:
        logger.exception("Error saving to pickle file %s: %s", pickle_path, e)

    logger.info("Hard negatives saved successfully")
    logger.info("JSON output: %s", json_path)
    logger.info("Pickle output: %s", pickle_path)


if __name__ == "__main__":
    main()
