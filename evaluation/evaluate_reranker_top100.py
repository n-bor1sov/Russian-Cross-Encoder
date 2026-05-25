#!/usr/bin/env python3
"""Evaluate rerankers on RusBEIR using saved first-stage retrieval JSON files.

Supports multiple reranker backends (sequence classification, sentence-transformers
CrossEncoder, Qwen3 yes/no CausalLM, Jina v3). Elasticsearch is not used.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from generate_bm25_top100 import CORE_IR_QA_DATASETS, DEFAULT_K_VALUES, selected_datasets
from local_assets import (
    DEFAULT_OFFLINE_ASSETS_DIR,
    assets_exist,
    load_corpus,
    load_qrels,
    load_queries,
    validate_offline_bundle,
)

MODEL_TYPES = (
    "auto",
    "seq-classification",
    "cross-encoder",
    "qwen-reranker",
    "jina-v3",
)

_QWEN_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def resolve_device(requested_device: str) -> str:
    import torch

    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip("/")).strip("-")


def default_output_dir(input_dir: Path, model_path: str) -> Path:
    model_name = safe_name(Path(model_path).name or model_path)
    return input_dir.with_name(f"{input_dir.name}-{model_name}-reranked")


def detect_model_type(model_path: str) -> str:
    """Infer reranker backend; default is sentence-transformers CrossEncoder."""
    path_lower = model_path.lower().replace("\\", "/")

    if "jina-reranker-v3" in path_lower or path_lower.endswith("jina-reranker-v3"):
        return "jina-v3"

    config_path = Path(model_path) / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        architectures = [a.lower() for a in cfg.get("architectures", [])]
        auto_map = cfg.get("auto_map") or {}
        model_type = (cfg.get("model_type") or "").lower()

        if any("jinaforranking" in a for a in architectures) or "JinaForRanking" in str(
            auto_map
        ):
            return "jina-v3"
        if any("causallm" in a for a in architectures) or model_type in {
            "qwen3",
            "qwen2",
            "llama",
        }:
            if "reranker" in path_lower or "rerank" in path_lower:
                return "qwen-reranker"

    if "qwen" in path_lower and "reranker" in path_lower:
        return "qwen-reranker"

    return "cross-encoder"


def resolve_model_type(model_path: str, requested: str) -> str:
    if requested != "auto":
        return requested
    detected = detect_model_type(model_path)
    logging.info("Auto-detected model type: %s", detected)
    return detected


class RerankerBackend(ABC):
    @abstractmethod
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score a flat list of (query, doc) pairs; return one float per pair."""

    def score_query(self, query: str, doc_texts: list[str]) -> list[float]:
        return self.score_pairs([(query, doc) for doc in doc_texts])


class SeqClassificationBackend(RerankerBackend):
    def __init__(
        self,
        model_path: str,
        device: str,
        batch_size: int,
        max_length: int,
        score_column: int | None,
        trust_remote_code: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.score_column = score_column
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        ).to(device)
        self.model.eval()
        self._torch = torch

    def _score_batch(self, logits) -> list[float]:
        if logits.ndim == 1:
            batch_scores = logits
        elif logits.shape[-1] == 1:
            batch_scores = logits[:, 0]
        else:
            col = (
                self.score_column
                if self.score_column is not None
                else logits.shape[-1] - 1
            )
            batch_scores = logits[:, col]
        return batch_scores.detach().cpu().float().tolist()

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        from tqdm import tqdm

        if not pairs:
            return []
        scores: list[float] = []
        batches = range(0, len(pairs), self.batch_size)
        with self._torch.no_grad():
            for start in tqdm(batches, desc="scoring batches", unit="batch"):
                batch = pairs[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs).logits
                scores.extend(self._score_batch(logits))
        return scores


class CrossEncoderBackend(RerankerBackend):
    def __init__(
        self,
        model_path: str,
        device: str,
        batch_size: int,
        max_length: int,
        trust_remote_code: bool,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if device == "cuda":
            model_kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
        self.model = CrossEncoder(
            model_path,
            device=device,
            max_length=max_length,
            **model_kwargs,
        )
        self.batch_size = batch_size

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        raw = self.model.predict(
            [list(p) for p in pairs],
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        if getattr(raw, "ndim", 1) == 2:
            return raw[:, 1].tolist() if raw.shape[1] > 1 else raw[:, 0].tolist()
        return [float(x) for x in raw]


class QwenRerankerBackend(RerankerBackend):
    def __init__(
        self,
        model_path: str,
        device: str,
        batch_size: int,
        max_length: int,
        reranker_instruction: str | None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.batch_size = batch_size
        self.instruction = reranker_instruction or _QWEN_DEFAULT_INSTRUCTION
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        load_kwargs: dict[str, Any] = {}
        if device == "cuda":
            load_kwargs["dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, **load_kwargs
        ).eval().to(device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query "
            'and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "\n"
            "<|im_start|>user\n"
        )
        suffix = "\n<|im_start|>assistant\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.max_content_len = max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        self._torch = torch

    def _format_pair(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        from tqdm import tqdm

        if not pairs:
            return []
        scores: list[float] = []
        batches = range(0, len(pairs), self.batch_size)
        with self._torch.no_grad():
            for start in tqdm(batches, desc="scoring batches", unit="batch"):
                batch = pairs[start : start + self.batch_size]
                texts = [self._format_pair(q, d) for q, d in batch]
                enc = self.tokenizer(
                    texts,
                    padding=False,
                    truncation=True,
                    max_length=self.max_content_len,
                    return_tensors=None,
                    return_attention_mask=False,
                )
                input_ids = [
                    self.prefix_tokens + ids + self.suffix_tokens
                    for ids in enc["input_ids"]
                ]
                inputs = self.tokenizer.pad(
                    {"input_ids": input_ids},
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits[:, -1, :]
                batch_scores = self._torch.nn.functional.log_softmax(
                    self._torch.stack(
                        [
                            logits[:, self.token_false_id],
                            logits[:, self.token_true_id],
                        ],
                        dim=1,
                    ),
                    dim=1,
                )[:, 1].exp()
                scores.extend(batch_scores.cpu().float().tolist())
        return scores


def resolve_jina_doc_batch_size(
    device: str, jina_doc_batch_size: int, fallback_batch_size: int
) -> int:
    if jina_doc_batch_size > 0:
        return jina_doc_batch_size
    if device == "mps":
        return min(fallback_batch_size, 2)
    if device == "cuda":
        return min(fallback_batch_size, 8)
    return min(fallback_batch_size, 4)


class JinaV3Backend(RerankerBackend):
    def __init__(
        self,
        model_path: str,
        device: str,
        trust_remote_code: bool,
        doc_batch_size: int,
        max_doc_chars: int,
    ) -> None:
        import torch
        from transformers import AutoModel

        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if device == "cuda":
            kwargs["dtype"] = torch.bfloat16
        self.model = AutoModel.from_pretrained(model_path, **kwargs).eval()
        if device != "cpu":
            self.model = self.model.to(device)
        self.device = device
        self.doc_batch_size = doc_batch_size
        self.max_doc_chars = max_doc_chars
        logging.info(
            "Jina v3: doc_batch_size=%d, max_doc_chars=%d",
            doc_batch_size,
            max_doc_chars,
        )

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch
        from collections import defaultdict

        if not pairs:
            return []

        # model.rerank is per-query, so group indices by query text
        query_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, (query, _) in enumerate(pairs):
            query_to_indices[query].append(i)

        scores = [0.0] * len(pairs)
        for query, indices in query_to_indices.items():
            docs = [pairs[i][1][: self.max_doc_chars] for i in indices]
            for start in range(0, len(docs), self.doc_batch_size):
                chunk = docs[start : start + self.doc_batch_size]
                chunk_indices = indices[start : start + self.doc_batch_size]
                results = self.model.rerank(query, chunk, top_n=len(chunk))
                score_map = {r["document"]: r["relevance_score"] for r in results}
                for doc, idx in zip(chunk, chunk_indices):
                    scores[idx] = float(score_map.get(doc, 0.0))
                if self.device == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
        return scores


def build_backend(
    model_path: str,
    model_type: str,
    device: str,
    batch_size: int,
    max_length: int,
    score_column: int | None,
    trust_remote_code: bool,
    reranker_instruction: str | None,
    jina_doc_batch_size: int = 0,
) -> RerankerBackend:
    resolved = resolve_model_type(model_path, model_type)
    logging.info("Using reranker backend: %s", resolved)

    if resolved == "seq-classification":
        return SeqClassificationBackend(
            model_path, device, batch_size, max_length, score_column, trust_remote_code
        )
    if resolved == "cross-encoder":
        return CrossEncoderBackend(
            model_path, device, batch_size, max_length, trust_remote_code
        )
    if resolved == "qwen-reranker":
        return QwenRerankerBackend(
            model_path, device, batch_size, max_length, reranker_instruction
        )
    if resolved == "jina-v3":
        doc_batch = resolve_jina_doc_batch_size(device, jina_doc_batch_size, batch_size)
        max_doc_chars = max(512, max_length * 4)
        return JinaV3Backend(
            model_path,
            device,
            trust_remote_code=True,
            doc_batch_size=doc_batch,
            max_doc_chars=max_doc_chars,
        )
    raise ValueError(f"Unsupported model type: {resolved}")


def load_results(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_results(path: Path, results: dict[str, dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def load_dataset_texts(
    dataset_name: str,
    dataset_args: tuple[str, str, str],
    offline_assets_dir: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    _corpus_repo, _qrels_repo, split = dataset_args
    if not assets_exist(offline_assets_dir, dataset_name, split):
        raise FileNotFoundError(
            f"Missing offline assets for {dataset_name}_{split} under "
            f"{offline_assets_dir}. Run prepare_offline_bundle.sh once locally."
        )
    return (
        load_corpus(offline_assets_dir, dataset_name, split),
        load_queries(offline_assets_dir, dataset_name, split),
    )


def rerank_dataset(
    dataset_name: str,
    dataset_args: tuple[str, str, str],
    input_dir: Path,
    output_dir: Path,
    backend: RerankerBackend,
    top_k: int,
    text_type: str,
    overwrite: bool,
    offline_assets_dir: Path,
    max_queries: int | None = None,
) -> None:
    _corpus_repo, _qrels_repo, split = dataset_args
    input_file = input_dir / f"results_{dataset_name}_{split}.json"
    output_file = output_dir / f"results_{dataset_name}_{split}_reranked.json"

    if not input_file.exists():
        raise FileNotFoundError(f"Missing first-stage results: {input_file}")
    if output_file.exists() and not overwrite:
        logging.info("Skipping existing %s", output_file.name)
        return

    logging.info("Loading dataset %s (%s)", dataset_name, split)
    corpus, queries = load_dataset_texts(
        dataset_name, dataset_args, offline_assets_dir
    )
    first_stage = load_results(input_file)
    query_items = list(first_stage.items())
    if max_queries is not None:
        query_items = query_items[:max_queries]
        logging.info(
            "Limited to first %d queries (of %d) for %s",
            len(query_items),
            len(first_stage),
            dataset_name,
        )

    # Collect all (query, doc) pairs across every query up front so the backend
    # can fill its full batch_size across query boundaries instead of being
    # capped at top_k (≤100) pairs per forward pass.
    all_query_ids: list[str] = []
    all_doc_id_lists: list[list[str]] = []
    all_pairs: list[tuple[str, str]] = []

    for query_id, candidates in query_items:
        items = sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:top_k]
        doc_ids = [doc_id for doc_id, _ in items]
        query_text = queries[query_id]
        for doc_id in doc_ids:
            doc = corpus[doc_id]
            all_pairs.append(
                (query_text, (doc.get("title", "") + " " + doc.get("text", "")).strip())
            )
        all_query_ids.append(query_id)
        all_doc_id_lists.append(doc_ids)

    logging.info(
        "Scoring %d pairs across %d queries for %s",
        len(all_pairs), len(all_query_ids), dataset_name,
    )
    all_scores = backend.score_pairs(all_pairs)

    reranked: dict[str, dict[str, float]] = {}
    offset = 0
    for query_id, doc_ids in zip(all_query_ids, all_doc_id_lists):
        n = len(doc_ids)
        reranked[query_id] = {
            doc_id: float(score)
            for doc_id, score in zip(doc_ids, all_scores[offset : offset + n])
        }
        offset += n

    write_results(output_file, reranked)
    logging.info("Wrote %s", output_file)


def evaluate_outputs(
    datasets: dict,
    output_dir: Path,
    metrics_json: Path | None,
    offline_assets_dir: Path,
) -> None:
    from rusBeIR.beir.retrieval.evaluation import EvaluateRetrieval

    retriever = EvaluateRetrieval(k_values=DEFAULT_K_VALUES)
    metric_sums = {
        "ndcg": {f"NDCG@{k}": 0.0 for k in DEFAULT_K_VALUES},
        "map": {f"MAP@{k}": 0.0 for k in DEFAULT_K_VALUES},
        "recall": {f"Recall@{k}": 0.0 for k in DEFAULT_K_VALUES},
        "precision": {f"P@{k}": 0.0 for k in DEFAULT_K_VALUES},
        "mrr": {f"MRR@{k}": 0.0 for k in DEFAULT_K_VALUES},
    }
    per_dataset: dict[str, dict] = {}
    processed = 0

    for dataset_name, dataset_args in datasets.items():
        split = dataset_args[2]
        result_file = (
            output_dir / f"results_{dataset_name}_{split}_reranked.json"
        )
        if not result_file.exists():
            logging.warning("No reranked file for %s", dataset_name)
            continue

        with result_file.open("r", encoding="utf-8") as f:
            results = json.load(f)

        qrels = load_qrels(offline_assets_dir, dataset_name, split)

        ndcg, _map, recall, precision = retriever.evaluate(
            qrels=qrels, results=results, k_values=DEFAULT_K_VALUES
        )
        mrr = retriever.evaluate_custom(
            qrels, results, DEFAULT_K_VALUES, "mrr"
        )

        dataset_key = f"{dataset_name}_{split}"
        per_dataset[dataset_key] = {
            "dataset": dataset_name,
            "split": split,
            "NDCG": dict(ndcg),
            "MAP": dict(_map),
            "Recall": dict(recall),
            "Precision": dict(precision),
            "MRR": dict(mrr),
        }

        for k in DEFAULT_K_VALUES:
            metric_sums["ndcg"][f"NDCG@{k}"] += ndcg[f"NDCG@{k}"]
            metric_sums["map"][f"MAP@{k}"] += _map[f"MAP@{k}"]
            metric_sums["recall"][f"Recall@{k}"] += recall[f"Recall@{k}"]
            metric_sums["precision"][f"P@{k}"] += precision[f"P@{k}"]
            metric_sums["mrr"][f"MRR@{k}"] += mrr[f"MRR@{k}"]
        processed += 1

        logging.info(
            "Metrics %s: NDCG@10=%.4f Recall@10=%.4f MRR@10=%.4f",
            dataset_key,
            ndcg["NDCG@10"],
            recall["Recall@10"],
            mrr["MRR@10"],
        )

    if processed == 0:
        logging.error("No datasets evaluated.")
        return

    metrics_results = {
        name: {k: metric_sums[name][k] / processed for k in metric_sums[name]}
        for name in ("ndcg", "map", "recall", "precision", "mrr")
    }
    for metric, values in {
        "NDCG": metrics_results["ndcg"],
        "MAP": metrics_results["map"],
        "Recall": metrics_results["recall"],
        "Precision": metrics_results["precision"],
        "MRR": metrics_results["mrr"],
    }.items():
        print(f"Average {metric}:")
        for k, val in values.items():
            print(f"{k}: {val}")
        print()

    if metrics_json is not None:
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "NDCG": metrics_results["ndcg"],
            "MAP": metrics_results["map"],
            "Recall": metrics_results["recall"],
            "Precision": metrics_results["precision"],
            "MRR": metrics_results["mrr"],
            "average": {
                "NDCG": metrics_results["ndcg"],
                "MAP": metrics_results["map"],
                "Recall": metrics_results["recall"],
                "Precision": metrics_results["precision"],
                "MRR": metrics_results["mrr"],
                "num_datasets": processed,
            },
            "per_dataset": per_dataset,
        }
        with metrics_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
        logging.info("Wrote metrics to %s", metrics_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank saved RusBEIR first-stage results and evaluate metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--model-type",
        default="cross-encoder",
        choices=MODEL_TYPES,
        help=(
            "Reranker backend (default: cross-encoder via sentence-transformers). "
            "Use auto to detect Jina/Qwen; seq-classification for raw HF logits only."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default="results/results-orig/rusBeIR-bm25-top100-results",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    parser.add_argument("--score-column", type=int, default=None)
    parser.add_argument("--text-type", default="text")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(CORE_IR_QA_DATASETS),
    )
    parser.add_argument(
        "--reranker-instruction",
        default=None,
        help="[qwen-reranker] Task instruction in the prompt.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only compute metrics from existing *_reranked.json (skip model load/rerank).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Rerank only the first N queries per dataset (for smoke tests).",
    )
    parser.add_argument(
        "--offline-assets-dir",
        default=str(DEFAULT_OFFLINE_ASSETS_DIR),
        help=(
            "Offline bundle from prepare_rusbeir_offline_bundle.py "
            "(queries.json, corpus.json, qrels.json per dataset). Required."
        ),
    )
    parser.add_argument(
        "--jina-doc-batch-size",
        type=int,
        default=0,
        help=(
            "Jina v3: passages scored per forward pass (0 = auto: 2 on mps, 8 on cuda). "
            "Lower this if you hit MPS/CUDA OOM."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    input_dir = Path(args.input_dir)
    offline_assets_dir = Path(args.offline_assets_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else default_output_dir(input_dir, args.model_path)
    )
    metrics_json = (
        Path(args.metrics_json)
        if args.metrics_json is not None
        else output_dir / "metrics.json"
    )
    datasets = selected_datasets(args.datasets)
    device = resolve_device(args.device)

    logging.info("Model path: %s", args.model_path)
    logging.info("Device: %s", device)
    logging.info("Input directory: %s", input_dir)
    logging.info("Output directory: %s", output_dir)
    logging.info("Selected datasets: %s", ", ".join(datasets))
    logging.info("Offline assets: %s", offline_assets_dir)
    validate_offline_bundle(offline_assets_dir, datasets)
    if args.max_queries is not None:
        logging.info("Max queries per dataset: %d", args.max_queries)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.evaluate_only:
        backend = build_backend(
            model_path=args.model_path,
            model_type=args.model_type,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            score_column=args.score_column,
            trust_remote_code=args.trust_remote_code,
            reranker_instruction=args.reranker_instruction,
            jina_doc_batch_size=args.jina_doc_batch_size,
        )

        for dataset_name, dataset_args in datasets.items():
            rerank_dataset(
                dataset_name=dataset_name,
                dataset_args=dataset_args,
                input_dir=input_dir,
                output_dir=output_dir,
                backend=backend,
                top_k=args.top_k,
                text_type=args.text_type,
                overwrite=args.overwrite,
                max_queries=args.max_queries,
                offline_assets_dir=offline_assets_dir,
            )

    if not args.no_evaluate:
        evaluate_outputs(
            datasets, output_dir, metrics_json, offline_assets_dir=offline_assets_dir
        )


if __name__ == "__main__":
    main()
