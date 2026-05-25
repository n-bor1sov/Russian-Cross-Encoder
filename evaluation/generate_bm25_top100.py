#!/usr/bin/env python3
"""Generate portable BM25 top-100 candidates for selected RusBEIR datasets.

The generated JSON files can be copied to another machine and used for
cross-encoder reranking without Elasticsearch.
"""

from __future__ import annotations

import argparse
import json
import logging
import tomllib
from pathlib import Path
from typing import Iterable


TOP_K = 100
DEFAULT_K_VALUES = [1, 3, 5, 10, TOP_K]

# Core IR + QA subset selected for textual query-document reranker evaluation.
CORE_IR_QA_DATASETS = {
    "rus-nfcorpus": ("kaengreg/rus-nfcorpus", "kaengreg/rus-nfcorpus-qrels", "test"),
    "rus-scidocs": ("kaengreg/rus-scidocs", "kaengreg/rus-scidocs-qrels", "test"),
    "rus-trec-covid": ("kaengreg/rus-trec-covid", "kaengreg/rus-trec-covid-qrels", "test"),
    "sberquad-retrieval": (
        "kaengreg/sberquad-retrieval",
        "kaengreg/sberquad-retrieval-qrels",
        "validation",
    ),
    "ruscibench-retrieval": (
        "kaengreg/ruSciBench-retrieval",
        "kaengreg/ruSciBench-retrieval-qrels",
        "dev",
    ),
    "rubq": ("kaengreg/rubq", "kaengreg/rubq-qrels", "test"),
    "ria-news": ("kaengreg/ria-news", "kaengreg/ria-news-qrels", "test"),
    "wikifacts-articles": (
        "kaengreg/wikifacts-articles",
        "kaengreg/wikifacts-articles-qrels",
        "dev",
    ),
    "wikifacts-sents": (
        "kaengreg/wikifacts-sents",
        "kaengreg/wikifacts-sents-qrels",
        "dev",
    ),
    "rus-fiqa": ("kaengreg/rus-fiqa", "kaengreg/rus-fiqa-qrels", "dev"),
    "rus-xquad": ("kaengreg/rus-xquad", "kaengreg/rus-xquad-qrels", "dev"),
    "rus-xquad-sentenes": (
        "kaengreg/rus-xquad-sentences",
        "kaengreg/rus-xquad-sentences-qrels",
        "dev",
    ),
}


def selected_datasets(names: Iterable[str]) -> dict:
    missing = [name for name in names if name not in CORE_IR_QA_DATASETS]
    if missing:
        raise KeyError(f"Unknown configured dataset key(s): {', '.join(missing)}")
    return {name: CORE_IR_QA_DATASETS[name] for name in names}


def expected_result_files(datasets: dict, output_dir: Path) -> list[Path]:
    return [
        output_dir / f"results_{dataset_name}_{dataset_args[2]}.json"
        for dataset_name, dataset_args in datasets.items()
    ]


def truncate_results_to_top_k(
    datasets: dict, output_dir: Path, top_k: int
) -> int:
    """Trim each query's candidate list to top_k by BM25 score (descending)."""
    repaired_queries = 0

    for result_file in expected_result_files(datasets, output_dir):
        if not result_file.exists():
            continue

        with result_file.open("r", encoding="utf-8") as f:
            results = json.load(f)

        file_changed = False
        for query_id, candidates in results.items():
            if len(candidates) <= top_k:
                continue
            results[query_id] = dict(
                sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:top_k]
            )
            repaired_queries += 1
            file_changed = True

        if file_changed:
            with result_file.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logging.info("Repaired %s", result_file.name)

    return repaired_queries


def validate_results(datasets: dict, output_dir: Path, top_k: int) -> None:
    missing_files = []
    empty_files = []
    oversized_queries = []
    total_queries = 0

    for result_file in expected_result_files(datasets, output_dir):
        if not result_file.exists():
            missing_files.append(result_file)
            continue

        with result_file.open("r", encoding="utf-8") as f:
            results = json.load(f)

        if not results:
            empty_files.append(result_file)
            continue

        total_queries += len(results)
        for query_id, candidates in results.items():
            if len(candidates) > top_k:
                oversized_queries.append((result_file, query_id, len(candidates)))

    if missing_files or empty_files or oversized_queries:
        message_parts = []
        if missing_files:
            message_parts.append(
                "Missing result files:\n"
                + "\n".join(f"  - {path}" for path in missing_files)
            )
        if empty_files:
            message_parts.append(
                "Empty result files:\n"
                + "\n".join(f"  - {path}" for path in empty_files)
            )
        if oversized_queries:
            message_parts.append(
                "Queries with more than top_k candidates:\n"
                + "\n".join(
                    f"  - {path.name}: query_id={query_id}, candidates={count}"
                    for path, query_id, count in oversized_queries[:20]
                )
            )
        raise RuntimeError("\n\n".join(message_parts))

    logging.info(
        "Validation passed: %d files, %d queries, each query has <= %d candidates.",
        len(expected_result_files(datasets, output_dir)),
        total_queries,
        top_k,
    )


def generate_candidates(args: argparse.Namespace, datasets: dict, output_dir: Path) -> None:
    from rusBeIR.beir.retrieval.search.lexical import BM25Search as BM25
    from rusBeIR.benchmarking.model_benchmark import DatasetEvaluator

    class FreshIndexBM25(BM25):
        """BM25 wrapper that rebuilds Elasticsearch state for each dataset."""

        def search(self, corpus, queries, top_k, *search_args, **search_kwargs):
            self.results = {}
            if self.initialize:
                self.initialise()
            return super().search(corpus, queries, top_k, *search_args, **search_kwargs)

    bm25 = FreshIndexBM25(
        index_name=args.index_name,
        hostname=args.hostname,
        language=args.language,
        batch_size=args.batch_size,
        initialize=True,
        sleep_for=args.sleep_for,
        username=args.username,
        password=args.password,
    )
    evaluator = DatasetEvaluator(model=bm25, k_values=DEFAULT_K_VALUES)
    evaluator.datasets = datasets
    evaluator.retrieve(text_type=args.text_type, results_path=output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BM25 top-100 RusBEIR candidates for reranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="results/results-orig/rusBeIR-bm25-top100-results",
        help="Directory where result JSON files will be written.",
    )
    parser.add_argument(
        "--hostname",
        default="https://localhost:9200",
        help="Elasticsearch hostname used only during BM25 candidate generation.",
    )
    parser.add_argument(
        "--index-name",
        default="rusbeir-bm25-top100",
        help="Temporary Elasticsearch index name. It is rebuilt for each dataset.",
    )
    parser.add_argument("--username", default="elastic", help="Elasticsearch username.")
    parser.add_argument("--password", default="rusbeir", help="Elasticsearch password.")
    parser.add_argument(
        "--language",
        default="russian",
        help="Elasticsearch analyzer language setting passed to RusBEIR BM25.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="BM25 query batch size for Elasticsearch multisearch.",
    )
    parser.add_argument(
        "--sleep-for",
        type=int,
        default=2,
        help="Seconds to wait after Elasticsearch index operations.",
    )
    parser.add_argument(
        "--text-type",
        default="text",
        help="Corpus text field loaded from Hugging Face datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(CORE_IR_QA_DATASETS),
        help="DatasetEvaluator keys to process.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing output files; do not generate candidates.",
    )
    parser.add_argument(
        "--repair-top-k",
        action="store_true",
        help=(
            "Trim existing result files to top_k candidates per query by score, "
            "then validate. Does not require Elasticsearch."
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after candidate generation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML config file (configs/thesis/eval.toml). CLI flags override TOML values.",
    )
    args = parser.parse_args()

    if args.config is not None:
        _cfg = tomllib.loads(args.config.read_text())
        for _k, _v in _cfg.items():
            if getattr(args, _k, "_missing_sentinel") is None:
                setattr(args, _k, _v)

    return args


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    output_dir = Path(args.output_dir)
    datasets = selected_datasets(args.datasets)

    logging.info("Selected datasets: %s", ", ".join(datasets))
    logging.info("Output directory: %s", output_dir)

    if not args.validate_only and not args.repair_top_k:
        generate_candidates(args, datasets, output_dir)

    if args.repair_top_k:
        repaired = truncate_results_to_top_k(datasets, output_dir, TOP_K)
        logging.info("Trimmed %d oversized query result lists to top %d.", repaired, TOP_K)

    if not args.no_validate:
        validate_results(datasets, output_dir, TOP_K)


if __name__ == "__main__":
    main()
