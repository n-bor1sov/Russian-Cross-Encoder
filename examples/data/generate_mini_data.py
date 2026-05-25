#!/usr/bin/env python3
"""Generate synthetic mini-corpus for examples/run.sh.

Reproducible — fixed seed; outputs do not depend on environment.
Designed to be small (~100 query-passage pairs total) and have enough
structure that the training stage produces a meaningful loss curve.
"""

from __future__ import annotations

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SEED = 0xC0FFEE


def _pair_row(qid: int, pid: int, lang: str, dataset: str) -> dict:
    return {
        "query": f"query-{dataset}-{qid}-{lang}",
        "passage": f"passage-{dataset}-{pid}-{lang}",
        "query_id": f"{dataset}:q{qid}:{lang}",
        "passage_id": f"{dataset}:p{pid}:{lang}",
        "lang": lang,
        "dataset": dataset,
    }


def write_mini_dataset_a(out: Path) -> None:
    rows = []
    for qid in range(25):
        for lang in ("ru", "en"):
            rows.append(_pair_row(qid, qid, lang, "mini_a"))
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out, compression="snappy")


def write_mini_dataset_b(out: Path) -> None:
    rows = [_pair_row(qid, qid, "ru", "mini_b") for qid in range(50)]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out, compression="snappy")


def write_mini_compiled(out: Path) -> None:
    """Pre-compiled (post-stage-5) training parquet.

    Schema: query, positive, negative_1 ... negative_8.
    Negatives are deterministic non-positives from the same synthetic pool.
    """
    rng = random.Random(SEED)
    queries = []
    for dataset in ("mini_a", "mini_b"):
        for qid in range(20):
            lang = "ru"
            row = {
                "query": f"query-{dataset}-{qid}-{lang}",
                "positive": f"passage-{dataset}-{qid}-{lang}",
            }
            negatives_pool = [
                f"passage-{dataset}-{j}-{lang}" for j in range(100) if j != qid
            ]
            rng.shuffle(negatives_pool)
            for i, neg in enumerate(negatives_pool[:8], start=1):
                row[f"negative_{i}"] = neg
            queries.append(row)
    table = pa.Table.from_pylist(queries)
    pq.write_table(table, out, compression="snappy")


def main() -> None:
    out_dir = Path(__file__).parent
    write_mini_dataset_a(out_dir / "mini_dataset_a.parquet")
    write_mini_dataset_b(out_dir / "mini_dataset_b.parquet")
    write_mini_compiled(out_dir / "mini_compiled.parquet")
    print("Wrote mini_dataset_a.parquet, mini_dataset_b.parquet, mini_compiled.parquet")


if __name__ == "__main__":
    main()
