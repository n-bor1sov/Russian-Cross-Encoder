import argparse
import random
import re
from pathlib import Path

import pyarrow.parquet as pq


NEGATIVE_COLUMNS = [f"negative{i}" for i in range(1, 9)]
TEXT_COLUMNS = ["query_id", "dataset", "query", "positive", *NEGATIVE_COLUMNS]


def clean(value) -> str:
    return "" if value is None else str(value).replace("\n", " ").strip()


def short(value, limit: int) -> str:
    text = clean(value)
    return text[:limit] + ("..." if len(text) > limit else "")


def tokens(value) -> set[str]:
    return set(re.findall(r"[\wА-Яа-яЁё]+", clean(value).lower()))


def jaccard(left, right) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def show_row(title: str, row: dict, *, extra: str = "", text_limit: int = 260) -> None:
    suffix = f" {extra}" if extra else ""
    print(f"\n=== {title}{suffix} ===")
    print(f"query_id={row.get('query_id')!r} dataset={row.get('dataset')!r}")
    print(f"query:    {short(row.get('query'), 300)}")
    print(f"positive: {short(row.get('positive'), 360)}")
    for idx, column in enumerate(NEGATIVE_COLUMNS, start=1):
        print(f"neg{idx}:     {short(row.get(column), text_limit)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect final cross-encoder parquet examples.")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to final parquet, e.g. final_dataset_parts/filtered_top_30/train/mmarco.parquet",
    )
    parser.add_argument("--examples", type=int, default=8, help="Number of examples to print per section.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random examples.")
    parser.add_argument("--text-limit", type=int, default=260, help="Printed length for negatives.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20_000,
        help="Maximum rows to inspect. Keeps the script fast on large parquet files.",
    )
    parser.add_argument(
        "--row-groups",
        type=int,
        default=4,
        help="Maximum row groups to sample from the parquet file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print(f"Reading final dataset: {args.path}")
    parquet_file = pq.ParquetFile(args.path)
    print(f"Rows: {parquet_file.metadata.num_rows:,}")
    print(f"Row groups: {parquet_file.metadata.num_row_groups:,}")
    print("Schema:")
    print(parquet_file.schema)

    n_row_groups = parquet_file.metadata.num_row_groups
    if n_row_groups <= args.row_groups:
        sampled_row_groups = list(range(n_row_groups))
    else:
        sampled_row_groups = sorted(random.sample(range(n_row_groups), args.row_groups))

    rows = []
    dataset_counts: dict[str, int] = {}
    for row_group in sampled_row_groups:
        if len(rows) >= args.max_rows:
            break
        table = parquet_file.read_row_group(row_group, columns=TEXT_COLUMNS)
        remaining = args.max_rows - len(rows)
        if table.num_rows > remaining:
            table = table.slice(0, remaining)

        batch_rows = table.to_pylist()
        rows.extend(batch_rows)
        for row in batch_rows:
            dataset = clean(row.get("dataset"))
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    print(
        f"\nLoaded sampled rows: {len(rows):,} "
        f"from row groups {sampled_row_groups} (max_rows={args.max_rows:,})"
    )

    print("\nDataset counts in sample:")
    for dataset, count in sorted(dataset_counts.items(), key=lambda item: item[1], reverse=True)[:20]:
        print(f"  {dataset!r}: {count:,}")

    empty_counts = {column: 0 for column in TEXT_COLUMNS}
    duplicate_negative_rows: list[tuple[int, dict]] = []
    positive_equals_negative_rows: list[tuple[int, dict]] = []
    suspicious_rows: list[tuple[tuple[float, int, float, float], int, dict]] = []

    for row_idx, row in enumerate(rows):
        for column in TEXT_COLUMNS:
            if not clean(row.get(column)):
                empty_counts[column] += 1

        query = clean(row.get("query"))
        positive = clean(row.get("positive"))
        negatives = [clean(row.get(column)) for column in NEGATIVE_COLUMNS]
        non_empty_negatives = [negative for negative in negatives if negative]

        if len(set(non_empty_negatives)) < len(non_empty_negatives):
            duplicate_negative_rows.append((row_idx, row))

        if positive and any(positive == negative for negative in non_empty_negatives):
            positive_equals_negative_rows.append((row_idx, row))

        best = (-1.0, 0, 0.0, 0.0)
        for neg_idx, negative in enumerate(negatives, start=1):
            if not negative:
                continue
            positive_negative_overlap = jaccard(positive, negative)
            query_negative_overlap = jaccard(query, negative)
            score = max(positive_negative_overlap, query_negative_overlap)
            if score > best[0]:
                best = (score, neg_idx, positive_negative_overlap, query_negative_overlap)

        suspicious_rows.append((best, row_idx, row))

    print("\nEmpty field counts:")
    printed_empty = False
    for column, count in empty_counts.items():
        if count:
            printed_empty = True
            print(f"  {column}: {count:,}")
    if not printed_empty:
        print("  none")

    print(f"\nRows with duplicate negatives: {len(duplicate_negative_rows):,}")
    print(f"Rows where positive exactly equals a negative: {len(positive_equals_negative_rows):,}")

    print("\n######## RANDOM EXAMPLES ########")
    for row_idx in random.sample(range(len(rows)), min(args.examples, len(rows))):
        show_row("Random row", rows[row_idx], extra=f"idx={row_idx}", text_limit=args.text_limit)

    print("\n######## DUPLICATE NEGATIVES EXAMPLES ########")
    for row_idx, row in duplicate_negative_rows[: args.examples]:
        show_row("Duplicate negatives", row, extra=f"idx={row_idx}", text_limit=args.text_limit)

    print("\n######## POSITIVE == NEGATIVE EXAMPLES ########")
    for row_idx, row in positive_equals_negative_rows[: args.examples]:
        show_row("Positive equals negative", row, extra=f"idx={row_idx}", text_limit=args.text_limit)

    print("\n######## MOST SUSPICIOUS LEXICAL OVERLAP ########")
    for best, row_idx, row in sorted(suspicious_rows, reverse=True)[: args.examples]:
        score, neg_idx, positive_negative_overlap, query_negative_overlap = best
        show_row(
            "Suspicious overlap",
            row,
            extra=(
                f"idx={row_idx} best_negative=negative{neg_idx} "
                f"score={score:.3f} "
                f"pos_neg_jaccard={positive_negative_overlap:.3f} "
                f"query_neg_jaccard={query_negative_overlap:.3f}"
            ),
            text_limit=args.text_limit,
        )


if __name__ == "__main__":
    main()
