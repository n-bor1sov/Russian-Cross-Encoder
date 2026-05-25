from __future__ import annotations

import os
import argparse
import glob
from collections import defaultdict

import networkx as nx
import numpy as np
from datasets import DatasetDict, load_dataset


def read_all_datasets_from_directory(directory_path: str):
    parquet_files = sorted(glob.glob(os.path.join(directory_path, "*.parquet")))
    datasets = [load_dataset("parquet", data_files=pq_file) for pq_file in parquet_files]
    return datasets, parquet_files


def component_split_bipartite(
    ds,
    query_id_col="query_id",
    passage_id_col="passage_id",
    lang_col="lang",
    val_size=0.04,
    seed=42,
):
    """
    Split dataset by query_id and passage_id components, ensuring all languages are preserved.

    Only train and validation; remaining pairs (~1 - val_size) go to train.
    """
    assert 0.0 < val_size < 1.0

    pair_to_langs = defaultdict(set)
    pair_to_rows = defaultdict(list)

    n = len(ds)
    for i in range(n):
        query_id = ds[i][query_id_col]
        passage_id = ds[i][passage_id_col]
        lang = ds[i].get(lang_col, None) if lang_col else None

        pair = (query_id, passage_id)
        pair_to_langs[pair].add(lang)
        pair_to_rows[pair].append(i)

    all_langs = set()
    for i in range(n):
        lang = ds[i].get(lang_col, None) if lang_col else None
        if lang is not None:
            all_langs.add(lang)

    if lang_col and all_langs:
        valid_pairs = []
        filtered_count = 0

        for pair, langs in pair_to_langs.items():
            if langs == all_langs:
                valid_pairs.append(pair)
            else:
                filtered_count += len(pair_to_rows[pair])

        print(f"Filtered out {filtered_count} rows with missing languages")
        print(f"Kept {len(valid_pairs)} (query_id, passage_id) pairs present in all languages")
    else:
        valid_pairs = list(pair_to_langs.keys())
        print(f"No language filtering applied (lang_col={lang_col}, all_langs={all_langs})")
        print(f"Processing {len(valid_pairs)} (query_id, passage_id) pairs")

    G = nx.Graph()

    for pair in valid_pairs:
        query_id = pair[0]
        passage_id = pair[1]
        qn = ("q", query_id)
        pn = ("p", passage_id)
        G.add_node(qn, bipartite=0)
        G.add_node(pn, bipartite=1)
        G.add_edge(qn, pn, pair=pair)

    comps = list(nx.connected_components(G))

    comp_pairs = []
    for nodes in comps:
        sub = G.subgraph(nodes)
        pairs = [data["pair"] for _, _, data in sub.edges(data=True)]
        if pairs:
            comp_pairs.append(pairs)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(comp_pairs)).tolist()
    comp_pairs = [comp_pairs[i] for i in order]

    total_pairs = sum(len(p) for p in comp_pairs)
    val_target = int(round(val_size * total_pairs))
    val_target = min(max(val_target, 0), total_pairs)
    train_target = total_pairs - val_target

    train_pairs, val_pairs = set(), set()
    cur = 0

    for pairs in comp_pairs:
        if cur < train_target:
            train_pairs.update(pairs)
        else:
            val_pairs.update(pairs)
        cur += len(pairs)

    train_idx, val_idx = [], []

    for pair in train_pairs:
        train_idx.extend(pair_to_rows[pair])
    for pair in val_pairs:
        val_idx.extend(pair_to_rows[pair])

    train_idx = sorted(set(train_idx))
    val_idx = sorted(set(val_idx))

    return DatasetDict({
        "train": ds.select(train_idx),
        "validation": ds.select(val_idx),
    })


def filter_single_language_pairs(
    ds,
    query_id_col="query_id",
    passage_id_col="passage_id",
    lang_col="lang",
):
    """Строки только для пар (query_id, passage_id), которые встречаются ровно в одном (непустом) языке."""
    pair_to_langs = defaultdict(set)
    pair_to_rows = defaultdict(list)

    n = len(ds)
    for i in range(n):
        query_id = ds[i][query_id_col]
        passage_id = ds[i][passage_id_col]
        lang = ds[i].get(lang_col, None) if lang_col else None

        pair = (query_id, passage_id)
        pair_to_langs[pair].add(lang)
        pair_to_rows[pair].append(i)

    if lang_col:
        single_lang_pairs = []
        filtered_count = 0

        for pair, langs in pair_to_langs.items():
            langs_without_none = {l for l in langs if l is not None}
            if len(langs_without_none) == 1:
                single_lang_pairs.append(pair)
            else:
                filtered_count += len(pair_to_rows[pair])

        print(f"Filtered out {filtered_count} rows from pairs with multiple languages")
        print(f"Kept {len(single_lang_pairs)} (query_id, passage_id) pairs present in exactly one language")
        valid_pairs = single_lang_pairs
    else:
        valid_pairs = list(pair_to_langs.keys())
        print(f"No language filtering applied (lang_col={lang_col})")
        print(f"Processing {len(valid_pairs)} (query_id, passage_id) pairs")

    filtered_idx = []
    for pair in valid_pairs:
        filtered_idx.extend(pair_to_rows[pair])

    filtered_idx = sorted(set(filtered_idx))

    return ds.select(filtered_idx)


def save_splits(dataset_dict: DatasetDict, output_dir: str, basename: str) -> None:
    for split_name, split_dataset in dataset_dict.items():
        split_subdir = os.path.join(output_dir, split_name)
        os.makedirs(split_subdir, exist_ok=True)
        output_path = os.path.join(split_subdir, basename)
        split_dataset.to_parquet(output_path)
        print(f"Saved split ({len(split_dataset)} rows) to {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        default="../nik_datasets/parquets/cleaned_datasets/no_split/",
        help="Каталог с исходными *.parquet",
    )
    p.add_argument(
        "--output-dir",
        default="../nik_datasets/parquets/cleaned_datasets/splits/",
        help="Куда писать train/validation/",
    )
    p.add_argument("--val-size", type=float, default=0.04, help="Доля пар на validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--extra-single-lang-source",
        default="ru_sci_bench_merged.parquet",
        help="Basename файла из input-dir для одноязычного хвоста; пустая строка — пропустить",
    )
    p.add_argument(
        "--extra-single-lang-output",
        default="ru_sci_bench_ru.parquet",
        help="Имя выходного parquet для одноязычного сплита",
    )
    return p.parse_args()


def main():
    args = parse_args()

    datasets, parquet_files = read_all_datasets_from_directory(args.input_dir)
    print(f"Loaded {len(datasets)} dataset(s) from {args.input_dir!r}")

    total_rows = 0
    for dataset, path in zip(datasets, parquet_files):
        n = len(dataset["train"])
        total_rows += n
        print(n, os.path.basename(path))
    print("Total:", total_rows)

    os.makedirs(args.output_dir, exist_ok=True)

    new_datasets = []
    for dataset in datasets:
        new_datasets.append(
            component_split_bipartite(dataset["train"], val_size=args.val_size, seed=args.seed)
        )

    for dataset_dict, original_path in zip(new_datasets, parquet_files):
        save_splits(dataset_dict, args.output_dir, os.path.basename(original_path))

    if args.extra_single_lang_source:
        src_base = args.extra_single_lang_source
        match_idx = None
        for i, path in enumerate(parquet_files):
            if os.path.basename(path) == src_base:
                match_idx = i
                break
        if match_idx is None:
            print(f"Skipping extra single-lang split: no file named {src_base!r} in input list")
        else:
            ru_sci = filter_single_language_pairs(datasets[match_idx]["train"])
            ru_sci_splits = component_split_bipartite(ru_sci, lang_col=None, val_size=args.val_size, seed=args.seed)
            save_splits(ru_sci_splits, args.output_dir, args.extra_single_lang_output)


if __name__ == "__main__":
    main()

# nohup python split_datasets.py  > split_datasets.log 2>&1 &