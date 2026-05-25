import os
import gc
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import load_dataset as _load_dataset
from tqdm.auto import tqdm

try:
    import orjson
    def _json_load(f):
        return orjson.loads(f.read())
except ImportError:
    _json_load = json.load





def load_hard_negatives_for_dataset(dataset_name, negatives_folder):
    """Load hard negatives for a single dataset only."""
    candidates = [
        os.path.join(negatives_folder, f"hard_negatives_{dataset_name}.json"),
        os.path.join(negatives_folder, f"{dataset_name}.json"),
    ]

    for file_path in candidates:
        if os.path.exists(file_path):
            print(f"  Loading hard negatives: {os.path.basename(file_path)}")
            with open(file_path, "rb") as f:
                return _json_load(f)

    print(f"  Warning: Hard negatives file not found for {dataset_name}")
    return []


# Paths are relative to a root that you set via FILTERING_DATA_ROOT and HNM_ROOT env vars,
# or as absolute paths. Update these to match your actual data layout.
_FILTERING_ROOT = os.environ.get("FILTERING_DATA_ROOT", "/path/to/consistency_filtering_data")
_HNM_ROOT = os.environ.get("HNM_ROOT", "/path/to/hnm/mined_negatives")

SPLIT_CONFIGS = {
    "train": {
        "filtering_name": "filtered_top_5",
        "hard_negatives_input_folder": os.path.join(_HNM_ROOT, "train/filtered_top_5"),
        "datasets_paths": [
            ["mmarco", os.path.join(_FILTERING_ROOT, "filtered_top_5/mmarco.parquet")],
            ["9111_questions", os.path.join(_FILTERING_ROOT, "filtered_top_5/9111_questions.parquet")],
            ["habr_qna", os.path.join(_FILTERING_ROOT, "filtered_top_5/habr_qna.parquet")],
            ["nq", os.path.join(_FILTERING_ROOT, "filtered_top_5/nq.parquet")],
            ["ru_news", os.path.join(_FILTERING_ROOT, "filtered_top_5/ru_news.parquet")],
            ["ru_sci_bench", os.path.join(_FILTERING_ROOT, "filtered_top_5/ru_sci_bench.parquet")],
            ["ru_stackoverflow", os.path.join(_FILTERING_ROOT, "filtered_top_5/ru_stackoverflow.parquet")],
            ["swim_ir", os.path.join(_FILTERING_ROOT, "filtered_top_5/swim_ir.parquet")],
            ["yandex_q", os.path.join(_FILTERING_ROOT, "filtered_top_5/yandex_q.parquet")],
            ["habr", os.path.join(_FILTERING_ROOT, "filtered_top_5/habr.parquet")],
        ],
    },

    # "validation": {
    #     "filtering_name": "cleaned",
    #     "hard_negatives_input_folder": os.path.join(_HNM_ROOT, "validation"),
    #     "datasets_paths": [
    #         ["mmarco", "/path/to/cleaned_datasets/splits/validation/mmarco.parquet"],
    #         ["9111_questions", "/path/to/cleaned_datasets/splits/validation/legal_9111_unified.parquet"],
    #         ["habr_qna", "/path/to/cleaned_datasets/splits/validation/habr_qna_unified.parquet"],
    #         ["nq", "/path/to/cleaned_datasets/splits/validation/nq_unified.parquet"],
    #         ["ru_news", "/path/to/cleaned_datasets/splits/validation/ru_news_cleaned.parquet"],
    #         ["ru_sci_bench_ru", "/path/to/cleaned_datasets/splits/validation/ru_sci_bench_ru.parquet"],
    #         ["ru_sci_bench_merged", "/path/to/cleaned_datasets/splits/validation/ru_sci_bench_merged.parquet", "ru_sci_bench"],
    #         ["ru_stackoverflow", "/path/to/cleaned_datasets/splits/validation/ru_stackoverflow.parquet"],
    #         ["swim_ir", "/path/to/cleaned_datasets/splits/validation/swim_ir_ru_en_unified.parquet"],
    #         ["yandex_q", "/path/to/cleaned_datasets/splits/validation/yandex_q_merged.parquet"],
    #         ["habr", "/path/to/cleaned_datasets/splits/validation/habr.parquet"],
    #     ],
    # },
}

# ---------------------------
# Configuration
# ---------------------------
CONFIG = {
    "max_hard_negatives": 8,
    "permutation_mode": "all",  # "all" or "same_only"
    "hard_negative_lang_policy": "query_lang",  # "query_lang" or "pair_lang"
    "max_chars": 32768,  # truncate text fields before large_string -> string cast
}

OUTPUT_PARTS_ROOT = Path("final_dataset_parts")
ITER_BATCH = 50_000      # rows per batch when scanning the source dataset
QUERY_CHUNK = 50_000     # queries to process before flushing to disk


# ---------------------------
# Helper utilities
# ---------------------------
def build_maps(ds, default_dataset_name: str | None = None):
    """
    Build index maps using pandas vectorised ops — 10-50x faster than a
    Python loop over HF dataset rows.  Only reads metadata columns.
    """
    meta_cols = ["query_id", "passage_id", "lang"]
    has_dataset_col = "dataset" in ds.column_names
    if has_dataset_col:
        meta_cols.append("dataset")

    print("  Loading metadata into pandas …")
    df = ds.select_columns(meta_cols).to_pandas()

    # Vectorised key columns
    df["qid"] = df["query_id"].astype(str)
    df["pid"] = df["passage_id"].astype(str)
    df["lang"] = df["lang"].astype(str)
    if has_dataset_col:
        dataset_values = df["dataset"].where(df["dataset"].notna(), default_dataset_name)
        df["dataset_name"] = dataset_values.astype(str).str.strip()
        if default_dataset_name is not None:
            df.loc[df["dataset_name"] == "", "dataset_name"] = str(default_dataset_name)
    else:
        df["dataset_name"] = "" if default_dataset_name is None else str(default_dataset_name)

    df["qid_dataset"] = df["qid"].str.cat(df["dataset_name"], sep="::")
    df["hn_qid"] = df["qid"].str.cat(df["lang"], sep="_")
    df["qid_lang"] = df["qid_dataset"].str.cat(df["lang"], sep="_")
    df["pid_lang"] = df["pid"].str.cat(df["lang"], sep="_")
    df["row_idx"] = range(len(df))

    print(f"  Building maps for {len(df)} rows …")

    # --- simple 1-to-1 maps (last occurrence wins, same as original loop) ---
    query_row_idx  = dict(zip(df["qid_lang"], df["row_idx"], strict=False))
    passage_row_idx = dict(zip(df["pid_lang"], df["row_idx"], strict=False))
    query_lang_map  = dict(zip(df["qid_lang"], df["lang"], strict=False))
    query_base_id_map = dict(zip(df["qid_lang"], df["qid"], strict=False))
    query_hn_id_map = dict(zip(df["qid_lang"], df["hn_qid"], strict=False))
    query_output_id_map = dict(zip(df["qid_lang"], df["qid_dataset"], strict=False))
    passage_lang_map = dict(zip(df["pid_lang"], df["lang"], strict=False))
    query_dataset = dict(zip(df["qid_lang"], df["dataset_name"], strict=False))

    # --- dataset_langs / query_dataset ---
    dataset_langs = defaultdict(set)
    for ds_name, lang in zip(df["dataset_name"], df["lang"], strict=False):
        dataset_langs[ds_name].add(lang)

    # --- positives: qid_lang -> {pid_lang, …} (shared across lang variants) ---
    # groupby in Python over pre-computed arrays is still far cheaper than
    # per-row Arrow deserialisation.
    base_qpairs = defaultdict(set)     # query_id::dataset -> {pid_lang}
    base_qid_langs = defaultdict(set)  # query_id::dataset -> {qid_lang}
    for qid_dataset, pid_lang, qid_lang in zip(df["qid_dataset"], df["pid_lang"], df["qid_lang"], strict=False):
        base_qpairs[qid_dataset].add(pid_lang)
        base_qid_langs[qid_dataset].add(qid_lang)

    del df
    gc.collect()

    positives = {}
    for base_qid, qid_langs in base_qid_langs.items():
        shared_pos = base_qpairs[base_qid]
        for ql in qid_langs:
            positives[ql] = set(shared_pos)

    del base_qpairs, base_qid_langs
    gc.collect()

    return (
        query_row_idx, passage_row_idx,
        query_lang_map, query_base_id_map, passage_lang_map,
        positives, dataset_langs, query_dataset, query_output_id_map, query_hn_id_map,
    )


def get_permutations(langs: Iterable[str], mode="all"):
    langs = sorted(langs)
    if mode == "same_only" or len(langs) == 1:
        return [(lang, lang) for lang in langs]
    return [(ql, pl) for ql in langs for pl in langs]


def select_hard_negative_ids(hard_list, lang_policy, q_lang, pair_lang, max_n, exclude_ids):
    selected = []
    seen = set(exclude_ids)
    target = q_lang if lang_policy == "query_lang" else pair_lang

    for hn in hard_list:
        if hn["lang"] == target:
            pid = hn["passage_id"]
            if pid not in seen:
                selected.append(pid)
                seen.add(pid)
                if len(selected) >= max_n:
                    return selected

    for hn in hard_list:
        pid = hn["passage_id"]
        if pid not in seen:
            selected.append(pid)
            seen.add(pid)
            if len(selected) >= max_n:
                break
    return selected


def prepare_text_column(
    column: pa.ChunkedArray,
    max_chars: int | None,
    column_name: str,
) -> tuple[pa.ChunkedArray, int]:
    """
    Prepare query/passage column before recipe expansion:
    - optional truncation to max_chars
    - cast to large_string for safe take() on very large batches
    """
    if not (pa.types.is_large_string(column.type) or pa.types.is_string(column.type)):
        raise TypeError(f"Column '{column_name}' must be string/large_string, got: {column.type}")

    truncated_values = 0
    prepared = column

    if max_chars is not None:
        too_long = pc.greater(pc.utf8_length(prepared), max_chars)
        truncated_sum = pc.sum(pc.cast(too_long, pa.int64()))
        truncated_values = int(truncated_sum.as_py() or 0)
        prepared = pc.utf8_slice_codeunits(prepared, start=0, stop=max_chars)

    # Keep as large_string so downstream take() can exceed 2GB offsets safely.
    prepared = pc.cast(prepared, pa.large_string())
    return prepared, truncated_values


def cast_table_columns_to_string(table: pa.Table) -> pa.Table:
    """Arrow-native cast for output batches (no Python string materialization)."""
    for idx, field in enumerate(table.schema):
        if not pa.types.is_string(field.type):
            table = table.set_column(idx, field.name, pc.cast(table.column(idx), pa.string()))
    return table


# ---------------------------
# Main builder — streams output to a Parquet file
# ---------------------------
def build_dataset(hf_dataset, mined_hard_negatives, output_path, config=CONFIG, dataset_name: str | None = None):
    """
    Build the final dataset and write it incrementally to *output_path*.

    Text resolution uses pure Arrow take() — data flows from the
    memory-mapped source straight to the Parquet writer without ever
    becoming Python strings.
    """
    (
        query_row_idx, passage_row_idx,
        query_lang, query_base_id_map, passage_lang,
        positives, dataset_langs, query_dataset, query_output_id_map, query_hn_id_map,
    ) = build_maps(hf_dataset, default_dataset_name=dataset_name)

    # Passage pools by lang
    all_passages_by_lang = defaultdict(list)
    for pid, lang in passage_lang.items():
        all_passages_by_lang[lang].append(pid)

    # Pre-group positives by passage language
    positives_by_lang = {}
    for qid, pos_ids in positives.items():
        by_lang = defaultdict(list)
        for pid in pos_ids:
            by_lang[passage_lang[pid]].append(pid)
        positives_by_lang[qid] = dict(by_lang)

    hn_by_qid = {d["query_id"]: d.get("hard_negatives", []) for d in mined_hard_negatives}

    max_hn = config["max_hard_negatives"]
    max_neg = max_hn
    hn_lang_policy = config["hard_negative_lang_policy"]
    perm_mode = config["permutation_mode"]
    max_chars = config.get("max_chars", 32768)
    write_batch_rows = config.get("write_batch_rows", 2048)
    if max_chars is not None and max_chars <= 0:
        raise ValueError("config['max_chars'] must be > 0 or None")
    if write_batch_rows <= 0:
        raise ValueError("config['write_batch_rows'] must be > 0")

    # Convert source text columns once (Arrow-native) before recipe expansion.
    arrow_query, truncated_query_values = prepare_text_column(
        hf_dataset.data.column("query"),
        max_chars=max_chars,
        column_name="query",
    )
    arrow_passage, truncated_passage_values = prepare_text_column(
        hf_dataset.data.column("passage"),
        max_chars=max_chars,
        column_name="passage",
    )

    neg_col_names = [f"negative{i}" for i in range(1, max_neg + 1)]

    fields = [
        ("query_id", pa.string()),
        ("dataset", pa.string()),
        ("query", pa.string()),
        ("positive", pa.string()),
    ]
    fields += [(name, pa.string()) for name in neg_col_names]
    schema = pa.schema(fields)

    writer = pq.ParquetWriter(str(output_path), schema)
    total_rows = 0
    total_queries = 0
    written_queries = 0
    dropped_insufficient_hn = 0
    dropped_incomplete_after_policy = 0
    truncated_values_total = truncated_query_values + truncated_passage_values
    all_qids = list(positives.keys())

    n_chunks = (len(all_qids) + QUERY_CHUNK - 1) // QUERY_CHUNK
    for chunk_start in tqdm(
        range(0, len(all_qids), QUERY_CHUNK),
        desc="  Processing queries", total=n_chunks, unit="chunk", leave=False,
    ):
        chunk_qids = all_qids[chunk_start : chunk_start + QUERY_CHUNK]

        # ---- 1. Build recipes (row-index tuples only, no text) ----
        recipes = []  # (query_id, dataset, query_row, positive_row, [neg_rows…])
        for qid in chunk_qids:
            total_queries += 1
            pos_ids = positives[qid]
            q_lang = query_lang[qid]
            query_id = query_output_id_map[qid]
            ds_name = query_dataset.get(qid)
            langs = dataset_langs.get(ds_name) if ds_name else None
            if langs is None:
                langs = set(all_passages_by_lang.keys()) or {q_lang}
            perms = get_permutations(langs, perm_mode)

            hard_list = hn_by_qid.get(query_hn_id_map[qid])
            if hard_list is None:
                hard_list = hn_by_qid.get(query_base_id_map[qid], [])
            pos_by_lang = positives_by_lang[qid]
            q_ridx = query_row_idx[qid]
            excluded_base = set(pos_ids)

            # If fewer than max_hn usable hard negatives exist for this query,
            # drop the query entirely to avoid NULL negatives in output.
            unique_usable_hn = set()
            for hn in hard_list:
                pid = hn.get("passage_id")
                if pid in excluded_base or pid in unique_usable_hn:
                    continue
                if pid in passage_row_idx:
                    unique_usable_hn.add(pid)
                if len(unique_usable_hn) >= max_hn:
                    break
            if len(unique_usable_hn) < max_hn:
                dropped_insufficient_hn += 1
                continue

            query_recipes = []
            invalid_query = False

            for ql, pl in perms:
                if ql != q_lang:
                    continue
                pos_in_lang = pos_by_lang.get(pl)
                if not pos_in_lang:
                    continue

                hn_ids = select_hard_negative_ids(
                    hard_list, hn_lang_policy, q_lang, pl, max_hn, pos_ids,
                )

                neg_ridxs = [
                    passage_row_idx[nid]
                    for nid in hn_ids
                    if nid in passage_row_idx
                ]

                if len(neg_ridxs) < max_hn:
                    invalid_query = True
                    break

                for pid in pos_in_lang:
                    query_recipes.append((query_id, ds_name, q_ridx, passage_row_idx[pid], neg_ridxs))

            if invalid_query:
                dropped_incomplete_after_policy += 1
                continue

            if query_recipes:
                written_queries += 1
                recipes.extend(query_recipes)

        if not recipes:
            continue

        # ---- 2. Build numpy index arrays from recipes ----
        n = len(recipes)
        query_ids = []
        dataset_values = []
        q_idx = np.empty(n, dtype=np.int64)
        p_idx = np.empty(n, dtype=np.int64)
        neg_idx = np.zeros((max_neg, n), dtype=np.int64)

        for r, (qid, ds_name, qi, pi, nis) in enumerate(recipes):
            query_ids.append(qid)
            dataset_values.append(ds_name)
            q_idx[r] = qi
            p_idx[r] = pi
            for k in range(max_neg):
                neg_idx[k, r] = nis[k]

        del recipes

        # ---- 3. Pure Arrow take → Parquet (no Python strings at all) ----
        col_qid = pa.array(query_ids, type=pa.string())
        col_dataset = pa.array(dataset_values, type=pa.string())
        col_q = pc.take(arrow_query, pa.array(q_idx))
        col_p = pc.take(arrow_passage, pa.array(p_idx))

        columns = {"query_id": col_qid, "dataset": col_dataset, "query": col_q, "positive": col_p}
        for i, name in enumerate(neg_col_names):
            columns[name] = pc.take(arrow_passage, pa.array(neg_idx[i]))

        raw_table = pa.table(columns)
        for batch in raw_table.to_batches(max_chunksize=write_batch_rows):
            out_table = cast_table_columns_to_string(pa.Table.from_batches([batch]))
            writer.write_table(out_table)
        total_rows += n

        del query_ids, dataset_values, q_idx, p_idx, neg_idx, col_qid, col_dataset, col_q, col_p, columns, raw_table
        gc.collect()

    writer.close()

    del arrow_query, arrow_passage
    del query_row_idx, passage_row_idx, query_lang, query_base_id_map, query_output_id_map, query_hn_id_map, passage_lang
    del positives, positives_by_lang, all_passages_by_lang, hn_by_qid
    gc.collect()

    stats = {
        "total_queries": total_queries,
        "written_queries": written_queries,
        "dropped_insufficient_hn": dropped_insufficient_hn,
        "dropped_incomplete_after_policy": dropped_incomplete_after_policy,
        "truncated_values": truncated_values_total,
        "total_rows": total_rows,
    }
    print(f"  Wrote {total_rows} rows → {output_path}")
    print(
        "  Query stats: "
        f"total={total_queries}, "
        f"written={written_queries}, "
        f"dropped_insufficient_hn={dropped_insufficient_hn}, "
        f"dropped_after_lang_policy={dropped_incomplete_after_policy}, "
        f"truncated_values={truncated_values_total}"
    )
    return stats

# ---------------------------
# Usage: Process one-by-one (including loading), then save (no final shuffle)
# ---------------------------

for split_name, split_cfg in  SPLIT_CONFIGS.items():
    print("\n" + "=" * 70)
    print(f"Processing split: {split_name}")
    print("=" * 70)

    filtering_name = split_cfg["filtering_name"]
    temp_dir = OUTPUT_PARTS_ROOT / filtering_name / split_name
    hard_negatives_input_folder = split_cfg["hard_negatives_input_folder"]
    datasets_paths = split_cfg["datasets_paths"]

    # FIXME:
    # if temp_dir.exists():
    #     shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)

    parquet_paths = []
    split_stats = defaultdict(dict)

    for dataset_entry in datasets_paths:
        dataset_name, dataset_path = dataset_entry[0], dataset_entry[1]
        hn_name = dataset_entry[2] if len(dataset_entry) > 2 else dataset_name
        print(f"\nProcessing dataset: {dataset_name}")

        mined = load_hard_negatives_for_dataset(hn_name, hard_negatives_input_folder)
        if not mined:
            print(f"  Warning: No hard negatives found for {dataset_name}, skipping...")
            continue

        # Load each dataset inside the loop to keep memory low
        dataset = _load_dataset("parquet", data_files=dataset_path, split="train")

        out_path = temp_dir / f"{dataset_name}.parquet"
        dataset_stats = build_dataset(dataset, mined, out_path, CONFIG, dataset_name=dataset_name)
        split_stats[split_name][dataset_name] = dataset_stats
        parquet_paths.append(str(out_path))

        del dataset, mined
        gc.collect()

    print("\n" + "-" * 70)
    print(f"Drop stats for split: {split_name}")
    for dataset_name, dataset_stats in split_stats[split_name].items():
        print(
            f"  {dataset_name}: "
            f"dropped_insufficient_hn={dataset_stats['dropped_insufficient_hn']}, "
            f"dropped_after_lang_policy={dataset_stats['dropped_incomplete_after_policy']}, "
            f"written_queries={dataset_stats['written_queries']}"
        )

# ex: nohup python3 final_dataset_compilation_top5.py > top5_compilation.log 2>&1 &
