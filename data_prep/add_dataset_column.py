#!/usr/bin/env python3
"""Добавляет колонку string `dataset` с одним значением во все строки parquet (по row groups, без полной загрузки файла в память)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Входной .parquet")
    parser.add_argument(
        "--value",
        default="mmarco",
        help='Значение для колонки dataset (по умолчанию: "mmarco")',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Куда записать. Если не указано — перезаписать --input атомарно через временный файл.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Файл не найден: {input_path}")

    pf = pq.ParquetFile(input_path)
    names = pf.schema_arrow.names
    if "dataset" in names:
        raise SystemExit(f"Колонка dataset уже есть в {input_path}")

    out_path = args.output.resolve() if args.output else input_path.with_name(
        f".{input_path.name}.tmp_add_dataset"
    )
    if out_path == input_path:
        raise SystemExit("--output не может совпадать с --input; для перезаписи не указывайте --output")

    writer: pq.ParquetWriter | None = None
    try:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            col = pa.repeat(pa.scalar(args.value, type=pa.string()), table.num_rows)
            table = table.append_column("dataset", col)
            if writer is None:
                writer = pq.ParquetWriter(
                    out_path,
                    table.schema,
                    compression="snappy",
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if args.output is None:
        os.replace(out_path, input_path)
        print(f"Обновлён {input_path}: добавлена колонка dataset={args.value!r}")
    else:
        print(f"Записано {out_path}: колонка dataset={args.value!r}")


if __name__ == "__main__":
    main()

# Example usage: python add_dataset_column.py --input /path/to/cleaned_datasets/splits/train/mmarco.parquet