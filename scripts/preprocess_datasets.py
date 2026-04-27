from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "adj_ratio",
    "adjectives",
    "adverbs",
    "avg_sentence_length",
    "avg_word_length",
    "camel_case",
    "capitalized_words",
    "characters",
    "gunning_fog",
    "lexical_density",
    "noun_ratio",
    "nouns",
    "oovs",
    "pos_num",
    "pos_x",
    "punctuations",
    "quality",
    "sentences",
    "stopwords",
    "symbols",
    "verb_ratio",
    "verbs",
    "words",
]


def _minmax_scale_numeric(df: pd.DataFrame, skip_cols: set[str] | None = None) -> pd.DataFrame:
    """Scale numeric columns to [0, 1] using min-max scaling."""
    out = df.copy()
    skip = skip_cols or set()

    for col in out.columns:
        if col in skip:
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue

        col_min = out[col].min()
        col_max = out[col].max()
        denom = col_max - col_min

        if pd.isna(denom) or float(denom) == 0.0:
            out[col] = 0.0
        else:
            out[col] = (out[col] - col_min) / denom

    return out


def preprocess_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    df = pd.read_csv(input_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        missing_txt = ", ".join(missing)
        raise ValueError(f"Missing required columns in {input_path.name}: {missing_txt}")

    out = df[REQUIRED_COLUMNS].copy()
    quality_norm = out["quality"].astype(str).str.strip().str.upper()
    out["Y"] = quality_norm.map({"HIGH": 1, "LOW": 0})

    before = len(out)
    out = out[out["Y"].notna()].copy()
    out["Y"] = out["Y"].astype(int)
    after = len(out)

    out = _minmax_scale_numeric(out, skip_cols={"Y"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["quality"]).to_csv(output_path, index=False)

    return before, after


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess datasets from input dir and write filtered CSVs with Y target"
    )
    parser.add_argument(
        "--input-dir",
        default="data",
        help="Directory with source CSV datasets",
    )
    parser.add_argument(
        "--output-dir",
        default="data_preprocessed",
        help="Directory where preprocessed CSV files will be saved",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional list of dataset file names (without .csv) or csv file names",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    if args.datasets:
        input_files: list[Path] = []
        for name in args.datasets:
            filename = name if name.endswith(".csv") else f"{name}.csv"
            input_files.append(input_dir / filename)
    else:
        input_files = sorted(input_dir.glob("*.csv"))

    if not input_files:
        raise SystemExit(f"No CSV files found in: {input_dir}")

    total_before = 0
    total_after = 0

    for csv_path in input_files:
        if not csv_path.exists():
            raise SystemExit(f"Dataset file not found: {csv_path}")

        out_path = output_dir / csv_path.name
        before, after = preprocess_file(csv_path, out_path)
        removed = before - after
        total_before += before
        total_after += after

        print(
            f"{csv_path.name}: before={before}, after={after}, removed={removed}, output={out_path}"
        )

    print("\nSummary:")
    print(f"  files: {len(input_files)}")
    print(f"  rows before: {total_before}")
    print(f"  rows after: {total_after}")
    print(f"  rows removed: {total_before - total_after}")


if __name__ == "__main__":
    main()
