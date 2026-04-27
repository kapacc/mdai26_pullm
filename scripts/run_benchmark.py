from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from py_puml.benchmark import (
    BENCHMARK_MODE_PER_DATASET,
    BENCHMARK_MODE_UNION_TRAIN,
    DEFAULT_PRETRAIN_SAMPLE_SIZE,
    benchmark_many_datasets,
    benchmark_many_datasets_for_c_calc_values,
    benchmark_many_datasets_for_strategies,
    benchmark_many_datasets_union_train,
    benchmark_many_datasets_union_train_for_c_calc_values,
    benchmark_many_datasets_union_train_for_strategies,
)
from py_puml.data_loader import load_datasets


CSV_PREPROCESSED_DEFAULT_DIR = "data_preprocessed"


def make_demo_dataset(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.9 * x1 - 0.6 * x2 + 0.2 * x3)))
    y = rng.binomial(1, p)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "Y": y})


def main(
    source: str = "demo",
    dataset_names: Optional[list[str]] = None,
    data_dir: str = "data",
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    c_calc: float = 0.5,
    threshold: float = 0.5,
    c_calc_values: Optional[tuple[float, ...]] = None,
    labelling_strategy: str = "mvc",
    labelling_strategies: Optional[tuple[str, ...]] = None,
    benchmark_mode: str = BENCHMARK_MODE_PER_DATASET,
    max_rows: Optional[int] = None,
    sample_size: Optional[int] = DEFAULT_PRETRAIN_SAMPLE_SIZE,
    use_smote: bool = False,
    output_suffix: Optional[str] = None,
) -> None:
    """
    Run benchmark on datasets.

    Parameters
    ----------
    source : str
        Data source: "demo", "speakleash", or "csv" (default: "demo").
    dataset_names : list[str], optional
        Dataset names to load (for speakleash only).
    data_dir : str
        Directory containing CSV files. For `source="csv"`, the default
        points to the preprocessed CSV directory with a `Y` column.
    seeds : tuple[int]
        Random seeds for reproducibility.
    c_calc : float
        Target labeling rate for PU labels.
    threshold : float
        Classification threshold used for threshold-based metrics (F1, TP/TN/FP/FN, FMI).
    c_calc_values : tuple[float], optional
        Optional multi-value mode. When provided, runs all listed c_calc
        values in one pass.
    labelling_strategy : str
        Labelling scheme identifier: "mvc" (default) or "classic".
    labelling_strategies : tuple[str], optional
        Optional multi-scheme mode. When provided, runs all listed
        labelling schemes in one pass.
    benchmark_mode : str
        Benchmark mode: "per-dataset" or "union-train".
    max_rows : int, optional
        For speakleash source: max rows per dataset to read from stream.
    sample_size : int, optional
        For csv source: number of rows to sample before labelling and train/test split.
    use_smote : bool
        When True, apply SMOTE oversampling to the training fold before
        labelling. Mirrors the SMOTE preprocessing used in mdai26/main.R.
    output_suffix : str, optional
        Optional suffix appended to output file names, e.g. "1" produces
        benchmark_runs_1.csv and benchmark_summary_1.csv.
    """
    print(f"Loading datasets from source: {source}")

    if source == "demo":
        datasets = {
            "demo_a": make_demo_dataset(700, 10),
            "demo_b": make_demo_dataset(900, 20),
        }
    else:
        effective_data_dir = data_dir
        if source == "csv" and data_dir == "data":
            effective_data_dir = CSV_PREPROCESSED_DEFAULT_DIR
        if source == "speakleash" and data_dir == "data":
            effective_data_dir = "data/speakleash_object"
        datasets = load_datasets(
            source=source,
            dataset_names=dataset_names,
            data_dir=effective_data_dir,
            max_rows=max_rows,
        )

    if source == "csv":
        invalid = [name for name, df in datasets.items() if "Y" not in df.columns]
        if invalid:
            raise ValueError(
                "CSV benchmark input must already contain a 'Y' column. "
                "Run scripts/preprocess_datasets.py first or pass --data-dir "
                f"{CSV_PREPROCESSED_DEFAULT_DIR}. Invalid datasets: {invalid}"
            )

    if source == "speakleash":
        invalid = [name for name, df in datasets.items() if "Y" not in df.columns]
        if invalid:
            raise ValueError(
                "SpeakLeash raw datasets do not contain 'Y' label required for benchmark. "
                "Prepare labeled feature tables first (e.g. save to CSV and run with --source csv). "
                f"Invalid datasets: {invalid}"
            )

    print(f"Loaded {len(datasets)} dataset(s)")
    for name, df in datasets.items():
        print(f"  {name}: {df.shape}")

    effective_labelling_strategies = labelling_strategies or (labelling_strategy,)
    effective_c_calc_values = c_calc_values or (c_calc,)

    strategy_label = ", ".join(effective_labelling_strategies)
    c_calc_label = ", ".join(str(value) for value in effective_c_calc_values)
    print(
        f"\nRunning benchmark with {len(seeds)} seeds, c_calc value(s): {c_calc_label}, "
        f"labelling scheme(s): {strategy_label}, mode: {benchmark_mode}, smote: {use_smote}, threshold: {threshold}..."
    )

    benchmark_runner_single = (
        benchmark_many_datasets
        if benchmark_mode == BENCHMARK_MODE_PER_DATASET
        else benchmark_many_datasets_union_train
    )
    benchmark_runner_strategies = (
        benchmark_many_datasets_for_strategies
        if benchmark_mode == BENCHMARK_MODE_PER_DATASET
        else benchmark_many_datasets_union_train_for_strategies
    )
    benchmark_runner_c_calc = (
        benchmark_many_datasets_for_c_calc_values
        if benchmark_mode == BENCHMARK_MODE_PER_DATASET
        else benchmark_many_datasets_union_train_for_c_calc_values
    )

    if len(effective_c_calc_values) > 1:
        runs, summary = benchmark_runner_c_calc(
            datasets=datasets,
            c_calc_values=effective_c_calc_values,
            labelling_strategies=effective_labelling_strategies,
            seeds=seeds,
            threshold=threshold,
            sample_size=sample_size if source == "csv" else None,
            use_smote=use_smote,
        )
    elif len(effective_labelling_strategies) == 1:
        runs, summary = benchmark_runner_single(
            datasets=datasets,
            seeds=seeds,
            c_calc=effective_c_calc_values[0],
            threshold=threshold,
            labelling_strategy=effective_labelling_strategies[0],
            sample_size=sample_size if source == "csv" else None,
            use_smote=use_smote,
        )
    else:
        runs, summary = benchmark_runner_strategies(
            datasets=datasets,
            labelling_strategies=effective_labelling_strategies,
            seeds=seeds,
            c_calc=effective_c_calc_values[0],
            threshold=threshold,
            sample_size=sample_size if source == "csv" else None,
            use_smote=use_smote,
        )

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""
    runs_path = out_dir / f"benchmark_runs{suffix}.csv"
    summary_path = out_dir / f"benchmark_summary{suffix}.csv"

    runs.to_csv(runs_path, index=False, float_format="%.3f")
    summary.to_csv(summary_path, index=False, float_format="%.3f")

    print("\nSaved:")
    print(f"  {runs_path} ({len(runs)} rows)")
    print(f"  {summary_path} ({len(summary)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PU learning benchmark on datasets from various sources."
    )
    parser.add_argument(
        "--source",
        choices=["demo", "speakleash", "csv"],
        default="demo",
        help="Data source: demo (synthetic), csv (local files with Y), or speakleash (requires datasets with Y)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to load (for csv or speakleash source; speakleash datasets must contain Y)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help=(
            "Directory containing CSV files. For csv source, the default is "
            f"{CSV_PREPROCESSED_DEFAULT_DIR} when --data-dir is not overridden"
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Random seeds for reproducibility",
    )
    parser.add_argument(
        "--c-calc",
        type=float,
        default=0.5,
        help="Target labeling rate for PU labels",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold used for threshold-based metrics",
    )
    parser.add_argument(
        "--c-calc-values",
        nargs="+",
        type=float,
        default=None,
        help="Run one or more c_calc values in a single benchmark pass; overrides --c-calc if set",
    )
    parser.add_argument(
        "--labelling-strategy",
        choices=["mvc", "classic"],
        default="mvc",
        help="Labelling scheme used to generate PU labels on train fold",
    )
    parser.add_argument(
        "--labelling-strategies",
        nargs="+",
        choices=["mvc", "classic"],
        default=None,
        help="Run one or more labelling schemes in a single benchmark pass; overrides --labelling-strategy if set",
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=[BENCHMARK_MODE_PER_DATASET, BENCHMARK_MODE_UNION_TRAIN],
        default=BENCHMARK_MODE_PER_DATASET,
        help="Benchmark mode: per-dataset (fit/test per dataset) or union-train (fit once on union train, evaluate per dataset)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="For speakleash source: optional max rows per dataset",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_PRETRAIN_SAMPLE_SIZE,
        help="For csv source: sample this many rows before labelling and train/test split",
    )
    parser.add_argument(
        "--use-smote",
        action="store_true",
        default=False,
        help="Apply SMOTE oversampling to the training fold before labelling (mirrors mdai26/main.R preprocessing)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Optional suffix for output filenames, e.g. 1 -> benchmark_runs_1.csv",
    )

    args = parser.parse_args()

    main(
        source=args.source,
        dataset_names=args.datasets,
        data_dir=args.data_dir,
        seeds=tuple(args.seeds),
        c_calc=args.c_calc,
        threshold=args.threshold,
        c_calc_values=tuple(args.c_calc_values) if args.c_calc_values else None,
        labelling_strategy=args.labelling_strategy,
        labelling_strategies=tuple(args.labelling_strategies) if args.labelling_strategies else None,
        benchmark_mode=args.benchmark_mode,
        max_rows=args.max_rows,
        sample_size=args.sample_size,
        use_smote=args.use_smote,
        output_suffix=args.output_suffix,
    )
