from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from py_puml.labelling import non_scar_labelling_classic, non_scar_labelling_mvc

DEFAULT_DATASETS = [
    "ISAP_corpus",
    "job_offers_pl_corpus",
    "open_subtitles_corpus",
    "plwiki",
    "ulotki_medyczne",
    "wolne_lektury_corpus",
]
DEFAULT_STRATEGIES = ["classic", "mvc"]
DEFAULT_SEEDS = [11, 22, 33, 44, 55]


def apply_labelling(df: pd.DataFrame, strategy: str, c_calc: float, seed: int) -> pd.DataFrame:
    if strategy == "classic":
        return non_scar_labelling_classic(df, target_c_calc=c_calc, random_state=seed)
    if strategy == "mvc":
        return non_scar_labelling_mvc(df, target_c_calc=c_calc, n_vars=2)
    raise ValueError(f"Unsupported strategy: {strategy}")


def compute_achieved_c(labelled_df: pd.DataFrame) -> float:
    y = pd.to_numeric(labelled_df["Y"], errors="raise").astype(int)
    s = pd.to_numeric(labelled_df["S"], errors="raise").astype(int)

    positives = y == 1
    if int(positives.sum()) == 0:
        raise ValueError("Dataset has no positive instances (Y=1), cannot compute achieved c.")

    return float(s.loc[positives].mean())


def format_latex_table(summary_df: pd.DataFrame, c_calc: float) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append(
        "\\caption{Stability of non-SCAR labelling on six preprocessed datasets "
        + f"($c_{{calc}}={c_calc}$, five seeds)."
        + "}\\label{tab:non_scar_stability}"
    )
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Method & Achieved $c$ & $|c_{calc}-c|$ & Time [s] " + "\\\\")
    lines.append("\\midrule")

    for _, row in summary_df.iterrows():
        dataset = row["dataset"]
        method = row["strategy"]
        achieved = f"${row['achieved_c_mean']:.3f} \\pm {row['achieved_c_std']:.3f}$"
        deviation = f"${row['abs_deviation_mean']:.3f} \\pm {row['abs_deviation_std']:.3f}$"
        elapsed = f"${row['time_seconds_mean']:.3f} \\pm {row['time_seconds_std']:.3f}$"

        dataset_tex = dataset.replace("_", "\\_")
        lines.append(
            f"\\texttt{{{dataset_tex}}} & \\texttt{{{method}}} & {achieved} & {deviation} & {elapsed} \\\\"  # noqa: E501
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate non-SCAR labelling stability report for multiple datasets."
    )
    parser.add_argument(
        "--data-dir",
        default="data_preprocessed",
        help="Directory containing preprocessed CSV files with Y column.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names (CSV stems) to include.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        choices=["classic", "mvc"],
        help="Labelling strategies to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds used in repeated stability runs.",
    )
    parser.add_argument(
        "--c-calc",
        type=float,
        default=0.2,
        help="Target c_calc value for non-SCAR labelling.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where report artifacts are saved.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for dataset in args.datasets:
        dataset_path = Path(args.data_dir) / f"{dataset}.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        df = pd.read_csv(dataset_path)
        if "Y" not in df.columns:
            raise ValueError(f"Dataset must contain Y column: {dataset_path}")

        for strategy in args.strategies:
            for seed in args.seeds:
                start = time.perf_counter()
                labelled = apply_labelling(df=df, strategy=strategy, c_calc=args.c_calc, seed=seed)
                elapsed = time.perf_counter() - start

                achieved_c = compute_achieved_c(labelled)
                rows.append(
                    {
                        "dataset": dataset,
                        "strategy": strategy,
                        "seed": seed,
                        "c_calc": args.c_calc,
                        "achieved_c": achieved_c,
                        "abs_deviation": abs(args.c_calc - achieved_c),
                        "time_seconds": elapsed,
                    }
                )

    runs_df = pd.DataFrame(rows)
    summary_df = (
        runs_df.groupby(["dataset", "strategy"], as_index=False)
        .agg(
            achieved_c_mean=("achieved_c", "mean"),
            achieved_c_std=("achieved_c", "std"),
            abs_deviation_mean=("abs_deviation", "mean"),
            abs_deviation_std=("abs_deviation", "std"),
            time_seconds_mean=("time_seconds", "mean"),
            time_seconds_std=("time_seconds", "std"),
        )
        .sort_values(["dataset", "strategy"])
        .reset_index(drop=True)
    )

    runs_path = output_dir / "non_scar_stability_runs.csv"
    summary_path = output_dir / "non_scar_stability_summary.csv"
    latex_path = output_dir / "non_scar_stability_table_latex.txt"

    runs_df.to_csv(runs_path, index=False, float_format="%.6f")
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")
    latex_path.write_text(format_latex_table(summary_df, c_calc=args.c_calc), encoding="utf-8")

    expected_rows = len(args.datasets) * len(args.strategies) * len(args.seeds)
    print(f"Saved runs: {runs_path} ({len(runs_df)} rows, expected {expected_rows})")
    print(f"Saved summary: {summary_path} ({len(summary_df)} rows)")
    print(f"Saved LaTeX table: {latex_path}")


if __name__ == "__main__":
    main()
