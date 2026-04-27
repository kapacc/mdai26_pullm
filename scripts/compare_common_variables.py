from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DATASETS = [
    "plwiki",
    "open_subtitles_corpus",
    "job_offers_pl_corpus",
    "ISAP_corpus",
    "ulotki_medyczne",
    "wolne_lektury_corpus",
]


def load_selected_datasets(data_dir: Path, dataset_names: list[str]) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for name in dataset_names:
        csv_path = data_dir / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        datasets[name] = df
    return datasets


def get_common_numeric_columns(datasets: dict[str, pd.DataFrame], include_target: bool) -> list[str]:
    common_cols: set[str] | None = None

    for df in datasets.values():
        numeric_cols = set(df.select_dtypes(include=["number"]).columns)
        common_cols = numeric_cols if common_cols is None else (common_cols & numeric_cols)

    if common_cols is None:
        return []

    common_list = sorted(common_cols)

    if not include_target:
        common_list = [c for c in common_list if c != "Y"]

    return common_list


def compute_stats(datasets: dict[str, pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for dataset_name, df in datasets.items():
        for col in columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()

            rows.append(
                {
                    "dataset": dataset_name,
                    "variable": col,
                    "count": int(series.shape[0]),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)),
                    "median": float(series.median()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            )

    return pd.DataFrame(rows)


def make_condensed_figure(stats_df: pd.DataFrame, out_png: Path) -> None:
    median_pivot = stats_df.pivot(index="variable", columns="dataset", values="median")
    iqr_pivot = (stats_df["q75"] - stats_df["q25"]).to_frame("iqr")
    iqr_pivot["variable"] = stats_df["variable"].values
    iqr_pivot["dataset"] = stats_df["dataset"].values
    iqr_pivot = iqr_pivot.pivot(index="variable", columns="dataset", values="iqr")

    fig, axes = plt.subplots(1, 2, figsize=(14, max(8, 0.35 * len(median_pivot.index))), dpi=150)

    im1 = axes[0].imshow(median_pivot.values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    axes[0].set_title("Median value")
    axes[0].set_yticks(range(len(median_pivot.index)))
    axes[0].set_yticklabels(median_pivot.index)
    axes[0].set_xticks(range(len(median_pivot.columns)))
    axes[0].set_xticklabels(median_pivot.columns, rotation=45, ha="right")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(iqr_pivot.values, aspect="auto", cmap="OrRd", vmin=0.0, vmax=1.0)
    axes[1].set_title("IQR (Q75 - Q25)")
    axes[1].set_yticks(range(len(iqr_pivot.index)))
    axes[1].set_yticklabels(iqr_pivot.index)
    axes[1].set_xticks(range(len(iqr_pivot.columns)))
    axes[1].set_xticklabels(iqr_pivot.columns, rotation=45, ha="right")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Condensed distribution comparison across datasets", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare common numeric variables across datasets and produce condensed distribution visuals"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_preprocessed"),
        help="Directory with preprocessed dataset CSV files",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Dataset names (without .csv) to include in the comparison",
    )
    parser.add_argument(
        "--include-target",
        action="store_true",
        help="Include Y in the common variable comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for tables and figures",
    )

    args = parser.parse_args()

    datasets = load_selected_datasets(args.data_dir, args.datasets)
    common_cols = get_common_numeric_columns(datasets, include_target=args.include_target)

    if not common_cols:
        raise SystemExit("No common numeric columns found across selected datasets")

    stats_df = compute_stats(datasets, common_cols).sort_values(["variable", "dataset"])
    stats_path = args.output_dir / "common_variables_stats.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(stats_path, index=False)

    median_pivot = stats_df.pivot(index="variable", columns="dataset", values="median")
    iqr_series = stats_df["q75"] - stats_df["q25"]
    iqr_pivot = (
        stats_df.assign(iqr=iqr_series)
        .pivot(index="variable", columns="dataset", values="iqr")
        .sort_index()
    )

    median_pivot_path = args.output_dir / "common_variables_median_pivot.csv"
    iqr_pivot_path = args.output_dir / "common_variables_iqr_pivot.csv"
    median_pivot.sort_index().to_csv(median_pivot_path)
    iqr_pivot.to_csv(iqr_pivot_path)

    condensed_path = args.output_dir / "common_variables_condensed_distributions.png"
    make_condensed_figure(stats_df, condensed_path)

    print(f"Selected datasets: {', '.join(args.datasets)}")
    print(f"Common numeric variables: {len(common_cols)}")
    print(f"Saved stats table: {stats_path}")
    print(f"Saved median pivot table: {median_pivot_path}")
    print(f"Saved IQR pivot table: {iqr_pivot_path}")
    print(f"Saved condensed figure: {condensed_path}")


if __name__ == "__main__":
    main()