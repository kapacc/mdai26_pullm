"""
Generate benchmark visualization (boxplots) and summary table for paper.

Produces:
1. Boxplot figure (2x2 grid, all 4 datasets, AUC & PR-AUC metrics)
2. Summary table (LaTeX format with mean ± std for all methods)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Configuration
BENCHMARK_RUNS = Path(__file__).parent.parent / "outputs" / "benchmark_runs.csv"
BENCHMARK_SUMMARY = Path(__file__).parent.parent / "outputs" / "benchmark_summary.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

METHODS = ["naive", "clust", "strict-lassclust", "non-strict-lassclust", "lassojoint", "spy"]
DATASETS = ["plwiki", "open_subtitles_corpus", "job_offers_pl_corpus", "forum_cdaction_pl_corpus"]
METRICS = ["auc", "pr_auc"]
C_CALC_VALUES = [0.3, 0.5, 0.7]
LABELLING_STRATEGY = "mvc"

# Color palette for methods
COLORS = {
    "naive": "#FF7F0E",
    "clust": "#2CA02C",
    "strict-lassclust": "#1F77B4",
    "non-strict-lassclust": "#9467BD",
    "lassojoint": "#D62728",
    "spy": "#8C564B"
}

METHOD_LABELS = {
    "naive": "Naive",
    "clust": "Clust",
    "strict-lassclust": "S-LassClust",
    "non-strict-lassclust": "NS-LassClust",
    "lassojoint": "LassoJoint",
    "spy": "Spy"
}


def load_data():
    """Load benchmark runs data."""
    df = pd.read_csv(BENCHMARK_RUNS)
    df = df[df["labelling_strategy"] == LABELLING_STRATEGY]
    return df


def create_boxplot_figure(df):
    """Create 2x2 grid boxplot figure with AUC and PR-AUC side-by-side per dataset."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        df_dataset = df[df["dataset"] == dataset].copy()
        
        # Prepare data for boxplot
        positions = []
        data_to_plot = []
        labels = []
        colors_to_use = []
        
        pos = 1
        for c_calc in C_CALC_VALUES:
            # Group header position
            group_start = pos
            
            for method_idx, method in enumerate(METHODS):
                df_subset = df_dataset[
                    (df_dataset["c_calc"] == c_calc) & 
                    (df_dataset["method"] == method)
                ]
                
                if len(df_subset) > 0:
                    # Collect AUC values
                    auc_values = df_subset["auc"].values
                    data_to_plot.append(auc_values)
                    positions.append(pos)
                    colors_to_use.append(COLORS[method])
                    pos += 1
            
            # Add spacing between c_calc groups
            pos += 0.5
        
        # Create boxplot
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                       showfliers=False, notch=False)
        
        # Color boxes
        for patch, color in zip(bp["boxes"], colors_to_use):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Styling
        ax.set_ylabel("AUC", fontsize=11, fontweight="bold")
        ax.set_xlabel("c_calc", fontsize=11, fontweight="bold")
        ax.set_ylim([0, 1.0])
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_title(f"{dataset.replace('_', ' ').title()}", fontsize=12, fontweight="bold", pad=10)
        
        # Set x-axis labels for c_calc values
        c_calc_positions = []
        c_calc_labels = []
        pos = 1
        for c_calc in C_CALC_VALUES:
            group_start = pos
            pos += len(METHODS)
            group_center = (group_start + pos - 1) / 2
            c_calc_positions.append(group_center)
            c_calc_labels.append(f"{c_calc:.1f}")
            pos += 0.5
        
        ax.set_xticks(c_calc_positions)
        ax.set_xticklabels(c_calc_labels)
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker="s", color="w", 
                                  markerfacecolor=COLORS[method], markersize=8, 
                                  label=METHOD_LABELS[method])
                      for method in METHODS]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6, 
              frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle("Benchmark Results: AUC Across Methods and Datasets", 
                fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.99])
    
    # Save figure
    fig_path = OUTPUT_DIR / "benchmark_boxplots_all_datasets.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"✓ Boxplot figure saved to {fig_path}")
    
    # Also save PDF for LaTeX
    pdf_path = OUTPUT_DIR / "benchmark_boxplots_all_datasets.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"✓ PDF version saved to {pdf_path}")
    
    plt.close()


def create_summary_table(df_summary):
    """Create LaTeX summary table with mean ± std for all methods and datasets."""
    
    df_summary = pd.read_csv(BENCHMARK_SUMMARY)
    
    # Build table structure: rows=methods, columns=dataset x c_calc x metric
    table_data = []
    
    for method in METHODS:
        row = [METHOD_LABELS[method]]
        
        for dataset in DATASETS:
            for metric in ["auc", "pr_auc"]:
                for c_calc in C_CALC_VALUES:
                    subset = df_summary[
                        (df_summary["method"] == method) &
                        (df_summary["dataset"] == dataset) &
                        (df_summary["c_calc"] == c_calc) &
                        (df_summary["labelling_strategy"] == LABELLING_STRATEGY)
                    ]
                    
                    if len(subset) > 0:
                        mean_col = f"{metric}_mean"
                        std_col = f"{metric}_std"
                        mean = subset[mean_col].values[0]
                        std = subset[std_col].values[0]
                        
                        # Format as mean ± std
                        if pd.notna(std) and std > 0:
                            cell_val = f"{mean:.3f}$\\pm${std:.3f}"
                        else:
                            cell_val = f"{mean:.3f}"
                    else:
                        cell_val = "—"
                    
                    row.append(cell_val)
        
        table_data.append(row)
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\caption{Benchmark Results: Mean $\\pm$ Std Over 10 Seeds.}")
    latex_lines.append("\\label{tab:benchmark_results}")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    
    # Build column definition
    # Method | plwiki(auc@0.3, auc@0.5, auc@0.7, pr@0.3, pr@0.5, pr@0.7) | open_sub... | job_off... | forum...
    n_cols = 1 + len(DATASETS) * len(METRICS) * len(C_CALC_VALUES)
    col_spec = "l" + "r" * (n_cols - 1)
    
    latex_lines.append(f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    
    # Header row 1: Dataset names
    header1 = "Method"
    for dataset in DATASETS:
        dataset_short = dataset.replace("_", "\\_")
        header1 += f" & \\multicolumn{{{len(METRICS) * len(C_CALC_VALUES)}}}" + "{c}{" + dataset_short + "}"
    header1 += " \\\\"
    latex_lines.append(header1)
    
    # Header row 2: Metrics (AUC, PR-AUC) and c_calc
    header2 = ""
    for dataset in DATASETS:
        for metric in METRICS:
            metric_label = "AUC" if metric == "auc" else "PR-AUC"
            for c_calc in C_CALC_VALUES:
                header2 += f" & {metric_label}@{c_calc}"
    header2 = "Method" + header2 + " \\\\"
    latex_lines.append(header2)
    latex_lines.append("\\midrule")
    
    # Data rows
    for row in table_data:
        latex_row = " & ".join(row) + " \\\\"
        latex_lines.append(latex_row)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabularx}")
    latex_lines.append("\\end{table}")
    
    latex_code = "\n".join(latex_lines)
    
    # Save to file
    table_path = OUTPUT_DIR / "benchmark_table_latex.txt"
    with open(table_path, "w") as f:
        f.write(latex_code)
    
    print(f"✓ LaTeX table code saved to {table_path}")
    return latex_code


def main():
    """Main execution."""
    print("🔄 Loading benchmark data...")
    df = load_data()
    print(f"   Loaded {len(df)} runs")
    
    print("\n📊 Generating boxplot figure...")
    create_boxplot_figure(df)
    
    print("\n📋 Generating summary table...")
    df_summary = pd.read_csv(BENCHMARK_SUMMARY)
    latex_table = create_summary_table(df_summary)
    
    print("\n✅ Done! Generated:")
    print(f"   - Boxplot PNG: outputs/benchmark_boxplots_all_datasets.png")
    print(f"   - Boxplot PDF: outputs/benchmark_boxplots_all_datasets.pdf")
    print(f"   - LaTeX table: outputs/benchmark_table_latex.txt")


if __name__ == "__main__":
    main()
