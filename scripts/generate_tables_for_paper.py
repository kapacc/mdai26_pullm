"""
Generate simplified tables for main.tex and detailed per-dataset tables for supplement.tex
"""

import pandas as pd
from pathlib import Path

# Load data
BENCHMARK_SUMMARY = Path(__file__).parent.parent / "outputs" / "benchmark_summary.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

df = pd.read_csv(BENCHMARK_SUMMARY)
df = df[df["labelling_strategy"] == "mvc"]

METHODS = ["naive", "clust", "strict-lassclust", "non-strict-lassclust", "lassojoint", "spy"]
DATASETS = ["plwiki", "open_subtitles_corpus", "job_offers_pl_corpus", "forum_cdaction_pl_corpus"]
METRICS = ["auc", "pr_auc", "f1"]

# Shorten dataset names for tables
DATASET_SHORT = {
    "plwiki": "plwiki",
    "open_subtitles_corpus": "open\\_subtitles",
    "job_offers_pl_corpus": "job\\_offers",
    "forum_cdaction_pl_corpus": "forum"
}

METHOD_LABELS = {
    "naive": "Naive",
    "clust": "Clust",
    "strict-lassclust": "S-LassClust",
    "non-strict-lassclust": "NS-LassClust",
    "lassojoint": "LassoJoint",
    "spy": "Spy"
}

def generate_main_tex_table():
    """Generate simplified table for main.tex showing average AUC per dataset."""
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\caption{Benchmark Results Summary: Average AUC Over All Labeling Rates.}")
    lines.append("\\label{tab:benchmark_summary}")
    lines.append("\\centering")
    lines.append("\\normalsize")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{plwiki} & \\textbf{open\\_subtitles} & \\textbf{job\\_offers} & \\textbf{forum} \\\\")
    lines.append("\\midrule")
    
    for method in METHODS:
        row = [METHOD_LABELS[method]]
        
        for dataset in DATASETS:
            subset = df[(df["method"] == method) & (df["dataset"] == dataset)]
            
            if len(subset) > 0:
                avg_auc = subset["auc_mean"].mean()
                row.append(f"{avg_auc:.3f}")
            else:
                row.append("—")
        
        lines.append(" & ".join(row) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_supplement_tables():
    """Generate detailed per-dataset tables for supplement.tex with AUC, PR-AUC, F1."""
    
    lines = []
    lines.append("\\documentclass[11pt]{article}")
    lines.append("\\usepackage[margin=1in]{geometry}")
    lines.append("\\usepackage{booktabs}")
    lines.append("\\usepackage{array}")
    lines.append("\\usepackage{multirow}")
    lines.append("")
    lines.append("\\title{Supplementary Material: Detailed Benchmark Results}")
    lines.append("\\author{}")
    lines.append("\\date{}")
    lines.append("")
    lines.append("\\begin{document}")
    lines.append("\\maketitle")
    lines.append("")
    
    for dataset in DATASETS:
        lines.append(f"\\section{{{DATASET_SHORT[dataset]}}}")
        lines.append("")
        
        # Create table for this dataset
        lines.append("\\begin{table}[htbp]")
        lines.append(f"\\caption{{Detailed results for {DATASET_SHORT[dataset]} across all labeling rates and metrics (mean $\\pm$ std over 10 seeds).}}")
        lines.append(f"\\label{{tab:details_{dataset}}}")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\begin{tabular}{lrrrrrrrrr}")
        lines.append("\\toprule")
        lines.append("\\textbf{Method} & \\multicolumn{3}{c}{\\textbf{AUC}} & \\multicolumn{3}{c}{\\textbf{PR-AUC}} & \\multicolumn{3}{c}{\\textbf{F1}} \\\\")
        lines.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}")
        lines.append(" & 0.3 & 0.5 & 0.7 & 0.3 & 0.5 & 0.7 & 0.3 & 0.5 & 0.7 \\\\")
        lines.append("\\midrule")
        
        for method in METHODS:
            row = [METHOD_LABELS[method]]
            
            for c_calc in [0.3, 0.5, 0.7]:
                for metric in ["auc", "pr_auc", "f1"]:
                    subset = df[
                        (df["method"] == method) & 
                        (df["dataset"] == dataset) & 
                        (df["c_calc"] == c_calc)
                    ]
                    
                    if len(subset) > 0:
                        mean_col = f"{metric}_mean"
                        std_col = f"{metric}_std"
                        mean = subset[mean_col].values[0]
                        std = subset[std_col].values[0]
                        
                        if pd.notna(std) and std > 0:
                            cell_val = f"{mean:.3f}$\\pm${std:.3f}"
                        else:
                            cell_val = f"{mean:.3f}"
                    else:
                        cell_val = "—"
                    
                    row.append(cell_val)
            
            lines.append(" & ".join(row) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")
    
    lines.append("\\end{document}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Generate main.tex table
    main_table = generate_main_tex_table()
    with open(OUTPUT_DIR / "benchmark_main_table.txt", "w") as f:
        f.write(main_table)
    print("[OK] Generated main.tex table: outputs/benchmark_main_table.txt")
    
    # Generate supplement.tex
    supplement = generate_supplement_tables()
    with open(OUTPUT_DIR.parent / "paper" / "supplement.tex", "w") as f:
        f.write(supplement)
    print("[OK] Generated supplement.tex: paper/supplement.tex")
