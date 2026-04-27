# Repo Description (PU SpeakLeash Benchmark)

This file describes the current state of the repository and project workflow: from data and preprocessing, through PU learning benchmark, to artifacts for publication.

## 1) Repository Purpose

The repo is used for:
- benchmarking PU learning methods on text data,
- comparing methods: naive, clust, strict-lassclust, non-strict-lassclust, lassojoint, spy,
- comparing labeling schemes (mvc, classic),
- comparing c_calc variants and benchmark modes,
- generating output materials for the paper (tables, plots, CSV).

## 2) Quick Start

### 2.1 Smoke path (quick validation)

- Quick pipeline test on toy data:
  - python scripts/run_smoke_benchmark.py

- Minimal benchmark on preprocessed CSV:
  - python scripts/run_benchmark.py --source csv --data-dir data_preprocessed --seeds 1 --sample-size 100 --labelling-strategy mvc --c-calc 0.5

### 2.2 Full path (complete workflow)

1. (Optional) Download SpeakLeash datasets:
   - python scripts/download_speakleash_datasets.py --datasets plwiki open_subtitles_corpus job_offers_pl_corpus ISAP_corpus ulotki_medyczne wolne_lektury_corpus
2. Preprocess CSV to benchmark format:
   - python scripts/preprocess_datasets.py --input-dir data --output-dir data_preprocessed
3. Main benchmark:
   - python scripts/run_benchmark.py --source csv --data-dir data_preprocessed --seeds 1 2 3 4 5 --sample-size 10000 --labelling-strategies mvc classic --c-calc-values 0.3 0.5 0.7
4. Visualizations and tables for report:
   - python scripts/visualize_benchmark_results.py
   - python scripts/generate_tables_for_paper.py
5. Analysis of common variables across datasets:
   - python scripts/compare_common_variables.py --data-dir data_preprocessed
6. Separate non-SCAR stability report:
  - python scripts/generate_non_scar_stability_report.py --data-dir data_preprocessed --output-dir outputs

## 3) Structure and Responsibilities

### 3.1 Library Code (src/py_puml)

- src/py_puml/labelling.py
  - Implements PU labeling: non_scar_labelling_mvc, non_scar_labelling_classic, scar_labelling.
  - Decision embedded in MVC code:
    - n_vars is limited to p-1 when p > 1,
    - fallback to 1 when p = 1,
    - this ensures ranking always remains defined.
    - quantile levels for interpolation: [0.1, 0.3, 0.5, 0.7, 0.9] (5 points).

- src/py_puml/benchmark.py
  - Orchestrates benchmark per dataset and union-train.
  - Decisions embedded in benchmark code:
    - Order without leakage: sampling -> split by Y -> labeling S only on train.
    - union-train requires identical feature schema across datasets.
    - Multi-strategy and multi-c_calc are executed in one pass and merged to common output files.
    - Metric aggregation groups by dataset, method, c_calc, labelling_strategy and additionally by smote and benchmark_mode, if columns are present.

- src/py_puml/methods.py
  - Runs all methods and returns unified output format.
  - Contains lassojoint implementation with helper functions (_sigma, _joint_neg_log_likelihood, _joint_gradient, _fit_joint).

- src/py_puml/metrics.py
  - Computes metrics: ROC AUC, PR-AUC, and F1.

- src/py_puml/data_loader.py
  - Loads data from csv or SpeakLeash.
  - With source=csv allows filtering by dataset names through dataset_names.

### 3.2 CLI Scripts (scripts)

- scripts/download_speakleash_datasets.py
  - Downloads specified SpeakLeash datasets,
  - caches objects to data/speakleash_object,
  - exports CSV to data.

- scripts/preprocess_datasets.py
  - Requires a set of feature columns and quality,
  - maps quality -> Y (HIGH=1, LOW=0, other values are discarded),
  - performs min-max scaling of numeric columns,
  - saves to data_preprocessed.

- scripts/run_benchmark.py
  - Main benchmark runner.
  - Sources: demo, csv, speakleash.
  - Modes: per-dataset, union-train.
  - Labeling schemes: mvc, classic (single or multi via --labelling-strategies).
  - c_calc variants: single via --c-calc or multi via --c-calc-values.
  - For csv default directory is data_preprocessed.
  - For speakleash requires presence of Y column (raw text-only datasets will not pass validation).
  - --use-smote option enables oversampling of training fold before labeling.

- scripts/run_smoke_benchmark.py
  - Quick end-to-end test on synthetic data,
  - runs methods and prints basic prediction summary.

- scripts/visualize_benchmark_results.py
  - Reads outputs/benchmark_runs.csv and outputs/benchmark_summary.csv,
  - filters on labelling_strategy=mvc,
  - generates boxplot charts and LaTeX/text tables:
    - outputs/benchmark_boxplots_all_datasets.png,
    - outputs/benchmark_boxplots_all_datasets.pdf,
    - outputs/benchmark_table_latex.txt.

- scripts/generate_tables_for_paper.py
  - Reads outputs/benchmark_summary.csv,
  - generates:
    - outputs/benchmark_main_table.txt,
    - detailed benchmark tables per dataset to paper/supplement.tex,
    - paper/supplement.tex.

- scripts/compare_common_variables.py
  - Compares common numeric variables across datasets,
  - generates statistics and pivots and a condensed plot:
    - outputs/common_variables_stats.csv,
    - outputs/common_variables_median_pivot.csv,
    - outputs/common_variables_iqr_pivot.csv,
    - outputs/common_variables_condensed_distributions.png,
    - outputs/common_variables_condensed_distributions.pdf.

- scripts/generate_non_scar_stability_report.py
  - Evaluates non-SCAR labeling stability across multiple datasets, strategies and seeds,
  - computes achieved c, deviation from c_calc and execution time,
  - generates artifacts:
    - outputs/non_scar_stability_runs.csv,
    - outputs/non_scar_stability_summary.csv,
    - outputs/non_scar_stability_table_latex.txt.

### 3.3 Tests (tests)

- tests/test_labelling.py
  - Tests correctness of S and labeling reproducibility.

- tests/test_methods_smoke.py
  - Smoke tests for PU methods.

- tests/test_metrics.py
  - Tests of evaluation metrics.

- tests/test_benchmark.py
  - Tests of benchmark structure and contract.

### 3.4 Paper Layer (paper)

- paper/main.tex
  - Main manuscript.
  - Contains sections on common variables, non-SCAR stability and main benchmark.

- paper/supplement.tex
  - Supplementary material with detailed benchmark tables per dataset.

- paper/main.pdf, paper/supplement.pdf
  - Current generated PDFs.

- Other .aux, .fls, .fdb_latexmk files
  - LaTeX compilation artifacts (not treated as key analytical artifacts).

## 4) Data and Experiment Pipeline

1. Download (optional) and export to CSV:
   - scripts/download_speakleash_datasets.py
2. Preprocessing and Y creation:
   - scripts/preprocess_datasets.py
3. Benchmark:
   - scripts/run_benchmark.py
   - for source=csv: sampling, train/test split, labeling on train, fit methods, metrics on test
   - for source=speakleash: requires ready Y
4. Results aggregation:
   - outputs/benchmark_runs.csv
   - outputs/benchmark_summary.csv
5. Reporting:
   - scripts/visualize_benchmark_results.py
   - scripts/generate_tables_for_paper.py
   - scripts/compare_common_variables.py
6. Non-SCAR stability:
  - scripts/generate_non_scar_stability_report.py
7. Integration to paper:
   - paper/main.tex
   - paper/supplement.tex

## 5) Data Contracts

- Y column is binary target 0/1.
- S column is PU label 0/1, generated from Y and features.
- In source=csv data benchmark expects Y to exist.
- In preprocessing quality is mapped:
  - HIGH -> Y=1,
  - LOW -> Y=0,
  - other quality values are removed.

## 6) Key Artifacts in Outputs

- outputs/benchmark_runs.csv
  - Raw results per run (dataset, seed, method, c_calc, labelling_strategy, smote, benchmark_mode, run_time_min, auc, pr_auc, f1).

- outputs/benchmark_summary.csv
  - Aggregation of mean/std across seeds and configuration, including run_time_min_mean and run_time_min_std.

- outputs/benchmark_boxplots_all_datasets.png
- outputs/benchmark_boxplots_all_datasets.pdf
  - Comparative plots of methods and datasets.

- outputs/benchmark_table_latex.txt
  - LaTeX table with benchmark summary.

- outputs/benchmark_main_table.txt
  - Abbreviated results table for main section.

- outputs/dataset_summary_preprocessed.csv
- outputs/dataset_summary_preprocessed.tex
  - Dataset summaries after preprocessing.

- outputs/common_variables_stats.csv
- outputs/common_variables_median_pivot.csv
- outputs/common_variables_iqr_pivot.csv
- outputs/common_variables_condensed_distributions.png
- outputs/common_variables_condensed_distributions.pdf
  - Artifacts of comparison of common features across datasets.

- outputs/common_variables_cv_pivot.csv
  - Artifact present in repo; may come from previous runs/versions of analysis.

- outputs/non_scar_stability_runs.csv
  - Raw non-SCAR stability results per dataset, method, seed and c_calc.

- outputs/non_scar_stability_summary.csv
  - Aggregation of mean/std for achieved c, deviation from c_calc and execution time.

- outputs/non_scar_stability_table_latex.txt
  - LaTeX table with non-SCAR stability summary.

## 7) Practical Notes

- sklearn warnings (FutureWarning, ConvergenceWarning) may appear in longer runs and do not necessarily indicate critical errors.
- For quick tests reduce sample-size.
- For final results run full passes and monitor convergence quality.
- Code reading order for onboarding:
  1. pyproject.toml
  2. src/py_puml/
  3. scripts/run_benchmark.py
  4. tests/
  5. paper/main.tex

## 8) Repository State (last verified experiment)

Last described full run (2026-04-17):
- 4 datasets,
- 5 seeds,
- 2 labeling schemes,
- 6 methods,
- sample-size=1000,
- artifacts: outputs/benchmark_runs.csv and outputs/benchmark_summary.csv.

The current repository also contains a separate workflow for analyzing common variables across 6 datasets and a non-SCAR stability report, so the main benchmark is not the only described workflow.
