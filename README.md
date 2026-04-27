# mdai26_pullm: PU Learning Benchmark for Polish Text Classification

Minimal reproducible repository for PU (Positive-Unlabeled) learning benchmarks on Polish text datasets from SpeakLeash. This repository contains all code, tests, and pre-computed results needed to reproduce the benchmark published in the paper.

**Paper**: "PU-Based Quality Classifier for LLM Training Texts"

## Quick Start

### 1. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/mdai26_pullm.git
cd mdai26_pullm
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` ensures the `py_puml` package is importable by scripts and tests.

**Dependencies:**
- Python 3.10+
- numpy, pandas, scikit-learn, imbalanced-learn
- matplotlib (for visualizations)
- pytest (for testing)

### 2. Validate Installation (Smoke Test)

Run a quick end-to-end test on synthetic data:

```bash
python scripts/run_smoke_benchmark.py
```

This should complete in <1 minute and print a prediction summary.

### 3. Run Benchmark on Your Own CSV Data

To run the benchmark on your preprocessed CSV files:

**Step A: Prepare your data**

Create CSV files in the `data_preprocessed/` directory with the following structure:
- Numeric feature columns (min-max scaled recommended)
- Column named `Y` with binary labels (0=negative, 1=positive)

Example:
```
feature_1,feature_2,feature_3,...,Y
0.5,0.3,0.8,...,1
0.2,0.7,0.1,...,0
...
```

See [data/README.md](data/README.md) for detailed format specifications.

**Step B: Run benchmark**

```bash
python scripts/run_benchmark.py \
  --source csv \
  --data-dir data_preprocessed \
  --seeds 1 2 3 4 5 \
  --sample-size 1000 \
  --labelling-strategy mvc \
  --c-calc 0.5
```

**Output files:**
- `outputs/benchmark_runs.csv` — Raw results per seed and method
- `outputs/benchmark_summary.csv` — Aggregated mean/std statistics

## Available Methods & Parameters

### Methods
Six PU learning methods are evaluated:
1. **naive** — Standard logistic regression on PU labels
2. **clust** — Clustering-based method
3. **strict-lassclust** — Lasso + strict clustering
4. **non-strict-lassclust** — Lasso + non-strict clustering
5. **lassojoint** — Joint logistic regression with feature selection
6. **spy** — EM-based spy method

### Labeling Schemes
- **mvc** (Most Variable Columns) — Non-SCAR labeling using top-variance features
- **classic** — Non-SCAR labeling with logistic propensity calibration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--source` | `csv` | Data source: `csv` or `speakleash` |
| `--data-dir` | `data_preprocessed` | Directory containing CSV files |
| `--seeds` | `1 2 3 4 5` | Random seeds for reproducibility |
| `--sample-size` | `10000` | Rows per dataset; `None` for all rows |
| `--labelling-strategy` | `mvc` | Labeling scheme: `mvc` or `classic` |
| `--labelling-strategies` | `mvc classic` | Multiple schemes (runs in one pass) |
| `--c-calc` | `0.5` | Target labeling probability |
| `--c-calc-values` | `0.3 0.5 0.8` | Multiple c values (multi-run) |
| `--use-smote` | (flag) | Enable SMOTE oversampling on train fold |

### Example: Full Multi-Strategy Benchmark

```bash
python scripts/run_benchmark.py \
  --source csv \
  --data-dir data_preprocessed \
  --seeds 1 2 3 4 5 \
  --sample-size 1000 \
  --labelling-strategies mvc classic \
  --c-calc-values 0.3 0.5 0.7
```

This runs all combinations in a single pass and saves to `outputs/benchmark_runs.csv` and `outputs/benchmark_summary.csv`.


## Results & Output Files

### Benchmark Results CSVs

**benchmark_runs.csv** — Raw results per seed
```
dataset,seed,method,c_calc,labelling_strategy,auc,pr_auc,f1,run_time_min,...
```

**benchmark_summary.csv** — Aggregated mean ± std
```
dataset,method,c_calc,labelling_strategy,auc_mean,auc_std,pr_auc_mean,pr_auc_std,f1_mean,f1_std,...
```

### Metrics

- **AUC** — Area Under the ROC Curve
- **PR-AUC** — Area Under the Precision-Recall Curve
- **F1** — F1 Score (harmonic mean of precision & recall)
- **runtime** — Execution time in minutes

### Supplement PDF

The [paper/supplement.pdf](paper/supplement.pdf) contains detailed benchmark tables for all 6 datasets (ISAP, job_offers, open_subtitles, plwiki, ulotki_medyczne, wolne_lektury) across both labeling schemes and multiple c_calc values.

## Testing

Run the test suite to validate the implementation:

```bash
pytest tests/ -v
```

**Test coverage:**
- Labeling correctness and reproducibility (`test_labelling.py`)
- Method smoke tests (`test_methods_smoke.py`)
- Metrics computation (`test_metrics.py`)
- Benchmark structure & contract (`test_benchmark.py`)

## Core Concepts

### PU Learning
Positive-Unlabeled learning handles datasets where only positive labels (Y=1) are observed, while the rest are treated as unlabeled. This repository implements both:
- **SCAR** (Selected At Random) — older assumption
- **Non-SCAR** — more realistic assumption

### Labeling Schemes
Both labeling schemes generate artificial "labeled" sets (S) from true labels (Y) for benchmarking:

**MVC (Most Variable Columns):**
- Ranks rows by top-variance features
- Linear calibration to target c_calc value

**Classic:**
- Logistic propensity e(x) = sigmoid(α + θ·x)
- Grid search to find α matching target c_calc
- Bernoulli sampling S ~ Bernoulli(e(x)) for Y=1

### Data Requirements
- Input must have binary Y column (0/1)
- Features should be numeric
- Preprocessing (scaling, feature engineering) happens before this benchmark
- Typical use: 1000-10000 samples per dataset

## Advanced Usage

### Single Dataset Run
```bash
python scripts/run_benchmark.py \
  --source csv \
  --data-dir data_preprocessed \
  --seeds 1 \
  --sample-size 100 \
  --labelling-strategy mvc
```

### Union-Train Mode (fit on combined training set, evaluate per dataset)
```bash
python -c "
from src.py_puml.benchmark import benchmark_many_datasets_union_train
from src.py_puml.data_loader import load_datasets_from_csv
datasets = load_datasets_from_csv('data_preprocessed')
runs, summary = benchmark_many_datasets_union_train(
    datasets, 
    seeds=[1,2,3,4,5],
    sample_size=1000
)
"
```

### Visualization
After running benchmarks, generate plots and LaTeX tables:

```bash
python scripts/visualize_benchmark_results.py
python scripts/generate_tables_for_paper.py
```

## Common Issues & Solutions

**ImportError: No module named 'speakleash'**
- Only needed if `--source speakleash`. For CSV mode, skip this.
- Install with: `pip install speakleash`

**FileNotFoundError: No CSV files found**
- Ensure your data directory structure matches `--data-dir`
- CSV files should have `.csv` extension
- Example: `data_preprocessed/dataset_name.csv`

**Convergence warnings from sklearn**
- These are normal for longer benchmarks and don't indicate errors
- Consider setting `--sample-size` smaller for quick tests

**Tests fail on import**
- Ensure you're in the repo root directory
- `export PYTHONPATH=$(pwd):$PYTHONPATH` (on Linux/Mac)
- Or: `set PYTHONPATH=%cd%;%PYTHONPATH%` (on Windows)

## Reproducing Paper Results

The pre-computed results are in:
- `outputs/benchmark_runs.csv` — 1440+ runs (6 datasets × 5 seeds × 6 methods × 8 configurations)
- `outputs/benchmark_summary.csv` — Aggregated statistics
- `paper/supplement.pdf` — Full results tables

To regenerate all results from scratch:

```bash
python scripts/run_benchmark.py \
  --source csv \
  --data-dir data_preprocessed \
  --seeds 1 2 3 4 5 \
  --sample-size 10000 \
  --labelling-strategies mvc classic \
  --c-calc-values 0.3 0.5 0.7

python scripts/visualize_benchmark_results.py
python scripts/generate_tables_for_paper.py
```

**Note:** Full benchmarks require your own preprocessed CSV files in `data_preprocessed/`. The repository includes code and results but not raw data (too large for version control).

## Publication & Citation

This work is part of research on quality classification for LLM training texts using Polish corpora from SpeakLeash.

**Related files:**
- [repo_description_en.md](repo_description_en.md) — Detailed technical workflow documentation
- [paper/main.tex](paper/main.tex) — Full manuscript LaTeX source
- [paper/supplement.pdf](paper/supplement.pdf) — Supplementary materials with detailed results


---

**Last Updated:** April 2026  
**Version:** 1.0 (Minimal Reproducible Repository)
