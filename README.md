# Kolmogorov-Arnold Networks vs. Multi-Layer Perceptrons for Tabular Classification

This repository contains the replication code, experimental pipeline, and raw results for the comparative evaluation of Kolmogorov-Arnold Networks (KANs) and Multi-Layer Perceptrons (MLPs) on medical tabular data.

The benchmark evaluates 11 model architectures across 7 clinical classification datasets under a 3x5 Repeated Stratified K-Fold Cross-Validation protocol with nested Bayesian hyperparameter optimization.

## Repository structure

```
kan_mlp_comparison/
├── notebooks/
│   ├── 01_eda.ipynb                     # Exploratory data analysis of benchmark cohorts
│   └── 02_results_analysis.ipynb        # Post-hoc analysis of metric distributions
├── results/
│   ├── benchmark_master_20261013-0013.csv # Full raw cross-validation results (N=1,155)
│   ├── summary_metrics.csv              # Metric means and standard deviations
│   ├── stats_bayesian_rope.csv          # Bayesian correlated t-test with ROPE analysis
│   ├── stats_wilcoxon_posthoc.csv       # Pairwise Wilcoxon signed-rank tests
│   ├── ablation_results.csv             # Training data scarcity ablation results
│   ├── plots/                           # Generated publication-quality vector plots (PDF)
│   └── artifacts/                       # Per-fold confusion matrices and learning curves
├── scripts/
│   ├── run_full_pipeline.py             # Master orchestration script
│   ├── automate_benchmark.py            # Main 3x5 CV benchmark with Optuna HPO
│   ├── run_ablation_study.py            # Sample efficiency / data scarcity evaluation
│   ├── generate_report.py               # Summary metric aggregation and figure generator
│   ├── plot_advanced.py                 # Pareto frontier, CD diagrams, rank heatmaps
│   └── plot_precision_recall.py         # Precision-Recall curve generation
├── src/
│   ├── config.py                        # Path configurations and dataset metadata
│   ├── data/
│   │   ├── loader.py                    # PyTorch Dataset definitions and scikit-learn pipelines
│   │   └── preprocessor.py              # UCI/OpenML automated fetching and standardization
│   ├── models/
│   │   ├── base.py                      # Abstract base classes for tabular models
│   │   ├── mlp.py                       # StandardMLP baseline architecture
│   │   └── kan_variants/                # KAN implementations (TabKAN, FastKAN, ChebyKAN,
│   │                                    # JacobiKAN, LegendreKAN, GramKAN, TaylorKAN,
│   │                                    # WavKAN, ReLUKAN)
│   ├── training/
│   │   ├── trainer.py                   # PyTorch training loop with MPS/CUDA support
│   │   ├── cross_validation.py          # Repeated Stratified K-Fold logic
│   │   ├── hpo.py                       # Optuna TPE Bayesian hyperparameter optimization
│   │   └── early_stopping.py            # Validation loss early stopping
│   └── evaluation/
│       ├── metrics.py                   # AUPRC, AUROC, Brier Score, MCC, F1, Balanced Acc
│       ├── stats.py                     # Frequentist Wilcoxon and Friedman/Nemenyi tests
│       ├── bayesian_stats.py            # Bayesian correlated t-test with ROPE
│       └── plot_*.py                    # Plotting modules and styling definitions
└── requirements.txt                     # Python dependencies
```

## Environment Setup

The codebase requires Python 3.10 or newer. PyTorch supports CUDA, Apple Silicon (MPS), and CPU devices.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing the results

### 1. Data acquisition and preprocessing

To download the raw datasets from the UCI Machine Learning Repository and OpenML, and format them for training:

```bash
python -m src.data.preprocessor
```

This creates processed CSV files in `data/processed/`.

### 2. Running the main experimental pipeline

To execute the entire experimental workflow (main benchmark, scarcity ablation, and visualization generation) sequentially:

```bash
python scripts/run_full_pipeline.py
```

Alternatively, individual components can be executed independently:

- **Main 3x5 CV Benchmark:**
  ```bash
  python scripts/automate_benchmark.py
  ```
- **Data Scarcity Ablation:**
  ```bash
  python scripts/run_ablation_study.py
  ```

### 3. Generating figures and statistical reports

To process raw benchmark outputs and generate publication plots (PDF format in `results/plots/`) and statistical tables:

```bash
python scripts/generate_report.py
```

Advanced standalone plots (Pareto frontiers, Critical Difference diagrams, and Precision-Recall curves) can be generated with:

```bash
python scripts/plot_advanced.py
python scripts/plot_precision_recall.py
```

> **Note:** Trained model weights (`*.pth`) and raw logs are excluded from version control. You must run the benchmark pipeline to generate weights before plotting.

## Benchmark data

Pre-computed benchmark outputs are provided in the `results/` directory:

- `results/benchmark_master_20261013-0013.csv`: Primary raw benchmark evaluation containing fold-level metrics across all 11 models, 7 datasets, and 15 repeated folds (1,155 evaluations).
- `results/summary_metrics.csv`: Aggregate means and standard deviations for all evaluated performance and computational metrics.
- `results/stats_bayesian_rope.csv`: Bayesian Correlated t-test probabilities within the Region of Practical Equivalence (ROPE) for $\Delta \in [-0.01, 0.01]$.
- `results/stats_wilcoxon_posthoc.csv`: Pairwise Wilcoxon signed-rank test statistics with Holm-Bonferroni correction.
- `results/ablation_results.csv`: Metrics across training set fractions (10% to 100%).
