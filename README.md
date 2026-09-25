# Re-evaluating Kolmogorov-Arnold Networks for Clinical Decision Support: A Multi-Dataset Tabular Benchmark

**Author:** Rafał Wodziński, Paweł Szczerbiak
**Affiliation:** Department of Computer Science, Faculty of Computer Science and Mathematics, Cracow University of Technology, Cracow, Poland, Sano Science, Cracow, Poland

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/HPO-Optuna-%E2%89%A53.0-6857C6?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%E2%89%A51.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

Official research repository for a rigorous, reproducible benchmark comparing Kolmogorov-Arnold Network (KAN) architectures against a tuned Multi-Layer Perceptron (MLP) baseline and classical tree-based models on heterogeneous medical tabular data. The study encompasses 11 architectures, 7 UCI clinical datasets, and 1,155 independent fold evaluations under a 3×5 Repeated Stratified K-Fold Cross-Validation protocol with per-fold Bayesian hyperparameter optimisation.

---

## Table of Contents

- [Research Questions & Key Findings](#research-questions--key-findings)
- [Evaluated Architectures](#evaluated-architectures)
- [Clinical Datasets](#clinical-datasets)
- [Methodological Protocol](#methodological-protocol)
- [Quickstart & Reproducibility](#quickstart--reproducibility)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

---

## Research Questions & Key Findings

### Core Research Question

> Do modern KAN variants — employing B-spline, RBF, wavelet, and orthogonal-polynomial basis functions — offer a predictive or calibration advantage over a hyperparameter-optimised MLP and tree-based models on medical tabular data?

### Empirical Findings

#### 1 — Predictive Performance: Practical Equivalence

Macro-averaged AUROC across all 7 datasets shows that no KAN variant outperforms the tuned MLP beyond the Bayesian ROPE threshold of ±0.01:

| Model | Mean AUROC ↑ | Mean Brier Score ↓ |
|---|---|---|
| **StandardMLP** | **0.9517** | 0.0593 |
| TaylorKAN | 0.9473 | 0.0611 |
| TabKAN | 0.9470 | 0.0580 |
| WavKAN | 0.9467 | 0.0588 |
| ReLUKAN | 0.9456 | 0.0608 |
| FastKAN | 0.9454 | 0.0574 |
| GramKAN | 0.9451 | 0.0621 |
| LegendreKAN | 0.9439 | 0.0593 |
| JacobiKAN | 0.9423 | 0.0632 |
| ChebyKAN | 0.9387 | 0.0633 |
| RandomForest | 0.9477 | 0.0669 |

*Macro averages computed over 7 datasets (11 models × 7 datasets × 15 folds = 1,155 fold evaluations).*

The mean AUROC of the full KAN family is 0.9447, versus 0.9517 for the MLP — a gap of 0.007, which falls entirely within the Bayesian ROPE of ±0.01. Brier Score differences between the leading KAN variants and the MLP are < 0.002, confirming equivalent probabilistic calibration.

#### 2 — Computational Overhead

KAN variants impose a systematic inference latency and parameter count penalty relative to StandardMLP with no compensating accuracy gain:

| Model | Inference Time (ms/sample) | Relative Latency | Trainable Parameters | Param Ratio |
|---|---|---|---|---|
| StandardMLP | 0.0602 | 1.00× | ~7,676 | 1.00× |
| TabKAN | 0.0714 | 1.19× | ~64,114 | 8.35× |
| FastKAN | 0.0716 | 1.19× | ~40,845 | 5.32× |
| ReLUKAN | 0.0735 | 1.22× | ~70,110 | 9.13× |
| TaylorKAN | 0.0737 | 1.22× | ~23,360 | 3.04× |
| ChebyKAN | 0.0781 | 1.30× | ~24,392 | 3.18× |
| GramKAN | 0.0782 | 1.30× | ~23,879 | 3.11× |
| LegendreKAN | 0.0816 | 1.36× | ~24,408 | 3.18× |
| JacobiKAN | 0.0833 | 1.38× | ~22,992 | 3.00× |
| WavKAN | 0.0853 | 1.42× | ~108,858 | 14.18× |

#### 3 — Pareto Dominance

StandardMLP is the sole Pareto-optimal model in the AUROC × inference-latency objective space. No KAN variant achieves a better accuracy-latency trade-off simultaneously. This finding is visualised in `results/plots/pareto_frontier.png`.

#### Statistical Inference Summary

- Bayesian correlated *t*-test (Benavoli et al., 2017): Per-dataset comparisons of StandardMLP vs. the best-performing KAN variant confirm that the probability mass within ROPE (±0.01) exceeds 0.59 for AUROC and 0.73 for Brier Score — indicating practical equivalence rather than inferiority or superiority.
- Wilcoxon signed-rank test with Holm–Bonferroni correction (α = 0.05, aggregated over *n* = 7 datasets): No KAN variant yields a statistically significant improvement over StandardMLP on any primary metric after multiple-comparison correction.

---

## Evaluated Architectures

Eleven architectures are benchmarked, organised by the mathematical basis of their learnable univariate functions:

### KAN Variants (9)

| Architecture | Basis Function | Key Characteristic | Source File |
|---|---|---|---|
| **FastKAN** | Gaussian RBF | Fast B-spline approximation via radial basis functions | `src/models/kan_variants/fast_kan.py` |
| **WavKAN** | Wavelet (Mexican hat / Morlet) | Multi-resolution wavelet basis for non-stationary patterns | `src/models/kan_variants/wav_kan.py` |
| **ReLUKAN** | Piecewise linear (ReLU) | Efficient grid-based piecewise activations | `src/models/kan_variants/relu_kan.py` |
| **ChebyKAN** | Chebyshev polynomials | Degree-*d* minimax-optimal polynomial approximation | `src/models/kan_variants/cheby_kan.py` |
| **TaylorKAN** | Taylor series | Learnable Taylor coefficients per neuron | `src/models/kan_variants/taylor_kan.py` |
| **GramKAN** | Gram polynomials | Orthogonal polynomial basis via Gram–Schmidt | `src/models/kan_variants/gram_kan.py` |
| **JacobiKAN** | Jacobi polynomials | Generalised orthogonal polynomials (α, β parameters) | `src/models/kan_variants/jacobi_kan.py` |
| **LegendreKAN** | Legendre polynomials | Classical orthogonal polynomials on [−1, 1] | `src/models/kan_variants/legendre_kan.py` |
| **TabKAN** | Hybrid (Eslamian et al., 2025) | Tabular-data-dedicated KAN with feature-wise gating | `src/models/kan_variants/tab_kan.py` |

### Baselines (2)

| Architecture | Type | Description | Source File |
|---|---|---|---|
| **StandardMLP** | Deep neural network | Multi-Layer Perceptron with AdamW optimiser, early stopping, and Optuna HPO | `src/models/mlp.py` |
| **RandomForest** | Ensemble / tree-based | Classical strong baseline for tabular data (`scikit-learn`) | `scripts/automate_benchmark.py` |

---

## Clinical Datasets

All datasets are sourced from the UCI Machine Learning Repository (fetched automatically via `ucimlrepo`) or OpenML, and preprocessed with a hermetic per-fold pipeline (median imputation + StandardScaler, fit only on training data).

| # | Dataset | UCI ID | Task | Target Column | Source |
|---|---|---|---|---|---|
| 1 | **Breast Cancer Wisconsin** | 17 | Binary classification | `Diagnosis` | UCI |
| 2 | **Cardiotocography** | 193 | 3-class classification | `NSP` | UCI |
| 3 | **Chronic Kidney Disease** | 336 | Binary classification | `class` | UCI |
| 4 | **Heart Disease (Cleveland)** | 45 | Binary (0 vs 1–4) | `num` | UCI |
| 5 | **Parkinson's Disease** | 174 | Binary classification | `status` | UCI |
| 6 | **Pima Indians Diabetes** | — | Binary classification | `class` | OpenML (`diabetes`) |
| 7 | **Cervical Cancer Risk** | 383 | Binary classification | `Biopsy` | UCI |

Dataset loading and preprocessing are handled by [`src/data/loader.py`](src/data/loader.py) and [`src/data/preprocessor.py`](src/data/preprocessor.py). Dataset metadata and UCI identifiers are centralised in [`src/config.py`](src/config.py).

---

## Methodological Protocol

### Cross-Validation

- **Scheme:** 3×5 Repeated Stratified K-Fold Cross-Validation (`sklearn.model_selection.RepeatedStratifiedKFold`, `random_state=42`)
- **Scale:** 11 models × 7 datasets × 15 folds = 1,155 independent fold evaluations
- **Stratification:** Class balance maintained across splits for all binary and multiclass targets

### Hyperparameter Optimisation (HPO)

- **Framework:** Optuna (≥ 3.0) Bayesian TPE sampler
- **Trials per fold:** 15 Optuna trials per model-dataset-fold combination
- **Leakage prevention:** HPO is performed entirely within the training split of each fold; the validation set is never seen during tuning (zero data leakage)
- **Search space:** architecture depth, width, KAN-specific capacity parameters (`grid_size`, `num_grids`, `degree`, `num_wavelets`), learning rate, and weight decay
- **HPO artefacts:** Best hyperparameter sets are serialised per fold as JSON files in `results/artifacts/<dataset>/<model>/`

### Evaluation Metrics

Each fold is evaluated on:

| Metric | Description |
|---|---|
| **AUROC** | Area Under the ROC Curve — primary discrimination metric |
| **MCC** | Matthews Correlation Coefficient — balanced binary/multiclass quality |
| **F1 Score** | Weighted F1 — precision-recall balance |
| **Balanced Accuracy** | Mean per-class recall |
| **Brier Score** | Calibration / probabilistic loss |
| **Inference Time** | Per-sample latency (ms) on validation set |
| **Training Time** | Total wall-clock time (seconds) |
| **Trainable Parameters** | Model capacity count |

### Statistical Inference

**1. Bayesian Correlated *t*-Test with ROPE**
- Implementation: `src/evaluation/bayesian_stats.py`
- Correlation constant: ρ = 1/*k* = 0.2 (Nadeau & Bengio, 2003)
- Region of Practical Equivalence: **ROPE = ±0.01**
- Outputs: posterior probabilities P(A > B), P(B > A), P(ROPE) per dataset-metric pair
- Results: `results/stats_bayesian_rope.csv`

**2. Wilcoxon Signed-Rank Test + Holm–Bonferroni Correction**
- Implementation: `src/evaluation/stats.py`
- Non-parametric pairwise comparison aggregated across *n* = 7 datasets
- Family-wise error rate controlled at α = 0.05
- Results: `results/stats_wilcoxon_posthoc.csv`

---

## Quickstart & Reproducibility

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/kan_mlp_comparison.git
cd kan_mlp_comparison

# Create and activate conda environment
conda create -n kan_project python=3.10
conda activate kan_project

# Install all dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Datasets are fetched automatically from UCI ML Repository and OpenML on first run. No manual download is required.

### 3. Run Full Benchmark

Executes 3×5 Repeated Stratified K-Fold CV for all 11 models across all 7 datasets:

```bash
python scripts/automate_benchmark.py
```

Results are saved to `results/benchmark_master_<timestamp>.csv`.

### 4. Generate Statistical Reports

Computes Wilcoxon post-hoc tests, Bayesian ROPE analysis, and aggregated summary metrics:

```bash
python scripts/generate_report.py
```

Outputs: `results/stats_bayesian_rope.csv`, `results/stats_wilcoxon_posthoc.csv`, `results/summary_metrics.csv`.

### 5. Generate Advanced Visualisations

Produces Pareto frontier, Critical Difference Diagrams, radar charts, and per-model boxplots:

```bash
python scripts/plot_advanced.py
```

Outputs saved to `results/plots/`.

### 6. Sample Efficiency / Ablation Study

Evaluates model robustness under training data scarcity (varying `train_fraction`):

```bash
python scripts/run_ablation_study.py
```

Outputs: `results/ablation_results.csv` and corresponding plots in `results/plots/ablation_*/`.

---

## Project Structure

```
kan_mlp_comparison/
│
├── scripts/                          # Executable pipeline scripts
│   ├── automate_benchmark.py         # Main benchmark orchestrator (3×5 CV, all models)
│   ├── generate_report.py            # Statistical testing & metric aggregation
│   ├── plot_advanced.py              # Pareto frontier, CD diagrams, radar charts
│   ├── run_ablation_study.py         # Sample efficiency / data regime ablation
│   └── run_full_pipeline.py          # End-to-end pipeline runner
│
├── src/                              # Core research library
│   ├── config.py                     # Dataset registry (UCI IDs, targets, tasks)
│   │
│   ├── data/
│   │   ├── loader.py                 # Dataset fetching & PyTorch Dataset wrappers
│   │   └── preprocessor.py          # Hermetic imputation + scaling pipeline
│   │
│   ├── models/
│   │   ├── base.py                   # Abstract base class for all neural models
│   │   ├── mlp.py                    # StandardMLP implementation
│   │   └── kan_variants/
│   │       ├── base_kan.py           # Shared KAN base class
│   │       ├── fast_kan.py           # FastKAN (Gaussian RBF)
│   │       ├── wav_kan.py            # WavKAN (Wavelet)
│   │       ├── relu_kan.py           # ReLUKAN (Piecewise linear)
│   │       ├── cheby_kan.py          # ChebyKAN (Chebyshev polynomials)
│   │       ├── taylor_kan.py         # TaylorKAN (Taylor series)
│   │       ├── gram_kan.py           # GramKAN (Gram polynomials)
│   │       ├── jacobi_kan.py         # JacobiKAN (Jacobi polynomials)
│   │       ├── legendre_kan.py       # LegendreKAN (Legendre polynomials)
│   │       └── tab_kan.py            # TabKAN (tabular-dedicated KAN)
│   │
│   ├── training/
│   │   ├── cross_validation.py       # RepeatedStratifiedKFold CV engine
│   │   ├── trainer.py                # Training loop, early stopping, metric logging
│   │   ├── hpo.py                    # Optuna Bayesian HPO tuner
│   │   └── early_stopping.py        # Early stopping callback
│   │
│   └── evaluation/
│       ├── metrics.py                # AUROC, MCC, F1, Brier Score computation
│       ├── stats.py                  # Wilcoxon test + Holm–Bonferroni correction
│       ├── bayesian_stats.py         # Bayesian correlated t-test with ROPE
│       ├── interpretability.py       # Feature attribution utilities
│       ├── plot_bayesian.py          # Bayesian posterior visualisation
│       ├── plot_radar.py             # Radar / spider chart generation
│       └── plot_ablation.py          # Ablation study visualisation
│
├── results/
│   ├── benchmark_master_20260725-2124.csv    # Master fold-level results (1,155 rows)
│   ├── summary_metrics.csv                   # Per-model-dataset aggregated statistics
│   ├── stats_bayesian_rope.csv               # Bayesian ROPE posterior probabilities
│   ├── stats_wilcoxon_posthoc.csv            # Wilcoxon + Holm–Bonferroni p-values
│   ├── ablation_results.csv                  # Sample efficiency ablation results
│   ├── artifacts/                            # Per-fold best HPO parameters (JSON)
│   └── plots/                               # All generated figures (PNG)
│       ├── pareto_frontier.png
│       ├── critical_difference_diagram.png
│       ├── radar_chart_tradeoffs.png
│       ├── bayesian_rope_evidence.png
│       ├── boxplot_auroc.png
│       ├── boxplot_mcc.png
│       ├── boxplot_brier_score.png
│       ├── boxplot_inference_time_per_sample_ms.png
│       ├── cm_<dataset>_<model>.png          # Confusion matrices (77 files)
│       ├── learning_curves_<dataset>_<model>.png  # Learning curves (77 files)
│       └── ablation_<dataset>_<metric>.png   # Ablation plots (28 files)
│
├── requirements.txt
└── README.md
```

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `torch` | ≥ 2.0.0 | Neural network training (all KAN and MLP models) |
| `scikit-learn` | ≥ 1.3.0 | CV, preprocessing, Random Forest, metrics |
| `optuna` | ≥ 3.0.0 | Bayesian hyperparameter optimisation |
| `pandas` | ≥ 2.0.0 | Data manipulation and results aggregation |
| `numpy` | ≥ 1.24.0 | Numerical computation |
| `scipy` | ≥ 1.10.0 | Statistical tests (Wilcoxon) |
| `matplotlib` | ≥ 3.7.0 | Plot generation |
| `seaborn` | ≥ 0.12.0 | Statistical visualisations |
| `tqdm` | ≥ 4.65.0 | Progress reporting |
| `ucimlrepo` | ≥ 0.0.3 | Automatic UCI dataset fetching |

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this codebase or benchmark results in your research, please cite:

```bibtex
@misc{wodzinski2026kan,
  title         = {Re-evaluating {Kolmogorov-Arnold} Networks for Clinical Decision Support:
                   A Multi-Dataset Tabular Benchmark},
  author        = {Wodziński, Rafał and Szczerbiak, Paweł},
  year          = {2026},
  affiliation   = {Department of Computer Science, Faculty of Computer Science and Mathematics, Cracow University of Technology, Kraków, Poland (R.W.); Sano Centre for Computational Medicine, Kraków, Poland (P.S.)},
  note          = {Preprint}
}
```
