# Evaluating Kolmogorov-Arnold Networks (KAN) vs. Multi-Layer Perceptrons (MLP) on Medical Tabular Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Rigorous MLOps & Statistical Benchmarking Engine** for investigating the empirical capabilities of Kolmogorov-Arnold Networks across noisy, small-sample, and class-imbalanced medical tabular datasets.

---

## 1. Overview & Research Objectives

While deep learning architectures have achieved superhuman performance in computer vision and natural language processing, traditional tree-based ensembles and Multi-Layer Perceptrons (MLPs) remain the dominant paradigms for clinical tabular data. Medical tabular datasets present unique machine learning challenges:
- **Severe Data Scarcity (*Small Data* Phenomenon)**: Small patient cohorts resulting in high risk of overfitting.
- **High Noise & Missingness**: Imperfect clinical measurements and unrecorded diagnostic variables.
- **Complex Non-Linear Topologies**: Intricate interactions between physiological biomarkers.
- **Class Imbalance**: Rare pathological conditions compared to healthy baseline populations.

**Kolmogorov-Arnold Networks (KANs)** propose a fundamental shift from the Universal Approximation Theorem (UAT) to the Kolmogorov-Arnold Representation Theorem (KAM). Instead of static nodal activation functions ($\sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$) with linear synaptic weights, KANs replace linear weights with learnable univariate functions ($\phi_{q,p}(x_p)$) parameterized by B-splines, orthogonal polynomials, or wavelets along the edges of the network graph.

This repository implements a **hermetically sealed, zero-data-leakage benchmarking pipeline** designed to empirically answer: *Do learnable univariate edge activations in KAN architectures outperform traditional MLP representations and tree-based baselines under rigorous clinical validation constraints?*

---

## 2. Key Methodological Innovations

To guarantee publication-grade reproducibility and eliminate common experimental biases (such as data leakage and $p$-value hacking), the benchmarking framework enforces the following methodological standards:

### Zero-Leakage Hermetic Data Preprocessing
In many medical ML studies, global normalization or imputation before cross-validation introduces severe data leakage, artificially inflating test scores. Our pipeline guarantees isolation:
- All transformations (`StandardScaler`, `SimpleImputer`, `KNNImputer`, and `OneHotEncoder`) are dynamically encapsulated inside each training fold.
- Transformers are fitted (`fit_transform`) exclusively on the training split and applied blindly (`transform`) to validation validation splits.

### 3×5 Repeated Stratified K-Fold Cross-Validation
To account for variance in fold splitting on small patient cohorts, the entire evaluation suite runs a **3×5 Repeated Stratified K-Fold Cross-Validation** (15 independent training runs per model–dataset pair). This preserves the exact diagnostic class ratio across every validation fold.

### Automated Hyperparameter Optimization (Optuna HPO)
To ensure fair architectural comparisons without manual tuning bias, each CV fold dynamically executes a **15-trial Optuna hyperparameter search** using a leakage-free inner 80/20 validation split and `MedianPruner` early trial termination. Optimizable parameters include learning rate, weight decay, hidden dimensionality, and KAN-specific capacity hyperparameters (`grid_size`, `degree`, `num_wavelets`).

### Early Stopping & Optimal Weight Restoration
Every neural training loop is monitored by an `EarlyStopping` callback (`patience=10`, `min_delta=1e-4`) evaluated against validation loss. When training terminates, deep copies of the optimal network weights from the best-performing epoch are automatically restored and serialized.

### Holistic Clinical & Computational Metrics
Models are evaluated across a multi-dimensional spectrum of predictive accuracy and clinical deployment viability:
- **Matthews Correlation Coefficient (MCC)**: Robust consensus metric for severe clinical class imbalance.
- **Area Under the ROC Curve (AUROC)** & **F1-Score**: Threshold-independent classification discriminative power.
- **Brier Score**: Probabilistic calibration error (essential for clinical decision support systems).
- **Computational Latency & Complexity**: Per-sample inference time (ms), total training time (seconds), and total trainable parameter count.

---

## 3. Repository Architecture

```text
kan_mlp_comparison/
├── data/
│   ├── raw/                 # Original clinical datasets (.csv)
│   └── processed/           # Standardized tabular datasets formatted for loading
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis & preprocessing verification
│   └── 02_results_analysis.ipynb # Interactive statistical inspection of results
├── scripts/
│   ├── run_full_pipeline.py  # MASTER: 1-click orchestration of the entire pipeline
│   ├── automate_benchmark.py # PHASE 1: Master execution script for 3x5 CV benchmark
│   ├── generate_report.py    # PHASE 2: Statistical aggregator, table & plot generator
│   └── run_ablation_study.py # PHASE 3: Data scarcity degradation experiment runner
├── src/
│   ├── data/
│   │   └── loader.py        # PyTorch Dataset wrappers and leakage-free Preprocessor pipelines
│   ├── models/
│   │   ├── base.py          # Abstract base class enforcing unified logging & parameter counting
│   │   ├── mlp.py           # StandardMLP & TabResNet architectures with GELU and BatchNorm
│   │   └── kan_variants/    # 9 specialized Kolmogorov-Arnold Network implementations
│   │       ├── base_kan.py       # Core KAN base architecture
│   │       ├── cheby_kan.py      # Chebyshev polynomial KAN
│   │       ├── fast_kan.py       # Fast Radial Basis Function KAN
│   │       ├── gram_kan.py       # Gram polynomial KAN
│   │       ├── jacobi_kan.py     # Jacobi polynomial KAN
│   │       ├── legendre_kan.py   # Legendre polynomial KAN
│   │       ├── relu_kan.py       # ReLU-based spline approximation KAN
│   │       ├── tab_kan.py        # Tabular-optimized KAN architecture
│   │       ├── taylor_kan.py     # Taylor series expansion KAN
│   │       └── wav_kan.py        # Wavelet-based KAN (WavKAN)
│   ├── training/
│   │   ├── cross_validation.py # Repeated Stratified K-Fold engine with hermetic preprocessors
│   │   ├── early_stopping.py   # Validation loss monitor with deep-copy state restoration
│   │   ├── hpo.py              # Lightweight Optuna tuning engine with inner validation splitting
│   │   └── trainer.py          # Unified PyTorch & Scikit-Learn training/evaluator module
│   └── evaluation/
│       ├── metrics.py       # Clinical evaluation metrics (MCC, AUROC, Brier Score, Confusion Matrix)
│       ├── stats.py         # Frequentist statistics (Wilcoxon signed-rank with Holm-Bonferroni)
│       ├── bayesian_stats.py# Bayesian correlated t-test with Region of Practical Equivalence (ROPE)
│       ├── plot_radar.py    # Multi-dimensional architecture trade-off Radar (Spider) charts
│       ├── plot_bayesian.py # Bayesian posterior probability distribution visualizer
│       └── plot_ablation.py # Data scarcity performance degradation curve plotter
├── results/                 # Auto-generated outputs (CSVs, JSON logs, saved models, PNG plots)
├── requirements.txt         # Core dependencies
└── README.md                # Project documentation
```

---

## 4. Model Registry & Medical Datasets

### Model Architecture Suite (`MODELS` Registry)
The benchmark compares **11 distinct model architectures** across neural and non-neural paradigms:
1. **`StandardMLP`**: Classic Multi-Layer Perceptron baseline with Batch Normalization, Dropout, and GELU activations.
2. **`RandomForest`**: Non-neural tree ensemble baseline (`scikit-learn` Random Forest Classifier).
3. **`WavKAN`**: Wavelet-based Kolmogorov-Arnold Network utilizing Difference of Gaussians (DoG) or Mexican Hat wavelets.
4. **`FastKAN`**: Radial Basis Function (RBF) approximation KAN designed for accelerated forward training.
5. **`ChebyKAN`**: Orthogonal Chebyshev polynomial parametrization KAN.
6. **`JacobiKAN`**: Jacobi orthogonal polynomial KAN.
7. **`LegendreKAN`**: Legendre polynomial KAN.
8. **`GramKAN`**: Gram polynomial KAN.
9. **`TaylorKAN`**: Taylor series expansion KAN.
10. **`ReLUKAN`**: Piecewise ReLU linear spline approximation KAN.
11. **`TabKAN`**: Domain-tailored tabular KAN architecture with feature-wise gating.

### Clinical Benchmark Datasets
The suite evaluates models across 7 diverse clinical tasks spanning binary and multiclass diagnosis, variable sample volumes, and differing feature distributions:

| Dataset Name | Filename | Target Column | Sample Size ($N$) | Features ($d$) | Clinical Domain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Breast Cancer** | `breast_cancer_processed.csv` | `Diagnosis` | 569 | 30 | Oncology (Wisconsin FNA biopsy features) |
| **Pima Diabetes** | `pima_diabetes_processed.csv` | `class` | 768 | 8 | Endocrinology (Metabolic diagnostic markers) |
| **Heart Disease** | `heart_disease_processed.csv` | `num` | 303 | 13 | Cardiology (UCI Cleveland angiography records) |
| **Chronic Kidney Disease**| `chronic_kidney_disease_processed.csv`| `class` | 400 | 24 | Nephrology (Biochemical blood & urine tests) |
| **Parkinson's Disease** | `parkinsons_processed.csv` | `status` | 195 | 22 | Neurology (Biomedical voice acoustic measurements)|
| **Cervical Cancer** | `cervical_cancer_processed.csv` | `Biopsy` | 858 | 32 | Gynecology (Demographic & clinical risk factors) |
| **Cardiotocography** | `cardiotocography_processed.csv` | `NSP` | 2,126 | 21 | Obstetrics (Fetal heart rate & uterine contraction)|

---

## 5. Step-by-Step Reproduction Guide

Follow these sequential steps to reproduce the full scientific findings, statistical tests, and publication figures from scratch.

### Step 0: Environment Setup
Ensure Python 3.10+ is installed. Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/kan_mlp_comparison.git
cd kan_mlp_comparison

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option A: 1-Click Automated Reproduction (Recommended)
To execute the entire experimental lifecycle sequentially (Main 3×5 CV Benchmark -> Dataset Scarcity Ablation -> Report & Publication Plot Generation) with automated time tracking and real-time logging, run the master orchestration script:
```bash
python scripts/run_full_pipeline.py
```

### Option B: Manual Step-by-Step Execution
If you prefer to execute individual phases of the pipeline separately:

#### Step 1: Execute Main 3×5 CV Benchmark
Run the automated benchmark script. This executes 3×5 Repeated Stratified CV across all 7 datasets and 11 model architectures (incorporating per-fold Optuna HPO, Early Stopping, and artifact serialization):
```bash
python scripts/automate_benchmark.py
```
*Note: This generates granular validation records in `results/benchmark_master_YYYYMMDD-HHMM.csv` and serialized weights/confusion matrices inside `results/artifacts/`.*

### Step 2: Generate Analytical Report, Statistics & Publication Figures
Once the main benchmark finishes, execute the analytical report generator:
```bash
python scripts/generate_report.py
```
This automatically parses the latest benchmark master file and produces:
- **`results/summary_metrics.csv`**: Aggregated `Mean ± Std` table across MCC, AUROC, F1, Brier Score, Inference Time, and Parameter counts.
- **`results/stats_wilcoxon_posthoc.csv`**: Frequentist Wilcoxon signed-rank test results against StandardMLP with **Holm-Bonferroni correction**.
- **`results/stats_bayesian_rope.csv`**: Correlated Bayesian $t$-test estimation evaluating posterior probabilities ($P(\text{KAN} > \text{MLP})$, $P(\text{MLP} > \text{KAN})$, and $P(\text{ROPE})$).
- **`results/plots/`**: High-resolution PNG figures including metric boxplots, clinical confusion matrices, learning curves, multi-dimensional architecture radar charts, and Bayesian posterior distributions.

### Step 3: Run Data Scarcity Ablation Study (Degradation Analysis)
To test architectural robustness under extreme clinical data scarcity (*Small Data* regime), execute the subsampling ablation study:
```bash
python scripts/run_ablation_study.py
```
This systematically trains all 11 architectures across decreasing training data fractions (`1.0`, `0.8`, `0.6`, `0.4`, `0.2`, `0.1`), preserving stratified class ratios.
Once completed, re-run `python scripts/generate_report.py` to generate the data scarcity degradation curves (`results/plots/ablation_*_*.png`).

---

## 6. Statistical Methodology Note

Evaluating machine learning models across a limited number of clinical datasets ($n \le 7$) introduces fundamental statistical challenges when relying solely on frequentist hypothesis testing:
- **Frequentist Power Limitations**: As demonstrated by Demšar (2006), non-parametric frequentist tests (such as the Wilcoxon signed-rank test) suffer from severe statistical powerlessness when the number of datasets is small ($n < 10$). Even when an architecture consistently outperforms a baseline across all 7 datasets, frequentist $p$-values may fail to cross standard alpha thresholds ($\alpha = 0.05$) due to discrete rank distribution limits.
- **Bayesian ROPE Analysis**: To provide rigorous, publication-grade statistical inference without $p$-value hacking, our pipeline implements the **Correlated Bayesian Signed-Rank $t$-Test** (Corani & Benavoli, 2015; Benavoli et al., 2017). By defining a **Region of Practical Equivalence (ROPE)** (e.g., $\pm 0.01$ MCC or AUROC), the Bayesian framework computes exact posterior probabilities:
  - $P(\text{Model A is superior})$
  - $P(\text{Model B is superior})$
  - $P(\text{Models are practically equivalent within clinical tolerance})$

This dual reporting structure guarantees full transparency and adheres to the highest statistical standards of top-tier medical AI journals (e.g., *Nature Digital Medicine*, *Lancet Digital Health*, *IEEE Transactions on Medical Imaging*).

---

## 7. License & Citation

This research code is released under the **MIT License**. If you utilize this benchmarking suite, data loaders, or KAN implementations in your academic research, please cite:

```bibtex
@article{kan_vs_mlp_medical_2026,
  title={Evaluating Kolmogorov-Arnold Networks (KAN) vs. Multi-Layer Perceptrons (MLP) on Medical Tabular Data},
  author={Core Benchmark Contributors},
  journal={Medical AI Research Repository},
  year={2026},
  url={https://github.com/your-username/kan_mlp_comparison}
}
```
