# Re-evaluating Kolmogorov-Arnold Networks for Clinical Decision Support: A Multi-Dataset Tabular Benchmark

**Authors:** Rafał Wodziński, Paweł Szczerbiak  
**Affiliation:** Department of Computer Science, Faculty of Computer Science and Mathematics, Cracow University of Technology, Kraków, Poland. Sano Science, Cracow, Poland.
**Manuscript:** [Under Review / Pre-print]

---

## 1. Project Overview
This repository contains the complete, reproducible codebase for a rigorous empirical benchmark evaluating Kolmogorov-Arnold Networks (KANs) against classical Multi-Layer Perceptrons (MLPs) and Tree-based methods (Random Forest) on medical tabular data. The study addresses the lack of comprehensive evaluation of KANs in applied, real-world clinical tasks characterized by data scarcity, missing values, and class imbalance.

Our evaluation pipeline employs a **3×5 Repeated Stratified K-Fold** cross-validation, combined with nested Bayesian hyperparameter optimization (HPO), hermetic preprocessing to prevent data leakage, and principled statistical inference using both Frequentist and Bayesian frameworks (Correlated Bayesian t-Test with ROPE).

## 2. Evaluated Architectures (11 Models)

The benchmark systematically compares 11 model architectures:

**Kolmogorov-Arnold Network (KAN) Variants (9):**
*   **FastKAN:** Spline approximation via Gaussian Radial Basis Functions.
*   **WavKAN:** Discrete wavelet transforms for edge activations.
*   **ReLUKAN:** Reconstructed spline functions from GPU-native ReLU operations.
*   **ChebyKAN:** Chebyshev polynomial bases.
*   **TaylorKAN:** Taylor series expansions.
*   **JacobiKAN:** Orthogonal Jacobi polynomial families.
*   **LegendreKAN:** Orthogonal Legendre polynomial families.
*   **GramKAN:** Gram polynomial activations.
*   **TabKAN:** Domain-specific KAN architecture with specialized feature preprocessing modules.

**Baselines (2):**
*   **StandardMLP:** Fully-connected network (GELU, Batch Normalization, Dropout) — the primary neural baseline.
*   **Random Forest:** Non-neural ensemble tabular baseline.

## 3. Key Findings

*   **Empirical Equivalence in Discrimination:** The StandardMLP (Grand Mean AUROC: 0.9517) and the KAN family (Grand Mean AUROC: 0.9447) are practically indistinguishable in predictive performance.
*   **Calibration:** The Brier Scores for leading KAN variants and the StandardMLP are within 0.002 of each other, confirming no clinically meaningful calibration advantage for KANs.
*   **Computational Cost (Latency & Parameters):** All KAN variants impose a significant computational penalty. Per-sample inference latency is 1.19× to 1.42× higher than the MLP, and the trainable parameter count inflates by 3.0× to 14.2×.
*   **Pareto Dominance:** The StandardMLP occupies the sole non-dominated position on the performance-efficiency Pareto frontier (highest grand mean AUROC at the lowest inference cost).

## 4. Quickstart & Reproducibility

This project strictly adheres to Reproducible Research standards.

**Environment Setup:**
```bash
conda create -n kan_project python=3.10
conda activate kan_project
pip install -r requirements.txt
```

**Run Full Benchmark:**
Execute the complete pipeline (data loading, preprocessing, HPO, training, evaluation):
```bash
python scripts/automate_benchmark.py
```

**Generate Statistical Reports:**
Run statistical significance testing (Wilcoxon and Bayesian ROPE) on benchmark results:
```bash
python scripts/generate_report.py
```

**Generate Advanced Visualizations:**
Produce publication-ready figures (Pareto frontiers, Critical Difference diagrams, Boxplots):
```bash
python scripts/plot_advanced.py
```

## 5. Repository Structure

```
.
├── manuscript/             # LaTeX source files for the publication
├── src/                    # Core Python modules
│   ├── data/               # Dataset loaders and hermetic preprocessors
│   ├── models/             # PyTorch architecture implementations (MLP, KAN variants)
│   ├── training/           # Nested HPO, Early Stopping, and Cross-Validation
│   └── evaluation/         # Statistical testing (Bayesian ROPE, Wilcoxon) and Metrics
├── scripts/                # Entry-point scripts for automation and reporting
│   ├── automate_benchmark.py
│   ├── generate_report.py
│   └── plot_advanced.py
├── results/                # Output artifacts (CSVs, figures, raw metrics)
├── docs/                   # Documentation (see CODEBASE_GUIDE.md)
└── requirements.txt        # Categorized project dependencies
```
