"""
Precision-Recall (PR) Curve Generator for Best KAN vs. Baselines.
Produces standalone publication-ready vector PDFs for each medical dataset.
Calculates interpolated/exact PR curves across cross-validation folds
using saved preprocessors and trained model weights (zero retraining).
"""

import os
import sys
import glob
import json
import joblib
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import get_data_and_preprocessor
from src.evaluation.plot_style import apply_publication_style, save_pdf
from scripts.automate_benchmark import MODELS

def find_latest_benchmark() -> str:
    files = glob.glob("results/benchmark_master_*.csv")
    if not files:
        raise FileNotFoundError("No benchmark_master_*.csv files found in results/")
    return max(files, key=os.path.getmtime)

def load_and_predict(dataset_name: str, model_name: str, X_val_raw: pd.DataFrame, repeat: int, fold: int):
    """Loads fitted preprocessor and trained model weights to predict validation set probabilities."""
    artifact_dir = f"results/artifacts/{dataset_name}/{model_name}"
    prep_path = os.path.join(artifact_dir, f"{dataset_name}_{model_name}_Repeat{repeat}_Fold{fold}_preprocessor.joblib")
    if not os.path.exists(prep_path):
        return None
    preprocessor = joblib.load(prep_path)
    X_val_clean = preprocessor.transform(X_val_raw)

    params_path = os.path.join(artifact_dir, f"{dataset_name}_{model_name}_Repeat{repeat}_Fold{fold}_best_params.json")
    best_params = {}
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            best_params = json.load(f)

    model_class = MODELS.get(model_name)
    if model_class is None:
        return None

    if model_name == "RandomForest":
        weights_path = os.path.join(artifact_dir, f"{dataset_name}_{model_name}_Repeat{repeat}_Fold{fold}_weights.joblib")
        if not os.path.exists(weights_path):
            return None
        rf_model = joblib.load(weights_path)
        probs = rf_model.predict_proba(X_val_clean)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs

    # PyTorch neural models (MLP and KAN variants)
    weights_path = os.path.join(artifact_dir, f"{dataset_name}_{model_name}_Repeat{repeat}_Fold{fold}_weights.pth")
    if not os.path.exists(weights_path):
        return None

    input_dim = X_val_clean.shape[1]
    model_kwargs = {
        'input_dim': input_dim,
        'output_dim': 1
    }
    if 'hidden_dim' in best_params:
        model_kwargs['hidden_dims'] = [best_params['hidden_dim'], best_params['hidden_dim'] // 2]
    for kan_param in ('grid_size', 'num_grids', 'degree', 'num_wavelets'):
        if kan_param in best_params:
            model_kwargs[kan_param] = best_params[kan_param]

    try:
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X_val_clean, dtype=torch.float32)
            logits = model(X_tensor)
            probs = torch.sigmoid(logits.squeeze(dim=-1)).cpu().numpy()
            if probs.ndim == 0:
                probs = np.array([probs.item()])
        return probs
    except Exception as e:
        print(f"[Warning] Failed loading {model_name} on {dataset_name} R{repeat}F{fold}: {e}")
        return None

def generate_precision_recall_curves(output_dir: str = "results/plots"):
    """
    Generates single publication-ready PDF Precision-Recall curves comparing
    StandardMLP, RandomForest, and the Best KAN variant for each binary dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    bench_file = find_latest_benchmark()
    df_bench = pd.read_csv(bench_file)

    DATASETS_DIR = "data/processed/"
    dataset_files = sorted([f for f in os.listdir(DATASETS_DIR) if f.endswith('_processed.csv')])

    apply_publication_style("whitegrid")
    generated_files = []

    for dataset_file in dataset_files:
        dataset_name = dataset_file.replace("_processed.csv", "")
        df_ds = df_bench[df_bench['dataset'] == dataset_name]
        if df_ds.empty:
            continue

        # Check if binary classification
        data_path = os.path.join(DATASETS_DIR, dataset_file)
        X_raw, y_raw, _ = get_data_and_preprocessor(data_path, dataset_file)
        unique_classes = np.unique(y_raw)
        if len(unique_classes) != 2:
            # Skip multiclass datasets like cardiotocography for binary PR curves
            continue

        # Strictly highlight StandardMLP vs. TabKAN (and RandomForest baseline) for narrative consistency
        target_kan = "TabKAN"
        if target_kan not in df_ds['model'].unique():
            print(f"[Warning] {target_kan} not found for {dataset_name}. Skipping PR curve.")
            continue

        target_models = ["StandardMLP", target_kan]
        if "RandomForest" in df_ds['model'].unique():
            target_models.append("RandomForest")

        # Collect out-of-fold predictions across Repeat 1 (5 folds)
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
        model_predictions = {m: {'y_true': [], 'y_prob': []} for m in target_models}

        for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_raw, y_raw)):
            fold = fold_idx + 1
            X_val_raw = X_raw.iloc[val_idx]
            y_val = y_raw[val_idx]

            for model_name in target_models:
                probs = load_and_predict(dataset_name, model_name, X_val_raw, repeat=1, fold=fold)
                if probs is not None:
                    model_predictions[model_name]['y_true'].extend(y_val)
                    model_predictions[model_name]['y_prob'].extend(probs)

        # Plot PR curves
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = {"StandardMLP": "#1f77b4", "TabKAN": "#d62728", "RandomForest": "#2ca02c"}
        linestyles = {"StandardMLP": "-", "TabKAN": "-", "RandomForest": "--"}

        has_data = False
        prevalence = np.mean(y_raw == 1)

        for model_name in target_models:
            y_t = np.array(model_predictions[model_name]['y_true'])
            y_p = np.array(model_predictions[model_name]['y_prob'])
            if len(y_t) > 0 and len(y_p) > 0:
                has_data = True
                prec, rec, _ = precision_recall_curve(y_t, y_p)
                auprc = average_precision_score(y_t, y_p)
                label = f"{model_name} (AUPRC = {auprc:.3f})"
                ax.plot(rec, prec, label=label, color=colors.get(model_name, "black"),
                        linestyle=linestyles.get(model_name, "-"), linewidth=2.5)

        if not has_data:
            plt.close(fig)
            continue

        # Baseline: No-Skill classifier (prevalence line)
        ax.axhline(y=prevalence, color='navy', linestyle=':', label=f"No-Skill Baseline ({prevalence:.2f})")

        ax.set_xlabel("Recall", fontsize=15, labelpad=10)
        ax.set_ylabel("Precision", fontsize=15, labelpad=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()

        out_path = save_pdf(os.path.join(output_dir, f"pr_curve_{dataset_name}.pdf"), fig)
        plt.close(fig)
        print(f"[Info] Saved Precision-Recall curve to: {out_path}")
        
        # Synchronize directly to manuscript/figures/ if available
        manuscript_dir = "manuscript/figures"
        if os.path.exists(manuscript_dir) and os.path.abspath(output_dir) != os.path.abspath(manuscript_dir):
            dest_pdf = os.path.join(manuscript_dir, f"pr_curve_{dataset_name}.pdf")
            shutil.copy2(out_path, dest_pdf)
            print(f"[Info] Synchronized Precision-Recall curve to: {dest_pdf}")
            
        generated_files.append(out_path)

    return generated_files

if __name__ == "__main__":
    generate_precision_recall_curves()
