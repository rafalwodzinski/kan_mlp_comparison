"""
Module implementing a Case Study for the Interpretability phenomenon.
Compares the importance of individual features (Feature Importance) calculated based on the permutation method
for the classic StandardMLP model and the WavKAN model on the Parkinson's Disease dataset.
The final plot is saved to the results/plots/ folder.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef

# Add path so Python sees the src folder and its modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import get_data_and_preprocessor
from src.models.mlp import StandardMLP
from src.models.kan_variants.wav_kan import WavKAN
from src.evaluation.interpretability import ModelInterpreter

def main():
    """
    Main execution function of the script. Responsible for:
    1. Loading the dataset and validation split (recreating Fold 1).
    2. Loading pre-trained MLP and KAN model weights.
    3. Performing Permutation Feature Importance analysis.
    4. Generating and saving a bar chart for TOP 10 features.
    """
    print("Starting Case Study for Parkinson's dataset...")
    # Device selection (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data loading
    data_path = "data/processed/parkinsons_processed.csv"
    dataset_filename = "parkinsons_processed.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found. Make sure the data is available.")
        return

    # Get raw data and scikit-learn preprocessor
    X_raw, y_raw, preprocessor = get_data_and_preprocessor(data_path, dataset_filename)
    
    # 2. Reproduction of split into Fold 1
    # We use StratifiedKFold with seed 42, which gives an identical split as in the training process
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(X_raw, y_raw))
    
    X_train_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
    y_train, y_val = y_raw[train_idx], y_raw[val_idx]
    
    # 3. Data transformation (no data leakage - fitting only on train)
    X_train_clean = preprocessor.fit_transform(X_train_raw)
    X_val_clean = preprocessor.transform(X_val_raw)
    
    # Feature name extraction (cleaning prefixes added by ColumnTransformer)
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.split('__')[-1] for name in feature_names]
    
    # Conversion to tensor format required by PyTorch
    X_val_t = torch.tensor(X_val_clean, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    # 4. Configuration and definition of model structures
    input_dim = X_train_clean.shape[1]
    num_classes = len(np.unique(y_raw))
    is_binary = (num_classes == 2)
    output_dim = 1 if is_binary else num_classes
    
    # Initialization of classic MLP and KAN network (wavelet variant)
    mlp = StandardMLP(input_dim=input_dim, output_dim=output_dim)
    kan = WavKAN(input_dim=input_dim, output_dim=output_dim)
    
    # Paths to trained weights from the best fold (Fold 1)
    mlp_path = "results/artifacts/parkinsons/StandardMLP/parkinsons_StandardMLP_Fold1_weights.pth"
    kan_path = "results/artifacts/parkinsons/WavKAN/parkinsons_WavKAN_Fold1_weights.pth"
    
    if not os.path.exists(mlp_path) or not os.path.exists(kan_path):
        print("Error: Weights files not found in results/artifacts/parkinsons/")
        return

    # Loading dumped weight states
    mlp.load_state_dict(torch.load(mlp_path, map_location=device))
    kan.load_state_dict(torch.load(kan_path, map_location=device))
    
    # 5. Calculating Permutation Feature Importance (Fair comparison)
    print("Calculating Feature Importance for StandardMLP...")
    mlp_interpreter = ModelInterpreter(mlp, device)
    # Permutation randomly repeated 10 times for stability
    imp_mlp = mlp_interpreter.permutation_feature_importance(X_val_t, y_val_t, feature_names, metric_fn=matthews_corrcoef, n_repeats=10)
    
    print("Calculating Feature Importance for WavKAN...")
    kan_interpreter = ModelInterpreter(kan, device)
    imp_kan = kan_interpreter.permutation_feature_importance(X_val_t, y_val_t, feature_names, metric_fn=matthews_corrcoef, n_repeats=10)
    
    # 6. Visualization of TOP 10 features (sorting by importance relative to dominant WavKAN)
    top_n = min(10, len(feature_names))
    sort_idx = np.argsort(imp_kan)[::-1][:top_n]
    
    # Selection of names and values for top features
    top_features = [feature_names[i] for i in sort_idx]
    top_kan_vals = imp_kan[sort_idx]
    top_mlp_vals = imp_mlp[sort_idx]
    
    x = np.arange(len(top_features))
    width = 0.35  # Bar width
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, top_mlp_vals, width, label='StandardMLP', color='#888888', edgecolor='black')
    rects2 = ax.bar(x + width/2, top_kan_vals, width, label='WavKAN', color='#1f77b4', edgecolor='black')
    
    # Aesthetics and plot annotations
    ax.set_ylabel('MCC Drop (Feature Importance)', fontsize=12)
    ax.set_title("TOP 10 Features: WavKAN vs StandardMLP (Parkinson's Dataset - Fold 1)", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save image to results directory
    os.makedirs("results/plots", exist_ok=True)
    plot_path = "results/plots/case_study_feature_importance.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Finished! Plot saved in: {plot_path}")

if __name__ == "__main__":
    main()
