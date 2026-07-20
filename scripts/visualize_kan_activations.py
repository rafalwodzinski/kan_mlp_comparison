"""
Module responsible for visualization of one-dimensional activation functions (Edge Activation Functions) 
in KAN family networks. Allows to 'look under the hood' of the model and understand 
what specific clinical anomalies are detected by the model (XAI - Explainable AI).
The script operates on the WavKAN model trained on the Parkinson's Disease dataset.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Add path so Python sees the src folder and its modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import get_data_and_preprocessor
from src.models.kan_variants.wav_kan import WavKAN

def mexican_hat_wavelet(x: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mathematical value of the 'Mexican Hat' wavelet function
    (second derivative of the Gaussian function).
    
    Args:
        x (torch.Tensor): Normalized input vector.
    Returns:
        torch.Tensor: Values after passing through the wavelet.
    """
    x_sq = x ** 2
    return (1.0 - x_sq) * torch.exp(-0.5 * x_sq)

def main():
    """
    Main execution function responsible for:
    1. Loading the Parkinson's Disease dataset to retrieve feature metadata.
    2. Loading the weights of the first layer of the WavKAN model (Fold 1).
    3. Feature selection (most "active" according to the sum of weights on wavelets).
    4. Analytical recreation of 1D functions on edges for these features.
    5. Saving the subplot matrix to a visualization file.
    """
    print("Generating visualization of WavKAN activation functions (Parkinson's Dataset)...")
    # Visualization is done safely and fastest locally on CPU
    device = torch.device("cpu")
    
    data_path = "data/processed/parkinsons_processed.csv"
    dataset_filename = "parkinsons_processed.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found.")
        return

    # 1. Getting feature names and input dimensions
    X_raw, y_raw, preprocessor = get_data_and_preprocessor(data_path, dataset_filename)
    preprocessor.fit(X_raw)
    feature_names = preprocessor.get_feature_names_out()
    # Removing unnecessary prefixes from ColumnTransformer (e.g., 'num__')
    feature_names = [name.split('__')[-1] for name in feature_names]
    
    input_dim = len(feature_names)
    num_classes = len(np.unique(y_raw))
    is_binary = (num_classes == 2)
    output_dim = 1 if is_binary else num_classes
    
    # 2. WavKAN model initialization and loading pre-trained weights
    model = WavKAN(input_dim=input_dim, output_dim=output_dim)
    kan_path = "results/artifacts/parkinsons/WavKAN/parkinsons_WavKAN_Fold1_weights.pth"
    
    if not os.path.exists(kan_path):
        print(f"Error: Missing file {kan_path}")
        return
        
    model.load_state_dict(torch.load(kan_path, map_location=device))
    model.eval()
    
    # 3. Extraction of FIRST KAN layer parameters
    # Model structure: nn.Sequential(WavKANLinear, LayerNorm, WavKANLinear, ...)
    first_layer = model.network[0]
    
    base_weight = first_layer.base_weight.detach().cpu()         # Global trend [out, in]
    translation = first_layer.translation.detach().cpu()         # Wavelet locations [out, in, wavelets]
    scale = first_layer.scale.detach().cpu()                     # Wavelet widths [out, in, wavelets]
    wavelet_weight = first_layer.wavelet_weight.detach().cpu()   # Wavelet amplitudes [out, in, wavelets]
    
    # We are looking for the most interesting (most active) medical features.
    # We use the sum of absolute values of wavelet amplitudes across all nodes
    feature_activity = torch.sum(torch.abs(wavelet_weight), dim=(0, 2))
    # We take TOP 6 most decisive attributes
    top_indices = torch.argsort(feature_activity, descending=True)[:6].numpy()
    
    # We select the target hidden node for visualization (index j=0)
    target_node = 0
    
    # 4. Generating X points and calculating analytical Y curves
    x_vals = torch.linspace(-3, 3, 300) # z-score range +/- 3 std. deviations
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature_idx in enumerate(top_indices):
        fname = feature_names[feature_idx]
        ax = axes[idx]
        
        # Calculation of model base function (SiLU multiplied by global weight)
        base_val = F.silu(x_vals) * base_weight[target_node, feature_idx]
        
        # Calculation of iterative sum of all nonlinear wavelet functions
        wave_val = torch.zeros_like(x_vals)
        num_wavelets = wavelet_weight.shape[2]
        
        for w in range(num_wavelets):
            t = translation[target_node, feature_idx, w]
            s = scale[target_node, feature_idx, w].abs() + 1e-8 # Protection against division by zero
            weight = wavelet_weight[target_node, feature_idx, w]
            
            x_scaled = (x_vals - t) / s
            wave_val += mexican_hat_wavelet(x_scaled) * weight
            
        # Final learned function on the edge is their superposition
        total_val = base_val + wave_val
        
        # Drawing components and total sum
        ax.plot(x_vals.numpy(), base_val.numpy(), linestyle=':', color='gray', alpha=0.7, label='Base Function (SiLU)')
        ax.plot(x_vals.numpy(), wave_val.numpy(), linestyle='--', color='orange', alpha=0.8, label='Wavelet Sum (Wavelets)')
        ax.plot(x_vals.numpy(), total_val.numpy(), linestyle='-', color='blue', linewidth=2.5, label='Total KAN Activation')
        
        # Aesthetic settings of subplots
        ax.set_title(f"Feature: {fname}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Feature Value (z-score)", fontsize=10)
        ax.set_ylabel(f"Signal to Hidden Node {target_node}", fontsize=10)
        ax.grid(linestyle='--', alpha=0.5)
        if idx == 0:
            ax.legend(fontsize=9)
            
    plt.suptitle("The True Black Box: 1D Activation Functions on Edges (WavKAN)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 5. Image saving
    os.makedirs("results/plots", exist_ok=True)
    plot_path = "results/plots/kan_activation_functions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Finished! Plot saved in: {plot_path}")

if __name__ == "__main__":
    main()
