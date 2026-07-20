import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Any

class ModelInterpreter:
    """
    Module responsible for extracting knowledge from trained models.
    Supports SHAP analysis for MLP and marginal function visualization for KAN.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()



    def permutation_feature_importance(self, X: torch.Tensor, y: torch.Tensor, feature_names: List[str], metric_fn, n_repeats: int = 5) -> np.ndarray:
        """
        Calculates Permutation Feature Importance in a model-agnostic way.
        Works on PyTorch tensors and supports binary or multiclass classification.
        """
        self.model.eval()
        X = X.to(self.device)
        y = y.to(self.device).cpu().numpy()
        
        # Calculate baseline score
        with torch.no_grad():
            baseline_outputs = self.model(X)
            if baseline_outputs.shape[1] == 1:
                baseline_preds = (torch.sigmoid(baseline_outputs).squeeze() > 0.5).cpu().numpy().astype(int)
            else:
                baseline_preds = torch.argmax(baseline_outputs, dim=1).cpu().numpy()
        
        baseline_score = metric_fn(y, baseline_preds)
        
        importances = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            scores_permuted = []
            for _ in range(n_repeats):
                X_permuted = X.clone()
                # Permute i-th column
                idx = torch.randperm(X.shape[0])
                X_permuted[:, i] = X_permuted[idx, i]
                
                with torch.no_grad():
                    outputs = self.model(X_permuted)
                    if outputs.shape[1] == 1:
                        preds = (torch.sigmoid(outputs).squeeze() > 0.5).cpu().numpy().astype(int)
                    else:
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                scores_permuted.append(metric_fn(y, preds))
            
            importances[i] = baseline_score - np.mean(scores_permuted)
            
        return importances

    def extract_kan_activations(self, data: torch.Tensor, layer_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Attaches a hook (Forward Hook) on the selected KAN layer, 
        to capture input (x) and output (phi(x)) values on the edges.
        """
        data = data.to(self.device)
        activations = {}

        def hook_fn(module, input, output):
            # We save the raw input to the layer and its output
            activations['input'] = input[0].detach().cpu().numpy()
            activations['output'] = output.detach().cpu().numpy()

        # Find the appropriate layer in Sequential (assuming structure from our base.py)
        # Skip LayerNorm and Dropout layers
        kan_layers = [module for module in self.model.network.modules() if not isinstance(module, (nn.Sequential, nn.LayerNorm, nn.Dropout))]
        
        if layer_index >= len(kan_layers):
            raise ValueError(f"Model has only {len(kan_layers)} KAN layers.")
            
        target_layer = kan_layers[layer_index]
        
        # Register hook
        hook = target_layer.register_forward_hook(hook_fn)
        
        # Pass data through the model (trigger hook)
        with torch.no_grad():
            self.model(data)
            
        # Remove hook after collecting data
        hook.remove()
        
        return activations

    def plot_kan_edge_functions(self, data: torch.Tensor, feature_names: List[str], layer_index: int = 0):
        """
        Visualizes learned 1D activation functions on the edges for the first KAN layer.
        Shows how the model non-linearly transforms original medical features.
        """
        acts = self.extract_kan_activations(data, layer_index)
        x_vals = acts['input']
        
        # We assume we check the impact of input features on hidden nodes
        num_features = x_vals.shape[1]
        
        fig, axes = plt.subplots(1, num_features, figsize=(4 * num_features, 4))
        if num_features == 1:
            axes = [axes]
            
        for i in range(num_features):
            # Sort x values for smooth plotting
            x_feature = x_vals[:, i]
            sort_idx = np.argsort(x_feature)
            
            # In the first layer we usually show impact on hidden nodes (visual aggregation)
            axes[i].scatter(x_feature[sort_idx], acts['output'][sort_idx, i % acts['output'].shape[1]], alpha=0.5, s=10)
            axes[i].set_title(f"Function for: {feature_names[i]}")
            axes[i].set_xlabel("Value after standardization")
            axes[i].set_ylabel("Value after KAN activation")
            axes[i].grid(True, linestyle='--', alpha=0.6)
            
        plt.tight_layout()
        plt.show()