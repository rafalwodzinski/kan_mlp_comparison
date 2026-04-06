import torch
import torch.nn as nn
import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Any

class ModelInterpreter:
    """
    Moduł odpowiedzialny za wydobywanie wiedzy z wytrenowanych modeli.
    Obsługuje analizę SHAP dla MLP oraz wizualizację funkcji brzegowych dla KAN.
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def explain_with_shap(self, background_data: torch.Tensor, test_data: torch.Tensor, feature_names: List[str]):
        """
        Generuje i wyświetla wartości SHAP dla modeli typu czarna skrzynka (MLP).
        Wykorzystuje DeepExplainer zoptymalizowany pod sieci neuronowe.
        """
        background_data = background_data.to(self.device)
        test_data = test_data.to(self.device)

        # DeepExplainer wymaga funkcji, która zwraca logity
        explainer = shap.DeepExplainer(self.model, background_data)
        
        # Obliczenie wartości SHAP
        shap_values = explainer.shap_values(test_data)
        
        # Wizualizacja globalnego wpływu cech (Summary Plot)
        print("Generowanie wykresu SHAP Summary Plot...")
        shap.summary_plot(
            shap_values, 
            features=test_data.cpu().numpy(), 
            feature_names=feature_names,
            show=True
        )
        return shap_values

    def extract_kan_activations(self, data: torch.Tensor, layer_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Zaczepia hak (Forward Hook) na wybranej warstwie sieci KAN, 
        aby przechwycić wartości wejściowe (x) i wyjściowe (phi(x)) na krawędziach.
        """
        data = data.to(self.device)
        activations = {}

        def hook_fn(module, input, output):
            # Zapisujemy surowe wejście do warstwy i jej wyjście
            activations['input'] = input[0].detach().cpu().numpy()
            activations['output'] = output.detach().cpu().numpy()

        # Szukamy odpowiedniej warstwy w Sequential (zakładamy strukturę z naszego base.py)
        # Pomijamy warstwy LayerNorm i Dropout
        kan_layers = [module for module in self.model.network.modules() if not isinstance(module, (nn.Sequential, nn.LayerNorm, nn.Dropout))]
        
        if layer_index >= len(kan_layers):
            raise ValueError(f"Model posiada tylko {len(kan_layers)} warstw KAN.")
            
        target_layer = kan_layers[layer_index]
        
        # Rejestracja haka
        hook = target_layer.register_forward_hook(hook_fn)
        
        # Przepuszczenie danych przez model (wyzwolenie haka)
        with torch.no_grad():
            self.model(data)
            
        # Usunięcie haka po zebraniu danych
        hook.remove()
        
        return activations

    def plot_kan_edge_functions(self, data: torch.Tensor, feature_names: List[str], layer_index: int = 0):
        """
        Wizualizuje wyuczone jednowymiarowe funkcje aktywacji na krawędziach dla pierwszej warstwy KAN.
        Pokazuje, w jaki sposób model nieliniowo transformuje oryginalne cechy medyczne.
        """
        acts = self.extract_kan_activations(data, layer_index)
        x_vals = acts['input']
        
        # Zakładamy, że sprawdzamy wpływ cech wejściowych na ukryte węzły
        num_features = x_vals.shape[1]
        
        fig, axes = plt.subplots(1, num_features, figsize=(4 * num_features, 4))
        if num_features == 1:
            axes = [axes]
            
        for i in range(num_features):
            # Sortujemy wartości x dla płynnego wykresu
            x_feature = x_vals[:, i]
            sort_idx = np.argsort(x_feature)
            
            # W pierwszej warstwie zazwyczaj pokazujemy wpływ na węzły ukryte (agregacja wizualna)
            axes[i].scatter(x_feature[sort_idx], acts['output'][sort_idx, i % acts['output'].shape[1]], alpha=0.5, s=10)
            axes[i].set_title(f"Funkcja dla: {feature_names[i]}")
            axes[i].set_xlabel("Wartość po standaryzacji")
            axes[i].set_ylabel("Wartość po aktywacji KAN")
            axes[i].grid(True, linestyle='--', alpha=0.6)
            
        plt.tight_layout()
        plt.show()