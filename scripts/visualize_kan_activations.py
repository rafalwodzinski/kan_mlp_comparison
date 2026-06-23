import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Dodanie ścieżki, aby Python widział folder src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import get_data_and_preprocessor
from src.models.kan_variants.wav_kan import WavKAN

def mexican_hat_wavelet(x: torch.Tensor) -> torch.Tensor:
    x_sq = x ** 2
    return (1.0 - x_sq) * torch.exp(-0.5 * x_sq)

def main():
    print("Generowanie wizualizacji funkcji aktywacji WavKAN (Zbiór Parkinson's)...")
    device = torch.device("cpu") # Wizualizację bezpiecznie robimy na CPU
    
    data_path = "data/processed/parkinsons_processed.csv"
    dataset_filename = "parkinsons_processed.csv"
    
    if not os.path.exists(data_path):
        print(f"Błąd: Nie znaleziono pliku {data_path}.")
        return

    # 1. Pozyskanie nazw cech i wymiarów
    X_raw, y_raw, preprocessor = get_data_and_preprocessor(data_path, dataset_filename)
    preprocessor.fit(X_raw)
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.split('__')[-1] for name in feature_names]
    
    input_dim = len(feature_names)
    num_classes = len(np.unique(y_raw))
    is_binary = (num_classes == 2)
    output_dim = 1 if is_binary else num_classes
    
    # 2. Inicjalizacja modelu i wczytanie wag
    model = WavKAN(input_dim=input_dim, output_dim=output_dim)
    kan_path = "results/artifacts/parkinsons/WavKAN/parkinsons_WavKAN_Fold1_weights.pth"
    
    if not os.path.exists(kan_path):
        print(f"Błąd: Brak pliku {kan_path}")
        return
        
    model.load_state_dict(torch.load(kan_path, map_location=device))
    model.eval()
    
    # 3. Ekstrakcja parametrów PIERWSZEJ warstwy KAN
    # Struktura: nn.Sequential(WavKANLinear, LayerNorm, WavKANLinear, ...)
    first_layer = model.network[0]
    
    base_weight = first_layer.base_weight.detach().cpu()         # [out, in]
    translation = first_layer.translation.detach().cpu()         # [out, in, wavelets]
    scale = first_layer.scale.detach().cpu()                     # [out, in, wavelets]
    wavelet_weight = first_layer.wavelet_weight.detach().cpu()   # [out, in, wavelets]
    
    # Szukamy najciekawszych (najbardziej aktywnych) cech medycznych.
    # Obliczamy sumę absolutnych wag falkowych dla każdej cechy
    feature_activity = torch.sum(torch.abs(wavelet_weight), dim=(0, 2))
    top_indices = torch.argsort(feature_activity, descending=True)[:6].numpy()
    
    # Wybieramy jeden węzeł ukryty docelowy do wizualizacji (np. j=0)
    target_node = 0
    
    # 4. Generowanie punktów X i obliczanie Y
    x_vals = torch.linspace(-3, 3, 300)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature_idx in enumerate(top_indices):
        fname = feature_names[feature_idx]
        ax = axes[idx]
        
        # Obliczenie funkcji bazowej (SiLU * waga)
        base_val = F.silu(x_vals) * base_weight[target_node, feature_idx]
        
        # Obliczenie sumy funkcji falkowych
        wave_val = torch.zeros_like(x_vals)
        num_wavelets = wavelet_weight.shape[2]
        
        for w in range(num_wavelets):
            t = translation[target_node, feature_idx, w]
            s = scale[target_node, feature_idx, w].abs() + 1e-8
            weight = wavelet_weight[target_node, feature_idx, w]
            
            x_scaled = (x_vals - t) / s
            wave_val += mexican_hat_wavelet(x_scaled) * weight
            
        total_val = base_val + wave_val
        
        # Rysowanie składowych i sumy
        ax.plot(x_vals.numpy(), base_val.numpy(), linestyle=':', color='gray', alpha=0.7, label='Funkcja Bazowa (SiLU)')
        ax.plot(x_vals.numpy(), wave_val.numpy(), linestyle='--', color='orange', alpha=0.8, label='Suma Falek (Wavelets)')
        ax.plot(x_vals.numpy(), total_val.numpy(), linestyle='-', color='blue', linewidth=2.5, label='Całkowita Aktywacja KAN')
        
        ax.set_title(f"Cecha: {fname}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Wartość Cechy (z-score)", fontsize=10)
        ax.set_ylabel(f"Sygnał do Węzła Ukrytego {target_node}", fontsize=10)
        ax.grid(linestyle='--', alpha=0.5)
        if idx == 0:
            ax.legend(fontsize=9)
            
    plt.suptitle("Prawdziwa Czarna Skrzynka: 1D Funkcje Aktywacji na Krawędziach (WavKAN)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs("results/plots", exist_ok=True)
    plot_path = "results/plots/kan_activation_functions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Zakończono! Wykres zapisano w: {plot_path}")

if __name__ == "__main__":
    main()
