import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from typing import List

# Importujemy klasę bazową
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class WavKANLinear(nn.Module):
    """
    Warstwa KAN wykorzystująca ciągłą transformatę falkową (Wavelet Transform).
    Zaprojektowana do detekcji lokalnych, wysokoczęstotliwościowych anomalii w danych.
    """
    def __init__(self, in_features: int, out_features: int, num_wavelets: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_wavelets = num_wavelets
        
        # Parametry falki (Wavelet parameters)
        # 1. Przesunięcie (Translation - odpowiada za lokalizację na osi X)
        self.translation = nn.Parameter(torch.empty(out_features, in_features, num_wavelets))
        # 2. Skala (Scale - odpowiada za szerokość/częstotliwość falki)
        self.scale = nn.Parameter(torch.empty(out_features, in_features, num_wavelets))
        # 3. Waga (Weight - odpowiada za amplitudę falki w kombinacji liniowej)
        self.wavelet_weight = nn.Parameter(torch.empty(out_features, in_features, num_wavelets))
        
        # Klasyczna waga bazowa dla globalnego trendu
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Inicjalizacja przesunięć równomiernie w standardowym zakresie
        nn.init.uniform_(self.translation, -2.0, 2.0)
        # Skala początkowo ustawiona na 1.0 (z małymi wariacjami)
        nn.init.normal_(self.scale, mean=1.0, std=0.1)
        # Inicjalizacja Kaiming dla wag
        nn.init.kaiming_uniform_(self.wavelet_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def mexican_hat_wavelet(self, x: torch.Tensor) -> torch.Tensor:
        """Oblicza wartości falki 'Meksykański Kapelusz'."""
        # Wzór: (1 - x^2) * exp(-x^2 / 2)
        x_sq = x ** 2
        return (1.0 - x_sq) * torch.exp(-0.5 * x_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Globalna aktywacja bazowa (np. SiLU)
        base_output = F.linear(F.silu(x), self.base_weight)
        
        # 2. Transformata Falkowa (Wavelet Transform)
        # x shape: [batch, in_features] -> [batch, 1, in_features, 1] dla wektoryzacji
        x_expanded = x.unsqueeze(1).unsqueeze(-1)
        
        # translation i scale mają kształt [out_features, in_features, num_wavelets]
        # Przekształcamy x (z-score bazujący na wyuczalnych parametrach falki)
        # Dodajemy epsilon do skali, aby uniknąć dzielenia przez zero
        x_scaled = (x_expanded - self.translation) / (self.scale.abs() + 1e-8)
        
        # Obliczamy wartości falki
        wavelet_basis = self.mexican_hat_wavelet(x_scaled) # [batch, out, in, wavelets]
        
        # 3. Sumowanie ważone (mnożenie przez wagi falkowe i sumowanie po in_features i wavelets)
        # Tutaj wykonujemy ręczne rzutowanie z wymiarów, unikając einsum dla większej czytelności przy 4D
        wavelet_weighted = wavelet_basis * self.wavelet_weight
        wavelet_output = wavelet_weighted.sum(dim=(2, 3)) # sumujemy po wejściach i falkach
        
        return base_output + wavelet_output


class WavKAN(BaseTabularModel):
    """
    Architektura Wav-KAN.
    Łączy klasyczne aproksymacje sieci głębokich z analizą czasowo-częstotliwościową.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [64, 32],
        num_wavelets: int = 8,
        **kwargs
    ):
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, num_wavelets=num_wavelets, **kwargs)
        
        layers = []
        in_features = input_dim
        
        for h_dim in hidden_dims:
            layers.append(WavKANLinear(in_features, h_dim, num_wavelets=num_wavelets))
            layers.append(nn.LayerNorm(h_dim))
            in_features = h_dim
            
        # Warstwa wyjściowa
        layers.append(WavKANLinear(in_features, output_dim, num_wavelets=num_wavelets))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)