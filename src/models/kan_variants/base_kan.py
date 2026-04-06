import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
import sys
import os

# Dodajemy ścieżkę, aby móc zaimportować naszą klasę bazową
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class KANLinear(nn.Module):
    """
    Pojedyncza warstwa KAN. Zastępuje klasyczne wagi z nn.Linear 
    siecią uczących się funkcji jednowymiarowych (aproksymowanych B-splinami).
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Klasyczna waga aktywacji bazowej (np. SiLU)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Parametry dla B-splinów: każda krawędź ma własny zestaw współczynników
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        # Parametry normalizujące siatkę dziedziny
        self.grid = nn.Parameter(
            torch.linspace(-1, 1, grid_size + spline_order + 1), requires_grad=False
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

    def b_spline(self, x: torch.Tensor) -> torch.Tensor:
        """Oblicza wartości B-splinów dla danego wejścia."""
        # W uproszczeniu: mapowanie wejścia x na wektory bazowe na siatce (grid)
        x = x.unsqueeze(-1)
        bases = (x >= self.grid[:-1]) & (x < self.grid[1:])
        bases = bases.float()
        
        for k in range(1, self.spline_order + 1):
            left = (x - self.grid[:-k-1]) / (self.grid[k:-1] - self.grid[:-k-1] + 1e-8)
            right = (self.grid[k+1:] - x) / (self.grid[k+1:] - self.grid[1:-k] + 1e-8)
            bases = left * bases[..., :-1] + right * bases[..., 1:]
            
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Aktywacja bazowa (Base activation)
        base_output = F.linear(F.silu(x), self.base_weight)
        
        # 2. Aktywacja Spline'owa (Spline activation)
        spline_basis = self.b_spline(x) # [batch_size, in_features, grid_size+spline_order]
        
        # Mnożenie tensorowe przez wagi splinów i sumowanie
        spline_output = torch.einsum('biq,oiq->bo', spline_basis, self.spline_weight)
        
        # 3. Końcowa suma w węźle
        return base_output + spline_output


class BaseKAN(BaseTabularModel):
    """
    Oryginalna architektura Kolmogorov-Arnold Network (KAN) 
    dostosowana do danych tabelarycznych.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [64, 32],
        grid_size: int = 5,
        spline_order: int = 3,
        **kwargs
    ):
        # Inicjalizacja bazowego wrappera (logowanie hyperparametrów)
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, grid_size=grid_size, **kwargs)
        
        layers = []
        in_features = input_dim
        
        for h_dim in hidden_dims:
            layers.append(KANLinear(in_features, h_dim, grid_size, spline_order))
            layers.append(nn.LayerNorm(h_dim)) # LayerNorm jest stabilniejszy dla KAN niż BatchNorm
            in_features = h_dim
            
        # Ostatnia warstwa klasyfikacyjna (bez normalizacji na końcu)
        layers.append(KANLinear(in_features, output_dim, grid_size, spline_order))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)