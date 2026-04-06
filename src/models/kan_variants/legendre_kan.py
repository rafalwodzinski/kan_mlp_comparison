import torch
import torch.nn as nn
import sys
import os
from typing import List

# Import klasy bazowej z uwzględnieniem ścieżki
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class LegendreKANLinear(nn.Module):
    """
    Warstwa KAN oparta na ortogonalnych wielomianach Legendre'a.
    Zapewnia równomierną wagę błędu aproksymacji na całym przedziale dziedziny.
    """
    def __init__(self, in_features: int, out_features: int, degree: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # Trenowalne współczynniki wielomianów Legendre'a
        self.legendre_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        
        # Inicjalizacja Xaviera/Glorota (skalowana odwrotnie proporcjonalnie do stopnia wielomianu)
        nn.init.normal_(self.legendre_coeffs, mean=0.0, std=1.0 / (in_features * (degree + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Mapowanie dziedziny na [-1, 1] dla stabilności wielomianów
        x = torch.tanh(x)
        
        # 2. Rekurencyjne generowanie bazy wielomianów Legendre'a
        legendre_basis = [torch.ones_like(x), x]
        
        for n in range(2, self.degree + 1):
            # Formuła relacji rekurencyjnej dla wielomianów Legendre'a
            term = ((2 * n - 1) * x * legendre_basis[n-1] - (n - 1) * legendre_basis[n-2]) / n
            legendre_basis.append(term)
            
        # Złożenie do tensora: [batch_size, in_features, degree + 1]
        legendre_basis = torch.stack(legendre_basis, dim=-1)
        
        # 3. Sumowanie ważone poprzez optymalizowaną operację einsum
        # b: batch_size, i: in_features, d: degree + 1, o: out_features
        out = torch.einsum('bid,oid->bo', legendre_basis, self.legendre_coeffs)
        
        return out


class LegendreKAN(BaseTabularModel):
    """
    Architektura LegendreKAN.
    Alternatywa dla ChebyKAN, sprawdzająca się świetnie w przypadku 
    rozkładów danych zbliżonych do jednostajnych.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [64, 32],
        degree: int = 4,
        **kwargs
    ):
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, degree=degree, **kwargs)
        
        layers = []
        in_features = input_dim
        
        for h_dim in hidden_dims:
            layers.append(LegendreKANLinear(in_features, h_dim, degree))
            layers.append(nn.LayerNorm(h_dim)) # Normalizacja stabilizująca kolejne warstwy wielomianów
            in_features = h_dim
            
        # Ostatnia warstwa wyjściowa
        layers.append(LegendreKANLinear(in_features, output_dim, degree))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)