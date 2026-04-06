import torch
import torch.nn as nn
import sys
import os
from typing import List

# Importujemy klasę bazową
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class ChebyKANLinear(nn.Module):
    """
    Liniowa warstwa KAN wykorzystująca ortogonalne wielomiany Czebyszewa.
    Zapewnia wysoką stabilność gradientową i dobrą ekspresyjność.
    """
    def __init__(self, in_features: int, out_features: int, degree: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # Trenowalne współczynniki wielomianów Czebyszewa
        self.cheby_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        
        # Inicjalizacja Xaviera/Glorota dostosowana do wielomianów
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1.0 / (in_features * (degree + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Mapowanie dziedziny na [-1, 1] dla stabilności wielomianów Czebyszewa
        x = torch.tanh(x)
        
        # 2. Rekurencyjne generowanie bazy wielomianów Czebyszewa
        cheby_basis = [torch.ones_like(x), x]
        
        for i in range(2, self.degree + 1):
            next_term = 2 * x * cheby_basis[i-1] - cheby_basis[i-2]
            cheby_basis.append(next_term)
            
        # Kształt po stackowaniu: [batch_size, in_features, degree + 1]
        cheby_basis = torch.stack(cheby_basis, dim=-1)
        
        # 3. Ważenie bazy wielomianów za pomocą parametrów
        # b: batch_size, i: in_features, d: degree, o: out_features
        out = torch.einsum('bid,oid->bo', cheby_basis, self.cheby_coeffs)
        
        return out


class ChebyKAN(BaseTabularModel):
    """
    Architektura ChebyKAN (Chebyshev KAN).
    Minimalizuje błąd aproksymacji na brzegach dziedziny (odporność na outliers).
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
            layers.append(ChebyKANLinear(in_features, h_dim, degree))
            layers.append(nn.LayerNorm(h_dim)) # Ponownie LayerNorm dla zachowania stabilności
            in_features = h_dim
            
        # Warstwa wyjściowa
        layers.append(ChebyKANLinear(in_features, output_dim, degree))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)