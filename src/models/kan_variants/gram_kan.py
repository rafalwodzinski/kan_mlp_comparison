import torch
import torch.nn as nn
import sys
import os
from typing import List

# Import klasy bazowej
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class GramKANLinear(nn.Module):
    """
    Warstwa KAN oparta na dyskretnych wielomianach Grama.
    Architektura o silnych podstawach teoretycznych dla dyskretnych danych tabelarycznych.
    """
    def __init__(self, in_features: int, out_features: int, degree: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # Trenowalne współczynniki dla bazy wielomianów Grama
        self.gram_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Inicjalizacja stabilizująca wariancję (skalowana do rzędu wielomianu)
        nn.init.normal_(self.gram_coeffs, mean=0.0, std=1.0 / (self.in_features * (self.degree + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Rzutowanie dziedziny na [-1, 1] dla stabilności numerycznej
        x = torch.tanh(x)
        
        # 2. Generowanie bazy wielomianów Grama (aproksymacja dla DL)
        # Baza zerowa i pierwsza
        gram_basis = [torch.ones_like(x), x]
        
        # Kolejne stopnie wielomianu z trójczłonowej relacji rekurencyjnej
        for n in range(2, self.degree + 1):
            # Dla celów DL używamy uproszczonej formy dyskretnej ortogonalizacji 
            # przypominającej wielomiany dyskretne, stabilnej dla tensora x
            term = x * gram_basis[n-1] - (n**2 - 1) / (4 * n**2 - 1) * gram_basis[n-2]
            gram_basis.append(term)
            
        # Złożenie bazy do tensora: [batch_size, in_features, degree + 1]
        gram_basis = torch.stack(gram_basis, dim=-1)
        
        # 3. Sumowanie ważone poprzez optymalizowaną operację einsum
        # b: batch, i: wejścia, d: stopień, o: wyjścia
        out = torch.einsum('bid,oid->bo', gram_basis, self.gram_coeffs)
        
        return out


class GramKAN(BaseTabularModel):
    """
    Architektura GramKAN.
    Wykorzystuje matematykę dyskretnej ortogonalności, co czyni ją 
    wyjątkowo użyteczną do klasyfikacji tabelarycznych danych klinicznych.
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
            layers.append(GramKANLinear(in_features, h_dim, degree))
            layers.append(nn.LayerNorm(h_dim)) # Normalizacja chroniąca przed "wybuchem" wielomianów
            in_features = h_dim
            
        # Ostatnia warstwa wyjściowa
        layers.append(GramKANLinear(in_features, output_dim, degree))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)