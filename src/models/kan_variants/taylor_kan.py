import torch
import torch.nn as nn
import sys
import os
import math
from typing import List

# Import klasy bazowej
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class TaylorKANLinear(nn.Module):
    """
    Warstwa KAN wykorzystująca rozwinięcie w szereg Taylora (potęgi x).
    Bardzo szybka, wymaga jednak rygorystycznego ograniczenia dziedziny.
    """
    def __init__(self, in_features: int, out_features: int, degree: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # Trenowalne współczynniki szeregu Taylora (wagi dla poszczególnych potęg x^n)
        self.taylor_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Inicjalizacja Kaiming Uniform dla stabilności początkowej
        nn.init.kaiming_uniform_(self.taylor_coeffs, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Rzutowanie dziedziny na [-1, 1] - krytyczne zabezpieczenie przed eksplozją potęg!
        x = torch.tanh(x)
        
        # 2. Generowanie bazy Taylora: [1, x, x^2, x^3, ..., x^d]
        # Tworzymy listę kolejnych potęg, by uniknąć wielokrotnego mnożenia od zera
        taylor_basis = [torch.ones_like(x), x]
        
        for n in range(2, self.degree + 1):
            # Mnożymy poprzednią potęgę przez x, co jest optymalniejsze niż x**n
            taylor_basis.append(taylor_basis[-1] * x)
            
        # Złożenie do tensora: [batch_size, in_features, degree + 1]
        taylor_basis = torch.stack(taylor_basis, dim=-1)
        
        # 3. Kombinacja liniowa za pomocą einsum (mnożenie przez nauczone współczynniki w_n)
        # b: batch_size, i: in_features, d: stopień wielomianu (degree + 1), o: out_features
        out = torch.einsum('bid,oid->bo', taylor_basis, self.taylor_coeffs)
        
        return out


class TaylorKAN(BaseTabularModel):
    """
    Architektura TaylorKAN.
    Używa standardowych wielomianów (szeregu Taylora) do aproksymacji funkcji na krawędziach.
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
            layers.append(TaylorKANLinear(in_features, h_dim, degree))
            # Używamy LayerNorm, by utrzymać wariancję wejść dla kolejnych rozwinięć Taylora w normie
            layers.append(nn.LayerNorm(h_dim))
            in_features = h_dim
            
        # Ostatnia warstwa wyjściowa
        layers.append(TaylorKANLinear(in_features, output_dim, degree))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)