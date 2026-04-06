import torch
import torch.nn as nn
import sys
import os
from typing import List

# Import klasy bazowej
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class JacobiKANLinear(nn.Module):
    """
    Warstwa KAN wykorzystująca uogólnione wielomiany Jacobiego.
    Parametry alpha i beta pozwalają kontrolować asymetrię i zachowanie na brzegach dziedziny.
    """
    def __init__(self, in_features: int, out_features: int, degree: int = 4, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.alpha = alpha
        self.beta = beta
        
        # Trenowalne współczynniki wielomianów
        self.jacobi_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        
        # Inicjalizacja stabilizująca wariancję
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1.0 / (in_features * (degree + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Rzutowanie dziedziny na [-1, 1]
        x = torch.tanh(x)
        
        # 2. Inicjalizacja bazy wielomianów Jacobiego
        jacobi_basis = [torch.ones_like(x)]
        
        if self.degree > 0:
            p1 = 0.5 * (self.alpha - self.beta + (self.alpha + self.beta + 2.0) * x)
            jacobi_basis.append(p1)
            
        # 3. Rozwinięcie rekurencyjne dla n >= 2
        for n in range(2, self.degree + 1):
            # Obliczanie stałych pomocniczych dla danego kroku n (wzór Jacobiego)
            # Używamy zmiennej k dla obecnego indeksu (k = n - 1 w standardowym wzorze)
            k = n - 1 
            
            c1 = 2.0 * n * (n + self.alpha + self.beta) * (2.0 * n + self.alpha + self.beta - 2.0)
            
            a_n_num = (2.0 * n + self.alpha + self.beta - 1.0) * (2.0 * n + self.alpha + self.beta) * (2.0 * n + self.alpha + self.beta - 2.0)
            a_n = a_n_num / c1
            
            b_n_num = (self.alpha**2 - self.beta**2) * (2.0 * n + self.alpha + self.beta - 1.0)
            b_n = b_n_num / c1
            
            c_n_num = 2.0 * (n + self.alpha - 1.0) * (n + self.beta - 1.0) * (2.0 * n + self.alpha + self.beta)
            c_n = c_n_num / c1
            
            # Właściwy krok rekurencyjny
            p_n = (a_n * x + b_n) * jacobi_basis[n-1] - c_n * jacobi_basis[n-2]
            jacobi_basis.append(p_n)
            
        # Złożenie do tensora: [batch_size, in_features, degree + 1]
        jacobi_basis = torch.stack(jacobi_basis, dim=-1)
        
        # 4. Kombinacja liniowa (ważenie bazy)
        out = torch.einsum('bid,oid->bo', jacobi_basis, self.jacobi_coeffs)
        
        return out


class JacobiKAN(BaseTabularModel):
    """
    Architektura JacobiKAN.
    Dzięki modyfikacji parametrów alpha i beta, potrafi modelować 
    silnie asymetryczne rozkłady cech w danych medycznych.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [64, 32],
        degree: int = 4,
        alpha: float = 1.0,
        beta: float = 1.0,
        **kwargs
    ):
        # Przekazujemy wszystkie hiperparametry do logowania
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, degree=degree, alpha=alpha, beta=beta, **kwargs)
        
        layers = []
        in_features = input_dim
        
        for h_dim in hidden_dims:
            layers.append(JacobiKANLinear(in_features, h_dim, degree, alpha, beta))
            layers.append(nn.LayerNorm(h_dim))
            in_features = h_dim
            
        # Ostatnia warstwa wyjściowa
        layers.append(JacobiKANLinear(in_features, output_dim, degree, alpha, beta))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)