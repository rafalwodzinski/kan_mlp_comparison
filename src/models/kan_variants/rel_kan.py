import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import List
import math

# Importujemy klasę bazową
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class RelKANLinear(nn.Module):
    """
    Warstwa KAN wykorzystująca przesunięte funkcje ReLU jako bazę aproksymacyjną.
    Ekstremalnie szybka na architekturach GPU.
    """
    def __init__(self, in_features: int, out_features: int, grid_min: float = -2.0, grid_max: float = 2.0, num_grids: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Inicjalizacja siatki (grid) punktów przesunięcia
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        
        # Funkcje bazowe to: ReLU(x - grid) oraz ReLU(grid - x)
        # Zatem mamy 2 * num_grids funkcji bazowych na każdą cechę wejściową
        basis_dim = num_grids * 2
        
        # Trenowalne wagi dla kombinacji liniowej naszych bramek ReLU
        self.relu_weight = nn.Parameter(torch.Tensor(out_features, in_features, basis_dim))
        
        # Opcjonalna waga bazowa (ułatwia początkowy trening)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.relu_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Klasyczna aktywacja bazowa (np. SiLU dla gładkości globalnej)
        base_output = F.linear(F.silu(x), self.base_weight)
        
        # 2. Generowanie bazy ReLU (rozszerzenie wymiarów)
        # x shape: [batch, in_features, 1], grid shape: [num_grids]
        x_expanded = x.unsqueeze(-1)
        
        # Baza 1: max(0, x - grid)
        relu_pos = F.relu(x_expanded - self.grid)
        # Baza 2: max(0, grid - x)
        relu_neg = F.relu(self.grid - x_expanded)
        
        # Łączymy obie bazy: [batch, in_features, num_grids * 2]
        relu_basis = torch.cat([relu_pos, relu_neg], dim=-1)
        
        # 3. Sumowanie ważone (Einsum)
        # b: batch_size, i: in_features, d: basis_dim, o: out_features
        relu_output = torch.einsum('bid,oid->bo', relu_basis, self.relu_weight)
        
        return base_output + relu_output


class RelKAN(BaseTabularModel):
    """
    Architektura RelKAN (ReLU KAN).
    Minimalizuje narzut obliczeniowy zachowując paradygmat aktywacji na krawędziach.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [64, 32],
        num_grids: int = 8,
        **kwargs
    ):
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, num_grids=num_grids, **kwargs)
        
        layers = []
        in_features = input_dim
        
        for h_dim in hidden_dims:
            layers.append(RelKANLinear(in_features, h_dim, num_grids=num_grids))
            layers.append(nn.LayerNorm(h_dim))
            in_features = h_dim
            
        # Warstwa wyjściowa
        layers.append(RelKANLinear(in_features, output_dim, num_grids=num_grids))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)