import torch
import torch.nn as nn
import sys
import os
from typing import List

# Import klasy bazowej oraz zoptymalizowanej warstwy FastKANLinear z naszego repozytorium
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

# Bezpieczny import warstwy FastKANLinear
try:
    from fast_kan import FastKANLinear
except ImportError:
    from .fast_kan import FastKANLinear


class TabularGating(nn.Module):
    """
    Mechanizm bramkowania (Feature Gating).
    Uczy się maski, która przepuszcza tylko istotne cechy kliniczne, tłumiąc szum.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # Inicjalizujemy wagi tak, aby początkowo przepuszczały większość sygnału
        self.weight = nn.Parameter(torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Używamy funkcji sigmoid do skalowania wagi cech między 0 a 1
        return x * torch.sigmoid(self.weight)


class TabKAN(BaseTabularModel):
    """
    Architektura TabKAN.
    Docelowy wariant zaprojektowany stricte pod ustrukturyzowane, niejednorodne 
    medyczne zbiory danych, łączący KAN z mechanizmami sieci tabelarycznych.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [64, 64],
        num_grids: int = 8,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, num_grids=num_grids, **kwargs)
        
        # 1. Warstwa bramkowania cech
        self.gating = TabularGating(input_dim)
        
        # 2. Główny korpus sieci KAN
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        in_features = input_dim
        for h_dim in hidden_dims:
            self.layers.append(FastKANLinear(in_features, h_dim, num_grids=num_grids))
            self.norms.append(nn.LayerNorm(h_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            in_features = h_dim
            
        # 3. Głowica klasyfikacyjna
        self.output_layer = FastKANLinear(in_features, output_dim, num_grids=num_grids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A. Selekcja i tłumienie szumu w cechach wejściowych
        x = self.gating(x)
        
        # B. Przejście przez warstwy KAN z uwzględnieniem Skip Connections
        for layer, norm, drop in zip(self.layers, self.norms, self.dropouts):
            identity = x
            x = layer(x)
            x = norm(x)
            x = drop(x)
            
            # Połączenie rezydualne (tylko jeśli wymiary się pokrywają)
            if identity.shape == x.shape:
                x = x + identity
                
        # C. Klasyfikacja
        out = self.output_layer(x)
        return out