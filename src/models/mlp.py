import torch
import torch.nn as nn
from typing import List
from .base import BaseTabularModel

class StandardMLP(BaseTabularModel):
    """
    Klasyczny Wielowarstwowy Perceptron (MLP) z Batch Normalization i Dropoutem.
    Służy jako podstawowy punkt odniesienia (Baseline).
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super().__init__(input_dim, output_dim, **kwargs)
        
        layers = []
        in_features = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU()) # GELU często sprawdza się lepiej niż ReLU
            layers.append(nn.Dropout(dropout_rate))
            in_features = h_dim
            
        layers.append(nn.Linear(in_features, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualBlock(nn.Module):
    """Blok rezydualny dla danych tabelarycznych."""
    def __init__(self, dim: int, dropout_rate: float):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        return self.act(out + identity)


class TabResNet(BaseTabularModel):
    """
    MLP z połączeniami rezydualnymi (wzorowane na architekturze ResNet).
    Znacznie silniejszy punkt odniesienia dla medycznych danych tabelarycznych.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super().__init__(input_dim, output_dim, **kwargs)
        
        # Rzutowanie wejścia na wymiar ukryty
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Bloki rezydualne
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        
        # Warstwa wyjściowa
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x)