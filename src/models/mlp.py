import torch
import torch.nn as nn
from typing import List
from .base import BaseTabularModel

class StandardMLP(BaseTabularModel):
    """
    Classic Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout.
    Serves as the basic point of reference (Baseline).
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
            layers.append(nn.GELU()) # GELU often performs better than ReLU
            layers.append(nn.Dropout(dropout_rate))
            in_features = h_dim
            
        layers.append(nn.Linear(in_features, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block for tabular data."""
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
    MLP with residual connections (inspired by ResNet architecture).
    A much stronger baseline for medical tabular data.
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
        
        # Input projection to hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        
        # Output layer
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x)