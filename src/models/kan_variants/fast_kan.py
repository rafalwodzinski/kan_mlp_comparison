import torch
import torch.nn as nn
import math
import sys
import os
from typing import List

# Import base class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class RadialBasisFunction(nn.Module):
    """
    Radial Basis Function (RBF) layer.
    Generates basis vectors much faster than classic B-splines.
    """
    def __init__(self, grid_min: float = -2.0, grid_max: float = 2.0, num_grids: int = 8, denominator: float = None):
        super().__init__()
        # Initialization of equally spaced grid means
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        
        # Gaussian function width (variance/sigma). If None, calculated automatically.
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand dimensions to vectorize the distance operation
        # x shape: [batch_size, in_features, 1]
        # grid shape: [num_grids]
        # output shape: [batch_size, in_features, num_grids]
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLinear(nn.Module):
    """
    Linear FastKAN layer using RBF functions as edge weights.
    """
    def __init__(self, in_features: int, out_features: int, grid_min: float = -2.0, grid_max: float = 2.0, num_grids: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.base_activation = nn.SiLU()
        
        # Trainable weights: classic (base) and for RBF
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.rbf_weight = nn.Parameter(torch.Tensor(out_features, in_features, num_grids))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming initialization prevents gradient explosion at the beginning of training
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.rbf_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Classic base activation
        base_output = nn.functional.linear(self.base_activation(x), self.base_weight)
        
        # 2. RBF activation
        rbf_output = self.rbf(x) # [batch, in_features, num_grids]
        
        # 3. Summing weighted RBFs using optimized einsum
        # b: batch_size, i: in_features, g: num_grids, o: out_features
        rbf_weighted = torch.einsum('big,oig->bo', rbf_output, self.rbf_weight)
        
        return base_output + rbf_weighted


class FastKAN(BaseTabularModel):
    """
    FastKAN architecture. 
    An ideal compromise between Kolmogorov-Arnold network expressiveness and GPU performance.
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
            layers.append(FastKANLinear(in_features, h_dim, num_grids=num_grids))
            layers.append(nn.LayerNorm(h_dim))
            in_features = h_dim
            
        # Output layer
        layers.append(FastKANLinear(in_features, output_dim, num_grids=num_grids))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)