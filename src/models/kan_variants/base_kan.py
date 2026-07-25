import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
import sys
import os

# Add path to import our base class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class KANLinear(nn.Module):
    """
    Single KAN layer. Replaces classic weights from nn.Linear 
    with a network of learnable 1D functions (approximated by B-splines).
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Classic base activation weight (e.g., SiLU)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Parameters for B-splines: each edge has its own set of coefficients
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        # Parameters normalizing the domain grid
        self.grid = nn.Parameter(
            torch.linspace(-5, 5, grid_size + spline_order + 1), requires_grad=False
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

    def b_spline(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates B-spline values for a given input."""
        # Simplified: maps input x to basis vectors on the grid
        x = x.unsqueeze(-1)
        bases = (x >= self.grid[:-1]) & (x < self.grid[1:])
        bases = bases.float()
        
        for k in range(1, self.spline_order + 1):
            left = (x - self.grid[:-k-1]) / (self.grid[k:-1] - self.grid[:-k-1] + 1e-8)
            right = (self.grid[k+1:] - x) / (self.grid[k+1:] - self.grid[1:-k] + 1e-8)
            bases = left * bases[..., :-1] + right * bases[..., 1:]
            
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Base activation
        base_output = F.linear(F.silu(x), self.base_weight)
        
        # 2. Spline activation
        spline_basis = self.b_spline(x) # [batch_size, in_features, grid_size+spline_order]
        
        # Tensor multiplication by spline weights and summation
        spline_output = torch.einsum('biq,oiq->bo', spline_basis, self.spline_weight)
        
        # 3. Final node sum
        return base_output + spline_output


class BaseKAN(BaseTabularModel):
    """
    Original Kolmogorov-Arnold Network (KAN) architecture 
    adapted for tabular data.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [64, 32],
        grid_size: int = 5,
        spline_order: int = 3,
        **kwargs
    ):
        # Initialize base wrapper (hyperparameter logging)
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, grid_size=grid_size, **kwargs)
        
        layers = []
        in_features = input_dim
        
        for h_dim in hidden_dims:
            layers.append(KANLinear(in_features, h_dim, grid_size, spline_order))
            layers.append(nn.LayerNorm(h_dim)) # LayerNorm is more stable for KAN than BatchNorm
            in_features = h_dim
            
        # Final classification layer (no normalization at the end)
        layers.append(KANLinear(in_features, output_dim, grid_size, spline_order))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)