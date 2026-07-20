import torch
import torch.nn as nn
import sys
import os
import math
from typing import List

# Import base class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class TaylorKANLinear(nn.Module):
    """
    KAN layer using Taylor series expansion (powers of x).
    Very fast, but requires rigorous domain restriction.
    """
    def __init__(self, in_features: int, out_features: int, degree: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # Trainable Taylor series coefficients (weights for individual powers of x^n)
        self.taylor_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming Uniform initialization for initial stability
        nn.init.kaiming_uniform_(self.taylor_coeffs, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Project domain to [-1, 1] - critical safeguard against power explosion!
        x = torch.tanh(x)
        
        # 2. Generate Taylor basis: [1, x, x^2, x^3, ..., x^d]
        # We create a list of consecutive powers to avoid multiple multiplications from scratch
        taylor_basis = [torch.ones_like(x), x]
        
        for n in range(2, self.degree + 1):
            # We multiply previous power by x, which is more optimal than x**n
            taylor_basis.append(taylor_basis[-1] * x)
            
        # Assemble to tensor: [batch_size, in_features, degree + 1]
        taylor_basis = torch.stack(taylor_basis, dim=-1)
        
        # 3. Linear combination using einsum (multiplication by learned coefficients w_n)
        # b: batch_size, i: in_features, d: polynomial degree (degree + 1), o: out_features
        out = torch.einsum('bid,oid->bo', taylor_basis, self.taylor_coeffs)
        
        return out


class TaylorKAN(BaseTabularModel):
    """
    TaylorKAN architecture.
    Uses standard polynomials (Taylor series) to approximate functions on edges.
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
            # We use LayerNorm to maintain input variance for subsequent Taylor expansions in norm
            layers.append(nn.LayerNorm(h_dim))
            in_features = h_dim
            
        # Final output layer
        layers.append(TaylorKANLinear(in_features, output_dim, degree))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)