import torch
import torch.nn as nn
import sys
import os
from typing import List

# Import base class with path inclusion
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class LegendreKANLinear(nn.Module):
    """
    KAN layer based on orthogonal Legendre polynomials.
    Ensures uniform approximation error weight over the entire domain interval.
    """
    def __init__(self, in_features: int, out_features: int, degree: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # Trainable Legendre polynomial coefficients
        self.legendre_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        
        # Xavier/Glorot initialization (scaled inversely proportional to polynomial degree)
        nn.init.normal_(self.legendre_coeffs, mean=0.0, std=1.0 / (in_features * (degree + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Map domain to [-1, 1] for polynomials stability
        x = torch.tanh(x)
        
        # 2. Recursive generation of Legendre polynomials basis
        legendre_basis = [torch.ones_like(x), x]
        
        for n in range(2, self.degree + 1):
            # Legendre polynomials recurrence relation formula
            term = ((2 * n - 1) * x * legendre_basis[n-1] - (n - 1) * legendre_basis[n-2]) / n
            legendre_basis.append(term)
            
        # Assemble to tensor: [batch_size, in_features, degree + 1]
        legendre_basis = torch.stack(legendre_basis, dim=-1)
        
        # 3. Weighted summation via optimized einsum operation
        # b: batch_size, i: in_features, d: degree + 1, o: out_features
        out = torch.einsum('bid,oid->bo', legendre_basis, self.legendre_coeffs)
        
        return out


class LegendreKAN(BaseTabularModel):
    """
    LegendreKAN architecture.
    Alternative to ChebyKAN, works great for 
    nearly uniform data distributions.
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
            layers.append(LegendreKANLinear(in_features, h_dim, degree))
            layers.append(nn.LayerNorm(h_dim)) # Normalization stabilizing subsequent polynomial layers
            in_features = h_dim
            
        # Final output layer
        layers.append(LegendreKANLinear(in_features, output_dim, degree))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)