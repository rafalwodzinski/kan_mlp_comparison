import torch
import torch.nn as nn
import sys
import os
from typing import List

# Import base class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class ChebyKANLinear(nn.Module):
    """
    Linear KAN layer using orthogonal Chebyshev polynomials.
    Provides high gradient stability and good expressiveness.
    """
    def __init__(self, in_features: int, out_features: int, degree: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # Trainable Chebyshev polynomial coefficients
        self.cheby_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        
        # Xavier/Glorot initialization adapted for polynomials
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1.0 / (in_features * (degree + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Map domain to [-1, 1] for Chebyshev polynomials stability
        x = torch.tanh(x)
        
        # 2. Recursive generation of Chebyshev polynomial basis
        cheby_basis = [torch.ones_like(x), x]
        
        for i in range(2, self.degree + 1):
            next_term = 2 * x * cheby_basis[i-1] - cheby_basis[i-2]
            cheby_basis.append(next_term)
            
        # Shape after stacking: [batch_size, in_features, degree + 1]
        cheby_basis = torch.stack(cheby_basis, dim=-1)
        
        # 3. Weighting the polynomial basis with parameters
        # b: batch_size, i: in_features, d: degree, o: out_features
        out = torch.einsum('bid,oid->bo', cheby_basis, self.cheby_coeffs)
        
        return out


class ChebyKAN(BaseTabularModel):
    """
    ChebyKAN (Chebyshev KAN) architecture.
    Minimizes approximation error at domain boundaries (robust to outliers).
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
            layers.append(ChebyKANLinear(in_features, h_dim, degree))
            layers.append(nn.LayerNorm(h_dim)) # Again LayerNorm to maintain stability
            in_features = h_dim
            
        # Output layer
        layers.append(ChebyKANLinear(in_features, output_dim, degree))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)