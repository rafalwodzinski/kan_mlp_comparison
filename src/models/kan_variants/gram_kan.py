import torch
import torch.nn as nn
import sys
import os
from typing import List

# Import base class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class GramKANLinear(nn.Module):
    """
    KAN layer based on discrete Gram polynomials.
    Architecture with strong theoretical foundations for discrete tabular data.
    """
    def __init__(self, in_features: int, out_features: int, degree: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # Trainable coefficients for Gram polynomial basis
        self.gram_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Variance stabilizing initialization (scaled to polynomial degree)
        nn.init.normal_(self.gram_coeffs, mean=0.0, std=1.0 / (self.in_features * (self.degree + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Project domain to [-1, 1] for numerical stability
        x = torch.tanh(x)
        
        # 2. Generate Gram polynomial basis (approximation for DL)
        # Zero and first basis
        gram_basis = [torch.ones_like(x), x]
        
        # Subsequent polynomial degrees from three-term recurrence relation
        for n in range(2, self.degree + 1):
            # For DL purposes we use a simplified form of discrete orthogonalization 
            # resembling discrete polynomials, stable for tensor x
            term = x * gram_basis[n-1] - (n**2 - 1) / (4 * n**2 - 1) * gram_basis[n-2]
            gram_basis.append(term)
            
        # Assemble basis to tensor: [batch_size, in_features, degree + 1]
        gram_basis = torch.stack(gram_basis, dim=-1)
        
        # 3. Weighted summation via optimized einsum operation
        # b: batch, i: inputs, d: degree, o: outputs
        out = torch.einsum('bid,oid->bo', gram_basis, self.gram_coeffs)
        
        return out


class GramKAN(BaseTabularModel):
    """
    GramKAN architecture.
    Uses discrete orthogonality mathematics, which makes it 
    exceptionally useful for clinical tabular data classification.
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
            layers.append(GramKANLinear(in_features, h_dim, degree))
            layers.append(nn.LayerNorm(h_dim)) # Normalization protecting against polynomial "explosion"
            in_features = h_dim
            
        # Final output layer
        layers.append(GramKANLinear(in_features, output_dim, degree))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)