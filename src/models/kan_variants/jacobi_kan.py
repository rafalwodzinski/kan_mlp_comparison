import torch
import torch.nn as nn
import sys
import os
from typing import List

# Import base class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class JacobiKANLinear(nn.Module):
    """
    KAN layer using generalized Jacobi polynomials.
    Alpha and beta parameters allow to control asymmetry and behavior at domain boundaries.
    """
    def __init__(self, in_features: int, out_features: int, degree: int = 4, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.alpha = alpha
        self.beta = beta
        
        # Trainable polynomial coefficients
        self.jacobi_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))
        
        # Variance stabilizing initialization
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1.0 / (in_features * (degree + 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Project domain to [-1, 1]
        x = torch.tanh(x)
        
        # 2. Initialize Jacobi polynomials basis
        jacobi_basis = [torch.ones_like(x)]
        
        if self.degree > 0:
            p1 = 0.5 * (self.alpha - self.beta + (self.alpha + self.beta + 2.0) * x)
            jacobi_basis.append(p1)
            
        # 3. Recursive expansion for n >= 2
        for n in range(2, self.degree + 1):
            # Calculate auxiliary constants for a given step n (Jacobi formula)
            # We use variable k for the current index (k = n - 1 in standard formula)
            k = n - 1 
            
            c1 = 2.0 * n * (n + self.alpha + self.beta) * (2.0 * n + self.alpha + self.beta - 2.0)
            
            a_n_num = (2.0 * n + self.alpha + self.beta - 1.0) * (2.0 * n + self.alpha + self.beta) * (2.0 * n + self.alpha + self.beta - 2.0)
            a_n = a_n_num / c1
            
            b_n_num = (self.alpha**2 - self.beta**2) * (2.0 * n + self.alpha + self.beta - 1.0)
            b_n = b_n_num / c1
            
            c_n_num = 2.0 * (n + self.alpha - 1.0) * (n + self.beta - 1.0) * (2.0 * n + self.alpha + self.beta)
            c_n = c_n_num / c1
            
            # Proper recursive step
            p_n = (a_n * x + b_n) * jacobi_basis[n-1] - c_n * jacobi_basis[n-2]
            jacobi_basis.append(p_n)
            
        # Assemble to tensor: [batch_size, in_features, degree + 1]
        jacobi_basis = torch.stack(jacobi_basis, dim=-1)
        
        # 4. Linear combination (basis weighting)
        out = torch.einsum('bid,oid->bo', jacobi_basis, self.jacobi_coeffs)
        
        return out


class JacobiKAN(BaseTabularModel):
    """
    JacobiKAN architecture.
    Thanks to modifying alpha and beta parameters, it can model 
    highly asymmetric feature distributions in medical data.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [64, 32],
        degree: int = 4,
        alpha: float = 1.0,
        beta: float = 1.0,
        **kwargs
    ):
        # Pass all hyperparameters for logging
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, degree=degree, alpha=alpha, beta=beta, **kwargs)
        
        layers = []
        in_features = input_dim
        
        for h_dim in hidden_dims:
            layers.append(JacobiKANLinear(in_features, h_dim, degree, alpha, beta))
            layers.append(nn.LayerNorm(h_dim))
            in_features = h_dim
            
        # Final output layer
        layers.append(JacobiKANLinear(in_features, output_dim, degree, alpha, beta))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)