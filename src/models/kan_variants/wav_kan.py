import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
from typing import List

# Import base class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

class WavKANLinear(nn.Module):
    """
    KAN layer using continuous Wavelet Transform.
    Designed for detection of local, high-frequency data anomalies.
    """
    def __init__(self, in_features: int, out_features: int, num_wavelets: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_wavelets = num_wavelets
        
        # Wavelet parameters
        # 1. Translation - responsible for location on X axis
        self.translation = nn.Parameter(torch.empty(out_features, in_features, num_wavelets))
        # 2. Scale - responsible for wavelet width/frequency
        self.scale = nn.Parameter(torch.empty(out_features, in_features, num_wavelets))
        # 3. Weight - responsible for wavelet amplitude in linear combination
        self.wavelet_weight = nn.Parameter(torch.empty(out_features, in_features, num_wavelets))
        
        # Classic base weight for global trend
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Uniformly initialize translations in standard range
        nn.init.uniform_(self.translation, -2.0, 2.0)
        # Scale initially set to 1.0 (with small variations)
        nn.init.normal_(self.scale, mean=1.0, std=0.1)
        # Kaiming initialization for weights
        nn.init.kaiming_uniform_(self.wavelet_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def mexican_hat_wavelet(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates 'Mexican Hat' wavelet values."""
        # Formula: (1 - x^2) * exp(-x^2 / 2)
        x_sq = x ** 2
        return (1.0 - x_sq) * torch.exp(-0.5 * x_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Global base activation (e.g. SiLU)
        base_output = F.linear(F.silu(x), self.base_weight)
        
        # 2. Wavelet Transform
        # x shape: [batch, in_features] -> [batch, 1, in_features, 1] for vectorization
        x_expanded = x.unsqueeze(1).unsqueeze(-1)
        
        # translation and scale have shape [out_features, in_features, num_wavelets]
        # We transform x (z-score based on learnable wavelet parameters)
        # We add epsilon to scale to avoid division by zero
        x_scaled = (x_expanded - self.translation) / (self.scale.abs() + 1e-8)
        
        # Calculate wavelet values
        wavelet_basis = self.mexican_hat_wavelet(x_scaled) # [batch, out, in, wavelets]
        
        # 3. Weighted summation (multiplication by wavelet weights and summing over in_features and wavelets)
        # Here we perform manual dimension casting, avoiding einsum for better readability with 4D
        wavelet_weighted = wavelet_basis * self.wavelet_weight
        wavelet_output = wavelet_weighted.sum(dim=(2, 3)) # sum over inputs and wavelets
        
        return base_output + wavelet_output


class WavKAN(BaseTabularModel):
    """
    Wav-KAN architecture.
    Combines classic deep network approximations with time-frequency analysis.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [64, 32],
        num_wavelets: int = 8,
        **kwargs
    ):
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, num_wavelets=num_wavelets, **kwargs)
        
        layers = []
        in_features = input_dim
        
        for h_dim in hidden_dims:
            layers.append(WavKANLinear(in_features, h_dim, num_wavelets=num_wavelets))
            layers.append(nn.LayerNorm(h_dim))
            in_features = h_dim
            
        # Final output layer
        layers.append(WavKANLinear(in_features, output_dim, num_wavelets=num_wavelets))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)