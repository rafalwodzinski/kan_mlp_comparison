import torch
import torch.nn as nn
import sys
import os
from typing import List

# Import base class and optimized FastKANLinear layer from our repository
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base import BaseTabularModel

# Safe import of FastKANLinear layer
try:
    from fast_kan import FastKANLinear
except ImportError:
    from .fast_kan import FastKANLinear


class TabularGating(nn.Module):
    """
    Feature Gating mechanism.
    Learns a mask that passes only relevant clinical features, suppressing noise.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # Initialize weights so that they initially pass most of the signal
        self.weight = nn.Parameter(torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use sigmoid function to scale feature weight between 0 and 1
        return x * torch.sigmoid(self.weight)


class TabKAN(BaseTabularModel):
    """
    TabKAN Architecture.
    Target variant designed strictly for structured, heterogeneous 
    medical datasets, combining KAN with tabular network mechanisms.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [64, 64],
        num_grids: int = 8,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, num_grids=num_grids, **kwargs)
        
        # 1. Feature gating layer
        self.gating = TabularGating(input_dim)
        
        # 2. Main body of KAN network
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        in_features = input_dim
        for h_dim in hidden_dims:
            self.layers.append(FastKANLinear(in_features, h_dim, num_grids=num_grids))
            self.norms.append(nn.LayerNorm(h_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            in_features = h_dim
            
        # 3. Classification head
        self.output_layer = FastKANLinear(in_features, output_dim, num_grids=num_grids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A. Feature selection and noise suppression
        x = self.gating(x)
        
        # B. Forward pass through KAN layers including Skip Connections
        for layer, norm, drop in zip(self.layers, self.norms, self.dropouts):
            identity = x
            x = layer(x)
            x = norm(x)
            x = drop(x)
            
            # Residual connection (only if dimensions match)
            if identity.shape == x.shape:
                x = x + identity
                
        # C. Classification
        out = self.output_layer(x)
        return out