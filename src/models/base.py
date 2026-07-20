import abc
import torch
import torch.nn as nn
from typing import Dict, Any

class BaseTabularModel(nn.Module, abc.ABC):
    """
    Abstract base class for all models in the benchmark (MLP and KAN).
    Provides a unified interface (API) for the training loop and logging.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """
        Initialize the base model.
        
        Args:
            input_dim (int): Number of input features (data dimensionality).
            output_dim (int): Number of output classes (e.g., 1 for binary, N for multiclass).
            **kwargs: Additional hyperparameters specific to the model.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hyperparameters = kwargs

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main forward pass operation.
        Must be implemented in the inheriting class.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Logits (raw predictions) of shape (batch_size, output_dim).
        """
        pass

    def get_num_parameters(self) -> int:
        """
        Calculates the total number of trainable model parameters.
        Useful for 'Fair Comparison' evaluation (model capacity).
        
        Returns:
            int: Number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_config(self) -> Dict[str, Any]:
        """
        Returns the model configuration for logging in tracking systems (e.g., MLflow).
        
        Returns:
            Dict[str, Any]: Dictionary with architecture and hyperparameters.
        """
        return {
            "model_type": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_parameters": self.get_num_parameters(),
            **self.hyperparameters
        }