import abc
import torch
import torch.nn as nn
from typing import Dict, Any

class BaseTabularModel(nn.Module, abc.ABC):
    """
    Abstrakcyjna klasa bazowa dla wszystkich modeli w benchmarku (MLP oraz KAN).
    Zapewnia ujednolicony interfejs (API) dla pętli treningowej i logowania.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """
        Inicjalizacja bazowego modelu.
        
        Args:
            input_dim (int): Liczba cech wejściowych (wymiarowość danych).
            output_dim (int): Liczba klas wyjściowych (np. 1 dla binarnej, N dla multiclass).
            **kwargs: Dodatkowe hiperparametry specyficzne dla danego modelu.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hyperparameters = kwargs

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Główna operacja przejścia w przód (forward pass).
        Musi być zaimplementowana w klasie dziedziczącej.
        
        Args:
            x (torch.Tensor): Tensor wejściowy o kształcie (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Logity (surowe predykcje) o kształcie (batch_size, output_dim).
        """
        pass

    def get_num_parameters(self) -> int:
        """
        Oblicza całkowitą liczbę trenowalnych parametrów modelu.
        Przydatne do ewaluacji 'Fair Comparison' (model capacity).
        
        Returns:
            int: Liczba parametrów.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_config(self) -> Dict[str, Any]:
        """
        Zwraca konfigurację modelu do celów logowania w systemach trackujących (np. MLflow).
        
        Returns:
            Dict[str, Any]: Słownik z architekturą i hiperparametrami.
        """
        return {
            "model_type": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_parameters": self.get_num_parameters(),
            **self.hyperparameters
        }