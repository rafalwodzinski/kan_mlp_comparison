import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
from typing import Dict, Any, Optional

class TabularTrainer:
    """
    Modularna klasa treningowa dla modeli KAN i MLP w czystym PyTorch.
    Zapewnia pełną kontrolę nad pętlą uczącą i integruje się z MLflow.
    """
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module, 
        device: torch.device,
        experiment_name: str = "KAN_vs_MLP_Benchmark"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.experiment_name = experiment_name

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Przeprowadza jedną epokę treningową."""
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            
            # Wymiary mogą wymagać dostosowania w zależności od typu klasyfikacji
            loss = self.criterion(predictions.squeeze(), y_batch.float()) 
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Ewaluacja modelu na zbiorze walidacyjnym/testowym."""
        self.model.eval()
        total_loss = 0.0
        # W przyszłości dodamy tu zbieranie predykcji do obliczania AUROC, F1 itp.
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = self.model(X_batch)
                loss = self.criterion(predictions.squeeze(), y_batch.float())
                total_loss += loss.item()
                
        return {"val_loss": total_loss / len(dataloader)}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, run_params: dict):
        """Główna pętla treningowa z logowaniem do MLflow."""
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run():
            # Logowanie parametrów eksperymentu
            mlflow.log_params(run_params)
            
            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader)
                val_metrics = self.evaluate(val_loader)
                
                # Logowanie metryk co epokę
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_metrics["val_loss"], step=epoch)
                
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Logowanie ostatecznego modelu do MLflow
            mlflow.pytorch.log_model(self.model, "model")