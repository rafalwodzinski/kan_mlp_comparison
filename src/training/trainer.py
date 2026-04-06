import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# Dodanie ścieżki do importu z innych folderów src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.metrics import MedicalMetricsEvaluator

class TabularTrainer:
    """
    Modularna klasa treningowa dla modeli KAN i MLP.
    Zintegrowana z MLflow oraz zaawansowanym systemem metryk medycznych.
    """
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module, 
        device: torch.device,
        is_binary: bool = True,
        experiment_name: str = "KAN_vs_MLP_Benchmark"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.is_binary = is_binary
        self.experiment_name = experiment_name
        
        # Inicjalizacja naszego autorskiego ewaluatora
        self.evaluator = MedicalMetricsEvaluator(is_binary=self.is_binary)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Przeprowadza jedną epokę treningową."""
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            
            if self.is_binary:
                loss = self.criterion(logits.squeeze(), y_batch.float()) 
            else:
                loss = self.criterion(logits, y_batch)
                
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Ewaluacja modelu na zbiorze walidacyjnym/testowym z pełnymi metrykami."""
        self.model.eval()
        total_loss = 0.0
        
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                
                # Obliczanie straty (Loss) i prawdopodobieństw
                if self.is_binary:
                    logits_squeezed = logits.squeeze()
                    # Zabezpieczenie przed batch_size = 1
                    if logits_squeezed.dim() == 0:
                        logits_squeezed = logits_squeezed.unsqueeze(0)
                        
                    loss = self.criterion(logits_squeezed, y_batch.float())
                    probs = torch.sigmoid(logits_squeezed)
                else:
                    loss = self.criterion(logits, y_batch)
                    probs = torch.softmax(logits, dim=1)
                
                total_loss += loss.item()
                
                # Przesłanie danych na CPU w celu przetworzenia przez scikit-learn
                all_preds.append(probs.cpu().numpy())
                all_trues.append(y_batch.cpu().numpy())
                
        # Konkatenacja wyników z całej epoki
        y_prob_all = np.concatenate(all_preds, axis=0)
        y_true_all = np.concatenate(all_trues, axis=0)
        
        # Obliczenie zaawansowanych metryk (AUROC, MCC, F1 itp.)
        metrics = self.evaluator.calculate_metrics(y_true_all, y_prob_all)
        metrics["loss"] = total_loss / len(dataloader)
        
        # Zapisujemy również macierz pomyłek z ostatniej ewaluacji
        self.last_confusion_matrix = self.evaluator.get_confusion_matrix(y_true_all, y_prob_all)
                
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, run_params: dict):
        """Główna pętla treningowa z precyzyjnym logowaniem do MLflow."""
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run():
            mlflow.log_params(run_params)
            
            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader)
                val_metrics = self.evaluate(val_loader)
                
                # Logowanie straty treningowej
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                
                # Logowanie wszystkich metryk walidacyjnych dynamicznie
                for metric_name, metric_value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", metric_value, step=epoch)
                
                # Wypisywanie logów w konsoli (skrócone dla czytelności)
                print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | Val MCC: {val_metrics['mcc']:.4f} | "
                      f"Val AUROC: {val_metrics['auroc']:.4f}")
            
            # Po zakończeniu treningu zapisujemy model oraz macierz pomyłek jako logi do MLflow
            mlflow.pytorch.log_model(self.model, "model")
            np.savetxt("confusion_matrix.csv", self.last_confusion_matrix, delimiter=",")
            mlflow.log_artifact("confusion_matrix.csv")