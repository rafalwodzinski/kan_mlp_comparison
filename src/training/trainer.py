import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any
import sys
import os

# Dodanie ścieżki do importu z innych folderów src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.metrics import MedicalMetricsEvaluator

class TabularTrainer:
    """
    Modularna klasa treningowa dla modeli KAN i MLP.
    Wersja 'Lite' - bez zależności od MLflow. Wypisuje logi do konsoli.
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
        
        self.evaluator = MedicalMetricsEvaluator(is_binary=self.is_binary)
        self.last_confusion_matrix = None

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
                
                if self.is_binary:
                    logits_squeezed = logits.squeeze()
                    if logits_squeezed.dim() == 0:
                        logits_squeezed = logits_squeezed.unsqueeze(0)
                        
                    loss = self.criterion(logits_squeezed, y_batch.float())
                    probs = torch.sigmoid(logits_squeezed)
                else:
                    loss = self.criterion(logits, y_batch)
                    probs = torch.softmax(logits, dim=1)
                
                total_loss += loss.item()
                
                all_preds.append(probs.cpu().numpy())
                all_trues.append(y_batch.cpu().numpy())
                
        y_prob_all = np.concatenate(all_preds, axis=0)
        y_true_all = np.concatenate(all_trues, axis=0)
        
        metrics = self.evaluator.calculate_metrics(y_true_all, y_prob_all)
        metrics["loss"] = total_loss / len(dataloader)
        
        self.last_confusion_matrix = self.evaluator.get_confusion_matrix(y_true_all, y_prob_all)
                
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, run_params: dict):
        """Główna pętla treningowa (czysty PyTorch + print)."""
        print(f"\n[{self.experiment_name}] Rozpoczęcie treningu...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            # Wypisywanie logów w konsoli
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val MCC: {val_metrics['mcc']:.4f} | "
                  f"Val AUROC: {val_metrics['auroc']:.4f}")
        
        # Po zakończeniu treningu zapisujemy wagi modelu i macierz pomyłek na dysk
        weights_path = f"{self.experiment_name}_weights.pth"
        cm_path = f"{self.experiment_name}_confusion_matrix.csv"
        
        torch.save(self.model.state_dict(), weights_path)
        if self.last_confusion_matrix is not None:
            np.savetxt(cm_path, self.last_confusion_matrix, delimiter=",", fmt='%d')
            
        print(f"[{self.experiment_name}] Trening zakończony. Zapisano wagi do {weights_path}")