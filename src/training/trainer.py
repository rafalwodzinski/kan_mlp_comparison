import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any
import sys
import os
import json

# Dodanie ścieżki do importu z innych folderów src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.metrics import MedicalMetricsEvaluator

class TabularTrainer:
    """
    Modularna klasa treningowa dla modeli KAN i MLP.
    Wersja 'Lite' - bez zależności od MLflow. Wypisuje logi do konsoli 
    oraz zapisuje wagi i macierz pomyłek na dysk.
    """
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module, 
        device: torch.device,
        is_binary: bool = True,
        dataset_name: str = "Dataset",
        model_name: str = "Model",
        fold: int = 1
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.is_binary = is_binary
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.fold = fold
        self.experiment_name = f"{dataset_name}_{model_name}_Fold{fold}"
        
        self.evaluator = MedicalMetricsEvaluator(is_binary=self.is_binary)
        self.last_confusion_matrix = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_mcc': [], 'val_auroc': []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Przeprowadza jedną epokę treningową."""
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            
            if self.is_binary:
                # BEZPIECZNE ŚCISKANIE: dim=-1 chroni przed błędem gdy batch_size = 1
                loss = self.criterion(logits.squeeze(dim=-1), y_batch.float()) 
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
                    # BEZPIECZNE ŚCISKANIE (czystszy kod)
                    logits_squeezed = logits.squeeze(dim=-1)
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
            
            # Zapis do historii
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mcc'].append(val_metrics['mcc'])
            self.history['val_auroc'].append(val_metrics['auroc'])
            
            # Wypisywanie logów w konsoli
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val MCC: {val_metrics['mcc']:.4f} | "
                  f"Val AUROC: {val_metrics['auroc']:.4f}")
        
        # Ostateczna struktura katalogów artefaktów
        save_dir = f"results/artifacts/{self.dataset_name}/{self.model_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Ścieżki docelowe
        weights_path = os.path.join(save_dir, f"{self.experiment_name}_weights.pth")
        cm_path = os.path.join(save_dir, f"{self.experiment_name}_confusion_matrix.csv")
        history_path = os.path.join(save_dir, f"{self.experiment_name}_history.json")
        
        # Zapis Artefaktów
        torch.save(self.model.state_dict(), weights_path)
        
        if self.last_confusion_matrix is not None:
            np.savetxt(cm_path, self.last_confusion_matrix, delimiter=",", fmt='%d')
            
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=4)
            
        print(f"[{self.experiment_name}] Trening zakończony. Zapisano artefakty do {save_dir}")