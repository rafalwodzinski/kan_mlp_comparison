import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any
import sys
import os
import json
import time
import matplotlib.pyplot as plt
import joblib
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss

# Add path to import from other src folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.metrics import MedicalMetricsEvaluator
from training.early_stopping import EarlyStopping

class TabularTrainer:
    """
    Modular training class for KAN and MLP models.
    'Lite' version - without MLflow dependency. Logs to console 
    and saves weights and confusion matrix to disk.
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
        repeat: int = 1,
        fold: int = 1
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.is_binary = is_binary
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.repeat = repeat
        self.fold = fold
        self.experiment_name = f"{dataset_name}_{model_name}_Repeat{repeat}_Fold{fold}"
        
        self.is_sklearn = isinstance(model, BaseEstimator)
        if not self.is_sklearn:
            self.model = model.to(device)
        else:
            self.model = model
            
        self.evaluator = MedicalMetricsEvaluator(is_binary=self.is_binary)
        self.last_confusion_matrix = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_mcc': [], 'val_auroc': []}
        
        # Time tracking metrics
        self.avg_epoch_time_seconds = 0.0
        self.total_train_time_seconds = 0.0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Performs one training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            
            if self.is_binary:
                # SAFE SQUEEZE: dim=-1 protects against error when batch_size = 1
                loss = self.criterion(logits.squeeze(dim=-1), y_batch.float()) 
            else:
                loss = self.criterion(logits, y_batch)
                
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Model evaluation on validation/test set with full metrics."""
        if self.is_sklearn:
            all_preds = []
            all_trues = []
            for X_batch, y_batch in dataloader:
                X_batch_np = X_batch.numpy()
                probs = self.model.predict_proba(X_batch_np)
                if self.is_binary:
                    probs = probs[:, 1]
                all_preds.append(probs)
                all_trues.append(y_batch.numpy())
            
            y_prob_all = np.concatenate(all_preds, axis=0)
            y_true_all = np.concatenate(all_trues, axis=0)
            
            metrics = self.evaluator.calculate_metrics(y_true_all, y_prob_all)
            metrics["loss"] = log_loss(y_true_all, y_prob_all)
            self.last_confusion_matrix = self.evaluator.get_confusion_matrix(y_true_all, y_prob_all)
            return metrics
            
        self.model.eval()
        total_loss = 0.0
        
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                
                if self.is_binary:
                    # SAFE SQUEEZE (cleaner code)
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
        """Main training loop (pure PyTorch + print) or scikit-learn fitting."""
        print(f"\n[{self.experiment_name}] Starting training...")
        
        start_time = time.time()
        
        # Final artifacts directory structure
        save_dir = f"results/artifacts/{self.dataset_name}/{self.model_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        if self.is_sklearn:
            # Bypass PyTorch loops for scikit-learn
            X_all, y_all = [], []
            for X_batch, y_batch in train_loader:
                X_all.append(X_batch.numpy())
                y_all.append(y_batch.numpy())
            X_train = np.concatenate(X_all, axis=0)
            y_train = np.concatenate(y_all, axis=0)
            
            self.model.fit(X_train, y_train)
            
            end_time = time.time()
            self.total_train_time_seconds = end_time - start_time
            self.avg_epoch_time_seconds = 0.0
            
            val_metrics = self.evaluate(val_loader)
            print(f"[{self.experiment_name}] Sklearn Model Fit | Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val MCC: {val_metrics['mcc']:.4f} | Val AUROC: {val_metrics['auroc']:.4f}")
                  
            # Target paths for sklearn
            weights_path = os.path.join(save_dir, f"{self.experiment_name}_weights.joblib")
            cm_path = os.path.join(save_dir, f"{self.experiment_name}_confusion_matrix.csv")
            history_path = os.path.join(save_dir, f"{self.experiment_name}_history.json")
            
            joblib.dump(self.model, weights_path)
            
            if self.last_confusion_matrix is not None:
                np.savetxt(cm_path, self.last_confusion_matrix, delimiter=",", fmt='%d')
                
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=4)
                
            print(f"[{self.experiment_name}] Training finished. Artifacts saved to {save_dir}")
            return
            
        # Configure Early Stopping
        patience = run_params.get("patience", 10)
        early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, verbose=False)
        
        start_time = time.time()
        epoch_times = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader)
            epoch_end = time.time()
            epoch_times.append(epoch_end - epoch_start)
            val_metrics = self.evaluate(val_loader)
            
            # Save to history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mcc'].append(val_metrics['mcc'])
            self.history['val_auroc'].append(val_metrics['auroc'])
            
            # Early Stopping check
            early_stopping(val_metrics['loss'], self.model, epoch)
            
            # Print logs in console
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val MCC: {val_metrics['mcc']:.4f} | "
                  f"Val AUROC: {val_metrics['auroc']:.4f}")
                  
            if early_stopping.early_stop:
                print(f"[{self.experiment_name}] Early stopping triggered at epoch {epoch+1}")
                self.history['stopped_epoch'] = epoch
                self.history['best_epoch'] = early_stopping.best_epoch
                break
        
        end_time = time.time()
        self.total_train_time_seconds = end_time - start_time
        self.avg_epoch_time_seconds = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
        
        # Restore best weights if Early Stopping triggered
        if early_stopping.best_state is not None:
            self.model.load_state_dict(early_stopping.best_state)
        
        # Final artifacts directory structure
        save_dir = f"results/artifacts/{self.dataset_name}/{self.model_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot and save learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        if 'best_epoch' in self.history:
            plt.axvline(x=self.history['best_epoch'], color='r', linestyle='--', label='Best Epoch (Early Stopping)')
        plt.title(f"Learning Curve - {self.experiment_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(save_dir, f"{self.experiment_name}_learning_curve.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Target paths
        weights_path = os.path.join(save_dir, f"{self.experiment_name}_weights.pth")
        cm_path = os.path.join(save_dir, f"{self.experiment_name}_confusion_matrix.csv")
        history_path = os.path.join(save_dir, f"{self.experiment_name}_history.json")
        
        # Save Artifacts
        torch.save(self.model.state_dict(), weights_path)
        
        if self.last_confusion_matrix is not None:
            np.savetxt(cm_path, self.last_confusion_matrix, delimiter=",", fmt='%d')
            
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=4)
            
        print(f"[{self.experiment_name}] Training finished. Artifacts saved to {save_dir}")