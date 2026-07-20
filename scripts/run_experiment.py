"""
Module responsible for the model training lifecycle (Trainer).
The `TabularTrainer` class serves for hermetic training and evaluation of a single
model instance (within one cross-validation fold).
Isolates low-level PyTorch operations from experimental logic.
Lite version: prints logs directly to the console (instead of e.g. MLFlow) and dumps
resulting artifacts to the local disk.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any
import sys
import os

# Add path to import from other src folders (project root directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation.metrics import MedicalMetricsEvaluator

class TabularTrainer:
    """
    Modular training class for neural networks (both KAN and classic MLP).
    Responsible for forward/backward pass loop, loss tracking, weights optimization
    and generation of final evaluation metrics.
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
        """
        Initialization of training environment.
        
        Args:
            model (nn.Module): Neural network (e.g., WavKAN, StandardMLP).
            optimizer (torch.optim.Optimizer): Optimization algorithm (e.g., AdamW).
            criterion (nn.Module): Loss function (e.g., BCEWithLogitsLoss for binary).
            device (torch.device): Compute environment (CPU/CUDA).
            is_binary (bool): Flag determining classification problem (binary vs multiclass).
            experiment_name (str): Identification name affecting resulting file names.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.is_binary = is_binary
        self.experiment_name = experiment_name
        
        # Initialization of dedicated evaluator for medical metrics (MCC, AUROC, etc.)
        self.evaluator = MedicalMetricsEvaluator(is_binary=self.is_binary)
        self.last_confusion_matrix = None

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Performs a single network training epoch (pass through all batches).
        
        Args:
            dataloader (DataLoader): PyTorch loader of training samples.
            
        Returns:
            float: Average loss function value for a given epoch.
        """
        self.model.train() # Switch model to training mode (e.g., Dropout activation)
        total_loss = 0.0
        
        for X_batch, y_batch in dataloader:
            # Transfer tensors to VRAM (if GPU available)
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # Resetting gradient accumulator
            self.optimizer.zero_grad()
            
            # Forward pass (raw logits prediction)
            logits = self.model(X_batch)
            
            if self.is_binary:
                # SAFE SQUEEZE: dim=-1 protects against silent PyTorch broadcasting error,
                # when batch size 1 is processed at the end of the dataset
                loss = self.criterion(logits.squeeze(dim=-1), y_batch.float()) 
            else:
                loss = self.criterion(logits, y_batch)
                
            # Backward pass (error propagation) and weights update
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Validation without gradients (Inference mode). Calculates logits, transforms them
        into probabilities and generates a set of hard evaluation metrics.
        
        Args:
            dataloader (DataLoader): PyTorch loader of validation/test set.
            
        Returns:
            Dict[str, Any]: Dictionary with metrics results (MCC, AUROC, F1, Loss, etc.).
        """
        self.model.eval() # Disable stochastic elements like Dropout/LayerNorm
        total_loss = 0.0
        
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                
                if self.is_binary:
                    logits_squeezed = logits.squeeze(dim=-1)
                    loss = self.criterion(logits_squeezed, y_batch.float())
                    # Transformation of logits to probabilities (range 0-1) for binary classification
                    probs = torch.sigmoid(logits_squeezed)
                else:
                    loss = self.criterion(logits, y_batch)
                    # Recreation of probabilities structure for multiclass classification
                    probs = torch.softmax(logits, dim=1)
                
                total_loss += loss.item()
                
                # Dump results to CPU buffers for Scikit-Learn metrics analysis purposes
                all_preds.append(probs.cpu().numpy())
                all_trues.append(y_batch.cpu().numpy())
                
        # Concatenation from individual batches
        y_prob_all = np.concatenate(all_preds, axis=0)
        y_true_all = np.concatenate(all_trues, axis=0)
        
        # Calculation of clinical indicators
        metrics = self.evaluator.calculate_metrics(y_true_all, y_prob_all)
        metrics["loss"] = total_loss / len(dataloader)
        
        # Save confusion matrix to global class state (available after fit function completes)
        self.last_confusion_matrix = self.evaluator.get_confusion_matrix(y_true_all, y_prob_all)
                
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, run_params: dict):
        """
        Main training loop coordinating learning (train_epoch) and verification (evaluate).
        
        Args:
            train_loader (DataLoader): Training set.
            val_loader (DataLoader): Validation set for generalization assessment.
            epochs (int): Maximum number of learning iterations over the whole set.
            run_params (dict): Configuration parameters for logging.
        """
        print(f"\n[{self.experiment_name}] Starting training...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            # Live console monitoring allowing to diagnose e.g. premature overfitting
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val MCC: {val_metrics['mcc']:.4f} | "
                  f"Val AUROC: {val_metrics['auroc']:.4f}")
        
        # Permanent dump of experimental "hard evidence" (MLOps practices)
        weights_path = f"{self.experiment_name}_weights.pth"
        cm_path = f"{self.experiment_name}_confusion_matrix.csv"
        
        # Save learned parameters on edges (KAN) or connection weights (MLP)
        torch.save(self.model.state_dict(), weights_path)
        
        # Save patient diagnostic counts
        if self.last_confusion_matrix is not None:
            np.savetxt(cm_path, self.last_confusion_matrix, delimiter=",", fmt='%d')
            
        print(f"[{self.experiment_name}] Training finished. Weights saved to {weights_path}")