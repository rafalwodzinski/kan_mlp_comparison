import optuna
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss
from optuna.pruners import MedianPruner
import inspect

from src.data.loader import MedicalTabularDataset

class OptunaTuner:
    """
    Lightweight hyperparameter optimization using Optuna.
    Implements an inner train/validation split to prevent data leakage.
    Uses MedianPruner to early-terminate unpromising trials for PyTorch models.
    """
    def __init__(self, model_class, X_train_clean, y_train, args, is_binary, num_classes, input_dim):
        self.model_class = model_class
        self.X_train_clean = X_train_clean
        self.y_train = y_train
        self.args = args
        self.is_binary = is_binary
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.is_sklearn = issubclass(model_class, BaseEstimator)
        self.device = torch.device(args.device if hasattr(args, 'device') else "cpu")

    def objective(self, trial):
        # 1. Inner Split (80/20) to prevent leakage
        X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
            self.X_train_clean, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )

        if self.is_sklearn:
            # Random Forest search space
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            
            model = self.model_class(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            model.fit(X_inner_train, y_inner_train)
            
            probs = model.predict_proba(X_inner_val)
            if self.is_binary and probs.shape[1] == 2:
                probs = probs[:, 1]
            
            return log_loss(y_inner_val, probs)
            
        else:
            # PyTorch models search space
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            
            model_kwargs = {
                'input_dim': self.input_dim,
                'output_dim': 1 if self.is_binary else self.num_classes
            }
            
            # Architectural search space
            if "MLP" in self.model_class.__name__:
                hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
                model_kwargs['hidden_dims'] = [hidden_dim, hidden_dim // 2]
            elif "KAN" in self.model_class.__name__:
                # Dynamically detect the correct capacity hyperparameter
                # for each KAN variant (grid_size, num_grids, degree, num_wavelets)
                sig = inspect.signature(self.model_class.__init__)
                params = sig.parameters
                
                if 'grid_size' in params:
                    grid_size = trial.suggest_categorical('grid_size', [3, 5, 10])
                    model_kwargs['grid_size'] = grid_size
                elif 'num_grids' in params:
                    num_grids = trial.suggest_categorical('num_grids', [4, 8, 12])
                    model_kwargs['num_grids'] = num_grids
                elif 'degree' in params:
                    degree = trial.suggest_categorical('degree', [3, 4, 6])
                    model_kwargs['degree'] = degree
                elif 'num_wavelets' in params:
                    num_wavelets = trial.suggest_categorical('num_wavelets', [4, 8, 12])
                    model_kwargs['num_wavelets'] = num_wavelets
                
            model = self.model_class(**model_kwargs).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.BCEWithLogitsLoss() if self.is_binary else nn.CrossEntropyLoss()
            
            train_dataset = MedicalTabularDataset(X_inner_train, y_inner_train)
            val_dataset = MedicalTabularDataset(X_inner_val, y_inner_val)
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
            
            inner_epochs = 15
            for epoch in range(inner_epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    logits = model(X_batch)
                    if self.is_binary:
                        loss = criterion(logits.squeeze(dim=-1), y_batch.float())
                    else:
                        loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                # Evaluate on inner val
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        logits = model(X_batch)
                        if self.is_binary:
                            loss = criterion(logits.squeeze(dim=-1), y_batch.float())
                        else:
                            loss = criterion(logits, y_batch)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                # Report and Prune
                trial.report(avg_val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                    
            return avg_val_loss

    def optimize(self, n_trials=15):
        # Disable overly verbose logging from Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", pruner=MedianPruner())
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params
