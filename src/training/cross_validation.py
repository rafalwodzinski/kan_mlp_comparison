import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Type
import os
import json
from sklearn.base import BaseEstimator

from src.data.loader import MedicalTabularDataset, get_data_and_preprocessor
from src.training.hpo import OptunaTuner

class CrossValidator:
    """
    Engine handling Stratified K-Fold logic.
    Guarantees experiment purity in each fold by independently training
    the Preprocessor object (prevents data leakage).
    """
    def __init__(self, k_folds: int = 5, n_repeats: int = 3, random_state: int = 42):
        self.k_folds = k_folds
        self.n_repeats = n_repeats
        self.rskf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=n_repeats, random_state=random_state)

    def run(self, 
            model_class: Type, 
            trainer_class: Type, 
            args: Any) -> pd.DataFrame:
        
        # 1. Instead of passing DF from outside, the validator loads it itself using a new loader
        dataset_filename = os.path.basename(args.data_path)
        dataset_name = dataset_filename.replace("_processed.csv", "")
        
        X_raw, y_raw, preprocessor = get_data_and_preprocessor(args.data_path, dataset_filename)
        all_fold_metrics = []

        print(f"\n[CV] Starting {self.n_repeats}x{self.k_folds}-fold CV for {args.model_name} model...")
        print(f"[CV] Dataset: {dataset_name} | Sample size: {len(X_raw)}")

        # Main CV loop on RAW data
        for idx, (train_idx, val_idx) in enumerate(self.rskf.split(X_raw, y_raw)):
            repeat = (idx // self.k_folds) + 1
            fold = (idx % self.k_folds) + 1
            print(f"\n>>> REPEAT {repeat}/{self.n_repeats} | FOLD {fold}/{self.k_folds}")

            # Extract raw folds
            X_train_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
            y_train, y_val = y_raw[train_idx], y_raw[val_idx]

            # 1.b Dataset Size Ablation (Subsampling)
            train_fraction = getattr(args, 'train_fraction', 1.0)
            if train_fraction < 1.0:
                from sklearn.model_selection import train_test_split
                # We strictly truncate the training data, keeping stratify
                X_train_raw, _, y_train, _ = train_test_split(
                    X_train_raw, y_train, train_size=train_fraction, stratify=y_train, random_state=42
                )

            # 2. HERMETIC TRANSFORMATION (No Data Leakage!)
            # We learn how to impute and scale ONLY on the training set
            X_train_clean = preprocessor.fit_transform(X_train_raw)
            # We apply this knowledge to the test/validation set
            X_val_clean = preprocessor.transform(X_val_raw)
            
            # Input dimension for the network (may change due to One-Hot Encoding)
            input_dim = X_train_clean.shape[1]

            # 3. Packing into PyTorch DataLoaders
            train_dataset = MedicalTabularDataset(X_train_clean, y_train)
            val_dataset = MedicalTabularDataset(X_val_clean, y_val)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            # 4. Determine classes
            num_classes = len(np.unique(y_raw))
            is_binary = (num_classes == 2)

            # 4. Hyperparameter Optimization (Optuna Light)
            print(f"[CV] Running Optuna HPO for {args.model_name} (15 trials)...")
            tuner = OptunaTuner(
                model_class=model_class,
                X_train_clean=X_train_clean,
                y_train=y_train,
                args=args,
                is_binary=is_binary,
                num_classes=num_classes,
                input_dim=input_dim
            )
            best_params = tuner.optimize(n_trials=15)
            print(f"[CV] Best params for Repeat {repeat} Fold {fold}: {best_params}")

            # 5. Final Model Initialization (Reset weights per fold!)
            if issubclass(model_class, BaseEstimator):
                model = model_class(random_state=args.random_state if hasattr(args, 'random_state') else 42, **best_params)
                optimizer = None
                criterion = None
            else:
                model_kwargs = {
                    'input_dim': input_dim,
                    'output_dim': 1 if is_binary else num_classes
                }
                
                if 'hidden_dim' in best_params:
                    model_kwargs['hidden_dims'] = [best_params['hidden_dim'], best_params['hidden_dim'] // 2]
                
                # Propagate the correct KAN capacity hyperparameter from HPO
                for kan_param in ('grid_size', 'num_grids', 'degree', 'num_wavelets'):
                    if kan_param in best_params:
                        model_kwargs[kan_param] = best_params[kan_param]
                    
                model = model_class(**model_kwargs)
                
                # Optimizer and loss function configuration
                lr = best_params.get('lr', args.lr)
                weight_decay = best_params.get('weight_decay', 1e-4)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

            # 6. Initialization of your 'clean' Trainer (without MLflow)
            trainer = trainer_class(
                model=model, 
                optimizer=optimizer, 
                criterion=criterion, 
                device=torch.device(args.device), 
                is_binary=is_binary,
                dataset_name=dataset_name,
                model_name=args.model_name,
                repeat=repeat,
                fold=fold
            )
            
            run_params = vars(args)
            run_params["repeat"] = repeat
            run_params["fold"] = fold
            
            # 7. Training and Evaluation
            trainer.fit(train_loader, val_loader, epochs=args.epochs, run_params=run_params)
            metrics = trainer.evaluate(val_loader)
            
            # Save best_params to artifacts
            save_dir = f"results/artifacts/{dataset_name}/{args.model_name}"
            os.makedirs(save_dir, exist_ok=True)
            params_path = os.path.join(save_dir, f"{dataset_name}_{args.model_name}_Repeat{repeat}_Fold{fold}_best_params.json")
            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=4)
            
            # Add metadata to metrics, so we know what to merge this with
            metrics['repeat'] = repeat
            metrics['fold'] = fold
            metrics['model'] = args.model_name
            metrics['dataset'] = dataset_name
            metrics['train_fraction'] = getattr(args, 'train_fraction', 1.0)
            
            if isinstance(model, BaseEstimator):
                metrics['trainable_parameters'] = 0
            else:
                metrics['trainable_parameters'] = model.get_num_parameters()
                
            metrics['avg_epoch_time_seconds'] = trainer.avg_epoch_time_seconds
            metrics['total_train_time_seconds'] = trainer.total_train_time_seconds
            metrics['inference_time_ms'] = getattr(trainer, 'inference_time_ms', trainer.inference_time_per_sample_ms)
            metrics['inference_time_per_sample_ms'] = metrics['inference_time_ms']
            metrics['inference_time_total_seconds'] = trainer.inference_time_total_seconds
            metrics['brier_score'] = metrics.get('brier_score', float('nan'))
            
            all_fold_metrics.append(metrics)

        # Return a beautiful Pandas table with results from all folds
        return pd.DataFrame(all_fold_metrics)