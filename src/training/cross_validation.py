import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Any, Type
import os

class CrossValidator:
    """
    Silnik obsługujący logikę Stratified K-Fold.
    Gwarantuje czystość eksperymentu w każdym foldzie.
    """
    def __init__(self, k_folds: int = 5, random_state: int = 42):
        self.k_folds = k_folds
        self.skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    def run(self, 
            model_class: Type, 
            trainer_class: Type, 
            df: pd.DataFrame, 
            target_col: str, 
            feature_cols: List[str], 
            args: Any) -> pd.DataFrame:
        
        X = df[feature_cols].values
        y = df[target_col].values
        all_fold_metrics = []
        dataset_name = os.path.basename(args.data_path).replace("_processed.csv", "")

        print(f"\n[CV] Rozpoczynam {self.k_folds}-fold CV dla modelu {args.model_name}...")

        # Główna pętla CV
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            print(f"\n>>> FOLD {fold+1}/{self.k_folds}")

            # Wykorzystujemy istniejący loader na podzbiorach
            from src.data.loader import prepare_dataloaders
            train_loader, val_loader, _, input_dim = prepare_dataloaders(
                df=df.iloc[np.concatenate([train_idx, val_idx])],
                target_col=target_col,
                numerical_cols=feature_cols,
                test_size=0.0, # Cała reszta poza treningiem to walidacja w tym kroku
                val_size=len(val_idx)/len(df),
                batch_size=args.batch_size
            )

            # 1. Inicjalizacja Nowego Modelu (Reset wag!)
            num_classes = len(np.unique(y))
            is_binary = (num_classes == 2)
            model = model_class(input_dim=input_dim, output_dim=1 if is_binary else num_classes)
            
            # 2. Konfiguracja
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

            # 3. Trening
            trainer = trainer_class(
                model=model, optimizer=optimizer, criterion=criterion, 
                device=torch.device(args.device), is_binary=is_binary,
                experiment_name=f"CV_{dataset_name}"
            )
            
            run_params = vars(args)
            run_params["fold"] = fold + 1
            
            trainer.fit(train_loader, val_loader, epochs=args.epochs, run_params=run_params)
            
            # 4. Ewaluacja i zbieranie metryk
            metrics = trainer.evaluate(val_loader)
            all_fold_metrics.append(metrics)

        return pd.DataFrame(all_fold_metrics)