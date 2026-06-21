import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Type
import os

from src.data.loader import MedicalTabularDataset, get_data_and_preprocessor

class CrossValidator:
    """
    Silnik obsługujący logikę Stratified K-Fold.
    Gwarantuje czystość eksperymentu w każdym foldzie poprzez niezależne trenowanie
    obiektu Preprocesora (zapobiega wyciekowi danych).
    """
    def __init__(self, k_folds: int = 5, random_state: int = 42):
        self.k_folds = k_folds
        self.skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    def run(self, 
            model_class: Type, 
            trainer_class: Type, 
            args: Any) -> pd.DataFrame:
        
        # 1. Zamiast przekazywać DF z zewnątrz, walidator ładuje go sam za pomocą nowego loadera
        dataset_filename = os.path.basename(args.data_path)
        dataset_name = dataset_filename.replace("_processed.csv", "")
        
        X_raw, y_raw, preprocessor = get_data_and_preprocessor(args.data_path, dataset_filename)
        all_fold_metrics = []

        print(f"\n[CV] Rozpoczynam {self.k_folds}-fold CV dla modelu {args.model_name}...")
        print(f"[CV] Zbiór: {dataset_name} | Rozmiar próbek: {len(X_raw)}")

        # Główna pętla CV na SUROWYCH danych
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(X_raw, y_raw)):
            print(f"\n>>> FOLD {fold+1}/{self.k_folds}")

            # Wyodrębnienie surowych foldów
            X_train_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
            y_train, y_val = y_raw[train_idx], y_raw[val_idx]

            # 2. HERMETYCZNA TRANSFORMACJA (Brak Wycieku Danych!)
            # Uczymy jak imputować i skalować TYLKO na zbiorze treningowym
            X_train_clean = preprocessor.fit_transform(X_train_raw)
            # Aplikujemy tę wiedzę do zbioru testowego/walidacyjnego
            X_val_clean = preprocessor.transform(X_val_raw)
            
            # Wymiar wejścia dla sieci (może się zmienić przez One-Hot Encoding)
            input_dim = X_train_clean.shape[1]

            # 3. Pakowanie do DataLoaderów PyTorcha
            train_dataset = MedicalTabularDataset(X_train_clean, y_train)
            val_dataset = MedicalTabularDataset(X_val_clean, y_val)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            # 4. Inicjalizacja Nowego Modelu (Reset wag co fold!)
            num_classes = len(np.unique(y_raw))
            is_binary = (num_classes == 2)
            
            model = model_class(input_dim=input_dim, output_dim=1 if is_binary else num_classes)
            
            # 5. Konfiguracja optymalizatora i funkcji straty
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

            # 6. Inicjalizacja Twojego 'czystego' Trainera (bez MLflow)
            trainer = trainer_class(
                model=model, 
                optimizer=optimizer, 
                criterion=criterion, 
                device=torch.device(args.device), 
                is_binary=is_binary,
                dataset_name=dataset_name,
                model_name=args.model_name,
                fold=fold+1
            )
            
            run_params = vars(args)
            run_params["fold"] = fold + 1
            
            # 7. Trening i Ewaluacja
            trainer.fit(train_loader, val_loader, epochs=args.epochs, run_params=run_params)
            metrics = trainer.evaluate(val_loader)
            
            # Dodajemy metadane do metryk, żebyśmy wiedzieli z czym to złączyć
            metrics['fold'] = fold + 1
            metrics['model'] = args.model_name
            metrics['dataset'] = dataset_name
            
            all_fold_metrics.append(metrics)

        # Zwracamy piękną tabelę Pandas z wynikami wszystkich foldów
        return pd.DataFrame(all_fold_metrics)