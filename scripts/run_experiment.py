import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Dodanie katalogu głównego projektu do ścieżki Pythona
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importy z naszych modułów
from src.data.loader import prepare_dataloaders
from src.training.trainer import TabularTrainer
from src.training.cross_validation import CrossValidator

# Importy Modeli (Model Registry)
from src.models.mlp import StandardMLP, TabResNet
from src.models.kan_variants.base_kan import BaseKAN
from src.models.kan_variants.fast_kan import FastKAN
from src.models.kan_variants.cheby_kan import ChebyKAN
from src.models.kan_variants.rel_kan import RelKAN
from src.models.kan_variants.wav_kan import WavKAN
from src.models.kan_variants.legendre_kan import LegendreKAN
from src.models.kan_variants.jacobi_kan import JacobiKAN
from src.models.kan_variants.taylor_kan import TaylorKAN
from src.models.kan_variants.gram_kan import GramKAN
from src.models.kan_variants.tab_kan import TabKAN

MODEL_REGISTRY = {
    "StandardMLP": StandardMLP, "TabResNet": TabResNet,
    "BaseKAN": BaseKAN, "FastKAN": FastKAN, "ChebyKAN": ChebyKAN,
    "RelKAN": RelKAN, "WavKAN": WavKAN, "LegendreKAN": LegendreKAN,
    "JacobiKAN": JacobiKAN, "TaylorKAN": TaylorKAN, "GramKAN": GramKAN,
    "TabKAN": TabKAN
}

def parse_args():
    parser = argparse.ArgumentParser(description="Uruchomienie eksperymentu KAN vs MLP")
    parser.add_argument("--data_path", type=str, required=True, help="Ścieżka do pliku CSV z danymi")
    parser.add_argument("--target_col", type=str, required=True, help="Nazwa kolumny docelowej (target)")
    parser.add_argument("--model_name", type=str, required=True, choices=MODEL_REGISTRY.keys(), help="Wybór modelu")
    parser.add_argument("--epochs", type=int, default=50, help="Liczba epok treningowych")
    parser.add_argument("--batch_size", type=int, default=32, help="Rozmiar batcha")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (szybkość uczenia)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cv", action="store_true", help="Uruchom w trybie walidacji krzyżowej")
    parser.add_argument("--k_folds", type=int, default=5, help="Liczba foldów w CV")
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # 1. Wczytanie danych
    df = pd.read_csv(args.data_path)
    feature_cols = [col for col in df.columns if col != args.target_col]
    dataset_name = os.path.basename(args.data_path).replace("_processed.csv", "")

    if args.cv:
        # --- TRYB WALIDACJI KRZYŻOWEJ ---
        engine = CrossValidator(k_folds=args.k_folds)
        results = engine.run(
            model_class=MODEL_REGISTRY[args.model_name],
            trainer_class=TabularTrainer,
            df=df,
            target_col=args.target_col,
            feature_cols=feature_cols,
            args=args
        )
        
        # Zapisujemy zagregowane wyniki foldów (kluczowe dla Bayes'a)
        output_name = f"cv_results_{args.model_name}_{dataset_name}.csv"
        results.to_csv(output_name, index=False)
        
        print(f"\n[ZAKOŃCZONO CV]")
        print(f"Średni MCC: {results['mcc'].mean():.4f} +/- {results['mcc'].std():.4f}")
        print(f"Średni AUROC: {results['auroc'].mean():.4f}")
        print(f"Wyniki wszystkich foldów zapisane w: {output_name}")

    else:
        # --- TRYB POJEDYNCZEGO TESTU (Twój oryginalny kod) ---
        print(f"[{args.model_name}] Rozpoczynam pojedynczy eksperyment...")
        
        train_loader, val_loader, test_loader, input_dim = prepare_dataloaders(
            df=df, target_col=args.target_col, numerical_cols=feature_cols, 
            batch_size=args.batch_size
        )

        num_classes = len(np.unique(df[args.target_col].dropna()))
        is_binary = (num_classes == 2)
        output_dim = 1 if is_binary else num_classes
        
        model = MODEL_REGISTRY[args.model_name](input_dim=input_dim, output_dim=output_dim)
        criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        trainer = TabularTrainer(
            model=model, optimizer=optimizer, criterion=criterion, 
            device=device, is_binary=is_binary,
            experiment_name=f"PoC_{dataset_name}"
        )
        
        run_params = vars(args)
        run_params["num_parameters"] = model.get_num_parameters()
        trainer.fit(train_loader, val_loader, epochs=args.epochs, run_params=run_params)
        print("Eksperyment zakończony.")

if __name__ == "__main__":
    main()