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
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"[{args.model_name}] Rozpoczynam eksperyment na urządzeniu: {device}")

    # 1. Wczytanie danych
    df = pd.read_csv(args.data_path)
    
    # Prosta heurystyka wyboru kolumn (zakładamy, że po czyszczeniu reszta to numeryczne dla PoC)
    # W przyszłości można to podawać z pliku konfiguracyjnego JSON
    feature_cols = [col for col in df.columns if col != args.target_col]
    
    # 2. Przygotowanie DataLoaderów
    train_loader, val_loader, test_loader, input_dim = prepare_dataloaders(
        df=df, target_col=args.target_col, numerical_cols=feature_cols, 
        batch_size=args.batch_size
    )

    # 3. Analiza Targetu (Binarna vs Wieloklasowa)
    num_classes = len(np.unique(df[args.target_col].dropna()))
    is_binary = (num_classes == 2)
    output_dim = 1 if is_binary else num_classes
    
    print(f"Typ klasyfikacji: {'Binarna' if is_binary else 'Wieloklasowa'} (Klasy: {num_classes})")
    print(f"Wymiar wejściowy: {input_dim}")

    # 4. Inicjalizacja Modelu
    model_class = MODEL_REGISTRY[args.model_name]
    model = model_class(input_dim=input_dim, output_dim=output_dim)

    # 5. Konfiguracja Straty (Loss) i Optymalizatora
    if is_binary:
        # Pamiętaj, aby w trainer.py dla binarnej klasyfikacji wymiary outputu i targetu były zgodne (np. squeeze)
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 6. Uruchomienie Trenera
    trainer = TabularTrainer(
        model=model, optimizer=optimizer, criterion=criterion, 
        device=device, experiment_name=f"PoC_{os.path.basename(args.data_path)}"
    )
    
    run_params = vars(args)
    run_params["num_parameters"] = model.get_num_parameters()
    
    trainer.fit(train_loader, val_loader, epochs=args.epochs, run_params=run_params)
    print("Eksperyment zakończony. Logi zapisane do MLflow.")

if __name__ == "__main__":
    main()