"""
Moduł realizujący studium przypadku (Case Study) dla zjawiska wyjaśnialności (Interpretability).
Porównuje wagę poszczególnych cech (Feature Importance) wyliczoną na podstawie metody permutacji 
dla klasycznego modelu StandardMLP oraz modelu WavKAN na zbiorze Parkinson's Disease.
Wykres końcowy zapisywany jest do folderu results/plots/.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef

# Dodanie ścieżki, aby Python widział folder src i jego moduły
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import get_data_and_preprocessor
from src.models.mlp import StandardMLP
from src.models.kan_variants.wav_kan import WavKAN
from src.evaluation.interpretability import ModelInterpreter

def main():
    """
    Główna funkcja wykonawcza skryptu. Odpowiada za:
    1. Wczytanie zbioru danych i podział walidacyjny (odtworzenie Fold 1).
    2. Załadowanie pretrenowanych wag modeli MLP i KAN.
    3. Przeprowadzenie analizy Permutation Feature Importance.
    4. Wygenerowanie i zapis wykresu słupkowego dla TOP 10 cech.
    """
    print("Rozpoczynanie Studium Przypadku dla zbioru Parkinson's...")
    # Wybór urządzenia (GPU lub CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Wczytanie danych
    data_path = "data/processed/parkinsons_processed.csv"
    dataset_filename = "parkinsons_processed.csv"
    
    if not os.path.exists(data_path):
        print(f"Błąd: Nie znaleziono pliku {data_path}. Upewnij się, że dane są dostępne.")
        return

    # Pozyskanie danych surowych oraz preprocesora scikit-learn
    X_raw, y_raw, preprocessor = get_data_and_preprocessor(data_path, dataset_filename)
    
    # 2. Reprodukcja podziału na Fold 1
    # Korzystamy z StratifiedKFold z ziarnem 42, co daje identyczny podział jak w procesie treningu
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(X_raw, y_raw))
    
    X_train_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
    y_train, y_val = y_raw[train_idx], y_raw[val_idx]
    
    # 3. Transformacja danych (brak wycieku danych - dopasowanie tylko na train)
    X_train_clean = preprocessor.fit_transform(X_train_raw)
    X_val_clean = preprocessor.transform(X_val_raw)
    
    # Ekstrakcja nazw cech (czyszczenie prefixów dodawanych przez ColumnTransformer)
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.split('__')[-1] for name in feature_names]
    
    # Konwersja do formy tensorów wymaganych przez PyTorch
    X_val_t = torch.tensor(X_val_clean, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    # 4. Konfiguracja i definicja struktur modeli
    input_dim = X_train_clean.shape[1]
    num_classes = len(np.unique(y_raw))
    is_binary = (num_classes == 2)
    output_dim = 1 if is_binary else num_classes
    
    # Inicjalizacja klasycznego MLP oraz sieci KAN (wariant falkowy)
    mlp = StandardMLP(input_dim=input_dim, output_dim=output_dim)
    kan = WavKAN(input_dim=input_dim, output_dim=output_dim)
    
    # Ścieżki do wyuczonych wag z najlepszego foldu (Fold 1)
    mlp_path = "results/artifacts/parkinsons/StandardMLP/parkinsons_StandardMLP_Fold1_weights.pth"
    kan_path = "results/artifacts/parkinsons/WavKAN/parkinsons_WavKAN_Fold1_weights.pth"
    
    if not os.path.exists(mlp_path) or not os.path.exists(kan_path):
        print("Błąd: Nie znaleziono plików z wagami w results/artifacts/parkinsons/")
        return

    # Ładowanie zrzuconych stanów wag
    mlp.load_state_dict(torch.load(mlp_path, map_location=device))
    kan.load_state_dict(torch.load(kan_path, map_location=device))
    
    # 5. Obliczanie Permutation Feature Importance (Sprawiedliwe porównanie)
    print("Obliczanie Feature Importance dla StandardMLP...")
    mlp_interpreter = ModelInterpreter(mlp, device)
    # Permutacja losowo powtarzana 10 razy dla stabilności
    imp_mlp = mlp_interpreter.permutation_feature_importance(X_val_t, y_val_t, feature_names, metric_fn=matthews_corrcoef, n_repeats=10)
    
    print("Obliczanie Feature Importance dla WavKAN...")
    kan_interpreter = ModelInterpreter(kan, device)
    imp_kan = kan_interpreter.permutation_feature_importance(X_val_t, y_val_t, feature_names, metric_fn=matthews_corrcoef, n_repeats=10)
    
    # 6. Wizualizacja TOP 10 cech (sortowanie po ważności względem dominującego WavKAN)
    top_n = min(10, len(feature_names))
    sort_idx = np.argsort(imp_kan)[::-1][:top_n]
    
    # Wyselekcjonowanie nazw i wartości dla czołowych cech
    top_features = [feature_names[i] for i in sort_idx]
    top_kan_vals = imp_kan[sort_idx]
    top_mlp_vals = imp_mlp[sort_idx]
    
    x = np.arange(len(top_features))
    width = 0.35  # Szerokość słupka
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, top_mlp_vals, width, label='StandardMLP', color='#888888', edgecolor='black')
    rects2 = ax.bar(x + width/2, top_kan_vals, width, label='WavKAN', color='#1f77b4', edgecolor='black')
    
    # Estetyka i adnotacje wykresu
    ax.set_ylabel('Spadek MCC (Ważność Cechy)', fontsize=12)
    ax.set_title("TOP 10 Cech: WavKAN vs StandardMLP (Zbiór Parkinson's - Fold 1)", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Zapis obrazu do katalogu wynikowego
    os.makedirs("results/plots", exist_ok=True)
    plot_path = "results/plots/case_study_feature_importance.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Zakończono! Wykres zapisano w: {plot_path}")

if __name__ == "__main__":
    main()
