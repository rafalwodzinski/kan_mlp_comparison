"""
Główny skrypt orkiestrujący eksperymenty (Automate Benchmark).
Odpowiada za automatyczne odnalezienie wszystkich przetworzonych zbiorów danych,
zainicjowanie wszystkich planowanych architektur (StandardMLP oraz 9 wariantów KAN)
i systematyczne przeprowadzenie walidacji krzyżowej (5-Fold CV) dla każdej pary zbior-model.
Zabezpiecza wyniki i agreguje je w jednym, zunifikowanym pliku wyników master (.csv).
"""

import os
import sys
import time
import pandas as pd
import torch
from dataclasses import dataclass
from tqdm import tqdm

# Dodanie ścieżki, aby Python widział główny folder src projektu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.cross_validation import CrossValidator
from src.training.trainer import TabularTrainer

# Importy wszystkich planowanych modeli badawczych
from src.models.mlp import StandardMLP
from src.models.kan_variants.tab_kan import TabKAN
from src.models.kan_variants.fast_kan import FastKAN
from src.models.kan_variants.cheby_kan import ChebyKAN
from src.models.kan_variants.jacobi_kan import JacobiKAN
from src.models.kan_variants.legendre_kan import LegendreKAN
from src.models.kan_variants.gram_kan import GramKAN
from src.models.kan_variants.taylor_kan import TaylorKAN
from src.models.kan_variants.wav_kan import WavKAN
from src.models.kan_variants.relu_kan import ReLUKAN

@dataclass
class ExperimentArgs:
    """
    Struktura konfiguracji dla pojedynczego eksperymentu.
    Gwarantuje ustandaryzowane hiperparametry (reprodukcyjność).
    """
    data_path: str = ""
    model_name: str = ""
    epochs: int = 50                 # Liczba epok uczenia
    batch_size: int = 32             # Rozmiar partii danych
    lr: float = 1e-3                 # Współczynnik uczenia (Learning Rate)
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # Automatyczne wykrywanie GPU

def main():
    """
    Główna pętla sterująca całym benchmarkiem:
    1. Skanuje katalog data/processed/ w poszukiwaniu zbiorów.
    2. Definiuje słownik dostępnych modeli.
    3. Dla każdego zbioru i dla każdego modelu wykonuje pętlę 5-Fold CV.
    4. Zapisuje wyniki cząstkowe, a po zakończeniu pełen plik master.
    """
    # 1. Definicja przestrzeni badawczej
    DATASETS_DIR = "data/processed/"
    
    # Dynamiczne wyszukiwanie przetworzonych zbiorów danych
    if os.path.exists(DATASETS_DIR):
        datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith('_processed.csv')]
    else:
        datasets = []
        print(f"Ostrzeżenie: Folder {DATASETS_DIR} nie istnieje. Uruchom preprocessor.py najpierw.")
    
    # Kompletny rejestr 10 planowanych architektur do przetestowania
    MODELS = {
        "StandardMLP": StandardMLP,
        "TabKAN": TabKAN,
        "FastKAN": FastKAN,
        "ChebyKAN": ChebyKAN,
        "JacobiKAN": JacobiKAN,
        "LegendreKAN": LegendreKAN,
        "GramKAN": GramKAN,
        "TaylorKAN": TaylorKAN,
        "WavKAN": WavKAN,
        "ReLUKAN": ReLUKAN
    }

    os.makedirs("results", exist_ok=True)
    all_benchmark_results = []
    
    # Obliczamy całkowitą liczbę eksperymentów dla płynnego paska postępu
    total_experiments = len(datasets) * len(MODELS)
    
    print(f" Rozpoczynam wielki benchmark medyczny")
    print(f" Urządzenie: {ExperimentArgs.device.upper()}")
    print(f" Konfiguracja: {len(datasets)} zbiorów x {len(MODELS)} modeli = {total_experiments} testów CV\n")

    start_time = time.time()
    # Inicjalizacja walidatora z ziarnem 42 dla pełnej reprodukcyjności
    cv_engine = CrossValidator(k_folds=5, random_state=42)

    # Inicjalizacja paska postępu z biblioteki tqdm
    pbar = tqdm(total=total_experiments, desc="Całkowity postęp", unit="exp")

    # Główna pętla iterująca po plikach danych
    for dataset_file in datasets:
        data_path = os.path.join(DATASETS_DIR, dataset_file)
        dataset_name = dataset_file.replace("_processed.csv", "")
        
        # Ochrona na wypadek, gdyby plik nagle zniknął w trakcie trwania pętli
        if not os.path.exists(data_path):
            pbar.update(len(MODELS)) # Pomijamy wszystkie modele dla tego zbioru na pasku
            continue
            
        # Pętla iterująca po klasach modeli dla aktualnego zbioru
        for model_name, model_class in MODELS.items():
            # Bieżący status w konsoli, pozwalający na łatwe śledzenie postępu
            pbar.set_postfix_str(f"Obecnie: {model_name} na {dataset_name}")
            
            args = ExperimentArgs(data_path=data_path, model_name=model_name)
            
            try:
                # Uruchomienie rygorystycznej 5-Fold CV dla bieżącej pary zbior-model
                df_results = cv_engine.run(
                    model_class=model_class,
                    trainer_class=TabularTrainer,
                    args=args
                )
                all_benchmark_results.append(df_results)
                
            except Exception as e:
                # Ciche logowanie błędów, by awaria jednego modelu nie wywaliła całego wielogodzinnego benchmarku
                with open("results/error_log.txt", "a") as f:
                    f.write(f"Error: {model_name} on {dataset_name}: {str(e)}\n")
            
            pbar.update(1)

    pbar.close()

    # 2. Agregacja i zapis pliku wyjściowego
    if all_benchmark_results:
        final_df = pd.concat(all_benchmark_results, ignore_index=True)
        # Znacznik czasowy, by nie nadpisać testów z wczoraj
        timestamp = time.strftime("%Y%m%d-%H%M")
        results_path = f"results/benchmark_master_{timestamp}.csv"
        
        # Zrzut wszystkiego do pliku CSV gotowego do analizy
        final_df.to_csv(results_path, index=False)
        
        print("\n" + "#"*60)
        print(f" BENCHMARK ZAKOŃCZONY SUKCESEM")
        print(f" Wyniki zapisano w: {results_path}")
        print(f" Czas trwania: {(time.time() - start_time) / 60:.2f} minut")
        print("#"*60)
    else:
        print("\n Benchmark nie wygenerował żadnych wyników. Sprawdź results/error_log.txt")

if __name__ == "__main__":
    main()