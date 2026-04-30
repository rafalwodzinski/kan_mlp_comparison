import os
import sys
import time
import pandas as pd
import torch
from dataclasses import dataclass
from tqdm import tqdm

# Dodanie ścieżki, aby Python widział folder src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.cross_validation import CrossValidator
from src.training.trainer import TabularTrainer

# Importy wszystkich planowanych modeli
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
    data_path: str = ""
    model_name: str = ""
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # 1. Definicja przestrzeni badawczej
    DATASETS_DIR = "data/processed/"
    datasets = [
        "breast_cancer_processed.csv",
        "heart_disease_processed.csv",
        "chronic_kidney_disease_processed.csv",
        "parkinsons_processed.csv",
        "cervical_cancer_processed.csv",
        "cardiotocography_processed.csv",
        "pima_diabetes_processed.csv"
    ]
    
    # Kompletna lista 10 planowanych modeli (MLP + 9 wariantów KAN)
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
    
    # Obliczamy całkowitą liczbę eksperymentów dla paska postępu
    total_experiments = len(datasets) * len(MODELS)
    
    print(f" Rozpoczynam wielki benchmark medyczny")
    print(f" Urządzenie: {ExperimentArgs.device.upper()}")
    print(f" Konfiguracja: {len(datasets)} zbiorów x {len(MODELS)} modeli = {total_experiments} testów CV\n")

    start_time = time.time()
    cv_engine = CrossValidator(k_folds=5, random_state=42)

    # Główny pasek postępu (cały benchmark)
    pbar = tqdm(total=total_experiments, desc="Całkowity postęp", unit="exp")

    for dataset_file in datasets:
        data_path = os.path.join(DATASETS_DIR, dataset_file)
        dataset_name = dataset_file.replace("_processed.csv", "")
        
        if not os.path.exists(data_path):
            pbar.update(len(MODELS)) # Pomijamy wszystkie modele dla tego zbioru
            continue
            
        for model_name, model_class in MODELS.items():
            # Aktualizacja opisu paska o bieżący status
            pbar.set_postfix_str(f"Obecnie: {model_name} na {dataset_name}")
            
            args = ExperimentArgs(data_path=data_path, model_name=model_name)
            
            try:
                # Uruchomienie 5-Fold CV
                df_results = cv_engine.run(
                    model_class=model_class,
                    trainer_class=TabularTrainer,
                    args=args
                )
                all_benchmark_results.append(df_results)
                
            except Exception as e:
                # Zapisywanie błędów do pliku logu, aby nie zaśmiecać konsoli z paskiem postępu
                with open("results/error_log.txt", "a") as f:
                    f.write(f"Error: {model_name} on {dataset_name}: {str(e)}\n")
            
            pbar.update(1)

    pbar.close()

    # 2. Agregacja i zapis
    if all_benchmark_results:
        final_df = pd.concat(all_benchmark_results, ignore_index=True)
        # Dodajemy timestamp do nazwy, żeby nie nadpisywać starych testów
        timestamp = time.strftime("%Y%m%d-%H%M")
        results_path = f"results/benchmark_master_{timestamp}.csv"
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