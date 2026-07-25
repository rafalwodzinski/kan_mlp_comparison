"""
Main script orchestrating experiments (Automate Benchmark).
Responsible for automatically finding all processed datasets,
initializing all planned architectures (StandardMLP and 9 KAN variants)
and systematically running cross-validation (3×5 Repeated Stratified K-Fold CV)
for each dataset-model pair.
Saves results and aggregates them in one unified master results file (.csv).
"""

import os
import sys
import time
import pandas as pd
import torch
from dataclasses import dataclass
from tqdm import tqdm

# Add path so Python sees the main src folder of the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.cross_validation import CrossValidator
from src.training.trainer import TabularTrainer

# Imports of all planned research models
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
from sklearn.ensemble import RandomForestClassifier

# Complete registry of 11 architectures (module-level for importability by ablation/analysis scripts)
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
    "ReLUKAN": ReLUKAN,
    "RandomForest": RandomForestClassifier
}

def _detect_device() -> str:
    """Detect best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@dataclass
class ExperimentArgs:
    """
    Configuration structure for a single experiment.
    Guarantees standardized hyperparameters (reproducibility).
    """
    data_path: str = ""
    model_name: str = ""
    epochs: int = 50                 # Number of training epochs
    batch_size: int = 32             # Batch size
    lr: float = 1e-3                 # Learning Rate
    device: str = _detect_device()   # Automatic GPU/MPS detection

def main():
    """
    Main loop controlling the entire benchmark:
    1. Scans data/processed/ directory for datasets.
    2. Uses the module-level MODELS registry.
    3. For each dataset and for each model, executes a 3×5 Repeated Stratified K-Fold CV loop.
    4. Saves partial results, and a full master file upon completion.
    """
    # 1. Definition of research space
    DATASETS_DIR = "data/processed/"
    
    # Dynamic search for processed datasets
    if os.path.exists(DATASETS_DIR):
        datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith('_processed.csv')]
    else:
        datasets = []
        print(f"Warning: Directory {DATASETS_DIR} does not exist. Run preprocessor.py first.")

    os.makedirs("results", exist_ok=True)
    all_benchmark_results = []
    
    # Calculate total number of experiments for a smooth progress bar
    total_experiments = len(datasets) * len(MODELS)
    
    print(f" Starting grand medical benchmark")
    print(f" Device: {ExperimentArgs.device.upper()}")
    print(f" Config: {len(datasets)} datasets x {len(MODELS)} models = {total_experiments} CV tests\n")

    start_time = time.time()
    # Initialize validator with 3×5 Repeated Stratified K-Fold CV and seed 42 for full reproducibility
    cv_engine = CrossValidator(k_folds=5, n_repeats=3, random_state=42)

    # Initialize progress bar from tqdm library
    pbar = tqdm(total=total_experiments, desc="Total progress", unit="exp")

    # Main loop iterating over data files
    for dataset_file in datasets:
        data_path = os.path.join(DATASETS_DIR, dataset_file)
        dataset_name = dataset_file.replace("_processed.csv", "")
        
        # Protection in case the file suddenly disappears during loop execution
        if not os.path.exists(data_path):
            pbar.update(len(MODELS)) # Skip all models for this dataset on the bar
            continue
            
        # Loop iterating over model classes for the current dataset
        for model_name, model_class in MODELS.items():
            # Current status in console, allowing easy progress tracking
            pbar.set_postfix_str(f"Currently: {model_name} on {dataset_name}")
            
            args = ExperimentArgs(data_path=data_path, model_name=model_name)
            
            try:
                # Run rigorous 3×5 Repeated Stratified K-Fold CV for current dataset-model pair
                df_results = cv_engine.run(
                    model_class=model_class,
                    trainer_class=TabularTrainer,
                    args=args
                )
                all_benchmark_results.append(df_results)
                
            except Exception as e:
                # Silent error logging, so one model failure doesn't crash entire hours-long benchmark
                with open("results/error_log.txt", "a") as f:
                    f.write(f"Error: {model_name} on {dataset_name}: {str(e)}\n")
            
            pbar.update(1)

    pbar.close()

    # 2. Aggregation and saving of the output file
    if all_benchmark_results:
        final_df = pd.concat(all_benchmark_results, ignore_index=True)
        # Timestamp, to not overwrite yesterday's tests
        timestamp = time.strftime("%Y%m%d-%H%M")
        results_path = f"results/benchmark_master_{timestamp}.csv"
        
        # Dump everything to a CSV file ready for analysis
        final_df.to_csv(results_path, index=False)
        
        print("\n" + "#"*60)
        print(f" BENCHMARK FINISHED SUCCESSFULLY")
        print(f" Results saved in: {results_path}")
        print(f" Duration: {(time.time() - start_time) / 60:.2f} minutes")
        print("#"*60)
    else:
        print("\n Benchmark did not generate any results. Check results/error_log.txt")

if __name__ == "__main__":
    main()