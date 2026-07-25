import os
import pandas as pd
from typing import List
import torch

# Fix for imports if script run from root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.automate_benchmark import ExperimentArgs, MODELS
from src.training.cross_validation import CrossValidator
from src.training.trainer import TabularTrainer

def run_ablation_study():
    DATASETS_DIR = "data/processed/"
    
    if os.path.exists(DATASETS_DIR):
        datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith('_processed.csv')]
    else:
        print(f"Warning: Directory {DATASETS_DIR} does not exist.")
        return
        
    # User-requested models for ablation
    ablation_models = ["StandardMLP", "WavKAN", "RandomForest", "FastKAN", "ChebyKAN"]
    fractions = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    
    os.makedirs("results", exist_ok=True)
    all_ablation_results = []
    
    validator = CrossValidator(k_folds=5, n_repeats=3)
    
    for dataset_file in datasets:
        data_path = os.path.join(DATASETS_DIR, dataset_file)
        
        for model_name in ablation_models:
            if model_name not in MODELS:
                print(f"Skipping {model_name} (Not found in MODELS)")
                continue
                
            model_class = MODELS[model_name]
            
            for fraction in fractions:
                print(f"\n===========================================================")
                print(f"ABLATION STUDY: {dataset_file} | {model_name} | Fraction: {fraction}")
                print(f"===========================================================")
                
                # Base args configuration
                args = ExperimentArgs(
                    data_path=data_path,
                    model_name=model_name,
                    batch_size=64,
                    epochs=100, # Handled by Early Stopping
                    lr=0.001,
                    device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                )
                
                # Inject train_fraction parameter
                setattr(args, 'train_fraction', fraction)
                
                try:
                    df_metrics = validator.run(
                        model_class=model_class,
                        trainer_class=TabularTrainer,
                        args=args
                    )
                    all_ablation_results.append(df_metrics)
                except Exception as e:
                    print(f"Error during {model_name} on {dataset_file} (Frac: {fraction}): {str(e)}")
                    
                # Save partial results defensively
                if all_ablation_results:
                    pd.concat(all_ablation_results).to_csv("results/ablation_results_partial.csv", index=False)
                    
    if all_ablation_results:
        final_df = pd.concat(all_ablation_results)
        final_df.to_csv("results/ablation_results.csv", index=False)
        print("\n[Ablation] Study complete. Results saved to results/ablation_results.csv")
    else:
        print("\n[Ablation] No results were generated.")

if __name__ == "__main__":
    run_ablation_study()
