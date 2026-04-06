import subprocess
import os

# Definicje eksperymentu
DATASETS = [
    ("breast_cancer_processed.csv", "Diagnosis"),
    ("pima_diabetes_processed.csv", "class"),
    ("heart_disease_processed.csv", "num"),
    ("chronic_kidney_disease_processed.csv", "class"),
    ("parkinsons_processed.csv", "status"),
    ("cervical_cancer_processed.csv", "Biopsy"),
    ("cardiotocography_processed.csv", "NSP")
]

MODELS = [
    "StandardMLP", "TabResNet", "BaseKAN", "FastKAN", "ChebyKAN",
    "RelKAN", "WavKAN", "LegendreKAN", "JacobiKAN", "TaylorKAN", 
    "GramKAN", "TabKAN"
]

def run_benchmark():
    for ds_file, target in DATASETS:
        ds_path = os.path.join("data", "processed", ds_file)
        
        for model in MODELS:
            print(f"\n🚀 URUCHAMIAM: Model={model} na zbiorze={ds_file}")
            
            # Budowanie komendy
            cmd = [
                "python", "scripts/run_experiment.py",
                "--data_path", ds_path,
                "--target_col", target,
                "--model_name", model,
                "--epochs", "50",      # Sugerowana liczba epok dla stabilności
                "--cv",                # Zawsze używamy CV dla wyników naukowych
                "--k_folds", "5",      # 5-fold CV to złoty środek
                "--batch_size", "32",
                "--lr", "0.001"
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ BŁĄD przy {model} na {ds_file}: {e}")
                continue

if __name__ == "__main__":
    run_benchmark()