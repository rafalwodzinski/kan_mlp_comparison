import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dodanie ścieżki, aby Python widział folder src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.stats import FrequentistEvaluator
from src.evaluation.bayesian_stats import BayesianEvaluator

def find_latest_results():
    """Znajduje najnowszy plik wynikowy z eksperymentu w folderze results."""
    files = glob.glob("results/benchmark_master_*.csv")
    if not files:
        raise FileNotFoundError("Nie znaleziono pliku wyników: results/benchmark_master_*.csv")
    
    # Wybieramy plik o najpóźniejszej dacie modyfikacji
    latest_file = max(files, key=os.path.getmtime)
    print(f"[Info] Załadowano najnowszy plik wyników: {latest_file}")
    return latest_file

def generate_summary_table(df):
    """
    Tworzy tabelę agregacji grupując po zbiorze danych i modelu.
    Wylicza średnią oraz odchylenie standardowe dla kluczowych metryk.
    """
    # Sprawdzamy jakie metryki faktycznie są w pliku
    possible_metrics = ['mcc', 'auroc', 'f1_score', 'accuracy', 'loss']
    metrics = [m for m in possible_metrics if m in df.columns]
    
    if not metrics:
        print("[Ostrzeżenie] Nie znaleziono znanych metryk w pliku do agregacji.")
        return
        
    # Agregacja danych (średnia i odchylenie standardowe)
    grouped = df.groupby(['dataset', 'model'])[metrics].agg(['mean', 'std'])
    
    # Tworzenie przyjaznego dla publikacji formatu tekstowego "Średnia ± Std"
    summary_df = pd.DataFrame(index=grouped.index)
    for m in metrics:
        mean_col = grouped[m]['mean']
        std_col = grouped[m]['std']
        # Formatowanie do 4 miejsc po przecinku
        summary_df[m] = mean_col.map("{:.4f}".format) + " ± " + std_col.map("{:.4f}".format)
            
    summary_df.reset_index(inplace=True)
    out_path = "results/summary_metrics.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"[Info] Zapisano zagregowane tabele do: {out_path}")

def generate_statistical_reports(df):
    """
    Uruchamia rygorystyczne testy statystyczne na uzyskanych wynikach i zapisuje rezultaty.
    """
    frequentist = FrequentistEvaluator()
    bayesian = BayesianEvaluator()
    
    datasets = df['dataset'].unique()
    wilcoxon_results = []
    bayesian_results = []
    
    for dataset in datasets:
        df_ds = df[df['dataset'] == dataset]
        
        # Znalezienie najlepszego wariantu KAN (pod względem średniego MCC) na tym zbiorze
        kan_models = [m for m in df_ds['model'].unique() if "KAN" in m]
        if not kan_models:
            continue
            
        mean_mcc = df_ds.groupby('model')['mcc'].mean()
        best_kan = mean_mcc[kan_models].idxmax()
        baseline = "StandardMLP"
        
        if baseline not in df_ds['model'].unique():
            continue
            
        # 1. Wilcoxon Post-Hoc Test
        # Używamy FrequentistEvaluator żeby porównać bazę (StandardMLP) z najlepszym modelem KAN
        try:
            res_wilcoxon = frequentist.run_wilcoxon_post_hoc(
                df_ds, 
                baseline_model=baseline, 
                competitor_models=[best_kan], 
                metric='mcc'
            )
            # Dodajemy adnotację jakiego zbioru dotyczy ten wynik
            res_wilcoxon['dataset'] = dataset
            wilcoxon_results.append(res_wilcoxon)
        except Exception as e:
            print(f"[Ostrzeżenie] Wilcoxon test failed dla {dataset}: {e}")
        
        # 2. Bayesian Correlated t-test
        # Używamy BayesianEvaluator dla prawdopodobieństw scenariuszy z użyciem ROPE
        try:
            res_bayes = bayesian.bayesian_correlated_ttest(
                df_ds, 
                model_a=baseline, 
                model_b=best_kan, 
                metric='mcc'
            )
            res_bayes['dataset'] = dataset
            res_bayes['Model A (Baseline)'] = baseline
            res_bayes['Model B (Best KAN)'] = best_kan
            bayesian_results.append(res_bayes)
        except Exception as e:
            print(f"[Ostrzeżenie] Bayesian test failed dla {dataset}: {e}")
        
    # Eksport tabel statystycznych do plików CSV
    if wilcoxon_results:
        final_wilcoxon = pd.concat(wilcoxon_results, ignore_index=True)
        final_wilcoxon.to_csv("results/stats_wilcoxon_posthoc.csv", index=False)
        print("[Info] Zapisano testy częstotliwościowe (Wilcoxon) do: results/stats_wilcoxon_posthoc.csv")
        
    if bayesian_results:
        final_bayesian = pd.DataFrame(bayesian_results)
        final_bayesian.to_csv("results/stats_bayesian_rope.csv", index=False)
        print("[Info] Zapisano testy Bayesowskie (ROPE) do: results/stats_bayesian_rope.csv")

def generate_plots(df):
    """
    Tworzy wykresy o jakości publikacyjnej (300 DPI, przyjazne palety kolorów) 
    dla kluczowych metryk MCC i AUROC.
    """
    os.makedirs("results/plots", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    metrics_to_plot = ['mcc', 'auroc']
    
    for metric in metrics_to_plot:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(14, 8))
        
        # Boxplot z paletą przyjazną dla daltonistów (colorblind friendly)
        # Oś X to zbiór danych, co pozwoli szybko ocenić wyniki modeli na każdym zbiorze
        ax = sns.boxplot(
            x="dataset", 
            y=metric, 
            hue="model", 
            data=df, 
            palette="colorblind"
        )
        
        # Estetyka publikacyjna
        plt.title(f"Rozkład Metryki {metric.upper()} w Walidacji Krzyżowej (5-Fold CV)", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Medyczny Zbiór Danych", fontsize=14, labelpad=10)
        plt.ylabel(f"Wartość {metric.upper()}", fontsize=14, labelpad=10)
        
        # Rotacja nazw osi X dla czytelności
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
        # Wyrzucenie legendy poza obszar wykresu
        plt.legend(title="Model Architektury", title_fontsize='13', fontsize='11', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        
        out_path = f"results/plots/boxplot_{metric}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Info] Zapisano wykres: {out_path}")

def generate_confusion_matrices(df):
    """
    Agreguje macierze pomyłek z 5 foldów i generuje heatmapę dla każdej kombinacji zbiór-model.
    """
    os.makedirs("results/plots", exist_ok=True)
    datasets = df['dataset'].unique()
    models = df['model'].unique()
    
    for dataset in datasets:
        for model in models:
            save_dir = f"results/artifacts/{dataset}/{model}"
            if not os.path.exists(save_dir):
                continue
                
            cm_files = glob.glob(os.path.join(save_dir, "*_confusion_matrix.csv"))
            if not cm_files:
                continue
                
            total_cm = None
            for cm_file in cm_files:
                cm = np.loadtxt(cm_file, delimiter=",", dtype=int)
                if total_cm is None:
                    total_cm = cm
                else:
                    total_cm += cm
                    
            if total_cm is not None:
                plt.figure(figsize=(8, 6))
                sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Suma Macierzy Pomyłek (5-Fold CV)\nZbiór: {dataset} | Model: {model}", fontsize=14)
                plt.xlabel("Przewidziana Klasa", fontsize=12)
                plt.ylabel("Prawdziwa Klasa", fontsize=12)
                plt.tight_layout()
                
                out_path = f"results/plots/cm_{dataset}_{model}.png"
                plt.savefig(out_path, dpi=300)
                plt.close()
                print(f"[Info] Zapisano macierz pomyłek: {out_path}")

def generate_learning_curves(df):
    """
    Wczytuje historię uczenia dla każdego foldu, uśrednia wyniki i rysuje krzywe uczenia 
    (Train vs Val Loss) ze wstęgą odchylenia standardowego.
    """
    os.makedirs("results/plots", exist_ok=True)
    datasets = df['dataset'].unique()
    models = df['model'].unique()
    
    for dataset in datasets:
        for model in models:
            save_dir = f"results/artifacts/{dataset}/{model}"
            if not os.path.exists(save_dir):
                continue
                
            hist_files = glob.glob(os.path.join(save_dir, "*_history.json"))
            if not hist_files:
                continue
                
            all_train_loss = []
            all_val_loss = []
            
            for hf in hist_files:
                with open(hf, "r") as f:
                    try:
                        history = json.load(f)
                        all_train_loss.append(history['train_loss'])
                        all_val_loss.append(history['val_loss'])
                    except Exception as e:
                        print(f"[Ostrzeżenie] Nie udało się wczytać historii z {hf}: {e}")
            
            if not all_train_loss:
                continue
                
            # Na wypadek gdyby jeden fold miał mniej epok z powodu np. ew. błędu, bierzemy min_len
            min_len = min(len(t) for t in all_train_loss)
            
            train_loss_arr = np.array([t[:min_len] for t in all_train_loss])
            val_loss_arr = np.array([v[:min_len] for v in all_val_loss])
            
            train_mean = np.mean(train_loss_arr, axis=0)
            train_std = np.std(train_loss_arr, axis=0)
            val_mean = np.mean(val_loss_arr, axis=0)
            val_std = np.std(val_loss_arr, axis=0)
            
            epochs = np.arange(1, min_len + 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_mean, label="Trening (Loss)", color="blue")
            plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
            
            plt.plot(epochs, val_mean, label="Walidacja (Loss)", color="orange")
            plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, color="orange", alpha=0.2)
            
            plt.title(f"Krzywa Uczenia ze standardowym odchyleniem\nZbiór: {dataset} | Model: {model}", fontsize=14)
            plt.xlabel("Epoka", fontsize=12)
            plt.ylabel("Wartość Funkcji Straty", fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            out_path = f"results/plots/learning_curves_{dataset}_{model}.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"[Info] Zapisano krzywą uczenia: {out_path}")

def main():
    print("="*60)
    print(" AUTOMATYCZNY GENERATOR RAPORTÓW (ETAP 4) ")
    print("="*60)
    try:
        latest_file = find_latest_results()
        df = pd.read_csv(latest_file)
        
        print("\n--> 1. Generowanie tabel agregacyjnych...")
        generate_summary_table(df)
        
        print("\n--> 2. Generowanie raportów statystycznych...")
        generate_statistical_reports(df)
        
        print("\n--> 3. Generowanie wizualizacji do publikacji...")
        generate_plots(df)
        
        print("\n--> 4. Generowanie map pomyłek (Heatmaps)...")
        generate_confusion_matrices(df)
        
        print("\n--> 5. Generowanie krzywych uczenia (Learning Curves)...")
        generate_learning_curves(df)
        
        print("\n" + "="*60)
        print("[SUKCES] Pełny potok analityczny zakończył się bezbłędnie.")
        print("Wszystkie tabele i wykresy znajdują się w folderze 'results/'.")
        print("="*60)
        
    except Exception as e:
        print(f"\n[BŁĄD KRYTYCZNY] Proces generowania raportu przerwany: {e}")

if __name__ == "__main__":
    main()
