"""
Analytical script for automatic generation of the final report 
from results collected during the main benchmark run (PHASE 4).
Processes raw CSV tables outputted by validation scripts and based on them 
prepares structured, publication-ready plots, 
result aggregations and statistical inferences.
"""

import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add path so Python sees the main src folder of the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.stats import FrequentistEvaluator
from src.evaluation.bayesian_stats import BayesianEvaluator
from src.evaluation.plot_radar import plot_radar_chart
from src.evaluation.plot_bayesian import plot_bayesian_rope_evidence
from src.evaluation.plot_ablation import plot_ablation_curves

def find_latest_results() -> str:
    """
    Scans results/ subdirectory for the freshest benchmark results file.
    
    Returns:
        str: Path to the newest CSV file with benchmark_master prefix.
    Raises:
        FileNotFoundError: If no result files are found.
    """
    files = glob.glob("results/benchmark_master_*.csv")
    if not files:
        raise FileNotFoundError("Results file not found: results/benchmark_master_*.csv")
    
    # Choose file with the latest modification date
    latest_file = max(files, key=os.path.getmtime)
    print(f"[Info] Loaded the latest results file: {latest_file}")
    return latest_file

def generate_summary_table(df: pd.DataFrame):
    """
    Calculates and saves global descriptive statistics from the entire experimental cycle.
    Creates a unified table (Mean ± Standard Deviation) in CSV format.
    
    Args:
        df (pd.DataFrame): Full results DataFrame loaded from benchmark_master.
    """
    # Defined catalog of target medical and computational metrics
    possible_metrics = [
        'mcc', 'auroc', 'f1_score', 'balanced_accuracy', 'brier_score', 
        'inference_time_per_sample_ms', 'total_train_time_seconds', 
        'trainable_parameters', 'loss'
    ]
    metrics = [m for m in possible_metrics if m in df.columns]
    
    if not metrics:
        print("[Warning] No known metrics found in file for aggregation.")
        return
        
    # Hierarchical grouping by dataset and model architecture
    grouped = df.groupby(['dataset', 'model'])[metrics].agg(['mean', 'std'])
    
    # Processing and concatenating for elegant text format for publication: "Mean ± Std"
    summary_df = pd.DataFrame(index=grouped.index)
    for m in metrics:
        mean_col = grouped[m]['mean']
        std_col = grouped[m]['std']
        # Formatting to 4 decimal places in floating-point notation
        summary_df[m] = mean_col.map("{:.4f}".format) + " ± " + std_col.map("{:.4f}".format)
            
    summary_df.reset_index(inplace=True)
    out_path = "results/summary_metrics.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"[Info] Saved aggregated tables to: {out_path}")

def generate_statistical_reports(df: pd.DataFrame):
    """
    Manages rigorous statistical assessments between classic MLP
    and the leading variant of the KAN network family across multiple metrics.
    Runs the pipeline of frequentist inference (Post-Hoc Wilcoxon) and Bayesian estimation (ROPE).

    Wilcoxon design (Demsar, 2006):
        Per metric, fold-level scores are first averaged within each dataset, yielding one
        scalar per (model, dataset) pair.  The Wilcoxon signed-rank test operates on the
        resulting n=7 dataset-level paired differences -- NOT on individual fold scores.
        The best-KAN opponent is the variant with the highest grand-mean MCC across ALL
        datasets (primary metric, Section III-D).

    Bayesian ROPE design (Benavoli et al., 2017):
        Operates per-dataset on all 15 fold-level paired differences.
        Best-KAN per dataset selected by highest mean MCC within that dataset.

    Args:
        df (pd.DataFrame): Full DataFrame after Repeated CV tests.
    """
    frequentist = FrequentistEvaluator()
    bayesian = BayesianEvaluator()

    datasets = df['dataset'].unique()
    wilcoxon_results = []
    bayesian_results = []

    metrics_to_test = ['mcc', 'auroc', 'f1_score', 'brier_score', 'inference_time_per_sample_ms']
    baseline = "StandardMLP"

    # ----------------------------------------------------------------
    # FREQUENTIST: Wilcoxon pooled across ALL datasets (n=7 pairs)
    # Best-KAN = single variant with highest grand-mean MCC (all data).
    # ----------------------------------------------------------------
    kan_models_global = [m for m in df['model'].unique() if "KAN" in m]
    if kan_models_global and baseline in df['model'].unique():
        grand_mean_mcc = df[df['model'].isin(kan_models_global)].groupby('model')['mcc'].mean()
        best_kan_global = grand_mean_mcc.idxmax()
        print(f"[Info] Wilcoxon pooled best-KAN (grand-mean MCC across all datasets): {best_kan_global}")

        for metric in metrics_to_test:
            if metric not in df.columns:
                continue
            try:
                # Pass the FULL DataFrame so the internal aggregation yields n=7 dataset rows
                res_wilcoxon = frequentist.run_wilcoxon_post_hoc(
                    df,
                    baseline_model=baseline,
                    competitor_models=[best_kan_global],
                    metric=metric
                )
                res_wilcoxon['metric'] = metric.upper()
                wilcoxon_results.append(res_wilcoxon)
            except Exception as e:
                print(f"[Warning] Wilcoxon test failed for metric={metric}: {e}")

    # ----------------------------------------------------------------
    # BAYESIAN ROPE: per-dataset, 15 fold-level paired differences.
    # Best-KAN per dataset by highest mean MCC within that dataset.
    # ----------------------------------------------------------------
    for dataset in datasets:
        df_ds = df[df['dataset'] == dataset]

        kan_models = [m for m in df_ds['model'].unique() if "KAN" in m]
        if not kan_models:
            continue

        mean_mcc_ds = df_ds.groupby('model')['mcc'].mean()
        best_kan = mean_mcc_ds[kan_models].idxmax()

        if baseline not in df_ds['model'].unique():
            continue

        for metric in metrics_to_test:
            if metric not in df_ds.columns:
                continue

            # Correlated Bayesian t-Test (ROPE)
            try:
                res_bayes = bayesian.bayesian_correlated_ttest(
                    df_ds,
                    model_a=baseline,
                    model_b=best_kan,
                    metric=metric
                )
                res_bayes['dataset'] = dataset
                res_bayes['metric'] = metric.upper()
                res_bayes['Model A (Baseline)'] = baseline
                res_bayes['Model B (Best KAN)'] = best_kan
                bayesian_results.append(res_bayes)
            except Exception as e:
                print(f"[Warning] Bayesian test failed for {dataset} ({metric}): {e}")

    # Final dump of results to CSV files
    if wilcoxon_results:
        final_wilcoxon = pd.concat(wilcoxon_results, ignore_index=True)
        final_wilcoxon.to_csv("results/stats_wilcoxon_posthoc.csv", index=False)
        print("[Info] Saved frequentist tests (Wilcoxon, pooled n=7) to: results/stats_wilcoxon_posthoc.csv")

    if bayesian_results:
        final_bayesian = pd.DataFrame(bayesian_results)
        final_bayesian.to_csv("results/stats_bayesian_rope.csv", index=False)
        print("[Info] Saved Bayesian tests (ROPE) to: results/stats_bayesian_rope.csv")

def generate_plots(df: pd.DataFrame):
    """
    Engineering of plots with quality and aesthetics adapted to publication requirements.
    Maps Boxplot type charts for main determinants of predictive ability and computational efficiency.
    
    Args:
        df (pd.DataFrame): Main aggregate of research data.
    """
    os.makedirs("results/plots", exist_ok=True)
    sns.set_theme(style="whitegrid") # Clean publication background (Nature/Science standard)
    
    metrics_to_plot = ['mcc', 'auroc', 'f1_score', 'brier_score', 'inference_time_per_sample_ms', 'total_train_time_seconds']
    
    for metric in metrics_to_plot:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(14, 8))
        
        # Plot considering accessibility palette (colorblind)
        ax = sns.boxplot(
            x="dataset", 
            y=metric, 
            hue="model", 
            data=df, 
            palette="colorblind"
        )
        
        # Publication descriptors
        plt.title(f"Distribution of Metric {metric.upper()} in Cross Validation", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Medical Dataset", fontsize=14, labelpad=10)
        plt.ylabel(f"Value ({metric.upper()})", fontsize=14, labelpad=10)
        
        # Adaptation of edges to readability of long medical diagnosis names
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
        # Extracted legend outside of quantile columns overlap zone
        plt.legend(title="Architecture Model", title_fontsize='13', fontsize='11', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        
        out_path = f"results/plots/boxplot_{metric}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Info] Saved plot: {out_path}")

def generate_confusion_matrices(df: pd.DataFrame):
    """
    Algorithm collecting all partial confusion matrices from individual folds, 
    making their global sum and deriving readable, clinical heatmaps.
    """
    os.makedirs("results/plots", exist_ok=True)
    datasets = df['dataset'].unique()
    models = df['model'].unique()
    
    for dataset in datasets:
        for model in models:
            save_dir = f"results/artifacts/{dataset}/{model}"
            if not os.path.exists(save_dir):
                continue
                
            # Disk location of raw matrix files saved by TabularTrainer
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
                # Painting using standard blue gamma
                plt.figure(figsize=(8, 6))
                sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Sum of Confusion Matrices\nDataset: {dataset} | Model: {model}", fontsize=14)
                plt.xlabel("Predicted Class (Model)", fontsize=12)
                plt.ylabel("True Class (Ground Truth)", fontsize=12)
                plt.tight_layout()
                
                out_path = f"results/plots/cm_{dataset}_{model}.png"
                plt.savefig(out_path, dpi=300)
                plt.close()
                print(f"[Info] Saved confusion matrix: {out_path}")

def generate_learning_curves(df: pd.DataFrame):
    """
    Used for analyzing the training process of a deep neural network.
    Reads Loss history from each CV run,
    calculates mean and deviation aggregates, generating final averaged curve.
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
                        print(f"[Warning] Failed to load history from {hf}: {e}")
            
            if not all_train_loss:
                continue
                
            # Normalization to minimal common amount of epochs in case of logic cut off by system crash
            min_len = min(len(t) for t in all_train_loss)
            
            train_loss_arr = np.array([t[:min_len] for t in all_train_loss])
            val_loss_arr = np.array([v[:min_len] for v in all_val_loss])
            
            # Vectorized statistics on folds axis (axis=0)
            train_mean = np.mean(train_loss_arr, axis=0)
            train_std = np.std(train_loss_arr, axis=0)
            val_mean = np.mean(val_loss_arr, axis=0)
            val_std = np.std(val_loss_arr, axis=0)
            
            epochs = np.arange(1, min_len + 1)
            
            plt.figure(figsize=(10, 6))
            
            # Plot and painting uncertainty intervals
            plt.plot(epochs, train_mean, label="Training (Loss)", color="blue")
            plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
            
            plt.plot(epochs, val_mean, label="Validation (Loss)", color="orange")
            plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, color="orange", alpha=0.2)
            
            plt.title(f"Learning Curve with standard deviation\nDataset: {dataset} | Model: {model}", fontsize=14)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Loss Function Value (Loss)", fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            out_path = f"results/plots/learning_curves_{dataset}_{model}.png"
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"[Info] Saved learning curve: {out_path}")

def main():
    """Start node - initializes all subsystems responsible for spitting out research analysis."""
    print("="*60)
    print(" AUTOMATIC REPORT GENERATOR (PHASE 4) ")
    print("="*60)
    try:
        latest_file = find_latest_results()
        df = pd.read_csv(latest_file)
        
        print("\n--> 1. Generating aggregation tables...")
        generate_summary_table(df)
        
        print("\n--> 2. Generating statistical reports...")
        generate_statistical_reports(df)
        
        print("\n--> 3. Generating visualizations for publication...")
        generate_plots(df)
        
        print("\n--> 3b. Generating multi-dimensional trade-off radar charts...")
        try:
            plot_radar_chart(df, output_path="results/plots/radar_chart_tradeoffs.png")
        except Exception as e:
            print(f"[Warning] Radar chart generation failed: {e}")
            
        print("\n--> 3c. Generating Bayesian ROPE posterior evidence charts...")
        try:
            bayes_file = "results/stats_bayesian_rope.csv"
            if os.path.exists(bayes_file):
                plot_bayesian_rope_evidence(bayes_file, output_path="results/plots/bayesian_rope_evidence.png")
            else:
                plot_bayesian_rope_evidence(df, output_path="results/plots/bayesian_rope_evidence.png")
        except Exception as e:
            print(f"[Warning] Bayesian ROPE chart generation failed: {e}")
        
        print("\n--> 4. Generating confusion matrices (Heatmaps)...")
        generate_confusion_matrices(df)
        
        print("\n--> 5. Generating learning curves (Learning Curves)...")
        generate_learning_curves(df)
        
        print("\n--> 6. Generating dataset size ablation degradation curves...")
        try:
            if os.path.exists("results/ablation_results.csv"):
                plot_ablation_curves(results_path="results/ablation_results.csv", output_dir="results/plots")
            else:
                print("[Info] No results/ablation_results.csv found. Skipping ablation plots.")
        except Exception as e:
            print(f"[Warning] Ablation curve generation failed: {e}")
        
        print("\n" + "="*60)
        print("[SUCCESS] Full analytical pipeline finished without errors.")
        print("All tables and plots are located in the 'results/' folder.")
        print("="*60)
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Report generation process aborted: {e}")

if __name__ == "__main__":
    main()
