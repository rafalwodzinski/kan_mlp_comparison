import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from itertools import combinations

def run_statistical_analysis(results_path: str = "results/benchmark_results.csv", output_dir: str = "results"):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
        
    df = pd.read_csv(results_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # We want to identify the core metrics.
    # From metrics.py, we have: accuracy, balanced_accuracy, precision, recall, f1_score, mcc, auroc, loss
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "mcc", "auroc", "loss"]
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print("No valid metrics found in the results file.")
        return
        
    models = df['model'].unique()
    if len(models) < 3:
        print("Friedman test requires at least 3 models.")
        return
        
    datasets = df['dataset'].unique()
    print(f"Running statistical tests for {len(models)} models across {len(datasets)} datasets...")

    for metric in available_metrics:
        # Group by dataset and model, taking the mean across Repeated CV runs
        agg_df = df.groupby(['dataset', 'model'])[metric].mean().reset_index()
        
        # Pivot so rows are datasets, columns are models
        pivot_df = agg_df.pivot(index='dataset', columns='model', values=metric)
        
        # Drop rows (datasets) with missing values to ensure fair comparison
        pivot_df = pivot_df.dropna()
        if len(pivot_df) < 3:
            print(f"Skipping {metric} due to insufficient valid datasets ({len(pivot_df)}).")
            continue
            
        # 1. Friedman Test (Omnibus)
        # Convert columns to a list of arrays for scipy
        stat_data = [pivot_df[model].values for model in pivot_df.columns]
        stat, p_value = friedmanchisquare(*stat_data)
        
        with open(os.path.join(output_dir, f"statistical_significance_report_{metric}.txt"), "w") as f:
            f.write(f"=== Statistical Analysis for {metric.upper()} ===\n")
            f.write(f"Omnibus Friedman Test Statistic: {stat:.4f}\n")
            f.write(f"Omnibus Friedman p-value: {p_value:.4e}\n\n")
            
            if p_value >= 0.05:
                msg = "No statistically significant differences found among models overall (p >= 0.05). Post-hoc tests skipped.\n"
                f.write(msg)
                print(f"[{metric.upper()}] Friedman p={p_value:.4f} (Not Significant)")
                continue
                
            msg = "Statistically significant differences found! Proceeding with all-vs-all Wilcoxon post-hoc tests.\n"
            f.write(msg)
            print(f"[{metric.upper()}] Friedman p={p_value:.4e} (Significant)")
            
            # 2. All-vs-All Pairwise Wilcoxon Tests
            pairs = list(combinations(pivot_df.columns, 2))
            raw_p_values = []
            test_results = []
            
            for m1, m2 in pairs:
                # Calculate differences
                diff = pivot_df[m1] - pivot_df[m2]
                if np.all(diff == 0):
                    # Identical predictions, Wilcoxon cannot handle all zeros, p = 1.0
                    w_stat, w_p = np.nan, 1.0
                else:
                    try:
                        w_stat, w_p = wilcoxon(pivot_df[m1], pivot_df[m2])
                    except ValueError: # E.g., zero differences error not caught above
                        w_stat, w_p = np.nan, 1.0
                        
                raw_p_values.append(w_p)
                test_results.append({
                    "Model_1": m1,
                    "Model_2": m2,
                    "Wilcoxon_Stat": w_stat,
                    "Raw_p_value": w_p
                })
                
            # 3. Holm-Bonferroni Correction
            reject, pvals_corrected, _, _ = multipletests(raw_p_values, alpha=0.05, method='holm')
            
            # Compile results
            for i, result in enumerate(test_results):
                result["Adjusted_p_value"] = pvals_corrected[i]
                result["Significant"] = reject[i]
                
            results_df = pd.DataFrame(test_results)
            
            # Save CSV report
            csv_path = os.path.join(output_dir, f"statistical_significance_report_{metric}.csv")
            results_df.to_csv(csv_path, index=False)
            f.write(f"Detailed pairwise comparisons saved to: {csv_path}\n")
            
            # Write a neat table summary to the text file
            f.write("\n--- Pairwise Wilcoxon Results (Holm Corrected) ---\n")
            f.write(results_df.to_string(index=False))
            f.write("\n")

if __name__ == "__main__":
    run_statistical_analysis()
