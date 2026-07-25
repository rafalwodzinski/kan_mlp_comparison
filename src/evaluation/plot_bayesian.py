"""
Bayesian Posterior Evidence Visualization Module (ROPE Stacked Bar Charts).
Plots the posterior probability distributions comparing standard MLP against KAN variants
across all medical datasets: P(MLP > KAN), P(Practical Equivalence inside ROPE), and P(KAN > MLP).
Provides immediate visual validation of the Null Hypothesis (practical equivalence).
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from typing import Union, Optional

def plot_bayesian_rope_evidence(df_or_path: Union[str, pd.DataFrame], 
                                output_path: str = "results/plots/bayesian_rope_evidence.png", 
                                rope_interval: float = 0.01,
                                metric_name: str = "MCC") -> pd.DataFrame:
    """
    Generates a horizontal stacked bar chart illustrating Bayesian ROPE posterior probabilities.
    
    Args:
        df_or_path (Union[str, pd.DataFrame]): Path to stats_bayesian_rope.csv OR a raw benchmark dataframe.
        output_path (str): Path where the publication-ready PNG chart will be saved.
        rope_interval (float): The Region of Practical Equivalence half-width used in the evaluation.
        metric_name (str): Label of the clinical metric evaluated (e.g., 'MCC' or 'AUROC').
    Returns:
        pd.DataFrame: The processed Bayesian probability table used for plotting.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Resolve data source: either precomputed bayesian table or raw benchmark results
    if isinstance(df_or_path, str):
        if not os.path.exists(df_or_path):
            raise FileNotFoundError(f"Bayesian results file not found at: {df_or_path}")
        df_bayes = pd.read_csv(df_or_path)
    elif isinstance(df_or_path, pd.DataFrame):
        df_bayes = df_or_path.copy()
        # If passed raw benchmark results, compute Bayesian probabilities on the fly
        if 'prob_ROPE' not in df_bayes.columns:
            from src.evaluation.bayesian_stats import BayesianEvaluator
            evaluator = BayesianEvaluator(rope_interval=rope_interval)
            results = []
            
            for ds in df_bayes['dataset'].unique():
                df_ds = df_bayes[df_bayes['dataset'] == ds]
                kan_models = [m for m in df_ds['model'].unique() if "KAN" in m]
                if not kan_models or "StandardMLP" not in df_ds['model'].unique():
                    continue
                    
                # Pick the top-performing KAN variant in this dataset as competitor
                mean_scores = df_ds.groupby('model')[metric_name.lower() if metric_name.lower() in df_ds.columns else 'mcc'].mean()
                best_kan = mean_scores[kan_models].idxmax()
                
                try:
                    res = evaluator.bayesian_correlated_ttest(
                        df_ds, 
                        model_a="StandardMLP", 
                        model_b=best_kan, 
                        metric=metric_name.lower() if metric_name.lower() in df_ds.columns else 'mcc'
                    )
                    res['dataset'] = ds
                    res['Model A (Baseline)'] = "StandardMLP"
                    res['Model B (Best KAN)'] = best_kan
                    results.append(res)
                except Exception as e:
                    print(f"[Warning] On-the-fly Bayesian evaluation failed for {ds}: {e}")
            df_bayes = pd.DataFrame(results)
    else:
        raise TypeError("df_or_path must be a filepath string or a pandas DataFrame.")
        
    if df_bayes.empty or 'prob_ROPE' not in df_bayes.columns:
        raise ValueError("No valid Bayesian posterior probability columns ('prob_ROPE') found.")
        
    # 2. Extract probabilities and format percentages
    datasets = df_bayes['dataset'].values
    y_pos = np.arange(len(datasets))
    
    p_a = df_bayes['prob_A_better'].values * 100.0  # P(MLP better)
    p_rope = df_bayes['prob_ROPE'].values * 100.0   # P(Equivalence)
    p_b = df_bayes['prob_B_better'].values * 100.0  # P(KAN better)
    
    # Extract model names for legend
    model_a_name = df_bayes['Model A (Baseline)'].iloc[0] if 'Model A (Baseline)' in df_bayes.columns else "StandardMLP"
    model_b_label = "Best KAN Variant"
    if 'Model B (Best KAN)' in df_bayes.columns and df_bayes['Model B (Best KAN)'].nunique() == 1:
        model_b_label = df_bayes['Model B (Best KAN)'].iloc[0]
        
    # 3. Build horizontal stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Clinical / Nature palette: Blue for Baseline, Green/Teal for ROPE (Equivalence), Orange for Competitor
    color_a = '#377eb8'     # Classic Blue
    color_rope = '#4daf4a'  # Evidence Green (Null Hypothesis confirmed)
    color_b = '#ff7f00'     # Vibrant Orange
    
    bar_a = ax.barh(y_pos, p_a, height=0.6, color=color_a, 
                    label=f"{model_a_name} Superior (p > ROPE)", edgecolor='white', linewidth=1.2)
    bar_rope = ax.barh(y_pos, p_rope, left=p_a, height=0.6, color=color_rope, 
                       label=f"Practical Equivalence (within ±{rope_interval*100:.0f}% ROPE)", edgecolor='white', linewidth=1.2)
    bar_b = ax.barh(y_pos, p_b, left=p_a + p_rope, height=0.6, color=color_b, 
                    label=f"{model_b_label} Superior (p < -ROPE)", edgecolor='white', linewidth=1.2)
    
    # 4. Annotate internal segment percentages if wide enough (>= 7%)
    for i in range(len(datasets)):
        # Model A segment text
        if p_a[i] >= 7.0:
            ax.text(p_a[i] / 2.0, y_pos[i], f"{p_a[i]:.1f}%", 
                    va='center', ha='center', color='white', fontweight='bold', fontsize=10.5)
        # ROPE segment text
        if p_rope[i] >= 7.0:
            ax.text(p_a[i] + (p_rope[i] / 2.0), y_pos[i], f"{p_rope[i]:.1f}%", 
                    va='center', ha='center', color='white', fontweight='bold', fontsize=10.5)
        # Model B segment text
        if p_b[i] >= 7.0:
            ax.text(p_a[i] + p_rope[i] + (p_b[i] / 2.0), y_pos[i], f"{p_b[i]:.1f}%", 
                    va='center', ha='center', color='white', fontweight='bold', fontsize=10.5)
            
    # 5. Format axes and presentation aesthetics
    ax.set_yticks(y_pos)
    ax.set_yticklabels(datasets, fontsize=12, fontweight='bold', color='#222222')
    ax.set_xlabel("Posterior Probability Distribution (%)", fontsize=14, fontweight='bold', labelpad=12)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Ensure dataset order is displayed from top to bottom
    ax.invert_yaxis()
    
    plt.title(f"Bayesian Posterior Evidence: {model_a_name} vs. KAN Architectures\n"
              f"Evaluating Clinical Equivalence on {metric_name} (5-Fold Cross-Validation)", 
              fontsize=16, fontweight='bold', pad=25, color='#111111')
    
    # Legend centered below chart
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=3, fontsize=11.5, frameon=True, facecolor='white', framealpha=0.9, edgecolor='#cccccc')
              
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.grid(axis='y', visible=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Success] Saved Bayesian ROPE posterior chart to: {output_path}")
    
    return df_bayes

if __name__ == "__main__":
    # Standalone verification test
    rope_file = "results/stats_bayesian_rope.csv"
    if os.path.exists(rope_file):
        print(f"[Bayesian Script] Loading {rope_file} for standalone execution...")
        plot_bayesian_rope_evidence(rope_file)
    else:
        print(f"[Warning] {rope_file} not found. Attempting to generate from latest benchmark results...")
        csv_files = glob.glob("results/benchmark_master_*.csv")
        if csv_files:
            latest = max(csv_files, key=os.path.getmtime)
            df_test = pd.read_csv(latest)
            plot_bayesian_rope_evidence(df_test)
        else:
            print("[Error] No benchmark results found to test.")
