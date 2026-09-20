import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, studentized_range

def find_latest_results() -> str:
    """Finds the latest benchmark_master csv file."""
    files = glob.glob("results/benchmark_master_*.csv")
    if not files:
        raise FileNotFoundError("Could not find results/benchmark_master_*.csv")
    return max(files, key=os.path.getmtime)

def get_pareto_frontier(costs, utilities):
    """
    Finds the pareto frontier points.
    costs: list or array of cost values (minimize)
    utilities: list or array of utility values (maximize)
    Returns the indices of the pareto optimal points.
    """
    sorted_indices = np.argsort(costs)
    pareto_front_indices = []
    max_utility = -float('inf')
    for idx in sorted_indices:
        if utilities[idx] > max_utility:
            pareto_front_indices.append(idx)
            max_utility = utilities[idx]
    return pareto_front_indices

def plot_pareto_frontier(df: pd.DataFrame, output_dir: str):
    """
    Generates a Pareto Frontier plot (Inference Time vs. AUROC).
    """
    # Aggregate to get mean metrics per model
    agg_df = df.groupby("model").agg(
        mean_auroc=("auroc", "mean"),
        mean_inference_time=("inference_time_per_sample_ms", "mean")
    ).reset_index()

    costs = agg_df["mean_inference_time"].values
    utilities = agg_df["mean_auroc"].values
    
    pareto_idx = get_pareto_frontier(costs, utilities)
    
    # Sort for plotting the line
    pareto_costs = costs[pareto_idx]
    pareto_utilities = utilities[pareto_idx]
    sort_p_idx = np.argsort(pareto_costs)
    pareto_costs = pareto_costs[sort_p_idx]
    pareto_utilities = pareto_utilities[sort_p_idx]

    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    # Categorize models for color coding
    agg_df['is_baseline'] = agg_df['model'].apply(lambda x: "Baseline" if "MLP" in x or "RandomForest" in x else "KAN Variant")
    
    # Scatter plot
    sns.scatterplot(
        data=agg_df, 
        x="mean_inference_time", 
        y="mean_auroc", 
        hue="is_baseline", 
        style="is_baseline",
        s=200, 
        palette={"Baseline": "#1f77b4", "KAN Variant": "#d62728"}
    )

    # Plot the pareto line with markers and higher zorder so it's visible even for a single point
    plt.plot(pareto_costs, pareto_utilities, color='black', linestyle='--', linewidth=2, marker='X', markersize=15, label="Pareto Frontier", zorder=10)

    # Annotate points
    for i, row in agg_df.iterrows():
        plt.annotate(
            row["model"], 
            (row["mean_inference_time"], row["mean_auroc"]),
            textcoords="offset points", 
            xytext=(7,7), 
            ha='left',
            fontsize=11
        )

    plt.xscale('log')
    plt.title("Pareto Frontier: Inference Time vs. AUROC", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Mean Inference Time per Sample (ms) [Log Scale]", fontsize=14, labelpad=10)
    plt.ylabel("Mean AUROC", fontsize=14, labelpad=10)
    plt.legend(title="Model Type", loc="lower right")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "pareto_frontier.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Info] Saved Pareto Frontier plot to: {out_path}")

def compute_nemenyi_cd(k, N, alpha=0.05):
    """
    Computes the Critical Difference for the Nemenyi test.
    k: number of models
    N: number of datasets
    alpha: significance level
    """
    # q is the studentized range statistic critical value
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf)
    cd = (q_alpha / np.sqrt(2)) * np.sqrt((k * (k + 1)) / (6 * N))
    return cd

def plot_cd_diagram(df: pd.DataFrame, output_dir: str):
    """
    Generates a Demšar Critical Difference (CD) diagram.
    """
    # 1. First, aggregate folds to get one score per (dataset, model)
    dataset_model_df = df.groupby(['dataset', 'model'])['auroc'].mean().reset_index()
    
    # 2. Pivot to have datasets as rows, models as columns
    pivot_df = dataset_model_df.pivot(index='dataset', columns='model', values='auroc')
    
    # 3. Rank models per dataset (higher AUROC is better, so rank 1 is highest)
    # rankdata assigns rank 1 to smallest value. So we rank negative values.
    ranks_df = pivot_df.apply(lambda row: pd.Series(rankdata(-row), index=row.index), axis=1)
    
    # 4. Calculate average rank for each model
    avg_ranks = ranks_df.mean().sort_values()
    models = avg_ranks.index.values
    ranks = avg_ranks.values
    
    k = len(models)
    N = len(pivot_df)
    
    cd = compute_nemenyi_cd(k, N)
    
    # 5. Draw the CD Diagram
    # This is a custom matplotlib implementation of the standard CD diagram
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="white", font_scale=1.2)
    ax = plt.gca()
    
    # Plot an axis on top
    ax.spines['top'].set_position(('axes', 0.8))
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_visible(False)
    
    min_rank = 1
    max_rank = k
    ax.set_xlim(min_rank - 0.5, max_rank + 0.5) 
    ax.set_xticks(np.arange(min_rank, max_rank + 1, 1))
    
    # Plot rank points
    ax.scatter(ranks, [0.8] * k, color='black', zorder=5, s=50)
    
    # Draw CD indicator in top right
    ax.plot([max_rank - cd, max_rank], [0.95, 0.95], color='red', linewidth=3)
    ax.text(max_rank - cd/2, 0.98, f"CD = {cd:.2f}", ha='center', va='bottom', color='red', fontweight='bold')
    
    # Draw lines down to labels
    left_models = models[:k//2]
    right_models = models[k//2:]
    
    y_offset_step = 0.05
    y_start = 0.75
    
    # Left models
    for i, model in enumerate(left_models):
        y_pos = y_start - i * y_offset_step
        ax.plot([ranks[i], ranks[i]], [0.8, y_pos], color='black', linewidth=1)
        ax.plot([ranks[i], min_rank - 0.2], [y_pos, y_pos], color='black', linewidth=1)
        ax.text(min_rank - 0.3, y_pos, f"{model} ({ranks[i]:.2f})", ha='right', va='center', fontsize=11)
        
    # Right models
    for i, model in enumerate(right_models):
        idx = k//2 + i
        # For right side, we stagger starting from top again
        y_pos = y_start - i * y_offset_step
        ax.plot([ranks[idx], ranks[idx]], [0.8, y_pos], color='black', linewidth=1)
        ax.plot([ranks[idx], max_rank + 0.2], [y_pos, y_pos], color='black', linewidth=1)
        ax.text(max_rank + 0.3, y_pos, f"{model} ({ranks[idx]:.2f})", ha='left', va='center', fontsize=11)
        
    # Find and draw cliques (groups of models not significantly different)
    cliques = []
    for i in range(k):
        for j in range(i + 1, k):
            if ranks[j] - ranks[i] <= cd:
                continue
            else:
                if j - 1 > i:
                    cliques.append((i, j - 1))
                break
        else:
            if k - 1 > i:
                cliques.append((i, k - 1))
                
    # Filter non-maximal cliques
    maximal_cliques = []
    for c in cliques:
        is_maximal = True
        for other in cliques:
            if c != other and c[0] >= other[0] and c[1] <= other[1]:
                is_maximal = False
                break
        if is_maximal:
            maximal_cliques.append(c)
            
    # Draw cliques
    clique_y = 0.85
    for c in maximal_cliques:
        ax.plot([ranks[c[0]], ranks[c[1]]], [clique_y, clique_y], color='black', linewidth=4, alpha=0.7)
        clique_y += 0.02
        
    plt.title("Critical Difference (CD) Diagram (Average Rank across datasets)", fontsize=16, fontweight='bold', pad=40)
    
    out_path = os.path.join(output_dir, "critical_difference_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Info] Saved Critical Difference Diagram to: {out_path}")

def main():
    os.makedirs("results/plots", exist_ok=True)
    try:
        latest_file = find_latest_results()
        print(f"[Info] Loaded latest benchmark file: {latest_file}")
        df = pd.read_csv(latest_file)
        
        print("[Info] Generating Pareto Frontier plot...")
        plot_pareto_frontier(df, "results/plots")
        
        print("[Info] Generating Critical Difference diagram...")
        plot_cd_diagram(df, "results/plots")
        
        print("\n[SUCCESS] Advanced plots generated successfully.")
    except Exception as e:
        print(f"[Error] Failed to generate advanced plots: {e}")

if __name__ == "__main__":
    main()
