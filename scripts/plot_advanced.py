import os
import glob
import pandas as pd
import numpy as np
import sys
import shutil
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import rankdata, studentized_range

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation.plot_style import apply_publication_style, save_pdf

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

def plot_pareto_frontier(df: pd.DataFrame, output_dir: str, metric: str = "auprc"):
    """
    Generates a Pareto Frontier plot (Inference Time vs. AUPRC).
    """
    metric_label = "AUPRC" if metric == "auprc" else metric.upper()
    # Aggregate to get mean metrics per model
    agg_df = df.groupby("model").agg(
        mean_perf=(metric, "mean"),
        mean_inference_time=("inference_time_per_sample_ms", "mean")
    ).reset_index()

    costs = agg_df["mean_inference_time"].values
    utilities = agg_df["mean_perf"].values
    
    pareto_idx = get_pareto_frontier(costs, utilities)
    
    # Sort for plotting the line
    pareto_costs = costs[pareto_idx]
    pareto_utilities = utilities[pareto_idx]
    sort_p_idx = np.argsort(pareto_costs)
    pareto_costs = pareto_costs[sort_p_idx]
    pareto_utilities = pareto_utilities[sort_p_idx]

    # Categorize models
    agg_df['is_pareto'] = False
    agg_df.loc[pareto_idx, 'is_pareto'] = True
    agg_df['is_baseline'] = agg_df['model'].apply(lambda x: "Baseline" if "MLP" in x or "RandomForest" in x else "KAN Variant")

    apply_publication_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    palette = {"Baseline": "#1f77b4", "KAN Variant": "#d62728"}

    # Dominated models: standard dots
    dominated_df = agg_df[~agg_df['is_pareto']]
    sns.scatterplot(
        data=dominated_df,
        x="mean_inference_time",
        y="mean_perf",
        hue="is_baseline",
        palette=palette,
        marker='o',
        s=140,
        alpha=0.85,
        ax=ax,
        legend=False,
        zorder=5
    )

    # Models on Pareto frontier: distinct, highly visible bold cross 'X'
    pareto_df = agg_df[agg_df['is_pareto']]
    sns.scatterplot(
        data=pareto_df,
        x="mean_inference_time",
        y="mean_perf",
        hue="is_baseline",
        palette=palette,
        marker='X',
        s=300,
        edgecolor='black',
        linewidth=1.5,
        ax=ax,
        legend=False,
        zorder=12
    )

    # Plot the pareto line with matching 'X' markers and higher zorder
    ax.plot(
        pareto_costs, 
        pareto_utilities, 
        color='black', 
        linestyle='--', 
        linewidth=2.5, 
        marker='X', 
        markersize=13, 
        label="Pareto Frontier", 
        zorder=10
    )

    # Annotate points with offsets designed to avoid label collisions
    custom_offsets = {
        "TabKAN": (10, 5),
        "FastKAN": (10, -2),
        "ReLUKAN": (10, -12),
        "StandardMLP": (10, 5),
        "TaylorKAN": (10, 5),
        "WavKAN": (10, -12),
        "ChebyKAN": (10, -5),
        "GramKAN": (10, 5),
        "LegendreKAN": (10, -14),
        "JacobiKAN": (10, 5),
        "RandomForest": (10, 4)
    }

    for _, row in agg_df.iterrows():
        is_pareto = row["is_pareto"]
        fontweight = 'bold' if is_pareto else 'normal'
        offset = custom_offsets.get(row["model"], (8, 6))
        ax.annotate(
            row["model"], 
            (row["mean_inference_time"], row["mean_perf"]),
            textcoords="offset points", 
            xytext=offset, 
            ha='left',
            fontweight=fontweight,
            fontsize=12 if is_pareto else 11
        )

    ax.set_xscale('log')
    ax.set_title(f"Pareto Frontier: Inference Time vs. {metric_label}", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Mean Inference Time per Sample (ms) [Log Scale]", fontsize=15, labelpad=10)
    ax.set_ylabel(f"Mean {metric_label}", fontsize=15, labelpad=10)

    # Custom legend in the upper right corner
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Baseline (Dominated)', markerfacecolor=palette["Baseline"], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='KAN Variant (Dominated)', markerfacecolor=palette["KAN Variant"], markersize=10),
        Line2D([0], [0], marker='X', color='black', linestyle='--', linewidth=2.5, label='Pareto Frontier (Optimal)', markerfacecolor='black', markersize=12)
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, framealpha=0.92, fontsize=12, title="Model Type", title_fontsize=13)
    fig.tight_layout()

    out_path = save_pdf(os.path.join(output_dir, "pareto_frontier.pdf"), fig)
    plt.close(fig)
    print(f"[Info] Saved Pareto Frontier plot to: {out_path}")

    # Synchronize directly to manuscript/figures/ if available
    manuscript_dir = "manuscript/figures"
    if os.path.exists(manuscript_dir) and os.path.abspath(output_dir) != os.path.abspath(manuscript_dir):
        dest_pdf = os.path.join(manuscript_dir, "pareto_frontier.pdf")
        shutil.copy2(out_path, dest_pdf)
        print(f"[Info] Synchronized Pareto Frontier plot to: {dest_pdf}")

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

def compute_average_ranks(df: pd.DataFrame, metric: str, higher_is_better: bool = True):
    """Returns (pivot_df dataset x model, ranks_df dataset x model) with rank 1 = best."""
    dataset_model_df = df.groupby(['dataset', 'model'])[metric].mean().reset_index()
    pivot_df = dataset_model_df.pivot(index='dataset', columns='model', values=metric)
    sign = -1 if higher_is_better else 1
    ranks_df = pivot_df.apply(lambda row: pd.Series(rankdata(sign * row), index=row.index), axis=1)
    return pivot_df, ranks_df

def plot_rank_heatmap(df: pd.DataFrame, output_dir: str, metric: str = 'mcc', higher_is_better: bool = True):
    """
    NEW: Heatmap of per-dataset model ranks (rank 1 = best) with the average rank in the last row.
    Complements the CD diagram by showing where each average rank comes from.
    """
    _, ranks_df = compute_average_ranks(df, metric, higher_is_better)
    order = ranks_df.mean().sort_values().index
    ranks_df = ranks_df[order]
    ranks_df.loc['AVERAGE'] = ranks_df.mean()

    apply_publication_style("white")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(ranks_df, annot=True, fmt='.1f', cmap='viridis_r', cbar_kws={'label': f'Rank ({metric.upper()}, 1 = best)'}, ax=ax)
    ax.set_xlabel("Model")
    ax.set_ylabel("Dataset")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    out_path = save_pdf(os.path.join(output_dir, f"rank_heatmap_{metric}.pdf"), fig)
    plt.close(fig)
    print(f"[Info] Saved rank heatmap to: {out_path}")

def plot_cd_diagram(df: pd.DataFrame, output_dir: str, metric: str = 'auroc', higher_is_better: bool = True):
    """
    Generates a Demšar Critical Difference (CD) diagram for the given metric.
    """
    # 1-3. Aggregate folds per (dataset, model), pivot and rank per dataset (rank 1 = best)
    pivot_df, ranks_df = compute_average_ranks(df, metric, higher_is_better)
    
    # 4. Calculate average rank for each model
    avg_ranks = ranks_df.mean().sort_values()
    models = avg_ranks.index.values
    ranks = avg_ranks.values
    
    k = len(models)
    N = len(pivot_df)
    
    cd = compute_nemenyi_cd(k, N)
    
    # 5. Draw the CD Diagram
    # This is a custom matplotlib implementation of the standard CD diagram
    apply_publication_style("white")
    plt.figure(figsize=(12, 6))
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
        
    plt.title(f"Critical Difference (CD) Diagram - {metric.upper()} (Nemenyi, alpha=0.05)", fontweight='bold', pad=40)
    
    suffix = "" if metric == 'auroc' else f"_{metric}"
    out_path = save_pdf(os.path.join(output_dir, f"critical_difference_diagram{suffix}.pdf"))
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
        
        print("[Info] Generating Critical Difference diagrams (AUROC, MCC, AUPRC)...")
        plot_cd_diagram(df, "results/plots", metric='auroc')
        for m in ('mcc', 'auprc'):
            if m in df.columns and df[m].notna().any():
                plot_cd_diagram(df, "results/plots", metric=m)
        
        print("[Info] Generating rank heatmaps...")
        for m in ('mcc', 'auroc', 'auprc'):
            if m in df.columns and df[m].notna().any():
                plot_rank_heatmap(df, "results/plots", metric=m)
        
        print("\n[SUCCESS] Advanced plots generated successfully.")
    except Exception as e:
        print(f"[Error] Failed to generate advanced plots: {e}")

if __name__ == "__main__":
    main()
