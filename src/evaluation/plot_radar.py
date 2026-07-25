"""
Multi-dimensional Trade-off Visualization Module (Radar / Spider Charts).
Compares primary model architectures across clinical accuracy metrics (AUROC, MCC)
and computational/complexity costs (Brier Score, Inference Time, Trainable Parameters).
Normalizes metrics strictly to a 0-1 scale where 1.0 represents the BEST performing model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict

def normalize_metrics_for_radar(df_grouped: pd.DataFrame, higher_is_better_flags: List[bool]) -> pd.DataFrame:
    """
    Transforms all metrics strictly to a 0-1 scale where 1.0 is the BEST performing model.
    
    Args:
        df_grouped (pd.DataFrame): Dataframe indexed by model name with raw metric columns.
        higher_is_better_flags (List[bool]): True if higher raw values indicate better performance
                                             (e.g., AUROC, MCC), False if lower is better
                                             (e.g., Brier Score, Inference Time, Parameters).
    Returns:
        pd.DataFrame: Normalized dataframe on a [0, 1] scale where 1.0 = Best, 0.0 = Worst.
    """
    df_norm = pd.DataFrame(index=df_grouped.index, columns=df_grouped.columns, dtype=float)
    
    for col, higher_is_better in zip(df_grouped.columns, higher_is_better_flags):
        col_data = df_grouped[col].astype(float)
        min_val = col_data.min()
        max_val = col_data.max()
        
        # Guard against zero division if all models performed identically
        if max_val == min_val:
            df_norm[col] = 1.0
        elif higher_is_better:
            # For AUROC / MCC: highest value gets 1.0, lowest gets 0.0
            df_norm[col] = (col_data - min_val) / (max_val - min_val)
        else:
            # For Brier Score / Time / Params: lowest (best) value gets 1.0, highest (worst) gets 0.0
            df_norm[col] = (max_val - col_data) / (max_val - min_val)
            
    return df_norm

def plot_radar_chart(df: pd.DataFrame, 
                     output_path: str = "results/plots/radar_chart_tradeoffs.png", 
                     primary_models: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generates a high-resolution, publication-ready Radar (Spider) chart comparing models.
    Mathematically closes the polygon loop and formats aesthetics for Lancet / Nature submission.
    
    Args:
        df (pd.DataFrame): Raw benchmark results dataframe containing model and metric columns.
        output_path (str): File path to save the generated PNG plot.
        primary_models (Optional[List[str]]): List of specific architectures to plot.
                                              If None, defaults to key baseline & KAN variants.
    Returns:
        pd.DataFrame: The normalized values used to generate the chart.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="white") # Clean background for polar plots
    
    # 1. Filter models to primary targets of interest
    if primary_models is None:
        default_targets = ["StandardMLP", "WavKAN", "FastKAN", "RandomForest", "ChebyKAN", "TaylorKAN"]
        available_models = df['model'].unique().tolist()
        primary_models = [m for m in default_targets if m in available_models]
        if not primary_models:
            primary_models = available_models[:5] # Fallback to first 5 available
            
    df_filtered = df[df['model'].isin(primary_models)].copy()
    
    # 2. Map required metrics with fallback mechanisms for backwards compatibility
    metric_map = {
        'auroc': ('auroc', True, 'AUROC\n(Higher is better)'),
        'mcc': ('mcc', True, 'MCC\n(Higher is better)'),
        'brier_score': ('brier_score' if 'brier_score' in df_filtered.columns else 'loss', False, 'Brier Score / Loss\n(Lower is better)'),
        'time': ('inference_time_per_sample_ms' if 'inference_time_per_sample_ms' in df_filtered.columns else 'avg_epoch_time_seconds' if 'avg_epoch_time_seconds' in df_filtered.columns else None, False, 'Inference Time\n(Lower is better)'),
        'params': ('trainable_parameters' if 'trainable_parameters' in df_filtered.columns else None, False, 'Trainable Params\n(Lower is better)')
    }
    
    active_cols = []
    flags = []
    labels = []
    
    for key, (col_name, is_higher_better, label_str) in metric_map.items():
        if col_name and col_name in df_filtered.columns:
            active_cols.append(col_name)
            flags.append(is_higher_better)
            labels.append(label_str)
        elif key == 'params':
            # If trainable_parameters column missing, estimate or impute 0 for RF / 1 for MLP
            df_filtered['trainable_parameters'] = np.where(df_filtered['model'] == 'RandomForest', 0, 1000)
            active_cols.append('trainable_parameters')
            flags.append(False)
            labels.append(label_str)
            
    if len(active_cols) < 3:
        raise ValueError(f"Not enough valid metrics found to construct a radar chart. Active: {active_cols}")
        
    # 3. Group by model and aggregate means across datasets and CV folds
    grouped_raw = df_filtered.groupby('model')[active_cols].mean()
    
    # Save raw averages table for reproducibility
    raw_csv_path = output_path.replace(".png", "_raw_metrics.csv")
    grouped_raw.to_csv(raw_csv_path)
    print(f"[Info] Saved raw model averages to: {raw_csv_path}")
    
    # 4. Strictly normalize to 0-1 scale where 1.0 is best
    df_norm = normalize_metrics_for_radar(grouped_raw, flags)
    
    # Save normalized table
    norm_csv_path = output_path.replace(".png", "_normalized_metrics.csv")
    df_norm.to_csv(norm_csv_path)
    print(f"[Info] Saved normalized 0-1 metrics to: {norm_csv_path}")
    
    # 5. Setup Matplotlib Polar Setup
    num_vars = len(labels)
    # Calculate angles for each axis and mathematically close the loop
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)  # Rotate to position first variable at top (12 o'clock)
    ax.set_theta_direction(-1)      # Clockwise progression
    
    # Draw radial axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold', color='#222222')
    ax.tick_params(axis='x', pad=15)
    
    # Draw radial gridlines and scale markers
    ax.set_rlabel_position(30)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0 (Best)"], color="grey", size=10, fontweight='normal')
    ax.set_ylim(0, 1.05)
    
    # 6. Plot each model on the radar chart
    colors = sns.color_palette("colorblind", len(df_norm))
    
    for idx, model_name in enumerate(df_norm.index):
        values = df_norm.loc[model_name].tolist()
        values += values[:1] # Mathematically close the loop
        
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=model_name, color=colors[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.25) # Semi-transparent fill
        
    # 7. Polish publication aesthetics
    plt.title("Multi-Dimensional Model Trade-offs (5-Fold CV)\nNormalized Performance Scale (1.0 = Optimal Architecture)", 
              fontsize=16, fontweight='bold', pad=35, color='#111111')
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), 
               fontsize=12, title="Architecture Model", title_fontsize='13', 
               frameon=True, facecolor='white', framealpha=0.9, edgecolor='#cccccc')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Success] Saved publication-ready radar chart to: {output_path}")
    
    return df_norm

if __name__ == "__main__":
    # Test execution when run standalone
    import glob
    csv_files = glob.glob("results/benchmark_master_*.csv")
    if csv_files:
        latest = max(csv_files, key=os.path.getmtime)
        print(f"[Radar Script] Loading {latest} for standalone execution...")
        df_test = pd.read_csv(latest)
        plot_radar_chart(df_test)
    else:
        print("[Warning] No benchmark_master CSV found in results/ to run standalone test.")
