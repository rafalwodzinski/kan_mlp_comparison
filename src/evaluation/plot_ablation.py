import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.plot_style import apply_publication_style, save_pdf, global_ylim

def plot_ablation_curves(results_path: str = "results/ablation_results.csv", output_dir: str = "results/plots"):
    if not os.path.exists(results_path):
        print(f"[Ablation Plot] Error: {results_path} not found.")
        return
        
    df = pd.read_csv(results_path)
    
    # Ensure train_fraction is present
    if 'train_fraction' not in df.columns:
        print("[Ablation Plot] Error: 'train_fraction' column missing from results.")
        return
        
    datasets = df['dataset'].unique()
    os.makedirs(output_dir, exist_ok=True)
    
    apply_publication_style("whitegrid")
    
    # Plot for AUPRC, AUROC, MCC, F1 Score, and Brier Score
    for metric in ['auprc', 'auroc', 'mcc', 'f1_score', 'brier_score']:
        if metric not in df.columns:
            continue
        
        # Unified Y scale: global min/max of the metric across ALL datasets of the experiment
        # (mean +/- std envelope, matching what sns.lineplot draws)
        agg = df.groupby(['dataset', 'model', 'train_fraction'])[metric].agg(['mean', 'std']).fillna(0.0)
        y_min, y_max = global_ylim((agg['mean'] - agg['std']).values, (agg['mean'] + agg['std']).values)
        
        for dataset in datasets:
            ds_df = df[df['dataset'] == dataset]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use lineplot which automatically handles repeated CV folds (mean and standard deviation shading)
            sns.lineplot(
                data=ds_df, 
                x='train_fraction', 
                y=metric, 
                hue='model', 
                marker='o',
                errorbar='sd',
                linewidth=2,
                ax=ax
            )
            
            ax.set_title(f"Data Scarcity Degradation: {dataset} ({metric.upper()})")
            ax.set_xlabel("Training Data Fraction")
            ax.set_ylabel(metric.upper())
            ax.set_xlim(0.0, 1.05)
            ax.set_ylim(y_min, y_max)
            # Invert X axis to show degradation from 1.0 down to 0.1
            ax.invert_xaxis()
            
            ax.legend(title='Model Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
            
            plot_path = save_pdf(os.path.join(output_dir, f"ablation_{dataset}_{metric}.pdf"), fig)
            plt.close(fig)
            print(f"[Ablation Plot] Saved plot: {plot_path}")

            manuscript_dir = "manuscript/figures"
            if os.path.exists(manuscript_dir) and os.path.abspath(output_dir) != os.path.abspath(manuscript_dir):
                dest_pdf = os.path.join(manuscript_dir, f"ablation_{dataset}_{metric}.pdf")
                shutil.copy2(plot_path, dest_pdf)

if __name__ == "__main__":
    plot_ablation_curves()
