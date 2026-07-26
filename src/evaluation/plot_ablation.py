import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    sns.set_theme(style="whitegrid")
    
    for dataset in datasets:
        ds_df = df[df['dataset'] == dataset]
        
        # Plot for AUROC, MCC, F1 Score, and Brier Score
        for metric in ['auroc', 'mcc', 'f1_score', 'brier_score']:
            if metric not in ds_df.columns:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Use lineplot which automatically handles repeated CV folds (mean and standard deviation shading)
            sns.lineplot(
                data=ds_df, 
                x='train_fraction', 
                y=metric, 
                hue='model', 
                marker='o',
                errorbar='sd',
                linewidth=2
            )
            
            plt.title(f"Data Scarcity Degradation: {dataset} ({metric.upper()})")
            plt.xlabel("Training Data Fraction")
            plt.ylabel(metric.upper())
            plt.xlim(0.0, 1.05)
            # Invert X axis to show degradation from 1.0 down to 0.1
            plt.gca().invert_xaxis()
            
            plt.legend(title='Model Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            plot_path = os.path.join(output_dir, f"ablation_{dataset}_{metric}.png")
            plt.savefig(plot_path)
            plt.close()
            
            print(f"[Ablation Plot] Saved plot: {plot_path}")

if __name__ == "__main__":
    plot_ablation_curves()
