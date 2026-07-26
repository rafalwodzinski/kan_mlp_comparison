import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from typing import Dict, List, Tuple, Any

class FrequentistEvaluator:
    """
    Module for rigorous frequentist statistics of ML experiment results.
    Based on J. Demšar's (2006) recommendations for comparing classifiers.
    """
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha (float): Statistical significance level (default 5%).
        """
        self.alpha = alpha

    def run_friedman_test(self, df: pd.DataFrame, metric: str = 'mcc') -> Dict[str, Any]:
        """
        Performs non-parametric Friedman test for multiple models based on a dataframe.
        
        Args:
            df (pd.DataFrame): Dataframe with columns including 'dataset', 'model', and metric column.
            metric (str): Name of metric column (e.g., 'mcc', 'auroc').
            
        Returns:
            Dict: Statistical test results.
        """
        # Calculate mean score per dataset and per model (averaging folds before Friedman test)
        # Demšar recommends comparing models based on dataset results.
        agg_df = df.groupby(['dataset', 'model'])[metric].mean().unstack()
        
        # Extract scores as a list of arrays for each model
        model_scores = [agg_df[model].values for model in agg_df.columns]
        
        stat, p_value = friedmanchisquare(*model_scores)
        
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "conclusion": "Reject H0 - there is a statistically significant difference between models" if p_value < self.alpha else "No grounds to reject H0"
        }

    def run_wilcoxon_post_hoc(self, df: pd.DataFrame, baseline_model: str, competitor_models: List[str], metric: str = 'mcc') -> pd.DataFrame:
        """
        Performs Wilcoxon signed-rank test for paired samples (e.g. KAN vs MLP comparison) with Holm-Bonferroni correction.
        
        Args:
            df (pd.DataFrame): Experiment results.
            baseline_model (str): Name of baseline model (e.g. 'StandardMLP').
            competitor_models (List[str]): List of models to compare.
            metric (str): Selected metric.
            
        Returns:
            pd.DataFrame: Post-hoc test results with p-value corrections.
        """
        agg_df = df.groupby(['dataset', 'model'])[metric].mean().unstack()
        baseline_scores = agg_df[baseline_model].values
        
        results = []
        for competitor in competitor_models:
            if competitor not in agg_df.columns:
                continue
                
            comp_scores = agg_df[competitor].values
            differences = comp_scores - baseline_scores
            
            if np.all(differences == 0):
                stat, p_val = 0.0, 1.0
            else:
                stat, p_val = wilcoxon(baseline_scores, comp_scores, zero_method='zsplit')
                
            is_higher_better = metric.lower() not in ['brier_score', 'loss', 'inference_time_ms', 'inference_time_per_sample_ms', 'total_train_time_seconds', 'trainable_parameters']
            if is_higher_better:
                winner = competitor if np.median(comp_scores) > np.median(baseline_scores) else baseline_model
            else:
                winner = competitor if np.median(comp_scores) < np.median(baseline_scores) else baseline_model

            results.append({
                "Model A (Baseline)": baseline_model,
                "Model B": competitor,
                "Statistic": stat,
                "Unadjusted p-value": p_val,
                "Winner": winner
            })
            
        res_df = pd.DataFrame(results)
        
        if not res_df.empty:
            # Holm-Bonferroni correction
            res_df = res_df.sort_values("Unadjusted p-value").reset_index(drop=True)
            m = len(res_df)
            holm_p = [min(1.0, res_df.loc[i, "Unadjusted p-value"] * (m - i)) for i in range(m)]
            
            # Guarantee non-decreasing sequence
            for i in range(1, m):
                holm_p[i] = max(holm_p[i], holm_p[i-1])
                
            res_df["Holm-Bonferroni p-value"] = holm_p
            res_df["Significant"] = res_df["Holm-Bonferroni p-value"] < self.alpha
            
        return res_df