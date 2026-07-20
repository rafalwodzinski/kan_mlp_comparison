import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict

class BayesianEvaluator:
    """
    Implementation of Bayesian Correlated t-test for rigorous model comparison
    on cross-validation (k-Fold CV) results.
    Uses Benavoli et al. (2017) correction protecting against artificial confidence inflation.
    """
    def __init__(self, rope_interval: float = 0.01, k_folds: int = 5):
        """
        Args:
            rope_interval (float): Width of Region of Practical Equivalence (ROPE).
                                   E.g. 0.01 is 1% difference.
            k_folds (int): Number of folds in cross-validation.
        """
        self.rope_interval = rope_interval
        self.k_folds = k_folds

    def bayesian_correlated_ttest(self, df: pd.DataFrame, model_a: str, model_b: str, metric: str = 'mcc') -> Dict[str, float]:
        """
        Performs Bayesian Correlated t-test using FOLD-level results.
        
        Args:
            df (pd.DataFrame): Dataframe with results (required: dataset, model, fold, <metric>).
            model_a (str): First model.
            model_b (str): Second model.
            metric (str): Name of the analyzed metric.
            
        Returns:
            Dict: Probabilities of scenarios A>B, B>A and Tie (ROPE).
        """
        # We filter data for both models and ensure equal number of observations
        df_a = df[df['model'] == model_a].sort_values(['dataset', 'fold'])
        df_b = df[df['model'] == model_b].sort_values(['dataset', 'fold'])
        
        if len(df_a) == 0 or len(df_b) == 0:
            raise ValueError("No results for provided models.")
            
        scores_a = df_a[metric].values
        scores_b = df_b[metric].values
        
        if len(scores_a) != len(scores_b):
            raise ValueError("Different number of results for models (incomplete cross-validations?).")
            
        differences = scores_a - scores_b
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # In case of identical results (standard deviation = 0)
        if std_diff == 0:
            if mean_diff > self.rope_interval:
                return {"prob_A_better": 1.0, "prob_B_better": 0.0, "prob_ROPE": 0.0, "mean_diff": mean_diff}
            elif mean_diff < -self.rope_interval:
                return {"prob_A_better": 0.0, "prob_B_better": 1.0, "prob_ROPE": 0.0, "mean_diff": mean_diff}
            else:
                return {"prob_A_better": 0.0, "prob_B_better": 0.0, "prob_ROPE": 1.0, "mean_diff": mean_diff}
        
        # Benavoli et al. (2017) correction for correlated samples in k-Fold CV.
        # Independence of samples is violated by repeating the training set in CV.
        rho = 1 / self.k_folds
        
        # New standard deviation accounting for correlation
        adjusted_std = std_diff * np.sqrt((1/n) + (rho / (1 - rho)))
        
        # Degrees of freedom
        df_t = n - 1
        
        # We integrate the Student's t-distribution density in appropriate intervals
        # P(Difference falls within ROPE): P(-rope < diff < rope)
        prob_rope = stats.t.cdf(self.rope_interval, df_t, loc=mean_diff, scale=adjusted_std) - \
                    stats.t.cdf(-self.rope_interval, df_t, loc=mean_diff, scale=adjusted_std)
        
        # P(Model A > Model B + rope): P(diff > rope)
        prob_a_wins = 1 - stats.t.cdf(self.rope_interval, df_t, loc=mean_diff, scale=adjusted_std)
        
        # P(Model B > Model A + rope): P(diff < -rope)
        prob_b_wins = stats.t.cdf(-self.rope_interval, df_t, loc=mean_diff, scale=adjusted_std)
        
        return {
            "prob_A_better": float(prob_a_wins),
            "prob_B_better": float(prob_b_wins),
            "prob_ROPE": float(prob_rope),
            "mean_diff": float(mean_diff)
        }