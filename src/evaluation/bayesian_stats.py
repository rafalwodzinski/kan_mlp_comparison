import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict

class BayesianEvaluator:
    """
    Implementation of Bayesian Correlated t-test for rigorous model comparison
    on cross-validation (k-Fold CV or Repeated k-Fold CV) results.
    Uses Benavoli et al. (2017) correction protecting against artificial confidence inflation.
    
    For Repeated Stratified K-Fold CV (r × k):
    - Each repeat produces k paired observations (one per fold).
    - Total paired observations: n = r × k.
    - The correlation correction rho = 1/k (test set fraction) remains unchanged
      per Corani & Benavoli (2015), as it captures the train/test overlap *within*
      each fold regardless of repeat count.
    """
    def __init__(self, rope_interval: float = 0.01, k_folds: int = 5, n_repeats: int = 3):
        """
        Args:
            rope_interval (float): Width of Region of Practical Equivalence (ROPE).
                                   E.g. 0.01 is 1% difference.
            k_folds (int): Number of folds in cross-validation.
            n_repeats (int): Number of CV repetitions (1 for standard k-Fold, >1 for Repeated CV).
        """
        self.rope_interval = rope_interval
        self.k_folds = k_folds
        self.n_repeats = n_repeats

    def bayesian_correlated_ttest(self, df: pd.DataFrame, model_a: str, model_b: str, metric: str = 'mcc') -> Dict[str, float]:
        """
        Performs Bayesian Correlated t-test using FOLD-level results.
        
        Args:
            df (pd.DataFrame): Dataframe with results (required: dataset, model, fold, <metric>).
                               If 'repeat' column exists, it is used for proper pairing.
            model_a (str): First model.
            model_b (str): Second model.
            metric (str): Name of the analyzed metric.
            
        Returns:
            Dict: Probabilities of scenarios A>B, B>A and Tie (ROPE).
        """
        # Determine sort columns based on available data
        sort_cols = ['dataset']
        if 'repeat' in df.columns:
            sort_cols.append('repeat')
        sort_cols.append('fold')
        
        # We filter data for both models and ensure equal number of observations
        df_a = df[df['model'] == model_a].sort_values(sort_cols).reset_index(drop=True)
        df_b = df[df['model'] == model_b].sort_values(sort_cols).reset_index(drop=True)
        
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
        is_higher_better = metric.lower() not in ['brier_score', 'loss', 'inference_time_ms', 'inference_time_per_sample_ms', 'total_train_time_seconds', 'trainable_parameters']
        if std_diff == 0:
            if mean_diff > self.rope_interval:
                return {"prob_A_better": 1.0 if is_higher_better else 0.0, "prob_B_better": 0.0 if is_higher_better else 1.0, "prob_ROPE": 0.0, "mean_diff": mean_diff}
            elif mean_diff < -self.rope_interval:
                return {"prob_A_better": 0.0 if is_higher_better else 1.0, "prob_B_better": 1.0 if is_higher_better else 0.0, "prob_ROPE": 0.0, "mean_diff": mean_diff}
            else:
                return {"prob_A_better": 0.0, "prob_B_better": 0.0, "prob_ROPE": 1.0, "mean_diff": mean_diff}
        
        # Benavoli et al. (2017) / Corani & Benavoli (2015) correction.
        # rho = n_test / (n_train + n_test) = 1/k for k-fold CV.
        # This holds for both standard and Repeated CV — the correlation
        # stems from the train/test overlap fraction, not the repeat count.
        rho = 1 / self.k_folds
        
        # Corrected standard error accounting for correlated observations
        adjusted_std = std_diff * np.sqrt((1/n) + (rho / (1 - rho)))
        
        # Degrees of freedom
        df_t = n - 1
        
        # We integrate the Student's t-distribution density in appropriate intervals
        # P(Difference falls within ROPE): P(-rope < diff < rope)
        prob_rope = stats.t.cdf(self.rope_interval, df_t, loc=mean_diff, scale=adjusted_std) - \
                    stats.t.cdf(-self.rope_interval, df_t, loc=mean_diff, scale=adjusted_std)
        
        if is_higher_better:
            # P(Model A > Model B + rope): P(diff > rope)
            prob_a_wins = 1 - stats.t.cdf(self.rope_interval, df_t, loc=mean_diff, scale=adjusted_std)
            # P(Model B > Model A + rope): P(diff < -rope)
            prob_b_wins = stats.t.cdf(-self.rope_interval, df_t, loc=mean_diff, scale=adjusted_std)
        else:
            # For lower-is-better, A winning means diff < -rope
            prob_a_wins = stats.t.cdf(-self.rope_interval, df_t, loc=mean_diff, scale=adjusted_std)
            prob_b_wins = 1 - stats.t.cdf(self.rope_interval, df_t, loc=mean_diff, scale=adjusted_std)
        
        return {
            "prob_A_better": float(prob_a_wins),
            "prob_B_better": float(prob_b_wins),
            "prob_ROPE": float(prob_rope),
            "mean_diff": float(mean_diff)
        }