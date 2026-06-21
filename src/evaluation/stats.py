import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from typing import Dict, List, Tuple, Any

class FrequentistEvaluator:
    """
    Moduł do rygorystycznej statystyki częstościowej wyników eksperymentów ML.
    Oparty na rekomendacjach J. Demšara (2006) dla porównywania klasyfikatorów.
    """
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha (float): Poziom istotności statystycznej (domyślnie 5%).
        """
        self.alpha = alpha

    def run_friedman_test(self, df: pd.DataFrame, metric: str = 'mcc') -> Dict[str, Any]:
        """
        Przeprowadza nieparametryczny test Friedmana dla wielu modeli na podstawie dataframe'u.
        
        Args:
            df (pd.DataFrame): Ramka danych z kolumnami m.in. 'dataset', 'model', oraz kolumną z metryką.
            metric (str): Nazwa kolumny z metryką (np. 'mcc', 'auroc').
            
        Returns:
            Dict: Wyniki testu statystycznego.
        """
        # Obliczenie średniego wyniku per zbiór danych i per model (uśrednianie foldów przed testem Friedmana)
        # Demšar rekomenduje porównywanie modeli na podstawie wyników ze zbiorów danych.
        agg_df = df.groupby(['dataset', 'model'])[metric].mean().unstack()
        
        # Pobieramy wyniki jako listę tablic dla każdego modelu
        model_scores = [agg_df[model].values for model in agg_df.columns]
        
        stat, p_value = friedmanchisquare(*model_scores)
        
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "conclusion": "Odrzucamy H0 - istnieje statystycznie istotna różnica między modelami" if p_value < self.alpha else "Brak podstaw do odrzucenia H0"
        }

    def run_wilcoxon_post_hoc(self, df: pd.DataFrame, baseline_model: str, competitor_models: List[str], metric: str = 'mcc') -> pd.DataFrame:
        """
        Przeprowadza test Wilcoxona dla par powiązanych (np. porównanie KAN vs MLP) z poprawką Holm-Bonferroni.
        
        Args:
            df (pd.DataFrame): Wyniki eksperymentów.
            baseline_model (str): Nazwa modelu bazowego (np. 'StandardMLP').
            competitor_models (List[str]): Lista modeli do porównania.
            metric (str): Wybrana metryka.
            
        Returns:
            pd.DataFrame: Wyniki testów post-hoc z poprawkami p-value.
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
                
            results.append({
                "Model A (Baseline)": baseline_model,
                "Model B": competitor,
                "Statistic": stat,
                "Unadjusted p-value": p_val,
                "Winner": competitor if np.median(comp_scores) > np.median(baseline_scores) else baseline_model
            })
            
        res_df = pd.DataFrame(results)
        
        if not res_df.empty:
            # Poprawka Holm-Bonferroni
            res_df = res_df.sort_values("Unadjusted p-value").reset_index(drop=True)
            m = len(res_df)
            holm_p = [min(1.0, res_df.loc[i, "Unadjusted p-value"] * (m - i)) for i in range(m)]
            
            # Gwarancja niemalejącej sekwencji
            for i in range(1, m):
                holm_p[i] = max(holm_p[i], holm_p[i-1])
                
            res_df["Holm-Bonferroni p-value"] = holm_p
            res_df["Significant"] = res_df["Holm-Bonferroni p-value"] < self.alpha
            
        return res_df