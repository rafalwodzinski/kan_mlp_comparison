import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from typing import Dict, List, Tuple

class StatisticalAnalyzer:
    """
    Moduł do rygorystycznej oceny statystycznej wyników eksperymentów ML.
    Oparty na rekomendacjach J. Demšara (2006) dla porównywania klasyfikatorów.
    """
    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha (float): Poziom istotności statystycznej (domyślnie 5%).
        """
        self.alpha = alpha

    def run_friedman_test(self, results_df: pd.DataFrame, metric: str) -> Dict[str, float]:
        """
        Przeprowadza nieparametryczny test Friedmana dla wielu modeli.
        Sprawdza hipotezę zerową: "Wszystkie modele radzą sobie tak samo dobrze".
        
        Oczekiwany format results_df:
        Indeksy: Zbiory danych (np. Breast Cancer, Diabetes)
        Kolumny: Modele (np. StandardMLP, TabKAN)
        Wartości: Wynik metryki (np. AUROC)
        """
        # Pobieramy wyniki jako listę tablic dla każdego modelu
        model_scores = [results_df[model].values for model in results_df.columns]
        
        stat, p_value = friedmanchisquare(*model_scores)
        
        return {
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "conclusion": "Odrzucamy H0 - istnieje różnica między modelami" if p_value < self.alpha else "Brak podstaw do odrzucenia H0"
        }

    def run_wilcoxon_pairwise(self, model_a_scores: np.ndarray, model_b_scores: np.ndarray) -> Dict[str, float]:
        """
        Przeprowadza test Wilcoxona dla par powiązanych (np. porównanie KAN vs MLP na N zbiorach danych).
        """
        # Różnice między wynikami (dla zbadania równości)
        differences = model_a_scores - model_b_scores
        
        # Zabezpieczenie na wypadek, gdyby wyniki były identyczne (co psuje test)
        if np.all(differences == 0):
            return {"statistic": 0.0, "p_value": 1.0, "significant": False, "winner": "Tie"}
            
        stat, p_value = wilcoxon(model_a_scores, model_b_scores, zero_method='zsplit')
        
        winner = "Model A" if np.median(model_a_scores) > np.median(model_b_scores) else "Model B"
        
        return {
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "winner": winner if p_value < self.alpha else "Tie"
        }

    def generate_pairwise_matrix(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generuje macierz p-value dla wszystkich możliwych par modeli.
        Przydatne do wygenerowania tabeli do artykułu naukowego.
        """
        models = results_df.columns
        n_models = len(models)
        p_matrix = pd.DataFrame(np.ones((n_models, n_models)), index=models, columns=models)
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_a = models[i]
                model_b = models[j]
                
                res = self.run_wilcoxon_pairwise(results_df[model_a].values, results_df[model_b].values)
                
                p_matrix.loc[model_a, model_b] = res["p_value"]
                p_matrix.loc[model_b, model_a] = res["p_value"] # Macierz symetryczna
                
        return p_matrix