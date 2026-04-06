import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, Dict

class BayesianComparator:
    """
    Implementacja Bayesian Correlated t-test do porównywania modeli w CV.
    Pozwala na wyznaczenie prawdopodobieństwa wygranej oraz ROPE.
    """
    def __init__(self, rope_interval: float = 0.01):
        """
        Args:
            rope_interval (float): Szerokość przedziału praktycznej równoważności (np. 0.01 dla 1%).
        """
        self.rope_interval = rope_interval

    def compare_models(self, scores_a: np.ndarray, scores_b: np.ndarray, k_folds: int) -> Dict[str, float]:
        """
        Oblicza prawdopodobieństwa bayesowskie na podstawie różnic wyników w foldach.
        
        Używamy poprawki na korelację: 1/k_folds.
        """
        differences = scores_a - scores_b
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # Korekta Benavoli et al. (2017) dla skorelowanych prób w CV
        # rho = 1 / k_folds (zakładając standardowy podział CV)
        rho = 1 / k_folds
        adjusted_std = std_diff * np.sqrt((1/n) + (rho / (1 - rho)))
        
        # Stopnie swobody
        df = n - 1
        
        # Obliczanie prawdopodobieństw przy użyciu dystrybuanty rozkładu t-Studenta
        # P(Model B > Model A + rope)
        prob_b_wins = stats.t.cdf(-self.rope_interval, df, loc=mean_diff, scale=adjusted_std)
        
        # P(Model A > Model B + rope)
        prob_a_wins = 1 - stats.t.cdf(self.rope_interval, df, loc=mean_diff, scale=adjusted_std)
        
        # P(Różnica mieści się w ROPE)
        prob_rope = 1 - prob_a_wins - prob_b_wins
        
        return {
            "prob_a_better": float(prob_a_wins),
            "prob_b_better": float(prob_b_wins),
            "prob_rope": float(prob_rope),
            "mean_diff": float(mean_diff)
        }

    def plot_posterior(self, scores_a: np.ndarray, scores_b: np.ndarray, k_folds: int, names: Tuple[str, str]):
        """
        Wizualizuje rozkład a posteriori różnicy między modelami.
        """
        diffs = scores_a - scores_b
        n = len(diffs)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        rho = 1 / k_folds
        adjusted_std = std_diff * np.sqrt((1/n) + (rho / (1 - rho)))
        
        x = np.linspace(mean_diff - 4*adjusted_std, mean_diff + 4*adjusted_std, 100)
        y = stats.t.pdf(x, n-1, loc=mean_diff, scale=adjusted_std)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='Posterior Difference')
        plt.axvspan(-self.rope_interval, self.rope_interval, color='gray', alpha=0.2, label='ROPE')
        plt.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        plt.title(f"Bayesian Analysis: {names[0]} vs {names[1]}")
        plt.xlabel(f"Difference in Metric ({names[0]} - {names[1]})")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()