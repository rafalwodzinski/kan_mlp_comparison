import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict

class BayesianEvaluator:
    """
    Implementacja Bayesian Correlated t-test do rygorystycznego porównywania modeli
    na wynikach z walidacji krzyżowej (k-Fold CV).
    Korzysta z poprawki Benavoli et al. (2017) chroniącej przed sztucznym zawyżaniem pewności.
    """
    def __init__(self, rope_interval: float = 0.01, k_folds: int = 5):
        """
        Args:
            rope_interval (float): Szerokość przedziału praktycznej równoważności (ROPE).
                                   Np. 0.01 to 1% różnicy.
            k_folds (int): Liczba foldów w cross-walidacji.
        """
        self.rope_interval = rope_interval
        self.k_folds = k_folds

    def bayesian_correlated_ttest(self, df: pd.DataFrame, model_a: str, model_b: str, metric: str = 'mcc') -> Dict[str, float]:
        """
        Wykonuje Bayesian Correlated t-test używając wyników na poziomie FOLDÓW.
        
        Args:
            df (pd.DataFrame): Dataframe z wynikami (wymagane: dataset, model, fold, <metric>).
            model_a (str): Pierwszy model.
            model_b (str): Drugi model.
            metric (str): Nazwa analizowanej metryki.
            
        Returns:
            Dict: Prawdopodobieństwa scenariuszy A>B, B>A oraz Remis (ROPE).
        """
        # Filtrujemy dane dla obu modeli i upewniamy się, że są równe ilości obserwacji
        df_a = df[df['model'] == model_a].sort_values(['dataset', 'fold'])
        df_b = df[df['model'] == model_b].sort_values(['dataset', 'fold'])
        
        if len(df_a) == 0 or len(df_b) == 0:
            raise ValueError("Brak wyników dla podanych modeli.")
            
        scores_a = df_a[metric].values
        scores_b = df_b[metric].values
        
        if len(scores_a) != len(scores_b):
            raise ValueError("Różna liczba wyników dla modeli (niekompletne cross-walidacje?).")
            
        differences = scores_a - scores_b
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # W przypadku identycznych wyników (odchylenie standardowe = 0)
        if std_diff == 0:
            if mean_diff > self.rope_interval:
                return {"prob_A_better": 1.0, "prob_B_better": 0.0, "prob_ROPE": 0.0, "mean_diff": mean_diff}
            elif mean_diff < -self.rope_interval:
                return {"prob_A_better": 0.0, "prob_B_better": 1.0, "prob_ROPE": 0.0, "mean_diff": mean_diff}
            else:
                return {"prob_A_better": 0.0, "prob_B_better": 0.0, "prob_ROPE": 1.0, "mean_diff": mean_diff}
        
        # Korekta Benavoli et al. (2017) dla skorelowanych prób w k-Fold CV.
        # Niezależność prób jest naruszona przez powielanie zbioru treningowego w CV.
        rho = 1 / self.k_folds
        
        # Nowe odchylenie standardowe uwzględniające korelację
        adjusted_std = std_diff * np.sqrt((1/n) + (rho / (1 - rho)))
        
        # Stopnie swobody
        df_t = n - 1
        
        # Całkujemy gęstość rozkładu t-Studenta w odpowiednich przedziałach
        # P(Różnica mieści się w ROPE): P(-rope < diff < rope)
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