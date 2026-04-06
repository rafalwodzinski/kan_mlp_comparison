import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix
)
from typing import Dict, Any, Tuple, Optional

class MedicalMetricsEvaluator:
    """
    Klasa odpowiedzialna za rygorystyczne obliczanie metryk 
    dla medycznych modeli tabelarycznych.
    Obsługuje zarówno zadania binarne, jak i wieloklasowe.
    """
    def __init__(self, is_binary: bool = True):
        """
        Args:
            is_binary (bool): Flaga określająca typ problemu. 
                              Zmienia sposób agregacji (np. macro dla multiclass).
        """
        self.is_binary = is_binary
        self.average_method = 'binary' if is_binary else 'macro'

    def calculate_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Oblicza pełen zestaw metryk klasyfikacyjnych.
        
        Args:
            y_true (np.ndarray): Prawdziwe etykiety klas (1D).
            y_prob (np.ndarray): Prawdopodobieństwa klas (1D dla binarnej, 2D dla multiclass).
            
        Returns:
            Dict[str, float]: Słownik z wynikami poszczególnych metryk.
        """
        # Konwersja prawdopodobieństw na twarde predykcje (hard labels)
        if self.is_binary:
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_pred = np.argmax(y_prob, axis=1)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=self.average_method, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=self.average_method, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=self.average_method, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred)
        }

        # Obliczanie AUROC wymaga specjalnego potraktowania dla multiclass
        try:
            if self.is_binary:
                metrics["auroc"] = roc_auc_score(y_true, y_prob)
            else:
                metrics["auroc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            # Zabezpieczenie na wypadek, gdyby w batchu/foldzie była tylko jedna klasa
            metrics["auroc"] = np.nan

        return metrics

    def get_confusion_matrix(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        """
        Generuje macierz pomyłek do analizy przypadków brzegowych.
        
        Args:
            y_true (np.ndarray): Prawdziwe etykiety.
            y_prob (np.ndarray): Prawdopodobieństwa.
            
        Returns:
            np.ndarray: Macierz pomyłek w formacie numpy.
        """
        if self.is_binary:
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_pred = np.argmax(y_prob, axis=1)
            
        return confusion_matrix(y_true, y_pred)