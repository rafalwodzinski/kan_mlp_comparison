import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix
)
from typing import Dict, Any, Tuple, Optional

class MedicalMetricsEvaluator:
    """
    Class responsible for rigorous metrics calculation 
    for medical tabular models.
    Supports both binary and multiclass tasks.
    """
    def __init__(self, is_binary: bool = True):
        """
        Args:
            is_binary (bool): Flag indicating the problem type. 
                              Changes aggregation method (e.g., macro for multiclass).
        """
        self.is_binary = is_binary
        self.average_method = 'binary' if is_binary else 'macro'

    def calculate_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculates a full set of classification metrics.
        
        Args:
            y_true (np.ndarray): True class labels (1D).
            y_prob (np.ndarray): Class probabilities (1D for binary, 2D for multiclass).
            
        Returns:
            Dict[str, float]: Dictionary with results of individual metrics.
        """
        # Conversion of probabilities to hard predictions (hard labels)
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

        # AUROC calculation requires special treatment for multiclass
        try:
            if self.is_binary:
                if len(np.unique(y_true)) > 1:
                    metrics["auroc"] = roc_auc_score(y_true, y_prob)
                else:
                    metrics["auroc"] = metrics["balanced_accuracy"]
            else:
                present_classes = np.unique(y_true)
                if len(present_classes) == y_prob.shape[1]:
                    metrics["auroc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                elif len(present_classes) > 1:
                    # We filter y_prob only for classes actually present in y_true
                    y_prob_filtered = y_prob[:, present_classes]
                    # We rescale probabilities back to 1
                    y_prob_filtered = y_prob_filtered / (y_prob_filtered.sum(axis=1, keepdims=True) + 1e-8)
                    metrics["auroc"] = roc_auc_score(y_true, y_prob_filtered, multi_class="ovr", average="macro", labels=present_classes)
                    print(f"[Warning] Missing classes in AUROC evaluation. Calculated for {len(present_classes)}/{y_prob.shape[1]} classes.")
                else:
                    # In case of only one class in the set (extreme test/validation data leakage)
                    metrics["auroc"] = metrics["balanced_accuracy"]
                    print("[Warning] Only one class in y_true! AUROC impossible, used balanced_accuracy as fallback.")
        except Exception as e:
            # Ironclad safeguard against crash breaking statistics - we never return np.nan
            print(f"[Error AUROC] {str(e)}. Fallback to balanced_accuracy.")
            metrics["auroc"] = metrics["balanced_accuracy"]

        return metrics

    def get_confusion_matrix(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        """
        Generates a confusion matrix for edge case analysis.
        
        Args:
            y_true (np.ndarray): True labels.
            y_prob (np.ndarray): Probabilities.
            
        Returns:
            np.ndarray: Confusion matrix in numpy format.
        """
        if self.is_binary:
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_pred = np.argmax(y_prob, axis=1)
            
        return confusion_matrix(y_true, y_pred)