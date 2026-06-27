"""
Moduł odpowiedzialny za cykl życia treningu modelu (Trainer).
Klasa `TabularTrainer` służy do hermetycznego trenowania i ewaluowania pojedynczej 
instancji modelu (w obrębie jednego foldu walidacji krzyżowej).
Izoluje operacje niskopoziomowe PyTorcha od logiki eksperymentalnej.
Wersja Lite: wypisuje logi prosto na konsolę (zamiast np. do MLFlow) i zrzuca 
wynikowe artefakty na dysk lokalny.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any
import sys
import os

# Dodanie ścieżki do importu z innych folderów src (katalog główny projektu)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation.metrics import MedicalMetricsEvaluator

class TabularTrainer:
    """
    Modularna klasa treningowa dla sieci neuronowych (zarówno KAN jak i klasycznych MLP).
    Odpowiada za pętlę forward/backward pass, śledzenie strat, optymalizację wag
    oraz generację ostatecznych metryk ewaluacyjnych.
    """
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module, 
        device: torch.device,
        is_binary: bool = True,
        experiment_name: str = "KAN_vs_MLP_Benchmark"
    ):
        """
        Inicjalizacja środowiska treningowego.
        
        Argumenty:
            model (nn.Module): Sieć neuronowa (np. WavKAN, StandardMLP).
            optimizer (torch.optim.Optimizer): Algorytm optymalizacji (np. AdamW).
            criterion (nn.Module): Funkcja straty (np. BCEWithLogitsLoss dla binarnej).
            device (torch.device): Środowisko obliczeniowe (CPU/CUDA).
            is_binary (bool): Flaga określająca problem klasyfikacji (binarna vs wieloklasowa).
            experiment_name (str): Nazwa identyfikacyjna wpływająca na nazwę plików wynikowych.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.is_binary = is_binary
        self.experiment_name = experiment_name
        
        # Inicjalizacja dedykowanego ewaluatora pod medyczne metryki (MCC, AUROC, itd.)
        self.evaluator = MedicalMetricsEvaluator(is_binary=self.is_binary)
        self.last_confusion_matrix = None

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Przeprowadza pojedynczą epokę uczenia sieci (przejście przez wszystkie batche).
        
        Argumenty:
            dataloader (DataLoader): PyTorchowy loader próbek treningowych.
            
        Zwraca:
            float: Średnia wartość funkcji straty dla danej epoki.
        """
        self.model.train() # Przełączenie modelu w tryb treningu (np. aktywacja Dropout)
        total_loss = 0.0
        
        for X_batch, y_batch in dataloader:
            # Transfer tensorów do VRAM (jeśli dostępne GPU)
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # Resetowanie akumulatora gradientów
            self.optimizer.zero_grad()
            
            # Forward pass (predykcja surowych logitów)
            logits = self.model(X_batch)
            
            if self.is_binary:
                # BEZPIECZNE ŚCISKANIE: dim=-1 chroni przed cichym błędem broadcasting'u w PyTorch,
                # kiedy dojdzie do przetworzenia batcha wielkości 1 na końcu zbioru
                loss = self.criterion(logits.squeeze(dim=-1), y_batch.float()) 
            else:
                loss = self.criterion(logits, y_batch)
                
            # Backward pass (propagacja błędu) i aktualizacja wag
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Walidacja bez gradientów (Inference mode). Oblicza logity, transformuje je
        w prawdopodobieństwa i generuje zestaw twardych metryk ewaluacyjnych.
        
        Argumenty:
            dataloader (DataLoader): PyTorchowy loader zbioru walidacyjnego/testowego.
            
        Zwraca:
            Dict[str, Any]: Słownik z wynikami metryk (MCC, AUROC, F1, Loss, itd.).
        """
        self.model.eval() # Wyłączenie elementów stochastycznych typu Dropout/LayerNorm
        total_loss = 0.0
        
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                
                if self.is_binary:
                    logits_squeezed = logits.squeeze(dim=-1)
                    loss = self.criterion(logits_squeezed, y_batch.float())
                    # Transformacja logitów na prawdopodobieństwa (zakres 0-1) dla klasyfikacji binarnej
                    probs = torch.sigmoid(logits_squeezed)
                else:
                    loss = self.criterion(logits, y_batch)
                    # Odtworzenie struktury prawdopodobieństw dla klasyfikacji wieloklasowej
                    probs = torch.softmax(logits, dim=1)
                
                total_loss += loss.item()
                
                # Zrzut wyników do buforów CPU do celów analizy metryk Scikit-Learn
                all_preds.append(probs.cpu().numpy())
                all_trues.append(y_batch.cpu().numpy())
                
        # Konkatenacja z poszczególnych batchy
        y_prob_all = np.concatenate(all_preds, axis=0)
        y_true_all = np.concatenate(all_trues, axis=0)
        
        # Obliczenie wskaźników klinicznych
        metrics = self.evaluator.calculate_metrics(y_true_all, y_prob_all)
        metrics["loss"] = total_loss / len(dataloader)
        
        # Zapis macierzy pomyłek do globalnego stanu klas (dostępna po zakończeniu funkcji fit)
        self.last_confusion_matrix = self.evaluator.get_confusion_matrix(y_true_all, y_prob_all)
                
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, run_params: dict):
        """
        Główna pętla treningowa koordynująca uczenie (train_epoch) i weryfikację (evaluate).
        
        Argumenty:
            train_loader (DataLoader): Zbiór treningowy.
            val_loader (DataLoader): Zbiór walidacyjny do oceny generalizacji.
            epochs (int): Maksymalna liczba iteracji uczących po całym zbiorze.
            run_params (dict): Parametry konfiguracyjne do logowania.
        """
        print(f"\n[{self.experiment_name}] Rozpoczęcie treningu...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            # Monitoring konsolowy na żywo pozwalający zdiagnozować np. przedwczesne przeuczenie
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val MCC: {val_metrics['mcc']:.4f} | "
                  f"Val AUROC: {val_metrics['auroc']:.4f}")
        
        # Trwały zrzut "twardych dowodów" eksperymentalnych (MLOps practices)
        weights_path = f"{self.experiment_name}_weights.pth"
        cm_path = f"{self.experiment_name}_confusion_matrix.csv"
        
        # Zapis wyuczonych parametrów na krawędziach (KAN) lub wag w połączeniach (MLP)
        torch.save(self.model.state_dict(), weights_path)
        
        # Zapis zliczeń diagnostycznych pacjentów
        if self.last_confusion_matrix is not None:
            np.savetxt(cm_path, self.last_confusion_matrix, delimiter=",", fmt='%d')
            
        print(f"[{self.experiment_name}] Trening zakończony. Zapisano wagi do {weights_path}")