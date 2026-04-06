import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Optional

class MedicalTabularDataset(Dataset):
    """
    Standardowy Dataset PyTorch do tabelarycznych danych medycznych.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) # Zakładamy klasyfikację (long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def prepare_dataloaders(
    df: pd.DataFrame, 
    target_col: str, 
    numerical_cols: List[str], 
    categorical_cols: Optional[List[str]] = None,
    test_size: float = 0.2, 
    val_size: float = 0.1,
    batch_size: int = 32,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Dzieli dane, przetwarza je (zapobiegając wyciekowi danych) i tworzy DataLoadery.
    Zwraca również input_dim potrzebny do zainicjowania modelu.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # 1. Stratified Split (Trening vs Reszta)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=random_state
    )
    
    # Podział Reszty na Walidację i Test
    val_ratio_in_temp = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio_in_temp), stratify=y_temp, random_state=random_state
    )

    # 2. Definicja rurociągów transformacji (Pipelines)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Odporne na wartości odstające w medycynie
        ('scaler', StandardScaler())
    ])
    
    # Jeśli mamy zmienne kategoryczne, dodajemy je tutaj (np. OneHotEncoder)
    transformers = [('num', num_pipeline, numerical_cols)]
    if categorical_cols:
        from sklearn.preprocessing import OneHotEncoder
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # 3. FIT tylko na treningowym! Transform na pozostałych.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Wymiar wejściowy dla sieci neuronowych
    input_dim = X_train_processed.shape[1]

    # 4. Tworzenie obiektów Dataset
    train_dataset = MedicalTabularDataset(X_train_processed, y_train)
    val_dataset = MedicalTabularDataset(X_val_processed, y_val)
    test_dataset = MedicalTabularDataset(X_test_processed, y_test)

    # 5. Tworzenie DataLoaderów
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim