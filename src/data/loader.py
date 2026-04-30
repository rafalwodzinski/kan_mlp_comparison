import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Mapowanie plików na ich kolumny docelowe (target)
TARGET_COLS = {
    'breast_cancer_processed.csv': 'Diagnosis',
    'pima_diabetes_processed.csv': 'class',
    'heart_disease_processed.csv': 'num',
    'chronic_kidney_disease_processed.csv': 'class',
    'parkinsons_processed.csv': 'status',
    'cervical_cancer_processed.csv': 'Biopsy',
    'cardiotocography_processed.csv': 'NSP'
}

class MedicalTabularDataset(Dataset):
    """Prosty wrapper PyTorch dla przetworzonych już danych tabelarycznych (NumPy arrays)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data_and_preprocessor(filepath: str, dataset_filename: str):
    """
    Wczytuje surowy plik CSV, dzieli na X i y oraz buduje odpowiedni 
    Pipeline scikit-learn do imputacji i skalowania bez wycieku danych.
    """
    # 1. Wczytanie danych
    df = pd.read_csv(filepath)
    target_col = TARGET_COLS.get(dataset_filename)
    
    if target_col not in df.columns:
        raise ValueError(f"Błąd: Nie znaleziono kolumny targetu '{target_col}' w pliku {dataset_filename}")

    # 2. Specjalne reguły czyszczenia z naszej analizy EDA
    if dataset_filename == 'cervical_cancer_processed.csv':
        cols_to_drop = ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 3. Podział na cechy (X) i etykiety (y)
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    # 4. Automatyczna detekcja typów kolumn
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 5. Budowa rurociągów (Pipelines)
    # Wybór imputera numerycznego na podstawie specyfiki zbioru
    if dataset_filename in ['chronic_kidney_disease_processed.csv', 'cervical_cancer_processed.csv']:
        # Zbiory z trudnymi brakami - używamy algorytmu najbliższych sąsiadów
        num_imputer = KNNImputer(n_neighbors=5)
    else:
        # Zbiory czyste lub z małymi brakami - używamy odpornej na outliery mediany
        num_imputer = SimpleImputer(strategy='median')

    numeric_transformer = Pipeline(steps=[
        ('imputer', num_imputer),
        ('scaler', StandardScaler())
    ])

    # Transformator kategoryczny (zawsze używa mody i kodowania One-Hot)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 6. Złożenie w jeden główny ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Ignoruje kolumny, które nie pasują do żadnego typu (zabezpieczenie)
    )

    return X, y, preprocessors