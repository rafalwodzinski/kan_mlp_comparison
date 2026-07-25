import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Map files to their target columns
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
    """Simple PyTorch wrapper for preprocessed tabular data (NumPy arrays)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data_and_preprocessor(filepath: str, dataset_filename: str):
    """
    Loads raw CSV file, splits into X and y, and builds an appropriate 
    scikit-learn Pipeline for imputation and scaling without data leakage.
    """
    # 1. Load data
    df = pd.read_csv(filepath)
    target_col = TARGET_COLS.get(dataset_filename)
    
    if target_col not in df.columns:
        raise ValueError(f"Error: Target column '{target_col}' not found in file {dataset_filename}")

    # 2. Special cleaning rules from our EDA analysis
    if dataset_filename == 'cervical_cancer_processed.csv':
        cols_to_drop = ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 3. Split into features (X) and labels (y)
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    # 4. Automatic column type detection
    numeric_candidates = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    numeric_features = []
    categorical_features = []
    
    for col in X.columns:
        # Route to numeric ONLY if it is of numeric type AND has >= 10 unique values
        if col in numeric_candidates and X[col].nunique() >= 10:
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    # 5. Build Pipelines
    # Choose numeric imputer based on dataset specificity
    if dataset_filename in ['chronic_kidney_disease_processed.csv', 'cervical_cancer_processed.csv']:
        # Datasets with difficult missing values - use k-nearest neighbors
        num_imputer = KNNImputer(n_neighbors=5)
    else:
        # Clean datasets or small number of missing values - use outlier-resistant median
        num_imputer = SimpleImputer(strategy='median')

    # Imputation FIRST, then scaling — prevents NaN from contaminating scaler statistics.
    numeric_transformer = Pipeline(steps=[
        ('imputer', num_imputer),
        ('scaler', StandardScaler())
    ])

    # Categorical transformer (always uses mode and One-Hot encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 6. Assemble into one main ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Ignore columns that don't match any type (safety)
    )

    return X, y, preprocessor