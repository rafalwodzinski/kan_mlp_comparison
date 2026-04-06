import pandas as pd
import sys
import os
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml

# Importujemy konfigurację
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def process_and_save(df: pd.DataFrame, dataset_name: str, meta: dict):
    """Zapisuje surowe dane, czyści etykiety i zapisuje dane gotowe do treningu."""
    # Zapis Raw
    raw_path = config.RAW_DATA_DIR / f"{dataset_name}_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"[{dataset_name}] Zapisano surowe dane.")
    
    # Preprocessing
    df = df.dropna(subset=[meta['target_col']])
    target = df[meta['target_col']]
    
    # Standaryzacja etykiet zależnie od zbioru
    if dataset_name == "heart_disease":
        df[meta['target_col']] = (target > 0).astype(int)
    elif dataset_name == "breast_cancer":
        df[meta['target_col']] = target.map({'M': 1, 'B': 0})
    elif dataset_name == "chronic_kidney_disease":
        # W zbiorze CKD wartości czasem mają spacje w stringu
        df[meta['target_col']] = target.astype(str).str.strip().map({'ckd': 1, 'notckd': 0})
    elif dataset_name == "pima_diabetes":
        df[meta['target_col']] = target.map({'tested_positive': 1, 'tested_negative': 0})
    elif dataset_name == "cardiotocography":
        # PyTorch wymaga klas indeksowanych od 0 (więc 1,2,3 -> 0,1,2)
        df[meta['target_col']] = target.astype(int) - 1

    # Zapis Processed
    processed_path = config.PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"[{dataset_name}] Zapisano gotowe dane: {processed_path}\n")


def download_datasets():
    # 1. Pobieranie z UCI
    for dataset_name, meta in config.UCI_DATASETS.items():
        print(f"[{dataset_name}] Pobieranie z UCI (ID: {meta['uci_id']})...")
        try:
            dataset = fetch_ucirepo(id=meta['uci_id'])
            X = dataset.data.features
            y = dataset.data.targets
            df = pd.concat([X, y], axis=1)
            process_and_save(df, dataset_name, meta)
        except Exception as e:
            print(f"[BŁĄD] Nie udało się pobrać zbioru {dataset_name}: {str(e)}\n")

    # 2. Pobieranie Pima Indians z OpenML
    for dataset_name, meta in config.OPENML_DATASETS.items():
        print(f"[{dataset_name}] Pobieranie z OpenML (Name: {meta['openml_name']})...")
        try:
            # as_frame=True ładuje to od razu jako Pandas DataFrame
            dataset = fetch_openml(name=meta['openml_name'], version=1, as_frame=True, parser='auto')
            df = dataset.frame
            process_and_save(df, dataset_name, meta)
        except Exception as e:
            print(f"[BŁĄD] Nie udało się pobrać zbioru {dataset_name}: {str(e)}\n")

if __name__ == "__main__":
    download_datasets()
    print("Zakończono proces pozyskiwania pełnej listy 7 medycznych zbiorów danych.")