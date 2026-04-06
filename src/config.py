from pathlib import Path

# --- Ścieżki systemowe ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Upewniamy się, że foldery istnieją
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Metadane Zbiorów Danych ---
# Klucz to nasza wewnętrzna nazwa, uci_id to ID w repozytorium UCI
UCI_DATASETS = {
    "breast_cancer": {
        "uci_id": 17, 
        "target_col": "Diagnosis",
        "task": "binary"
    },
    "heart_disease": {
        "uci_id": 45, 
        "target_col": "num", 
        "task": "multiclass_to_binary" # Zmniejszamy 0-4 do 0 i 1
    },
    "chronic_kidney_disease": {
        "uci_id": 336,
        "target_col": "class",
        "task": "binary"
    },
    "parkinsons": {
        "uci_id": 174,
        "target_col": "status",
        "task": "binary"
    },
    "cervical_cancer": {
        "uci_id": 383,
        "target_col": "Biopsy",
        "task": "binary"
    },
    "cardiotocography": {
        "uci_id": 193,
        "target_col": "NSP",
        "task": "multiclass" # 3 klasy (Normal, Suspect, Pathologic)
    }
}

# Pima Indians Diabetes pobieramy z OpenML (bardziej niezawodne)
OPENML_DATASETS = {
    "pima_diabetes": {
        "openml_name": "diabetes",
        "target_col": "class",
        "task": "binary"
    }
}