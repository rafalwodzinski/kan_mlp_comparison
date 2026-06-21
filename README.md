# KAN vs MLP: Benchmark dla medycznych danych tabelarycznych

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## Cel projektu

Ten projekt przeprowadza benchmark, aby odpowiedzieć na palące pytanie we współczesnym uczeniu głębokim (Deep Learning): **Czy nowo wprowadzone Sieci Kołmogorowa-Arnolda (KAN) i ich warianty przewyższają klasyczne Wielowarstwowe Perceptrony (MLP) na medycznych zbiorach danych tabelarycznych?**

Medyczne dane tabelaryczne są niezwykle trudne do modelowania ze względu na ich heterogeniczność, niezbalansowanie klas oraz brakujące wartości. W tym projekcie wprowadzono rygorystyczne standardy MLOps, aby zagwarantować, że ewaluacja jest naukowo ważna. Kładziemy ogromny nacisk na zapobieganie wyciekowi danych (Data Leakage). Nasze strategie imputacji (KNN/Mediana) oraz skalowania są hermetycznie odizolowane – uczą się wyłącznie na foldach treningowych podczas walidacji krzyżowej (cross-validation), gwarantując tym samym, że nasze wyniki odzwierciedlają prawdziwą zdolność modeli do generalizacji, a nie błędy w metodologii.

## Architektura i struktura projektu

Repozytorium zostało zorganizowane zgodnie z najlepszymi praktykami MLOps:

```text
kan_mlp_comparison/
│
├── data/
│   ├── raw/               # Surowe medyczne zbiory danych
│   └── processed/         # Zbiory danych przetworzone i gotowe do benchmarku
│
├── scripts/
│   ├── automate_benchmark.py  # Główny potok (pipeline) uruchamiający testy dla wszystkich modeli i zbiorów
│   └── generate_report.py     # Automatyczny skrypt generujący tabele analityczne i wykresy do publikacji
│
├── src/
│   ├── data/
│   │   └── loader.py      # Bezpieczny potok preprocessingu (StandardScaler -> KNNImputer) chroniący przed wyciekiem danych
│   ├── models/
│   │   ├── base.py        # Bazowy interfejs dla modeli tabelarycznych
│   │   ├── mlp.py         # Architektura StandardMLP
│   │   └── kan_variants/  # 9 zaawansowanych architektur KAN (TabKAN, FastKAN, ChebyKAN, itd.)
│   ├── training/
│   │   ├── trainer.py     # TabularTrainer zarządzający pętlą uczenia, funkcją straty i zapisem artefaktów
│   │   └── cross_validation.py # Silnik Stratified 5-Fold CV zapewniający hermetyczny podział danych
│   └── evaluation/
│       ├── metrics.py     # Kuloodporne wyliczanie metryk medycznych (MCC, AUROC, F1)
│       ├── stats.py       # Statystyka częstościowa (Test Friedmana, Test Wilcoxona Post-Hoc)
│       └── bayesian_stats.py # Bayesowski skorelowany test t-Studenta z analizą ROPE dla CV
│
└── results/
    ├── artifacts/         # Zapisane wagi modeli, macierze pomyłek i historie uczenia (JSON) na fold
    └── plots/             # Automatycznie wygenerowane wykresy skrzynkowe (Boxplots), Heatmapy oraz Krzywe Uczenia
```

## Modele i zbiory danych

### Testowane architektury (10 Modeli)
1. **StandardMLP** (Baza porównawcza)
2. **TabKAN** (Dedykowany KAN zoptymalizowany pod cechy tabelaryczne i bramkowanie sygnału)
3. **FastKAN**
4. **ChebyKAN**
5. **JacobiKAN**
6. **LegendreKAN**
7. **GramKAN**
8. **TaylorKAN**
9. **WavKAN**
10. **ReLUKAN**

### Medyczne zbiory danych (7 Zbiorów)
- Rak Piersi (Breast Cancer)
- Choroba Parkinsona
- Kardiotokografia (CTG)
- Cukrzyca (Diabetes)
- Przewlekła Choroba Nerek (Chronic Kidney Disease)
- *(...oraz inne przetworzone zbiory znajdujące się w `data/processed/`)*

## Przygotowanie benchmarku

Postępuj zgodnie z poniższymi instrukcjami, aby odtworzyć środowisko i uruchomić benchmark lokalnie.

1. **Sklonuj repozytorium:**
   ```bash
   git clone https://github.com/your-username/kan_mlp_comparison.git
   cd kan_mlp_comparison
   ```

2. **Stwórz wirtualne środowisko:**
   Używając `conda`:
   ```bash
   conda create -n kan_benchmark python=3.10
   conda activate kan_benchmark
   ```
   Lub używając standardowego `venv`:
   ```bash
   python -m venv venv
   source venv/bin/activate  # W systemie Windows: venv\Scripts\activate
   ```

3. **Zainstaluj zależności:**
   ```bash
   pip install -r requirements.txt
   ```

## Uruchomienie benchmarku

Przepływ pracy jest podzielony na dwa zautomatyzowane skrypty. Upewnij się, że zbiory danych znajdują się w `data/processed/` przed startem.

### Krok 1: Uruchomienie potoku treningowego (Benchmark Pipeline)
Aby rozpocząć walidację krzyżową (CV) dla wszystkich zbiorów i modeli, wpisz:
```bash
python scripts/automate_benchmark.py
```
*Uwaga: Skrypt wykona Stratified 5-Fold CV. Automatycznie będzie on śledzić historię `Train Loss` oraz `Val Loss` i zapisywać wagi, historie epok oraz macierze pomyłek w odpowiednich folderach w `results/artifacts/`.*

### Krok 2: Wygenerowanie raportu analitycznego
Gdy benchmark zakończy pracę, wygeneruj tabele statystyczne i wykresy do publikacji:
```bash
python scripts/generate_report.py
```
Ten skrypt wykona:
- Agregację metryk (`results/summary_metrics.csv`).
- Obliczenia testów statystycznych (`results/stats_wilcoxon_posthoc.csv`, `results/stats_bayesian_rope.csv`).
- Generowanie Heatmap (Macierze Pomyłek), Krzywych Uczenia oraz Boxplotów dla Metryk, zapisując je w `results/plots/`.

## Metody statystyczne

Wykorzystujemy zaawansowane metodologie statystyczne, aby udowodnić wyższość modeli:
- **Podejście częstościowe:** Używamy **Testu Friedmana** do sprawdzenia ogólnej istotności różnic na wielu zbiorach danych, po którym następuje **Test Wilcoxona ze znakiem (Post-Hoc)** wyposażony w poprawkę Holm-Bonferroni, eliminującą ryzyko fałszywych wyników pozytywnych przy testowaniu wielu hipotez.
- **Podejście bayesowskie:** Wykorzystujemy **Bayesowski skorelowany test t-Studenta** (Benavoli et al., 2017) zaprojektowany specjalnie na potrzeby walidacji krzyżowej. Definiując Obszar Praktycznej Równoważności (ROPE), szacujemy dokładne prawdopodobieństwo, że dany model KAN jest znacząco lepszy od modelu bazowego MLP, unikając tym samym problemów klasycznego p-value.
