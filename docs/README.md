# GRD Prediction Project - Hospital El Pino

Machine Learning project for predicting GRD (Grupos Relacionados por Diagnóstico) from patient data at Hospital El Pino.

## Project Overview

This project implements a machine learning solution to predict the GRD (Diagnosis Related Groups) of patients based on their diagnoses, procedures, age, and sex.

## Dataset

- **Source**: Hospital El Pino (Chile)
- **Patients**: 14,561 records
- **Features**:
  - 35 diagnosis columns (1 principal + 34 secondary)
  - 30 procedure columns (1 principal + 29 secondary)
  - Age (in years)
  - Sex (Male/Female)
- **Target**: 526 unique GRD classes

## Project Structure

```
.
├── config/
│   └── requirements.txt                # Python dependencies
├── data/
│   ├── raw/                            # Raw dataset
│   │   ├── dataset_elpino.csv
│   │   ├── CIE-10.xlsx
│   │   ├── CIE-9.xlsx
│   │   └── Tablas maestras bases GRD.xlsx
│   └── processed/                      # Preprocessed data
│       ├── X_features.csv
│       ├── y_target.npy
│       ├── label_encoder.pkl
│       └── metadata.pkl
├── docs/
│   ├── README.md                       # This file
│   └── Informe_GRD_Prediction.md       # Full project report
├── models/                             # Trained models
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   ├── metadata.pkl
│   └── results.csv
├── src/                                # Source code
│   ├── preprocessing.py               # Data preprocessing pipeline
│   ├── eda_analysis.py                # Exploratory data analysis
│   ├── model_training.py              # Model training script
│   └── predict.py                     # Prediction demo
├── assets/
│   └── images/                        # EDA visualizations
│       ├── 01-completeness.png
│       ├── 02-demographics.png
│       ├── 03-grd-distribution.png
│       ├── 04-diagnosis-codes.png
│       ├── 05-procedure-codes.png
│       ├── 06-correlations.png
│       ├── 07-feature-stats.png
│       └── 08-class-imbalance.png
└── .gitignore
```

## Installation

```bash
pip install -r config/requirements.txt
```

## Usage

### 1. Preprocess Data

```bash
python src/preprocessing.py
```

### 2. Run EDA

```bash
python src/eda_analysis.py
```

### 3. Train Model

```bash
python src/model_training.py
```

### 4. Make Predictions

```bash
python src/predict.py
```

## Model Performance

- **Best Model**: Random Forest Classifier
- **Test Accuracy**: ~36%
- **Weighted F1 Score**: ~0.37
- **Top-5 Accuracy**: Available through prediction probabilities

Note: Given the high number of classes (526) and significant class imbalance (813:1), the model performs reasonably well.

## Key Findings

### Data Quality
- High completeness for primary diagnosis (100%) and primary procedure (100%)
- Secondary columns have lower completeness (typical in medical data)
- 3,470 unique diagnosis codes and 832 procedure codes identified

### Class Imbalance
- Extreme imbalance: most common GRD (Cesárea) has 813 samples
- 76 GRD classes have only 1 sample
- This significantly impacts model performance

### Feature Engineering
- Binary encoding of diagnosis codes (presence/absence)
- Binary encoding of procedure codes
- Age grouping (7 categories)
- Sex encoding

## Technical Details

- **Language**: Python 3.x
- **Dependencies**: pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn

## References

- GRD (Grupos Relacionados por Diagnóstico) is a patient classification system
- Similar approaches used in hospital management and healthcare analytics
- Random Forest selected for handling high-dimensional sparse data

## Authors

- CINF104 - Aprendizaje de Máquinas
- Universidad

## License

Academic project