#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for GRD Prediction
This script handles:
1. Loading and cleaning the raw data
2. Extracting diagnosis and procedure codes
3. Encoding categorical variables
4. Creating feature matrices for ML models
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_PATH = 'data/raw/dataset_elpino.csv'
OUTPUT_DIR = 'data/processed'

# Minimum frequency for codes to be included as features (helps reduce noise)
MIN_CODE_FREQUENCY = 10

# Maximum number of diagnosis/procedure codes to keep
MAX_DIAG_CODES = 500
MAX_PROC_CODES = 300

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data(path=DATA_PATH):
    """Load the GRD dataset"""
    print("Loading data...")
    df = pd.read_csv(path, delimiter=';', on_bad_lines='skip', encoding='utf-8')
    print(f"  Loaded {len(df)} records with {len(df.columns)} columns")
    return df

# ==============================================================================
# CODE EXTRACTION
# ==============================================================================

def extract_code(text):
    """Extract the ICD code from a diagnosis/procedure description"""
    if pd.isna(text) or text == '-' or str(text).strip() == '-':
        return None
    
    text = str(text).strip()
    
    # Pattern to match codes like "A41.8", "86.28", "041013"
    # ICD-10 codes: Letter + numbers (optionally with dots)
    # ICD-9 procedure codes: Numbers with dots
    
    # Try to match ICD-10 pattern (e.g., A41.8, U07.1, J12.8)
    match = re.match(r'^([A-Z]\d{2}(?:\.\d)?)', text)
    if match:
        return match.group(1).upper()
    
    # Try to match ICD-9 pattern (e.g., 86.28, 31.1, 96.72)
    match = re.match(r'^(\d{2}\.\d{2})', text)
    if match:
        return match.group(1)
    
    # Try to match GRD code pattern (e.g., 041013)
    match = re.match(r'^(\d{6})', text)
    if match:
        return match.group(1)
    
    return None

def extract_all_codes(df):
    """Extract all unique diagnosis and procedure codes from the dataset"""
    print("\nExtracting diagnosis codes...")
    
    diag_cols = [col for col in df.columns if 'Diag' in col]
    proc_cols = [col for col in df.columns if 'Proced' in col]
    
    # Extract all diagnosis codes
    all_diag_codes = []
    for col in diag_cols:
        for val in df[col].dropna():
            code = extract_code(val)
            if code:
                all_diag_codes.append(code)
    
    # Extract all procedure codes
    all_proc_codes = []
    for col in proc_cols:
        for val in df[col].dropna():
            code = extract_code(val)
            if code:
                all_proc_codes.append(code)
    
    # Count frequencies
    diag_code_counts = Counter(all_diag_codes)
    proc_code_counts = Counter(all_proc_codes)
    
    print(f"  Found {len(diag_code_counts)} unique diagnosis codes")
    print(f"  Found {len(proc_code_counts)} unique procedure codes")
    
    return diag_code_counts, proc_code_counts

# ==============================================================================
# FEATURE CREATION
# ==============================================================================

def create_code_features(df, diag_code_counts, proc_code_counts, 
                          min_freq=MIN_CODE_FREQUENCY,
                          max_diag=MAX_DIAG_CODES, max_proc=MAX_PROC_CODES):
    """Create binary features for each diagnosis and procedure code"""
    print("\nCreating code features...")
    
    diag_cols = [col for col in df.columns if 'Diag' in col]
    proc_cols = [col for col in df.columns if 'Proced' in col]
    
    # Filter codes by frequency
    frequent_diag_codes = {code for code, count in diag_code_counts.items() 
                           if count >= min_freq}
    frequent_proc_codes = {code for code, count in proc_code_counts.items() 
                           if count >= min_freq}
    
    # Sort by frequency and take top N
    top_diag_codes = [code for code, _ in 
                      sorted(diag_code_counts.items(), key=lambda x: -x[1]) 
                      if code in frequent_diag_codes][:max_diag]
    top_proc_codes = [code for code, _ in 
                     sorted(proc_code_counts.items(), key=lambda x: -x[1]) 
                     if code in frequent_proc_codes][:max_proc]
    
    print(f"  Using {len(top_diag_codes)} diagnosis codes as features")
    print(f"  Using {len(top_proc_codes)} procedure codes as features")
    
    # Create feature matrices
    diag_features = pd.DataFrame(0, index=df.index, 
                                 columns=[f'DIAG_{code}' for code in top_diag_codes])
    proc_features = pd.DataFrame(0, index=df.index, 
                                 columns=[f'PROC_{code}' for code in top_proc_codes])
    
    # Fill in features
    for col in diag_cols:
        codes = df[col].apply(extract_code)
        for idx, code in codes.items():
            if code in top_diag_codes:
                diag_features.loc[idx, f'DIAG_{code}'] = 1
    
    for col in proc_cols:
        codes = df[col].apply(extract_code)
        for idx, code in codes.items():
            if code in top_proc_codes:
                proc_features.loc[idx, f'PROC_{code}'] = 1
    
    print(f"  Diagnosis features shape: {diag_features.shape}")
    print(f"  Procedure features shape: {proc_features.shape}")
    
    return diag_features, proc_features, top_diag_codes, top_proc_codes

# ==============================================================================
# DEMOGRAPHIC FEATURES
# ==============================================================================

def create_demographic_features(df):
    """Create features from demographic variables"""
    print("\nCreating demographic features...")
    
    # Age
    age = df['Edad en años'].copy()
    
    # Handle potential outliers (age > 120 likely data error)
    age = age.clip(0, 120)
    
    # Create age groups
    age_groups = pd.cut(age, bins=[0, 1, 5, 18, 40, 60, 80, 120], 
                       labels=['neonate', 'infant', 'child', 'young_adult', 
                               'middle_adult', 'senior', 'elderly'],
                       include_lowest=True)
    
    age_dummies = pd.get_dummies(age_groups, prefix='AGE')
    
    # Sex
    sex = df['Sexo (Desc)'].fillna('Unknown')
    sex_encoded = (sex == 'Hombre').astype(int)
    sex_df = pd.DataFrame({'SEX_MALE': sex_encoded})
    
    # Combine
    demo_features = pd.concat([age_dummies, sex_df], axis=1)
    
    print(f"  Demographic features: {demo_features.columns.tolist()}")
    
    return demo_features

# ==============================================================================
# TARGET VARIABLE
# ==============================================================================

def create_target(df):
    """Encode the target variable (GRD)"""
    print("\nEncoding target variable...")
    
    grd = df['GRD'].fillna('UNKNOWN')
    le = LabelEncoder()
    grd_encoded = le.fit_transform(grd)
    
    print(f"  Number of GRD classes: {len(le.classes_)}")
    print(f"  Most common GRDs:")
    grd_counts = df['GRD'].value_counts()
    for grd_code, count in grd_counts.head(10).items():
        print(f"    {grd_code}: {count}")
    
    return grd_encoded, le

# ==============================================================================
# MAIN PREPROCESSING PIPELINE
# ==============================================================================

def preprocess_data():
    """Run the full preprocessing pipeline"""
    print("=" * 80)
    print("GRD PREDICTION - DATA PREPROCESSING")
    print("=" * 80)
    
    # Load data
    df = load_data()
    
    # Extract codes
    diag_counts, proc_counts = extract_all_codes(df)
    
    # Create features
    diag_features, proc_features, diag_codes, proc_codes = create_code_features(
        df, diag_counts, proc_counts)
    demo_features = create_demographic_features(df)
    
    # Create target
    y, label_encoder = create_target(df)
    
    # Combine all features
    print("\nCombining features...")
    X = pd.concat([demo_features, diag_features, proc_features], axis=1)
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"  - Demographic features: {demo_features.shape[1]}")
    print(f"  - Diagnosis features: {diag_features.shape[1]}")
    print(f"  - Procedure features: {proc_features.shape[1]}")
    
    # Save processed data
    print("\nSaving processed data...")
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save feature matrix
    X.to_csv(f'{OUTPUT_DIR}/X_features.csv', index=False)
    print(f"  Saved X_features.csv")
    
    # Save target
    np.save(f'{OUTPUT_DIR}/y_target.npy', y)
    print(f"  Saved y_target.npy")
    
    # Save label encoder
    with open(f'{OUTPUT_DIR}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"  Saved label_encoder.pkl")
    
    # Save code lists
    metadata = {
        'diag_codes': diag_codes,
        'proc_codes': proc_codes,
        'n_classes': len(label_encoder.classes_),
        'feature_names': X.columns.tolist()
    }
    with open(f'{OUTPUT_DIR}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  Saved metadata.pkl")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    
    return X, y, label_encoder, metadata

# Run preprocessing
if __name__ == "__main__":
    X, y, label_encoder, metadata = preprocess_data()
