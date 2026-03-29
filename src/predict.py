#!/usr/bin/env python3
"""
Prediction script for GRD model
Loads the trained model and makes predictions on new data
"""

import pickle
import pandas as pd
import numpy as np
import re
import sys

def load_model(model_dir='models'):
    """Load the trained model and associated files"""
    print("Loading model...")
    
    with open(f'{model_dir}/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(f'{model_dir}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    return model, label_encoder, metadata

def extract_code(text):
    """Extract ICD code from diagnosis/procedure text"""
    if pd.isna(text) or text == '-' or str(text).strip() == '-':
        return None
    
    text = str(text).strip()
    
    # Try ICD-10 pattern
    match = re.match(r'^([A-Z]\d{2}(?:\.\d)?)', text)
    if match:
        return match.group(1).upper()
    
    # Try ICD-9 pattern
    match = re.match(r'^(\d{2}\.\d{2})', text)
    if match:
        return match.group(1)
    
    return None

def preprocess_patient(diagnoses, procedures, age, sex, metadata):
    """Preprocess a single patient for prediction"""
    diag_codes = metadata['diag_codes']
    proc_codes = metadata['proc_codes']
    
    # Create feature vector
    features = {}
    
    # Age features
    age_bins = [0, 1, 5, 18, 40, 60, 80, 120]
    age_labels = ['AGE_neonate', 'AGE_infant', 'AGE_child', 'AGE_young_adult', 
                  'AGE_middle_adult', 'AGE_senior', 'AGE_elderly']
    
    age_clipped = min(max(age, 0), 120)
    for i in range(len(age_bins) - 1):
        if age_clipped >= age_bins[i] and age_clipped <= age_bins[i+1]:
            features[age_labels[i]] = 1
            break
    
    # Sex feature
    features['SEX_MALE'] = 1 if sex.lower() == 'hombre' else 0
    
    # Diagnosis features
    for code in diag_codes:
        features[f'DIAG_{code}'] = 0
    
    for diag in diagnoses:
        code = extract_code(diag)
        if code and f'DIAG_{code}' in features:
            features[f'DIAG_{code}'] = 1
    
    # Procedure features
    for code in proc_codes:
        features[f'PROC_{code}'] = 0
    
    for proc in procedures:
        code = extract_code(proc)
        if code and f'PROC_{code}' in features:
            features[f'PROC_{code}'] = 1
    
    # Create DataFrame in correct order
    feature_names = metadata['feature_names']
    X = pd.DataFrame([features])[feature_names]
    
    return X

def predict_grd(model, label_encoder, metadata, diagnoses, procedures, age, sex):
    """Predict GRD for a patient"""
    X = preprocess_patient(diagnoses, procedures, age, sex, metadata)
    
    # Make prediction
    y_pred = model.predict(X)
    grd_code = label_encoder.inverse_transform(y_pred)[0]
    
    # Get probabilities if available
    try:
        proba = model.predict_proba(X)[0]
        top_indices = np.argsort(proba)[::-1][:5]
        top_grds = label_encoder.inverse_transform(top_indices)
        top_probs = proba[top_indices]
        
        return grd_code, list(zip(top_grds, top_probs))
    except:
        return grd_code, None

def main():
    """Interactive prediction demo"""
    print("=" * 80)
    print("GRD PREDICTION - DEMO")
    print("=" * 80)
    
    # Load model
    model, label_encoder, metadata = load_model()
    
    # Example patient
    example_diagnoses = [
        'U07.1 - COVID-19, virus identificado',
        'J12.8 - Neumonía debida a otros virus',
        'R06.0 - Disnea',
        'R50.9 - Fiebre, no especificada',
        'J96.00 - Insuficiencia respiratoria aguda'
    ]
    
    example_procedures = [
        '96.72 - VENTILACION MECANICA CONTINUA',
        '31.1 - TRAQUEOSTOMIA TEMPORAL',
        '87.41 - TOMOGRAFIA AXIAL COMPUTERIZADA TORAX'
    ]
    
    example_age = 55
    example_sex = 'Hombre'
    
    print(f"\nExample Patient:")
    print(f"  Age: {example_age}")
    print(f"  Sex: {example_sex}")
    print(f"  Diagnoses: {len(example_diagnoses)}")
    print(f"  Procedures: {len(example_procedures)}")
    
    grd, top_predictions = predict_grd(
        model, label_encoder, metadata,
        example_diagnoses, example_procedures,
        example_age, example_sex
    )
    
    print(f"\n*** Predicted GRD: {grd} ***")
    
    if top_predictions:
        print("\nTop 5 Predictions:")
        for i, (code, prob) in enumerate(top_predictions, 1):
            print(f"  {i}. {code} (prob: {prob:.3f})")

if __name__ == "__main__":
    main()
