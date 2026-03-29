#!/usr/bin/env python3
"""
Fast Machine Learning Model Training for GRD Prediction
Optimized for speed with efficient models
"""

import pandas as pd
import numpy as np
import pickle
import time
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Try to import LightGBM (faster than XGBoost for this task)
try:
    import lightgbm as lgb
    HAS_LGB = True
    print("LightGBM available")
except ImportError:
    HAS_LGB = False
    print("LightGBM not available")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = 'models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data():
    """Load preprocessed data"""
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    X = pd.read_csv('data/processed/X_features.csv')
    y = np.load('data/processed/y_target.npy')
    
    with open('data/processed/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open('data/processed/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    return X, y, label_encoder, metadata

# ==============================================================================
# TRAIN/TEST SPLIT
# ==============================================================================

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split data into train/test sets"""
    print("\nSplitting data...")
    
    # Filter out classes with less than 2 samples
    from collections import Counter
    class_counts = Counter(y)
    valid_classes = [c for c, count in class_counts.items() if count >= 2]
    valid_mask = np.isin(y, valid_classes)
    
    X_filtered = X.values[valid_mask]
    y_filtered = y[valid_mask]
    
    print(f"  Removed {len(y) - len(y_filtered)} samples from rare classes")
    print(f"  Remaining: {len(y_filtered)} samples with {len(valid_classes)} classes")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_filtered
    )
    
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def train_lightgbm(X_train, y_train):
    """Train LightGBM model"""
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM")
    print("=" * 80)
    
    start_time = time.time()
    
    model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training time: {train_time:.1f} seconds")
    
    return model, train_time

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST")
    print("=" * 80)
    
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"  Training time: {train_time:.1f} seconds")
    
    return model, train_time

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Evaluate a model"""
    print(f"\nEvaluating {name}...")
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'name': name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_f1_macro': f1_score(y_train, y_train_pred, average='macro', zero_division=0.0),
        'test_f1_macro': f1_score(y_test, y_test_pred, average='macro', zero_division=0.0),
        'train_f1_weighted': f1_score(y_train, y_train_pred, average='weighted', zero_division=0.0),
        'test_f1_weighted': f1_score(y_test, y_test_pred, average='weighted', zero_division=0.0),
    }
    
    # Top-k accuracy
    try:
        y_proba = model.predict_proba(X_test)
        metrics['test_top5_accuracy'] = top_k_accuracy_score(y_test, y_proba, k=5)
        metrics['test_top10_accuracy'] = top_k_accuracy_score(y_test, y_proba, k=10)
    except Exception as e:
        print(f"  Warning: Could not compute top-k accuracy: {e}")
        metrics['test_top5_accuracy'] = 0
        metrics['test_top10_accuracy'] = 0
    
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  Test F1 (macro): {metrics['test_f1_macro']:.4f}")
    print(f"  Test F1 (weighted): {metrics['test_f1_weighted']:.4f}")
    print(f"  Top-5 Accuracy: {metrics['test_top5_accuracy']:.4f}")
    print(f"  Top-10 Accuracy: {metrics['test_top10_accuracy']:.4f}")
    
    return metrics, y_test_pred

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 80)
    print("GRD PREDICTION - FAST MODEL TRAINING")
    print("=" * 80)
    
    # Load data
    X, y, label_encoder, metadata = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train models
    results = {}
    trained_models = {}
    predictions = {}
    training_times = {}
    
    # Train LightGBM (usually fastest and best performance)
    if HAS_LGB:
        model, train_time = train_lightgbm(X_train, y_train)
        trained_models['LightGBM'] = model
        training_times['LightGBM'] = train_time
        metrics, y_pred = evaluate_model('LightGBM', model, X_train, X_test, y_train, y_test)
        results['LightGBM'] = metrics
        predictions['LightGBM'] = y_pred
    
    # Train Random Forest
    model, train_time = train_random_forest(X_train, y_train)
    trained_models['Random Forest'] = model
    training_times['Random Forest'] = train_time
    metrics, y_pred = evaluate_model('Random Forest', model, X_train, X_test, y_train, y_test)
    results['Random Forest'] = metrics
    predictions['Random Forest'] = y_pred
    
    # ==============================================================================
    # RESULTS COMPARISON
    # ==============================================================================
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    # Create comparison table
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)
    
    print("\nModel Performance Summary:")
    print(comparison_df[['test_accuracy', 'test_f1_macro', 'test_f1_weighted', 
                        'test_top5_accuracy', 'test_top10_accuracy']])
    
    # Select best model
    best_model_name = comparison_df['test_f1_weighted'].idxmax()
    best_score = comparison_df.loc[best_model_name, 'test_f1_weighted']
    
    print(f"\n*** Best Model: {best_model_name} ***")
    print(f"    Weighted F1 Score: {best_score:.4f}")
    print(f"    Training Time: {training_times[best_model_name]:.1f} seconds")
    
    # ==============================================================================
    # SAVE BEST MODEL
    # ==============================================================================
    
    print("\n" + "=" * 80)
    print("SAVING BEST MODEL")
    print("=" * 80)
    
    best_model = trained_models[best_model_name]
    
    # Save model
    model_path = f'{OUTPUT_DIR}/best_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Saved model to {model_path}")
    
    # Save label encoder
    with open(f'{OUTPUT_DIR}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Saved label encoder")
    
    # Save metadata
    with open(f'{OUTPUT_DIR}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata")
    
    # Save results
    comparison_df.to_csv(f'{OUTPUT_DIR}/results.csv')
    print(f"Saved results to results.csv")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    return best_model_name, best_model, results

if __name__ == "__main__":
    main()
