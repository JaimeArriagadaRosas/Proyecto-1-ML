#!/usr/bin/env python3
"""
Exploratory Data Analysis for GRD Prediction
Generates visualizations and statistics for the dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR = 'assets/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading data for EDA...")
df = pd.read_csv('data/raw/dataset_elpino.csv', delimiter=';', 
                 on_bad_lines='skip', encoding='utf-8')

with open('data/processed/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

with open('data/processed/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# ==============================================================================
# 1. DATA QUALITY ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("1. DATA QUALITY ANALYSIS")
print("=" * 80)

# Completeness analysis
print("\n1.1 Completeness Analysis")
diag_cols = [col for col in df.columns if 'Diag' in col]
proc_cols = [col for col in df.columns if 'Proced' in col]

def count_valid(series):
    """Count non-missing values (not '-' and not NaN)"""
    valid = series.apply(lambda x: pd.notna(x) and str(x).strip() != '-')
    return valid.sum()

diag_completeness = {col: count_valid(df[col]) / len(df) * 100 for col in diag_cols}
proc_completeness = {col: count_valid(df[col]) / len(df) * 100 for col in proc_cols}

print("Diagnosis columns completeness (first 10):")
for col in list(diag_cols)[:10]:
    print(f"  {col}: {diag_completeness[col]:.1f}%")

print("\nProcedure columns completeness (first 10):")
for col in list(proc_cols)[:10]:
    print(f"  {col}: {proc_completeness[col]:.1f}%")

# Plot completeness
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Diagnosis completeness
diag_df = pd.DataFrame({'Column': list(diag_completeness.keys()), 
                       'Completeness': list(diag_completeness.values())})
diag_df['Column'] = diag_df['Column'].str.replace(' (cod+des)', '')
axes[0].barh(diag_df['Column'][:15], diag_df['Completeness'][:15], color='steelblue')
axes[0].set_xlabel('Completeness (%)')
axes[0].set_title('Diagnosis Column Completeness')
axes[0].set_xlim(0, 100)

# Procedure completeness
proc_df = pd.DataFrame({'Column': list(proc_completeness.keys()), 
                       'Completeness': list(proc_completeness.values())})
proc_df['Column'] = proc_df['Column'].str.replace(' (cod+des)', '')
axes[1].barh(proc_df['Column'][:15], proc_df['Completeness'][:15], color='darkorange')
axes[1].set_xlabel('Completeness (%)')
axes[1].set_title('Procedure Column Completeness')
axes[1].set_xlim(0, 100)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_completeness.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 01_completeness.png")

# ==============================================================================
# 2. DEMOGRAPHIC STATISTICS
# ==============================================================================

print("\n" + "=" * 80)
print("2. DEMOGRAPHIC STATISTICS")
print("=" * 80)

# Age distribution
age = df['Edad en años'].clip(0, 120)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Age histogram
axes[0].hist(age, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
axes[0].axvline(age.mean(), color='red', linestyle='--', label=f'Mean: {age.mean():.1f}')
axes[0].axvline(age.median(), color='green', linestyle='--', label=f'Median: {age.median():.1f}')
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Age Distribution')
axes[0].legend()

# Age boxplot
axes[1].boxplot(age, vert=True)
axes[1].set_ylabel('Age (years)')
axes[1].set_title('Age Boxplot')

# Sex distribution
sex_counts = df['Sexo (Desc)'].value_counts()
axes[2].pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', 
            colors=['lightcoral', 'lightblue'])
axes[2].set_title('Sex Distribution')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_demographics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 02_demographics.png")

print("\nAge Statistics:")
print(f"  Mean: {age.mean():.2f}")
print(f"  Std: {age.std():.2f}")
print(f"  Min: {age.min()}")
print(f"  Max: {age.max()}")
print(f"  Median: {age.median()}")

# ==============================================================================
# 3. TARGET VARIABLE ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("3. TARGET VARIABLE ANALYSIS")
print("=" * 80)

# GRD distribution
grd_counts = df['GRD'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Top 20 GRDs bar chart
top_grd = grd_counts.head(20)
axes[0].barh(range(len(top_grd)), top_grd.values, color='steelblue')
axes[0].set_yticks(range(len(top_grd)))
axes[0].set_yticklabels([g[:40] + '...' if len(g) > 40 else g for g in top_grd.index], fontsize=8)
axes[0].set_xlabel('Frequency')
axes[0].set_title('Top 20 GRDs')
axes[0].invert_yaxis()

# Distribution of GRD frequencies
axes[1].hist(grd_counts.values, bins=50, color='darkorange', edgecolor='white', alpha=0.7)
axes[1].set_xlabel('GRD Frequency')
axes[1].set_ylabel('Number of GRDs')
axes[1].set_title('Distribution of GRD Frequencies')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_grd_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 03_grd_distribution.png")

print(f"\nTotal unique GRDs: {len(grd_counts)}")
print(f"GRDs with frequency >= 100: {(grd_counts >= 100).sum()}")
print(f"GRDs with frequency >= 50: {(grd_counts >= 50).sum()}")
print(f"GRDs with frequency >= 10: {(grd_counts >= 10).sum()}")

# ==============================================================================
# 4. DIAGNOSIS CODE ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("4. DIAGNOSIS CODE ANALYSIS")
print("=" * 80)

# Extract codes
def extract_code(text):
    if pd.isna(text) or text == '-' or str(text).strip() == '-':
        return None
    text = str(text).strip()
    match = re.match(r'^([A-Z]\d{2}(?:\.\d)?)', text)
    if match:
        return match.group(1).upper()
    match = re.match(r'^(\d{2}\.\d{2})', text)
    if match:
        return match.group(1)
    match = re.match(r'^(\d{6})', text)
    if match:
        return match.group(1)
    return None

import re
all_diag_codes = []
for col in diag_cols:
    for val in df[col].dropna():
        code = extract_code(val)
        if code:
            all_diag_codes.append(code)

diag_code_counts = Counter(all_diag_codes)

# Top diagnosis codes
top_diag = pd.DataFrame(diag_code_counts.most_common(30), columns=['Code', 'Count'])

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top_diag)), top_diag['Count'].values, color='steelblue')
ax.set_yticks(range(len(top_diag)))
ax.set_yticklabels(top_diag['Code'].values)
ax.set_xlabel('Frequency')
ax.set_title('Top 30 Diagnosis Codes')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_diagnosis_codes.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 04_diagnosis_codes.png")

print(f"\nTotal unique diagnosis codes: {len(diag_code_counts)}")

# ==============================================================================
# 5. PROCEDURE CODE ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("5. PROCEDURE CODE ANALYSIS")
print("=" * 80)

all_proc_codes = []
for col in proc_cols:
    for val in df[col].dropna():
        code = extract_code(val)
        if code:
            all_proc_codes.append(code)

proc_code_counts = Counter(all_proc_codes)

# Top procedure codes
top_proc = pd.DataFrame(proc_code_counts.most_common(30), columns=['Code', 'Count'])

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top_proc)), top_proc['Count'].values, color='darkorange')
ax.set_yticks(range(len(top_proc)))
ax.set_yticklabels(top_proc['Code'].values)
ax.set_xlabel('Frequency')
ax.set_title('Top 30 Procedure Codes')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_procedure_codes.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 05_procedure_codes.png")

print(f"\nTotal unique procedure codes: {len(proc_code_counts)}")

# ==============================================================================
# 6. CORRELATION ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("6. CORRELATION ANALYSIS")
print("=" * 80)

# Load processed features
X = pd.read_csv('data/processed/X_features.csv')
y = np.load('data/processed/y_target.npy')

# Add age and sex to analyze correlations with target
X_analysis = X.copy()
X_analysis['AGE_NUMERIC'] = df['Edad en años'].values
X_analysis['SEX'] = (df['Sexo (Desc)'] == 'Hombre').astype(int).values

# Correlation with target (use top features)
# Calculate correlation between each feature and target
correlations = []
for col in X_analysis.columns[:50]:  # Top 50 features
    corr = np.corrcoef(X_analysis[col].values, y)[0, 1]
    correlations.append((col, corr))

corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)

# Plot top correlations
fig, ax = plt.subplots(figsize=(10, 8))
top_corr = corr_df.head(20)
colors = ['green' if x > 0 else 'red' for x in top_corr['Correlation']]
ax.barh(range(len(top_corr)), top_corr['Correlation'].values, color=colors)
ax.set_yticks(range(len(top_corr)))
ax.set_yticklabels(top_corr['Feature'].values)
ax.set_xlabel('Correlation with Target')
ax.set_title('Top 20 Feature Correlations with GRD')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 06_correlations.png")

# ==============================================================================
# 7. FEATURE IMPORTANCE PREVIEW
# ==============================================================================

print("\n" + "=" * 80)
print("7. FEATURE STATISTICS")
print("=" * 80)

# Feature sparsity
sparsity = (X == 0).sum() / len(X) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Sparsity distribution
axes[0].hist(sparsity, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
axes[0].set_xlabel('Sparsity (%)')
axes[0].set_ylabel('Number of Features')
axes[0].set_title('Feature Sparsity Distribution')

# Feature count statistics per patient
diag_cols_feats = [c for c in X.columns if c.startswith('DIAG_')]
proc_cols_feats = [c for c in X.columns if c.startswith('PROC_')]

n_diag = X[diag_cols_feats].sum(axis=1)
n_proc = X[proc_cols_feats].sum(axis=1)

axes[1].hist(n_diag, bins=30, alpha=0.5, label='Diagnosis Codes', color='steelblue')
axes[1].hist(n_proc, bins=30, alpha=0.5, label='Procedure Codes', color='darkorange')
axes[1].set_xlabel('Number of Codes')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Number of Codes per Patient')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_feature_stats.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 07_feature_stats.png")

print(f"\nFeature Statistics:")
print(f"  Total features: {X.shape[1]}")
print(f"  Mean sparsity: {sparsity.mean():.1f}%")
print(f"  Mean diagnosis codes per patient: {n_diag.mean():.1f}")
print(f"  Mean procedure codes per patient: {n_proc.mean():.1f}")

# ==============================================================================
# 8. CLASS IMBALANCE ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("8. CLASS IMBALANCE ANALYSIS")
print("=" * 80)

# Class imbalance ratio
grd_counts = df['GRD'].value_counts()
imbalance_ratio = grd_counts.max() / grd_counts.min()

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(grd_counts.values, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
ax.set_xlabel('GRD Frequency')
ax.set_ylabel('Number of GRDs')
ax.set_title(f'GRD Class Distribution\n(Imbalance Ratio: {imbalance_ratio:.1f}:1)')
ax.axvline(grd_counts.median(), color='red', linestyle='--', 
            label=f'Median: {grd_counts.median():.0f}')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_class_imbalance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved 08_class_imbalance.png")

print(f"\nClass Imbalance:")
print(f"  Max frequency: {grd_counts.max()}")
print(f"  Min frequency: {grd_counts.min()}")
print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
print(f"  Median frequency: {grd_counts.median():.0f}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)
print(f"\nGenerated visualizations saved to {OUTPUT_DIR}/")
print("\nSummary:")
print(f"  - Dataset: {len(df)} patients, {len(df.columns)} raw columns")
print(f"  - Target: {len(grd_counts)} unique GRDs")
print(f"  - Age range: {age.min()}-{age.max()} years")
print(f"  - Sex distribution: {sex_counts.to_dict()}")
print(f"  - Unique diagnosis codes: {len(diag_code_counts)}")
print(f"  - Unique procedure codes: {len(proc_code_counts)}")
print(f"  - Features created: {X.shape[1]}")
