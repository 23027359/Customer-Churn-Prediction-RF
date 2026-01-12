#!/usr/bin/env python
# coding: utf-8

# In[1]:


# LEVEL 3 - TASK 2: SUPPORT VECTOR MACHINE (SVM) FOR CLASSIFICATION
# Dataset: Customer Churn (churn-bigml-80.csv + churn-bigml-20.csv)
# Objectives: Train SVM → Compare Linear vs RBF kernels → Visualize decision boundary → Evaluate thoroughly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.decomposition import PCA   # For 2D visualization
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ==================== 1. Load and Prepare Data ====================
train_df = pd.read_csv('churn-bigml-80.csv')
test_df  = pd.read_csv('churn-bigml-20.csv')

# Encode categorical variables
le = LabelEncoder()
for col in ['International plan', 'Voice mail plan']:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col]  = le.transform(test_df[col])

# Use only numerical + encoded categorical (drop 'State' to avoid high dimensionality)
features_to_use = ['Account length', 'International plan', 'Voice mail plan',
                   'Number vmail messages', 'Total day minutes', 'Total day calls',
                   'Total day charge', 'Total eve minutes', 'Total eve calls',
                   'Total eve charge', 'Total night minutes', 'Total night calls',
                   'Total night charge', 'Total intl minutes', 'Total intl calls',
                   'Total intl charge', 'Customer service calls']

X_train = train_df[features_to_use]
y_train = train_df['Churn'].astype(int)

X_test = test_df[features_to_use]
y_test = test_df['Churn'].astype(int)

# Scale features (VERY IMPORTANT FOR SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
print(f"Churn rate: {y_train.mean():.3f}")

# ==================== 2. Train & Compare Kernels ====================
kernels = ['linear', 'rbf']
results = []

plt.figure(figsize=(15, 6))

for i, kernel in enumerate(kernels):
    print(f"\nTraining SVM with {kernel.upper()} kernel...")

    svm = SVC(kernel=kernel, probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)

    y_pred = svm.predict(X_test_scaled)
    y_proba = svm.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        'Kernel': kernel.upper(),
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC': auc
    })

    print(f"{kernel.upper()} → Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# Results table
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(results_df.round(4))

# ==================== 3. Visualize Decision Boundary (2D PCA) ====================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Retrain best kernel (usually RBF) on 2D data for visualization
svm_viz = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
svm_viz.fit(X_pca, y_train)

# Create mesh grid
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = svm_viz.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='RdYlBu', edgecolors='k', s=60)
plt.title('SVM Decision Boundary (RBF Kernel) - PCA 2D Projection\n(Customer Churn Dataset)', fontsize=14, pad=20)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(['No Churn', 'Churn'], loc='upper right')
plt.tight_layout()
plt.show()

# ==================== 4. Final Confusion Matrix (RBF) ====================
y_pred_final = svm_viz.predict(pca.transform(X_test_scaled))  # using same model structure
cm = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - SVM (RBF Kernel)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[2]:


import pandas as pd

# === Option 1: Full absolute path (Windows style - use raw string r"" or double \\ ) ===

# Example - CHANGE THIS TO YOUR REAL PATH
train_df = pd.read_csv(r"C:\Users\Sindi\OneDrive\Codveda Matchine Learning Internship\Data Set For Task-20260112T101045Z-1-001.zip")
test_df  = pd.read_csv(r"C:\Users\Sindi\OneDrive\Codveda Matchine Learning Internship\Data Set For Task-20260112T101045Z-1-001.zip")

# Alternative style with double backslashes (also works)
# train_df = pd.read_csv("C:\\Users\\Sindisi\\OneDrive\\Codveda Matche Learning Internship\\datasets\\churn-bigml-80.csv")

print("Datasets loaded successfully!")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("\nFirst 3 rows of train:\n", train_df.head(3))


# In[4]:


# =============================================================================
# RANDOM FOREST CLASSIFIER
# Codveda Machine Learning Internship
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ──── Set visual style ──────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ──── 1. Load the datasets using your actual folder path ─────────────────────
folder = r"C:\Users\Sindi\OneDrive\Codveda Matchine Learning Internship\Churn Prdiction Data"

train_path = folder + r"\churn-bigml-80.csv"
test_path  = folder + r"\churn-bigml-20.csv"

try:
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    print("✓ Datasets loaded successfully from:")
    print("  ", train_path)
    print("  ", test_path)
except FileNotFoundError as e:
    print("File not found! Please check:")
    print("1. Files are really in this folder:", folder)
    print("2. File names are exactly: churn-bigml-80.csv and churn-bigml-20.csv")
    print("Error message:", e)
    raise  # stop execution if files missing

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape : {test_df.shape}")
print(f"Churn rate : {train_df['Churn'].mean():.3%}\n")

# ──── 2. Quick preprocessing ────────────────────────────────────────────────
# Encode Yes/No columns
for col in ['International plan', 'Voice mail plan']:
    train_df[col] = (train_df[col] == 'Yes').astype(int)
    test_df[col]  = (test_df[col]  == 'Yes').astype(int)

# Select useful features (dropping State to keep it simple)
features = [
    'Account length', 'International plan', 'Voice mail plan',
    'Number vmail messages', 'Total day minutes', 'Total day calls',
    'Total day charge', 'Total eve minutes', 'Total eve calls',
    'Total eve charge', 'Total night minutes', 'Total night calls',
    'Total night charge', 'Total intl minutes', 'Total intl calls',
    'Total intl charge', 'Customer service calls'
]

X_train = train_df[features]
y_train = train_df['Churn'].astype(int)

X_test = test_df[features]
y_test = test_df['Churn'].astype(int)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ──── 3. Train Random Forest with basic tuning ──────────────────────────────
print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

# ──── 4. Evaluate on test set ───────────────────────────────────────────────
y_pred = rf.predict(X_test_scaled)
y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

metrics = {
    'Accuracy':  accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall':    recall_score(y_test, y_pred),
    'F1-Score':  f1_score(y_test, y_pred),
    'ROC-AUC':   roc_auc_score(y_test, y_pred_proba)
}

print("\n" + "═"*40)
print("       TEST SET PERFORMANCE")
print("═"*40)
for name, value in metrics.items():
    print(f"{name:10}: {value:.4f}")
print("═"*40 + "\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10,7))
importances.plot.barh(color='cornflowerblue')
plt.title('Feature Importance - Customer Churn Prediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

print("\nLevel 3 Task 1 - Random Forest Classifier → COMPLETED!")
print("Save this notebook, take screenshots of the plots and metrics,")
print("record a short video explanation and you're ready to submit!")


# In[ ]:




