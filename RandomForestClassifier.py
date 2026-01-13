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








