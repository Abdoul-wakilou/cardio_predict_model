"""
3_train.py - Entraînement et optimisation du modèle
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import json

print("="*60)
print("ENTRAÎNEMENT - DATASET COMPLET (1018 patients)")
print("="*60)

# ============================================================================
# 1. CHARGEMENT
# ============================================================================

X_train = np.loadtxt('../data/X_train_preprocessed_full.csv', delimiter=',')
X_test  = np.loadtxt('../data/X_test_preprocessed_full.csv',  delimiter=',')
y_train = pd.read_csv('../data/y_train_full.csv')['target']
y_test  = pd.read_csv('../data/y_test_full.csv')['target']

print(f"\n X_train : {X_train.shape}")
print(f" X_test  : {X_test.shape}")
print(f" y_train : {y_train.shape}")
print(f" y_test  : {y_test.shape}")
print(f"\n✅ Nombre de features après transformation : {X_train.shape[1]}")

# ============================================================================
# 2. MODÈLES BASELINE
# ============================================================================

print("\n" + "="*60)
print("MODÈLES BASELINE")
print("="*60)

baseline_models = {
    "Régression Logistique": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest":         RandomForestClassifier(random_state=42, n_estimators=100),
    "SVM":                   SVC(random_state=42, probability=True)
}

baseline_results = {}

for name, model in baseline_models.items():
    print(f"\n🔹 {name}...")
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    baseline_results[name] = {'model': model, 'accuracy': acc, 'auc': auc}
    print(f"      Accuracy : {acc:.4f} | AUC : {auc:.4f}")

# ============================================================================
# 3. OPTIMISATION RANDOM FOREST
# ============================================================================

print("\n" + "="*60)
print("OPTIMISATION HYPERPARAMÈTRES — RANDOM FOREST")
print("="*60)

param_grid = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [None, 10, 15],
    'min_samples_split':[2, 5],
    'min_samples_leaf': [1, 2]
}

print("🔍 Grille de recherche :")
for k, v in param_grid.items():
    print(f"      {k} : {v}")

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

print("\n🔄 Recherche en cours...")
grid_search.fit(X_train, y_train)

print(f"\n✅ Meilleurs paramètres : {grid_search.best_params_}")
print(f"✅ Meilleure accuracy (CV) : {grid_search.best_score_:.4f}")

# ============================================================================
# 4. ÉVALUATION DU MODÈLE OPTIMISÉ
# ============================================================================

print("\n" + "="*60)
print("ÉVALUATION SUR LE TEST SET")
print("="*60)

best_rf  = grid_search.best_estimator_
y_pred   = best_rf.predict(X_test)
y_proba  = best_rf.predict_proba(X_test)[:, 1]

acc_best = accuracy_score(y_test, y_pred)
auc_best = roc_auc_score(y_test, y_proba)

print(f"\n Performance du modèle optimisé :")
print(f"   Accuracy : {acc_best:.4f}")
print(f"   AUC      : {auc_best:.4f}")

print(f"\n Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=["Sain", "Malade"]))

cm = confusion_matrix(y_test, y_pred)
print(f"\n Matrice de confusion :")
print(f"   Vrais Negatifs : {cm[0,0]}, Faux Positifs : {cm[0,1]}")
print(f"   Faux Negatifs  : {cm[1,0]}, Vrais Positifs : {cm[1,1]}")

# ============================================================================
# 5. SÉLECTION DU MODÈLE FINAL
# ============================================================================

print("\n" + "="*60)
print("SÉLECTION DU MODÈLE FINAL")
print("="*60)

acc_baseline = baseline_results['Random Forest']['accuracy']

if acc_best >= acc_baseline:
    final_model      = best_rf
    final_model_name = "Random Forest (optimisé)"
    final_accuracy   = acc_best
else:
    final_model      = baseline_results['Random Forest']['model']
    final_model_name = "Random Forest (baseline)"
    final_accuracy   = acc_baseline

print(f"\n🏆 Modèle sélectionné : {final_model_name} (accuracy = {final_accuracy:.4f})")

# ============================================================================
# 6. SAUVEGARDE
# ============================================================================

print("\n" + "="*60)
print("SAUVEGARDE")
print("="*60)

joblib.dump(final_model, '../model/best_model_full.joblib')
print("✅ Modèle sauvegardé : model/best_model_full.joblib")

params = {
    'best_model':    final_model_name,
    'accuracy':      float(final_accuracy),
    'auc':           float(auc_best),
    'best_params':   grid_search.best_params_,
    'cv_best_score': float(grid_search.best_score_)
}
with open('../model/model_info.json', 'w') as f:
    json.dump(params, f, indent=4)
print("✅ Informations sauvegardées : model/model_info.json")

print("\n" + "="*60)
print("✅ ENTRAÎNEMENT TERMINÉ")
print("="*60)