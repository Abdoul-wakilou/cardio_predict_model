"""
3_train.py — Entraînement et optimisation des modèles
======================================================
Dataset : UCI Heart Disease combiné (834 patients — labels corrects)
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix)

print("=" * 60)
print("ENTRAÎNEMENT — UCI HEART DISEASE COMBINÉ")
print("=" * 60)

# ============================================================================
# 1. CHARGEMENT
# ============================================================================
X_train = np.loadtxt('../data/X_train_preprocessed_full.csv', delimiter=',')
X_test  = np.loadtxt('../data/X_test_preprocessed_full.csv',  delimiter=',')
y_train = pd.read_csv('../data/y_train_full.csv')['target']
y_test  = pd.read_csv('../data/y_test_full.csv')['target']

print(f"\n  X_train : {X_train.shape}  |  X_test : {X_test.shape}")
print(f"  y_train : {y_train.shape}  |  y_test  : {y_test.shape}")
print(f"  Features après transformation : {X_train.shape[1]}")

# ============================================================================
# 2. MODÈLES BASELINE
# ============================================================================
print("\n" + "=" * 60)
print("MODÈLES BASELINE")
print("=" * 60)

baseline_models = {
    "Régression Logistique": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest":         RandomForestClassifier(random_state=42, n_estimators=100),
    "SVM":                   SVC(random_state=42, probability=True),
}

baseline_results = {}

for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cv_acc  = cross_val_score(model, X_train, y_train, cv=5,
                               scoring='accuracy').mean()
    baseline_results[name] = {
        'model': model, 'accuracy': acc, 'auc': auc, 'cv_accuracy': cv_acc
    }
    print(f"\n  {name}")
    print(f"    Accuracy test : {acc:.4f}  |  AUC : {auc:.4f}  |  CV-5 accuracy : {cv_acc:.4f}")

# ============================================================================
# 3. OPTIMISATION HYPERPARAMÈTRES — RANDOM FOREST
# ============================================================================
print("\n" + "=" * 60)
print("OPTIMISATION HYPERPARAMÈTRES — RANDOM FOREST (GridSearchCV 5-fold)")
print("=" * 60)

param_grid = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2],
}

print(f"  Grille : {param_grid}")
print("  Recherche en cours (patience)...")

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

print(f"\n  ✅ Meilleurs paramètres  : {grid_search.best_params_}")
print(f"  ✅ Meilleure accuracy CV : {grid_search.best_score_:.4f}")

# ============================================================================
# 4. ÉVALUATION DU MODÈLE OPTIMISÉ
# ============================================================================
print("\n" + "=" * 60)
print("ÉVALUATION SUR LE TEST SET — RANDOM FOREST OPTIMISÉ")
print("=" * 60)

best_rf = grid_search.best_estimator_
y_pred  = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

acc_best = accuracy_score(y_test, y_pred)
auc_best = roc_auc_score(y_test, y_proba)
cm       = confusion_matrix(y_test, y_pred)

print(f"\n  Accuracy : {acc_best:.4f}")
print(f"  AUC      : {auc_best:.4f}")
print(f"\n  Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=['Sain', 'Malade']))

print(f"  Matrice de confusion :")
print(f"    Vrais Négatifs (TN) : {cm[0,0]}  |  Faux Positifs (FP) : {cm[0,1]}")
print(f"    Faux Négatifs  (FN) : {cm[1,0]}  |  Vrais Positifs (TP) : {cm[1,1]}")

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print(f"\n  Sensibilité (recall malade) : {sensitivity:.4f}")
print(f"  Spécificité (recall sain)   : {specificity:.4f}")

# ============================================================================
# 5. SÉLECTION DU MODÈLE FINAL
# ============================================================================
print("\n" + "=" * 60)
print("SÉLECTION DU MODÈLE FINAL")
print("=" * 60)

acc_baseline_rf = baseline_results['Random Forest']['accuracy']

if acc_best >= acc_baseline_rf:
    final_model      = best_rf
    final_model_name = "Random Forest (optimisé)"
    final_accuracy   = acc_best
    final_auc        = auc_best
else:
    final_model      = baseline_results['Random Forest']['model']
    final_model_name = "Random Forest (baseline)"
    final_accuracy   = acc_baseline_rf
    final_auc        = baseline_results['Random Forest']['auc']

print(f"\n  🏆 Modèle sélectionné : {final_model_name}")
print(f"     Accuracy = {final_accuracy:.4f}  |  AUC = {final_auc:.4f}")

print("\n  Récapitulatif de tous les modèles :")
print(f"  {'Modèle':<30} {'Accuracy':<10} {'AUC':<10} {'CV Acc':<10}")
print("  " + "-" * 60)
for name, res in baseline_results.items():
    print(f"  {name:<30} {res['accuracy']:<10.4f} {res['auc']:<10.4f} {res['cv_accuracy']:.4f}")
print(f"  {'RF optimisé (GridSearch)':<30} {acc_best:<10.4f} {auc_best:<10.4f} {grid_search.best_score_:.4f}")

# ============================================================================
# 6. SAUVEGARDE DES MODÈLES ET MÉTRIQUES
# ============================================================================
print("\n" + "=" * 60)
print("SAUVEGARDE")
print("=" * 60)

joblib.dump(final_model, '../model/best_model_full.joblib')
# Sauvegarder aussi les baselines pour 4_evaluate.py
joblib.dump(baseline_results['Régression Logistique']['model'],
            '../model/best_model_lr.joblib')
joblib.dump(baseline_results['SVM']['model'],
            '../model/best_model_svm.joblib')
print("  ✅ Modèles sauvegardés (RF, LR, SVM)")

params = {
    'dataset':         'heart_uci_combined.csv',
    'n_patients_train': int(X_train.shape[0]),
    'n_patients_test':  int(X_test.shape[0]),
    'best_model':       final_model_name,
    'accuracy':         float(final_accuracy),
    'auc':              float(final_auc),
    'sensitivity':      float(sensitivity),
    'specificity':      float(specificity),
    'best_params':      grid_search.best_params_,
    'cv_best_score':    float(grid_search.best_score_),
}
with open('../model/model_info.json', 'w') as f:
    json.dump(params, f, indent=4)
print("  ✅ model_info.json sauvegardé")

print("\n" + "=" * 60)
print("✅ ENTRAÎNEMENT TERMINÉ")
print("=" * 60)