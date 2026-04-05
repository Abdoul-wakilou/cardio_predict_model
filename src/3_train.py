"""
3_train.py - Entraînement et optimisation du modèle
- Chargement des données prétraitées
- Entraînement de 3 modèles (Régression Logistique, Random Forest, SVM)
- Optimisation des hyperparamètres (GridSearchCV)
- Sélection du meilleur modèle
- Sauvegarde du modèle final et du préprocesseur
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
print("ENTRAÎNEMENT - DATASET COMPLET (1025 patients)")
print("="*60)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES PRÉTRAITÉES
# ============================================================================

print("\n1️⃣ CHARGEMENT DES DONNÉES")

X_train = np.loadtxt('../data/X_train_preprocessed_full.csv', delimiter=',')
X_test = np.loadtxt('../data/X_test_preprocessed_full.csv', delimiter=',')
y_train = pd.read_csv('../data/y_train_full.csv')['target']
y_test = pd.read_csv('../data/y_test_full.csv')['target']

print(f"   X_train : {X_train.shape}")
print(f"   X_test  : {X_test.shape}")
print(f"   y_train : {y_train.shape}")
print(f"   y_test  : {y_test.shape}")

# ============================================================================
# 2. MODÈLES BASELINE (paramètres par défaut)
# ============================================================================

print("\n2️⃣ MODÈLES BASELINE (paramètres par défaut)")

baseline_models = {
    "Régression Logistique": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "SVM": SVC(random_state=42, probability=True)
}

baseline_results = {}

for name, model in baseline_models.items():
    print(f"\n   🔹 {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    
    baseline_results[name] = {
        "model": model,
        "accuracy": accuracy,
        "auc": auc
    }
    
    print(f"      Accuracy : {accuracy:.4f}")
    if auc:
        print(f"      AUC      : {auc:.4f}")

# ============================================================================
# 3. OPTIMISATION DU MEILLEUR MODÈLE (Random Forest)
# ============================================================================

print("\n3️⃣ OPTIMISATION DES HYPERPARAMÈTRES (Random Forest)")

# Grille d'hyperparamètres
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("   🔍 Grille de recherche :")
for k, v in param_grid.items():
    print(f"      - {k} : {v}")

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, 
    param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1,
    verbose=0
)

print("\n   🔄 Recherche en cours...")
grid_search.fit(X_train, y_train)

print(f"\n   ✅ Meilleurs paramètres trouvés :")
for param, value in grid_search.best_params_.items():
    print(f"      {param} : {value}")

print(f"\n   ✅ Meilleure accuracy (CV) : {grid_search.best_score_:.4f}")

# ============================================================================
# 4. ÉVALUATION DU MODÈLE OPTIMISÉ
# ============================================================================

print("\n4️⃣ ÉVALUATION SUR LE TEST SET")

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

accuracy_best = accuracy_score(y_test, y_pred)
auc_best = roc_auc_score(y_test, y_proba)

print(f"\n   📊 Performance du modèle optimisé :")
print(f"      Accuracy : {accuracy_best:.4f}")
print(f"      AUC      : {auc_best:.4f}")

print(f"\n   📊 Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=["Sain", "Malade"]))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print(f"\n   📊 Matrice de confusion :")
print(f"      Vrais Negatifs : {cm[0,0]}, Faux Positifs : {cm[0,1]}")
print(f"      Faux Negatifs  : {cm[1,0]}, Vrais Positifs : {cm[1,1]}")

# ============================================================================
# 5. COMPARAISON AVEC LA LITTÉRATURE
# ============================================================================

print("\n" + "="*60)
print("COMPARAISON AVEC LA LITTÉRATURE")
print("="*60)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                    COMPARAISON DES PERFORMANCES                 │
├─────────────────────────────────────────────────────────────────┤
│ Modèle                    │ Accuracy  │ AUC      │ Écart       │
├─────────────────────────────────────────────────────────────────┤
│ Al-Waeli et al. (2025)    │ 97,66%    │ Non dispo │ Référence   │
│ Notre modèle (défaut)     │ {baseline_results['Random Forest']['accuracy']*100:.2f}%      │ {baseline_results['Random Forest']['auc']:.4f}    │ -{(97.66 - baseline_results['Random Forest']['accuracy']*100):.2f}% │
│ Notre modèle (optimisé)   │ {accuracy_best*100:.2f}%      │ {auc_best:.4f}    │ +{(accuracy_best*100 - baseline_results['Random Forest']['accuracy']*100):.2f}% │
└─────────────────────────────────────────────────────────────────┘
""")

# ============================================================================
# 6. SÉLECTION DU MODÈLE FINAL
# ============================================================================

print("\n5️⃣ SÉLECTION DU MODÈLE FINAL")

# Comparer les performances baseline vs optimisé
if accuracy_best > baseline_results['Random Forest']['accuracy']:
    final_model = best_rf
    final_model_name = "Random Forest (optimisé)"
    final_accuracy = accuracy_best
    print(f"   🏆 Modèle sélectionné : Random Forest optimisé (accuracy = {accuracy_best:.4f})")
else:
    final_model = baseline_models['Random Forest']
    final_model_name = "Random Forest (baseline)"
    final_accuracy = baseline_results['Random Forest']['accuracy']
    print(f"   🏆 Modèle sélectionné : Random Forest baseline (accuracy = {final_accuracy:.4f})")

# ============================================================================
# 7. SAUVEGARDE
# ============================================================================

print("\n6️⃣ SAUVEGARDE POUR FLASK")

# Charger le préprocesseur original
preprocessor = joblib.load('../model/preprocessor_full.joblib')

# Sauvegarder le modèle final
joblib.dump(final_model, '../model/best_model_full.joblib')
print(f"   ✅ Modèle sauvegardé : model/best_model_full.joblib")

# Sauvegarder le préprocesseur (déjà fait, mais on vérifie)
joblib.dump(preprocessor, '../model/preprocessor_full.joblib')
print(f"   ✅ Préprocesseur sauvegardé : model/preprocessor_full.joblib")

# Sauvegarder les paramètres pour la documentation
params = {
    'best_model': final_model_name,
    'accuracy': float(final_accuracy),
    'best_params': grid_search.best_params_ if final_model == best_rf else 'default',
    'cv_best_score': float(grid_search.best_score_) if final_model == best_rf else None
}
with open('../model/model_info.json', 'w') as f:
    json.dump(params, f, indent=4)
print(f"   ✅ Informations sauvegardées : model/model_info.json")

# ============================================================================
# 8. RÉSULTATS FINAUX
# ============================================================================

print("\n" + "="*60)
print("RÉSULTATS FINAUX")
print("="*60)

print(f"""
📊 PERFORMANCES FINALES :

Modèle sélectionné : {final_model_name}
Accuracy : {final_accuracy:.4f} ({final_accuracy*100:.2f}%)
AUC      : {auc_best:.4f}

📁 Fichiers sauvegardés :
   - model/best_model_full.joblib
   - model/preprocessor_full.joblib
   - model/model_info.json
""")

print("="*60)
print("✅ ENTRAÎNEMENT TERMINÉ")
print("="*60)