"""
4_evaluate.py - Évaluation complète des modèles après entraînement
- Courbes ROC comparatives (3 modèles)
- Matrices de confusion
- Importance des features (Random Forest)
- Métriques détaillées
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, 
    classification_report, ConfusionMatrixDisplay
)
import os

os.makedirs('../images', exist_ok=True)

print("="*60)
print("ÉVALUATION DES MODÈLES - POST ENTRAÎNEMENT")
print("="*60)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================

print("\n1️⃣ CHARGEMENT DES DONNÉES")

X_test = np.loadtxt('../data/X_test_preprocessed_full.csv', delimiter=',')
y_test = pd.read_csv('../data/y_test_full.csv')['target']

print(f"   X_test shape : {X_test.shape}")
print(f"   y_test shape : {y_test.shape}")

# ============================================================================
# 2. CHARGEMENT DES TROIS MODÈLES
# ============================================================================

print("\n2️⃣ CHARGEMENT DES MODÈLES")

# Modèle final (Random Forest)
model_rf = joblib.load('../model/best_model_full.joblib')
print(f"   ✅ Random Forest chargé : {type(model_rf).__name__}")

# Pour la comparaison, nous devons également charger les modèles baseline
# Si les modèles baseline ne sont pas sauvegardés, on les ré-entraîne rapidement

# Option 1 : Charger depuis des fichiers séparés (si sauvegardés)
try:
    model_lr = joblib.load('../model/best_model_lr.joblib')
    model_svm = joblib.load('../model/best_model_svm.joblib')
    print("   ✅ Modèles baseline chargés depuis les fichiers")
except:
    print("   ⚠️ Modèles baseline non trouvés, ré-entraînement rapide...")
    
    # Ré-entraîner les modèles baseline
    X_train = np.loadtxt('../data/X_train_preprocessed_full.csv', delimiter=',')
    y_train = pd.read_csv('../data/y_train_full.csv')['target']
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    model_lr = LogisticRegression(random_state=42, max_iter=1000)
    model_lr.fit(X_train, y_train)
    
    model_svm = SVC(random_state=42, probability=True)
    model_svm.fit(X_train, y_train)
    
    print("   ✅ Modèles baseline ré-entraînés")

# ============================================================================
# 3. PRÉDICTIONS ET PROBABILITÉS
# ============================================================================

print("\n3️⃣ PRÉDICTIONS ET PROBABILITÉS")

# Random Forest
y_pred_rf = model_rf.predict(X_test)
y_proba_rf = model_rf.predict_proba(X_test)[:, 1]
acc_rf = (y_pred_rf == y_test).mean()
auc_rf = roc_auc_score(y_test, y_proba_rf)

# Régression Logistique
y_pred_lr = model_lr.predict(X_test)
y_proba_lr = model_lr.predict_proba(X_test)[:, 1]
acc_lr = (y_pred_lr == y_test).mean()
auc_lr = roc_auc_score(y_test, y_proba_lr)

# SVM
y_pred_svm = model_svm.predict(X_test)
y_proba_svm = model_svm.predict_proba(X_test)[:, 1]
acc_svm = (y_pred_svm == y_test).mean()
auc_svm = roc_auc_score(y_test, y_proba_svm)

print(f"\n   📊 Résultats sur le test set :")
print(f"   Régression Logistique : Accuracy = {acc_lr:.4f}, AUC = {auc_lr:.4f}")
print(f"   SVM                   : Accuracy = {acc_svm:.4f}, AUC = {auc_svm:.4f}")
print(f"   Random Forest         : Accuracy = {acc_rf:.4f}, AUC = {auc_rf:.4f}")

# ============================================================================
# 4. MATRICES DE CONFUSION (les trois)
# ============================================================================

print("\n4️⃣ MATRICES DE CONFUSION")

models_confusion = [
    ("Régression Logistique", model_lr, y_pred_lr),
    ("SVM", model_svm, y_pred_svm),
    ("Random Forest", model_rf, y_pred_rf)
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, _, y_pred) in enumerate(models_confusion):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sain", "Malade"])
    disp.plot(ax=axes[idx], cmap='Blues', values_format='d')
    axes[idx].set_title(f'Matrice de confusion - {name}', fontsize=12)

plt.tight_layout()
plt.savefig('../images/matrices_confusion.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : images/matrices_confusion.png")

# ============================================================================
# 5. COURBE ROC COMPARATIVE (LES TROIS MODÈLES)
# ============================================================================

print("\n5️⃣ COURBE ROC COMPARATIVE")

# Calcul des courbes ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

# Création du graphique
plt.figure(figsize=(9, 7))

# Courbe pour la Régression Logistique
plt.plot(fpr_lr, tpr_lr, linewidth=2, 
         label=f'Régression Logistique (AUC = {auc_lr:.4f})', 
         color='#3498db', linestyle='-')

# Courbe pour SVM
plt.plot(fpr_svm, tpr_svm, linewidth=2, 
         label=f'SVM (AUC = {auc_svm:.4f})', 
         color='#e67e22', linestyle='--')

# Courbe pour Random Forest
plt.plot(fpr_rf, tpr_rf, linewidth=2, 
         label=f'Random Forest (AUC = {auc_rf:.4f})', 
         color='#2ecc71', linestyle='-')

# Ligne aléatoire (diagonale)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Modèle aléatoire (AUC = 0.5)')

# Configuration du graphique
plt.xlabel('Taux de faux positifs (1 - Spécificité)', fontsize=12)
plt.ylabel('Taux de vrais positifs (Sensibilité)', fontsize=12)
plt.title('Courbes ROC comparatives des trois modèles', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../images/courbe_roc_comparative.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : images/courbe_roc_comparative.png")

# ============================================================================
# 6. IMPORTANCE DES FEATURES (Random Forest)
# ============================================================================

if hasattr(model_rf, 'feature_importances_'):
    print("\n6️⃣ IMPORTANCE DES FEATURES")
    
    # Charger les noms des features
    try:
        with open('../model/feature_names_full.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
    except:
        feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
    
    # S'assurer que le nombre de noms correspond
    if len(feature_names) != len(model_rf.feature_importances_):
        feature_names = [f'Feature_{i}' for i in range(len(model_rf.feature_importances_))]
    
    # Trier par importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Affichage console
    print("\n   Top 10 des features les plus importantes :")
    for i, row in importance_df.head(10).iterrows():
        print(f"      {row['feature']:<25} : {row['importance']:.4f}")
    
    # Figure
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Importance des features - Random Forest', fontsize=14)
    plt.tight_layout()
    plt.savefig('../images/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✅ Figure sauvegardée : images/feature_importance.png")

# ============================================================================
# 7. RAPPORT DE CLASSIFICATION DÉTAILLÉ
# ============================================================================

print("\n7️⃣ RAPPORT DE CLASSIFICATION")

print("\n--- Régression Logistique ---")
print(classification_report(y_test, y_pred_lr, target_names=["Sain", "Malade"]))

print("\n--- SVM ---")
print(classification_report(y_test, y_pred_svm, target_names=["Sain", "Malade"]))

print("\n--- Random Forest ---")
print(classification_report(y_test, y_pred_rf, target_names=["Sain", "Malade"]))

# ============================================================================
# 8. RÉCAPITULATIF DES MÉTRIQUES
# ============================================================================

print("\n" + "="*60)
print("RÉCAPITULATIF DES MÉTRIQUES")
print("="*60)

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy, sensitivity, specificity, precision, f1

print("\n┌─────────────────────┬──────────┬────────────┬────────────┬───────────┬───────────┐")
print("│ Modèle              │ Accuracy │ Sensibilité│ Spécificité│ Précision │ F1-score  │")
print("├─────────────────────┼──────────┼────────────┼────────────┼───────────┼───────────┤")

acc_lr, sens_lr, spec_lr, prec_lr, f1_lr = compute_metrics(y_test, y_pred_lr)
print(f"│ Régression Logistique │ {acc_lr:.4f}   │ {sens_lr:.4f}     │ {spec_lr:.4f}     │ {prec_lr:.4f}    │ {f1_lr:.4f}    │")

acc_svm, sens_svm, spec_svm, prec_svm, f1_svm = compute_metrics(y_test, y_pred_svm)
print(f"│ SVM                 │ {acc_svm:.4f}   │ {sens_svm:.4f}     │ {spec_svm:.4f}     │ {prec_svm:.4f}    │ {f1_svm:.4f}    │")

acc_rf, sens_rf, spec_rf, prec_rf, f1_rf = compute_metrics(y_test, y_pred_rf)
print(f"│ Random Forest       │ {acc_rf:.4f}   │ {sens_rf:.4f}     │ {spec_rf:.4f}     │ {prec_rf:.4f}    │ {f1_rf:.4f}    │")
print("└─────────────────────┴──────────┴────────────┴────────────┴───────────┴───────────┘")

print("\n" + "="*60)
print("✅ ÉVALUATION TERMINÉE")
print("="*60)