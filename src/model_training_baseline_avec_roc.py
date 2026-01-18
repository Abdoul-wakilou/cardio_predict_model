# model_training_baseline_avec_roc.py
""" 
Modifié pour inclure la courbe ROC et le calcul de l'AUC
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score, auc)
import matplotlib.pyplot as plt

# 1. Charger les données préprocessées
X_train = np.loadtxt("../data/X_train_preprocessed.csv", delimiter=",")
X_test = np.loadtxt("../data/X_test_preprocessed.csv", delimiter=",")
y_train = pd.read_csv("../data/y_train.csv").squeeze()
y_test = pd.read_csv("../data/y_test.csv").squeeze()

print("Données chargées!")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# 2. Initialisation des modèles à tester
models = {
    "Régression Logistique": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42, probability=True)  # IMPORTANT: probability=True pour ROC
}

# 3. Boucle d'entraînement et d'évaluation
results = {}
plt.figure(figsize=(12, 5))

# Sous-figure pour les matrices de confusion
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, model) in enumerate(models.items()):
    print(f"\n--- Entraînement du modèle : {name} ---")
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédiction sur le jeu de test
    y_pred = model.predict(X_test)
    
    # Prédiction des probabilités (nécessaire pour ROC)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Stockage des résultats
    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "auc": auc_score,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }
    
    # Affichage des résultats
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print("Rapport de classification:")
    print(classification_report(y_test, y_pred, target_names=["Sain", "Malade"]))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sain", "Malade"])
    disp.plot(ax=axes[idx], cmap='Blues')
    axes[idx].set_title(f"Matrice - {name}")
    
    # Courbe ROC (on prépare pour le graphique séparé)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# 4. Afficher les matrices de confusion
plt.tight_layout()
plt.savefig('../figures/matrices_confusion.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Courbe ROC comparative
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {res["auc"]:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.5)')
plt.xlabel('Taux de Faux Positifs (1 - Spécificité)')
plt.ylabel('Taux de Vrais Positifs (Sensibilité)')
plt.title('Courbes ROC - Comparaison des modèles')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('../figures/courbe_roc.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Synthèse des performances
print("\n" + "="*60)
print("SYNTHÈSE DES PERFORMANCES")
print("="*60)

for name, res in results.items():
    print(f"{name}: Accuracy = {res['accuracy']:.4f}, AUC = {res['auc']:.4f}")

# 7. Identifier le meilleur modèle (par AUC ou Accuracy)
best_model_name_auc = max(results, key=lambda x: results[x]['auc'])
best_auc = results[best_model_name_auc]['auc']

best_model_name_acc = max(results, key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name_acc]['accuracy']

print(f"\n Meilleur modèle par AUC: {best_model_name_auc} avec AUC = {best_auc:.4f}")
print(f" Meilleur modèle par Accuracy: {best_model_name_acc} avec Accuracy = {best_accuracy:.4f}")

# 8. Sauvegarder les résultats détaillés pour la régression logistique
print(f"\n Détails pour la Régression Logistique:")
lr_results = results.get("Régression Logistique")
if lr_results:
    cm = confusion_matrix(y_test, lr_results['y_pred'])
    print(f"Matrice de confusion: [[{cm[0,0]} {cm[0,1]}]")
    print(f"                     [{cm[1,0]} {cm[1,1]}]]")
    print(f"Sensibilité (Rappel): {cm[1,1]/(cm[1,0]+cm[1,1]):.3f}")
    print(f"Spécificité: {cm[0,0]/(cm[0,0]+cm[0,1]):.3f}")