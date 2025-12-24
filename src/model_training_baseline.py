# model_training_baseline.py
""" 
Les modèles de base permettent d’établir une référence avant optimisation.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Charger les données préprocessées
X_train = np.loadtxt("../data/X_train_preprocessed.csv", delimiter=",")
X_test = np.loadtxt("../data/X_test_preprocessed.csv", delimiter=",")
y_train = pd.read_csv("../data/y_train.csv").squeeze() # .squeeze() pour convertir en Series
y_test = pd.read_csv("../data/y_test.csv").squeeze()

print("✅ Données chargées!")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# 2. Initialisation des modèles à tester
models = {
    "Régression Logistique": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

# 3. Boucle d'entraînement et d'évaluation
results = {}

for name, model in models.items():
    print(f"\n--- Entraînement du modèle : {name} ---")
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédiction sur le jeu de test
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Sain", "Malade"])
    
    # Stockage des résultats
    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "report": report,
        "y_pred": y_pred
    }
    
    # Affichage des résultats
    print(f"Accuracy: {accuracy:.4f}")
    print("Rapport de classification:\n", report)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sain", "Malade"])
    disp.plot(cmap='Blues')
    plt.title(f"Matrice de Confusion - {name}")
    plt.show()

# 4. Synthèse des performances
print("\n" + "="*50)
print("SYNTHÈSE DES PERFORMANCES")
print("="*50)

for name, res in results.items():
    print(f"{name}: Accuracy = {res['accuracy']:.4f}")

# 5. Identifier le meilleur modèle
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']
print(f"\n🎯 Meilleur modèle: {best_model_name} avec une accuracy de {best_accuracy:.4f}")