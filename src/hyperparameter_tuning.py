# hyperparameter_tuning.py
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score, classification_report

# Charger les données
X_train = np.loadtxt("../data/X_train_preprocessed.csv", delimiter=",")
y_train = pd.read_csv("../data/y_train.csv").squeeze()

# 1. OPTIMISATION RÉGRESSION LOGISTIQUE (à ajouter en premier)
print("=== OPTIMISATION RÉGRESSION LOGISTIQUE ===")
log_reg = LogisticRegression(random_state=42, max_iter=1000)

# Définition de l'espace des hyperparamètres
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Paramètre de régularisation (inverse)
    'penalty': ['l1', 'l2', 'elasticnet', None],  # Type de régularisation
    'solver': ['liblinear', 'saga'],  # Algorithmes d'optimisation
    'l1_ratio': [0, 0.25, 0.5, 0.75, 1]  # Pour elasticnet
}

# Note: Tous les solveurs ne supportent pas toutes les pénalités
# On va adapter la grille pour éviter les combinaisons incompatibles
grid_search_lr = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid_lr,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search_lr.fit(X_train, y_train)

print(f"\nMeilleurs paramètres pour Regression Logistique: {grid_search_lr.best_params_}")
print(f"Meilleure score CV (accuracy): {grid_search_lr.best_score_:.4f}")

# 2. OPTIMISATION RANDOM FOREST
print("\n=== OPTIMISATION RANDOM FOREST ===")
rf = RandomForestClassifier(random_state=42)

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search_rf.fit(X_train, y_train)
print(f"\nMeilleurs paramètres pour Random Forest: {grid_search_rf.best_params_}")
print(f"Meilleure score CV (accuracy): {grid_search_rf.best_score_:.4f}")

# 3. OPTIMISATION SVM
print("\n=== OPTIMISATION SVM ===")
svm = SVC(random_state=42)

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

grid_search_svm = GridSearchCV(
    estimator=svm,
    param_grid=param_grid_svm,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search_svm.fit(X_train, y_train)
print(f"\nMeilleurs paramètres pour SVM: {grid_search_svm.best_params_}")
print(f"Meilleure score CV (accuracy): {grid_search_svm.best_score_:.4f}")

# 4. ÉVALUATION FINALE SUR LE TEST SET
print("\n=== ÉVALUATION FINALE SUR LE JEU DE TEST ===")
X_test = np.loadtxt("../data/X_test_preprocessed.csv", delimiter=",")
y_test = pd.read_csv("../data/y_test.csv").squeeze()

# Meilleur modèle Regression Logistique
best_lr = grid_search_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test)
print("Regression Logistique Optimisée - Rapport de Test:\n", classification_report(y_test, y_pred_lr))

# Meilleur modèle Random Forest
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("Random Forest Optimisé - Rapport de Test:\n", classification_report(y_test, y_pred_rf))

# Meilleur modèle SVM
best_svm = grid_search_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test)
print("SVM Optimisé - Rapport de Test:\n", classification_report(y_test, y_pred_svm))

# Comparaison des performances
print("\n=== COMPARAISON FINALE ===")
print(f"Regression Logistique Optimisée - Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Random Forest Optimisé - Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"SVM Optimisé - Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")