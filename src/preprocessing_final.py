# preprocessing_final.py
"""
Ce script implémente le prétraitement final des données, incluant la standardisation,
l’encodage et la séparation train/test, garantissant un apprentissage équitable des modèles.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Charger les données nettoyées (après suppression des NaN et transformation de la cible)
df = pd.read_csv('../data/cleveland_clean.csv')

# 1. Définir les features (X) et la target (y)
X = df.drop('target', axis=1)
y = df['target']

# 2. Lister les colonnes par type
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['cp', 'restecg', 'slope', 'thal'] # Variables à one-hot encode
binary_features = ['sex', 'fbs', 'exang'] # Variables déjà binaires (0/1), à garder telles quelles

# 3. Définir le préprocesseur
# One-Hot Encoding pour les variables catégorielles, StandardScaler pour les numériques, 'passthrough' pour les binaires.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features), # drop='first' pour éviter la multicolinéarité
        ('bin', 'passthrough', binary_features)
    ])

# 4. Séparation Stratifiée Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Application du préprocessing
# Fit sur le train, transform sur le train et le test
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# 6. (Optionnel) Récupérer les noms des features après One-Hot Encoding pour analyse future
# Cela est utile pour comprendre l'importance des features plus tard
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names) + binary_features

# 7. Sauvegarde des jeux de données préprocessés (Optionnel, pour usage futur)
import numpy as np
np.savetxt("../data/X_train_preprocessed.csv", X_train_preprocessed, delimiter=",")
np.savetxt("../data/X_test_preprocessed.csv", X_test_preprocessed, delimiter=",")
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)

print("Préprocessing terminé !")
print(f"Shape X_train : {X_train_preprocessed.shape}")
print(f"Shape X_test : {X_test_preprocessed.shape}")

print(f"Valeurs uniques dans y_train: {y_train.unique()}")
print(f"Valeurs uniques dans y_test: {y_test.unique()}")