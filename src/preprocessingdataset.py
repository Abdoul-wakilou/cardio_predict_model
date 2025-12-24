# preprocessingdataset.py
"""Prétraitement du dataset Cleveland Heart Disease pour l'analyse et la modélisation.

Ce script assure le nettoyage initial du jeu de données et la transformation de la variable 
cible afin de rendre les données exploitables pour les étapes de modélisation.
"""
# Importation de la bibliothèque pandas pour la manipulation des données
import pandas as pd

# Définition des noms de colonnes selon la documentation UCI
noms_colonnes = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Chargement du dataset avec gestion des valeurs manquantes représentées par '?'
chemin_fichier = "../data/processed.cleveland.data"
dataset = pd.read_csv(
    chemin_fichier, 
    names=noms_colonnes, 
    header=None, 
    na_values='?'
)

# Suppression des lignes contenant des valeurs manquantes
dataset_propre = dataset.dropna()

# 🔥 TRANSFORMATION CRUCIALE : Conversion de la variable cible en binaire
# Les valeurs 1, 2, 3, 4 (présence de maladie) deviennent toutes 1
print("Valeurs uniques de 'target' AVANT transformation :", sorted(dataset_propre['target'].unique()))
dataset_propre['target'] = dataset_propre['target'].apply(lambda x: 0 if x == 0 else 1)
print("Valeurs uniques de 'target' APRES transformation :", sorted(dataset_propre['target'].unique()))

# Vérification des dimensions du dataset après nettoyage
print("Dimensions initiales :", dataset.shape)
print("Dimensions après nettoyage :", dataset_propre.shape)
print("\nValeurs manquantes après nettoyage :")
print(dataset_propre.isna().sum())

# Sauvegarde du dataset nettoyé ET transformé pour les étapes suivantes
dataset_propre.to_csv("../data/cleveland_clean.csv", index=False)

print("\n✅ Dataset nettoyé et transformé sauvegardé sous '../data/cleveland_clean.csv'")