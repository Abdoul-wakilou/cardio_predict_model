"""
1_preprocessing.py - Prétraitement
- Le dataset est déjà propre (pas de valeurs manquantes, target binaire)
- Standardisation des variables numériques
- OneHotEncoder pour variables catégorielles
- Séparation train/test
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

os.makedirs('../data', exist_ok=True)
os.makedirs('../model', exist_ok=True)

print("="*60)
print("PRÉTRAITEMENT - DATASET (1025 patients)")
print("="*60)

# ============================================================================
# 1. CHARGEMENT
# ============================================================================
df = pd.read_csv('../data/heart.csv')
print(f"\n Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ============================================================================
# 2. VÉRIFICATION (aucune correction nécessaire)
# ============================================================================
print("\n" + "="*60)
print("VÉRIFICATION")
print("="*60)

print(f"✅ Valeurs manquantes : {df.isnull().sum().sum()}")
print(f"✅ Target unique values : {sorted(df['target'].unique())}")
print(f"   - Sain (0) : {(df['target'] == 0).sum()} patients ({(df['target'] == 0).sum()/len(df)*100:.1f}%)")
print(f"   - Malade (1) : {(df['target'] == 1).sum()} patients ({(df['target'] == 1).sum()/len(df)*100:.1f}%)")

# ============================================================================
# 3. DÉFINITION DES FEATURES
# ============================================================================
print("\n" + "="*60)
print("DÉFINITION DES FEATURES")
print("="*60)

numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['cp', 'restecg', 'slope', 'thal']
binary_features = ['sex', 'fbs', 'exang']

print(f"✅ Features numériques ({len(numeric_features)}) : {numeric_features}")
print(f"✅ Features catégorielles ({len(categorical_features)}) : {categorical_features}")
print(f"✅ Features binaires ({len(binary_features)}) : {binary_features}")

# ============================================================================
# 4. SÉPARATION X / y
# ============================================================================
X = df.drop('target', axis=1)
y = df['target']
print(f"\n X shape : {X.shape}, y shape : {y.shape}")

# ============================================================================
# 5. CRÉATION DU PRÉPROCESSEUR
# ============================================================================
print("\n" + "="*60)
print("CRÉATION DU PRÉPROCESSEUR")
print("="*60)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
        ('bin', 'passthrough', binary_features)
    ])

print("✅ Préprocesseur créé (StandardScaler + OneHotEncoder)")

# ============================================================================
# 6. SÉPARATION TRAIN/TEST
# ============================================================================
print("\n" + "="*60)
print("SÉPARATION TRAIN/TEST")
print("="*60)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f" Split effectué (70% train, 30% test) :")
print(f"   X_train_raw : {X_train_raw.shape}")
print(f"   X_test_raw  : {X_test_raw.shape}")

# ============================================================================
# 7. APPLICATION DU PRÉTRAITEMENT
# ============================================================================
print("\n" + "="*60)
print("APPLICATION DU PRÉTRAITEMENT")
print("="*60)

X_train_processed = preprocessor.fit_transform(X_train_raw)
X_test_processed = preprocessor.transform(X_test_raw)

print(f"✅ Prétraitement appliqué :")
print(f"   X_train_processed : {X_train_processed.shape}")
print(f"   X_test_processed  : {X_test_processed.shape}")

# ============================================================================
# 8. SAUVEGARDE
# ============================================================================
print("\n" + "="*60)
print("SAUVEGARDE")
print("="*60)

np.savetxt('../data/X_train_preprocessed_full.csv', X_train_processed, delimiter=',')
np.savetxt('../data/X_test_preprocessed_full.csv', X_test_processed, delimiter=',')
y_train.to_csv('../data/y_train_full.csv', index=False)
y_test.to_csv('../data/y_test_full.csv', index=False)
print("✅ Données prétraitées sauvegardées")

joblib.dump(preprocessor, '../model/preprocessor_full.joblib')
print("✅ Préprocesseur sauvegardé : model/preprocessor_full.joblib")

if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_names) + binary_features
    with open('../model/feature_names_full.txt', 'w') as f:
        for name in all_feature_names:
            f.write(f"{name}\n")
    print(f"✅ {len(all_feature_names)} noms de features sauvegardés")

df.to_csv('../data/heart_clean_full.csv', index=False)
print("✅ Dataset nettoyé sauvegardé : data/heart_clean_full.csv")

print("\n" + "="*60)
print("✅ PRÉTRAITEMENT TERMINÉ")
print("="*60)