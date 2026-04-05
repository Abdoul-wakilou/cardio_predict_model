"""
1_preprocessing.py - Prétraitement complet du dataset (1025 patients)
- Chargement des données brutes
- Nettoyage (valeurs aberrantes, valeurs manquantes)
- Transformation de la cible en binaire
- Standardisation et encodage
- Séparation train/test
- Sauvegarde du préprocesseur et des données transformées
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

# Créer les dossiers nécessaires
os.makedirs('../data', exist_ok=True)
os.makedirs('../model', exist_ok=True)

print("="*60)
print("PRÉTRAITEMENT - DATASET COMPLET (1025 patients)")
print("="*60)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES BRUTES
# ============================================================================

df = pd.read_csv('../data/heart_disease_full.csv')
print(f"\n📊 Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"Colonnes : {list(df.columns)}")

# ============================================================================
# 2. NETTOYAGE
# ============================================================================

print("\n" + "="*60)
print("NETTOYAGE")
print("="*60)

# 2.1 Vérification des valeurs manquantes
print("\n🔍 Valeurs manquantes initiales :")
print(df.isnull().sum())

# 2.2 Correction des valeurs aberrantes (ca : normalement 0-3, thal : normalement 1-3)
if 'ca' in df.columns:
    n_ca_4 = (df['ca'] == 4).sum()
    df.loc[df['ca'] == 4, 'ca'] = np.nan
    print(f"\n✅ ca : {n_ca_4} valeur(s) '4' remplacée(s) par NaN")

if 'thal' in df.columns:
    n_thal_0 = (df['thal'] == 0).sum()
    df.loc[df['thal'] == 0, 'thal'] = np.nan
    print(f"✅ thal : {n_thal_0} valeur(s) '0' remplacée(s) par NaN")

# 2.3 Imputation par la médiane (moins de 1% de valeurs manquantes)
print("\n📊 Imputation par la médiane :")
for col in ['ca', 'thal']:
    if col in df.columns:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   - {col} : médiane = {median_val}")

# 2.4 Vérification finale
print("\n✅ Valeurs manquantes après nettoyage :")
print(df.isnull().sum())

# ============================================================================
# 3. TRANSFORMATION DE LA CIBLE
# ============================================================================

print("\n" + "="*60)
print("TRANSFORMATION DE LA CIBLE")
print("="*60)

print(f"Valeurs originales : {sorted(df['target'].unique())}")

# Transformation binaire : 0 = sain, 1 = malade
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

print(f"Valeurs après transformation : {sorted(df['target'].unique())}")
print(f"   - Sain (0) : {(df['target'] == 0).sum()} patients ({(df['target'] == 0).sum()/len(df)*100:.1f}%)")
print(f"   - Malade (1) : {(df['target'] == 1).sum()} patients ({(df['target'] == 1).sum()/len(df)*100:.1f}%)")

# ============================================================================
# 4. DÉFINITION DES FEATURES
# ============================================================================

print("\n" + "="*60)
print("DÉFINITION DES FEATURES")
print("="*60)

numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['cp', 'restecg', 'slope', 'thal']
binary_features = ['sex', 'fbs', 'exang']

# Vérifier la présence des colonnes
numeric_features = [f for f in numeric_features if f in df.columns]
categorical_features = [f for f in categorical_features if f in df.columns]
binary_features = [f for f in binary_features if f in df.columns]

print(f"\n✅ Features numériques ({len(numeric_features)}) : {numeric_features}")
print(f"✅ Features catégorielles ({len(categorical_features)}) : {categorical_features}")
print(f"✅ Features binaires ({len(binary_features)}) : {binary_features}")

# ============================================================================
# 5. SÉPARATION X / y
# ============================================================================

X = df.drop('target', axis=1)
y = df['target']

print(f"\n📊 X shape : {X.shape}")
print(f"📊 y shape : {y.shape}")

# ============================================================================
# 6. CRÉATION DU PRÉPROCESSEUR
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
# 7. SÉPARATION TRAIN/TEST (70/30 stratifié)
# ============================================================================

print("\n" + "="*60)
print("SÉPARATION TRAIN/TEST")
print("="*60)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"📊 Split effectué (70% train, 30% test) :")
print(f"   X_train_raw : {X_train_raw.shape}")
print(f"   X_test_raw  : {X_test_raw.shape}")
print(f"   y_train     : {y_train.shape}")
print(f"   y_test      : {y_test.shape}")

# ============================================================================
# 8. APPLICATION DU PRÉTRAITEMENT
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
# 9. SAUVEGARDE
# ============================================================================

print("\n" + "="*60)
print("SAUVEGARDE")
print("="*60)

# Sauvegarde des données prétraitées
np.savetxt('../data/X_train_preprocessed_full.csv', X_train_processed, delimiter=',')
np.savetxt('../data/X_test_preprocessed_full.csv', X_test_processed, delimiter=',')
y_train.to_csv('../data/y_train_full.csv', index=False)
y_test.to_csv('../data/y_test_full.csv', index=False)
print("✅ Données prétraitées sauvegardées")

# Sauvegarde du préprocesseur
joblib.dump(preprocessor, '../model/preprocessor_full.joblib')
print("✅ Préprocesseur sauvegardé : model/preprocessor_full.joblib")

# Sauvegarde des noms des features (pour debug)
if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_names) + binary_features
    with open('../model/feature_names_full.txt', 'w') as f:
        for name in all_feature_names:
            f.write(f"{name}\n")
    print(f"✅ {len(all_feature_names)} noms de features sauvegardés")

# Sauvegarde du dataset nettoyé
df.to_csv('../data/heart_clean_full.csv', index=False)
print("✅ Dataset nettoyé sauvegardé : data/heart_clean_full.csv")

print("\n" + "="*60)
print("✅ PRÉTRAITEMENT TERMINÉ")
print("="*60)