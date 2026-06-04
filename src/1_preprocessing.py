"""
1_preprocessing.py — Prétraitement du dataset UCI Heart Disease combiné
========================================================================
Dataset : heart_uci_combined.csv (834 patients, produit par 0_build_dataset.py)

Encodage en vigueur (convention 0-based après 0_build_dataset.py) :
  cp    : 0=typique, 1=atypique, 2=non-angineuse, 3=asymptomatique
  slope : 0=ascendante, 1=plate, 2=descendante
  thal  : 0=normal, 1=défaut fixe, 2=défaut réversible
  target: 0=sain, 1=malade  (CORRECT — vérifié médicalement)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

os.makedirs('../data',  exist_ok=True)
os.makedirs('../model', exist_ok=True)

print("=" * 60)
print("PRÉTRAITEMENT — UCI HEART DISEASE COMBINÉ (834 patients)")
print("=" * 60)

# ============================================================================
# 1. CHARGEMENT
# ============================================================================
df = pd.read_csv('../data/heart_uci_combined.csv')
print(f"\n✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ============================================================================
# 2. VÉRIFICATION RAPIDE
# ============================================================================
print("\n" + "=" * 60)
print("VÉRIFICATION")
print("=" * 60)

print(f"  Valeurs manquantes  : {df.isnull().sum().sum()}")
print(f"  Valeurs target      : {sorted(df['target'].unique())}  (attendu [0, 1])")
print(f"  Sain   (0) : {(df['target'] == 0).sum()} ({(df['target'] == 0).mean()*100:.1f}%)")
print(f"  Malade (1) : {(df['target'] == 1).sum()} ({(df['target'] == 1).mean()*100:.1f}%)")

# Sanity check médical rapide
g0_thal = df[df['target'] == 0]['thalach'].mean()
g1_thal = df[df['target'] == 1]['thalach'].mean()
ok = "✅" if g0_thal > g1_thal else "❌"
print(f"\n  {ok} Sanity check thalach : sain={g0_thal:.1f} bpm > malade={g1_thal:.1f} bpm")

g0_age = df[df['target'] == 0]['age'].mean()
g1_age = df[df['target'] == 1]['age'].mean()
ok = "✅" if g1_age > g0_age else "❌"
print(f"  {ok} Sanity check age    : malade={g1_age:.1f} ans > sain={g0_age:.1f} ans")

# ============================================================================
# 3. DÉFINITION DES FEATURES
# ============================================================================
print("\n" + "=" * 60)
print("DÉFINITION DES FEATURES")
print("=" * 60)

# Numériques : standardisation z-score
numeric_features     = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Catégorielles multi-classes : OneHotEncoding avec drop='first'
# cp (4 modalités : 0-3), restecg (3 : 0-2), slope (3 : 0-2), thal (3 : 0-2)
categorical_features = ['cp', 'restecg', 'slope', 'thal']

# Binaires (0/1) : passthrough — pas de transformation nécessaire
binary_features      = ['sex', 'fbs', 'exang']

# ca (0-3) : traité comme numérique car ordinal continu
# On l'ajoute aux numériques pour le StandardScaler
numeric_features_all = numeric_features + ['ca']

print(f"  Numériques ({len(numeric_features_all)}) : {numeric_features_all}")
print(f"  Catégorielles ({len(categorical_features)}) : {categorical_features}")
print(f"  Binaires ({len(binary_features)}) : {binary_features}")

# ============================================================================
# 4. SÉPARATION X / y
# ============================================================================
X = df.drop('target', axis=1)
y = df['target']
print(f"\n  X shape : {X.shape}  |  y shape : {y.shape}")

# ============================================================================
# 5. SÉPARATION TRAIN / TEST (70/30, stratifiée)
# ============================================================================
print("\n" + "=" * 60)
print("SÉPARATION TRAIN/TEST  (70 % / 30 %)")
print("=" * 60)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print(f"  X_train : {X_train_raw.shape}  |  y_train : {y_train.shape}")
print(f"  X_test  : {X_test_raw.shape}   |  y_test  : {y_test.shape}")
print(f"\n  Distribution y_train : sain={( y_train==0).sum()} / malade={(y_train==1).sum()}")
print(f"  Distribution y_test  : sain={(y_test ==0).sum()} / malade={(y_test ==1).sum()}")

# ============================================================================
# 6. CRÉATION ET APPLICATION DU PRÉPROCESSEUR
# ============================================================================
print("\n" + "=" * 60)
print("PRÉPROCESSEUR  (StandardScaler + OneHotEncoder)")
print("=" * 60)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),
         numeric_features_all),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
         categorical_features),
        ('bin', 'passthrough',
         binary_features)
    ],
    remainder='drop'
)

# fit sur train seulement, transform sur les deux
X_train_processed = preprocessor.fit_transform(X_train_raw)
X_test_processed  = preprocessor.transform(X_test_raw)

# Noms des features après transformation
cat_names   = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features_all + list(cat_names) + binary_features

print(f"  Features après transformation : {X_train_processed.shape[1]}")
print(f"  Détail : {len(numeric_features_all)} num  +  {len(cat_names)} OHE  +  {len(binary_features)} bin")
print(f"\n  Noms des features :")
for i, name in enumerate(all_feature_names):
    print(f"    [{i:02d}] {name}")

# ============================================================================
# 7. SAUVEGARDE
# ============================================================================
print("\n" + "=" * 60)
print("SAUVEGARDE")
print("=" * 60)

np.savetxt('../data/X_train_preprocessed_full.csv', X_train_processed, delimiter=',')
np.savetxt('../data/X_test_preprocessed_full.csv',  X_test_processed,  delimiter=',')
y_train.to_csv('../data/y_train_full.csv', index=False)
y_test.to_csv( '../data/y_test_full.csv',  index=False)
print("  ✅ Données prétraitées sauvegardées")

joblib.dump(preprocessor, '../model/preprocessor_full.joblib')
print("  ✅ Préprocesseur sauvegardé : model/preprocessor_full.joblib")

with open('../model/feature_names_full.txt', 'w') as f:
    for name in all_feature_names:
        f.write(f"{name}\n")
print(f"  ✅ {len(all_feature_names)} noms de features sauvegardés")

# Sauvegarder aussi le dataset propre (source de vérité pour 2_eda.py)
df.to_csv('../data/heart_clean_full.csv', index=False)
print("  ✅ Dataset nettoyé sauvegardé : data/heart_clean_full.csv")

print("\n" + "=" * 60)
print("✅ PRÉTRAITEMENT TERMINÉ")
print("=" * 60)