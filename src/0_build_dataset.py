"""
0_build_dataset.py — Construction du dataset UCI Heart Disease combiné (4 centres)
===================================================================================
Utilise les fichiers PROCESSED (14 colonnes) : plus fiables que les fichiers bruts.

Fichiers attendus dans ../data/raw/ :
  - processed.cleveland.data
  - processed.hungarian.data
  - processed.switzerland.data
  - processed.va.data   (= long-beach-va processed)

Source : https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/

Encodages UCI originaux (dans les fichiers processed) :
  cp    : 1=typique, 2=atypique, 3=non-angineuse, 4=asymptomatique
  slope : 1=ascendante, 2=plate, 3=descendante
  thal  : 3=normal, 6=défaut fixe, 7=défaut réversible
  num   : 0=sain, 1-4=malade → binarisé en target 0/1

Après recodage (convention 0-based pour compatibilité avec le reste du projet) :
  cp    : 0=typique, 1=atypique, 2=non-angineuse, 3=asymptomatique
  slope : 0=ascendante, 1=plate, 2=descendante
  thal  : 0=normal, 1=défaut fixe, 2=défaut réversible
"""

import pandas as pd
import numpy as np
import os

os.makedirs('../data/raw', exist_ok=True)
os.makedirs('../data', exist_ok=True)

print("=" * 60)
print("CONSTRUCTION DU DATASET UCI COMBINÉ (4 centres)")
print("=" * 60)

# ============================================================================
# ÉTAPE 1 : Chargement des 4 fichiers processed
# ============================================================================
# Les fichiers processed ont 14 colonnes, séparées par virgules,
# sans en-tête. Les valeurs manquantes sont codées "-9" ou "?".

COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
           'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
           'ca', 'thal', 'num']

# Nom fichier → nom du centre
SOURCE_MAP = {
    'processed.cleveland.data': 'Cleveland',
    'processed.hungarian.data': 'Hungarian',
    'processed.switzerland.data': 'Switzerland',
    'processed.va.data': 'Long Beach VA',
}

dfs = []

for fname, source in SOURCE_MAP.items():
    fpath = f'../data/raw/{fname}'
    if not os.path.exists(fpath):
        print(f'  ⚠️  Fichier absent : {fpath}  (sera ignoré)')
        continue

    # Lire en traitant "?" et "-9" comme NaN
    df_src = pd.read_csv(
        fpath,
        header=None,
        names=COLUMNS,
        na_values=['?', '-9', '-9.0']
    )
    df_src['source'] = source
    dfs.append(df_src)
    print(f'  ✅ {source:<15} : {len(df_src)} lignes chargées')

if not dfs:
    print('\n❌ Aucun fichier trouvé. Place les fichiers processed dans ../data/raw/')
    exit(1)

df = pd.concat(dfs, ignore_index=True)
print(f'\n  Total brut combiné : {len(df)} lignes')

# ============================================================================
# ÉTAPE 2 : Binarisation de la cible (num → target)
# ============================================================================
print(f"\n{'=' * 60}")
print("BINARISATION DE LA CIBLE")
print(f"{'=' * 60}")

print(f'\nDistribution originale de "num" (toutes sources) :')
print(df['num'].value_counts(dropna=False).sort_index().to_string())
print(f'\nRépartition par centre :')
print(df.groupby('source')['num'].value_counts().unstack(fill_value=0).to_string())

# Vérification de cohérence médicale AVANT binarisation
# (les malades doivent avoir une fréquence cardiaque plus basse, oldpeak plus élevé)
print(f'\nCheck cohérence AVANT binarisation (num=0 vs num>=1) :')
mask_sain = df['num'] == 0
mask_mal  = df['num'] >= 1
for col in ['thalach', 'oldpeak', 'age']:
    m0 = df.loc[mask_sain, col].mean()
    m1 = df.loc[mask_mal,  col].mean()
    sens = "✅" if (col == 'thalach' and m0 > m1) or \
                   (col in ['oldpeak', 'age'] and m0 < m1) else "❌"
    print(f'  {sens} {col:<10} sain={m0:.1f}  malade={m1:.1f}')

df['target'] = (df['num'] >= 1).astype(int)
df.drop(columns=['num'], inplace=True)

tc = df['target'].value_counts().sort_index()
print(f'\nAprès binarisation :')
print(f'  Sain   (0) : {tc.get(0,0):>4}  ({tc.get(0,0)/len(df)*100:.1f}%)')
print(f'  Malade (1) : {tc.get(1,0):>4}  ({tc.get(1,0)/len(df)*100:.1f}%)')

# ============================================================================
# ÉTAPE 3 : Recodage 0-based
# ============================================================================
print(f"\n{'=' * 60}")
print("RECODAGE VERS CONVENTION 0-BASED")
print(f"{'=' * 60}")

# cp : 1-4 → 0-3
df['cp'] = df['cp'].map({1: 0, 2: 1, 3: 2, 4: 3})
print('  ✅ cp    : 1-4 → 0-3  (0=typique, 1=atypique, 2=non-ang, 3=asympt)')

# slope : 1-3 → 0-2
df['slope'] = df['slope'].map({1: 0, 2: 1, 3: 2})
print('  ✅ slope : 1-3 → 0-2  (0=ascendante, 1=plate, 2=descendante)')

# thal : 3/6/7 → 0/1/2
df['thal'] = df['thal'].map({3: 0, 6: 1, 7: 2})
print('  ✅ thal  : 3/6/7 → 0/1/2  (0=normal, 1=fixe, 2=réversible)')

# ============================================================================
# ÉTAPE 4 : Gestion des valeurs manquantes
# ============================================================================
print(f"\n{'=' * 60}")
print("GESTION DES VALEURS MANQUANTES")
print(f"{'=' * 60}")

cols_14 = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
           'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
           'ca', 'thal', 'target']

df = df[cols_14 + ['source']].copy()

print(f'\nValeurs manquantes par colonne :')
miss = df[cols_14].isnull().sum()
for col, cnt in miss[miss > 0].items():
    print(f'  {col:<10} : {cnt:>4} ({cnt/len(df)*100:.1f}%)')
print(f'  Total     : {miss.sum()}')

# Supprimer les lignes où target est NaN (ne peut pas entraîner le modèle)
n0 = len(df)
df.dropna(subset=['target'], inplace=True)
print(f'\n  Lignes supprimées (target manquant) : {n0 - len(df)}')

# Supprimer les lignes avec trop de variables manquantes (>= 4 sur 13 features)
n0 = len(df)
df = df[df[cols_14[:-1]].isnull().sum(axis=1) < 4]
print(f'  Lignes supprimées (>= 4 features manquantes) : {n0 - len(df)}')

# Imputation des manquants restants
# Numériques → médiane par classe target (évite la fuite d'info inter-classes)
num_imp = ['trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'age']
for col in num_imp:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        df[col] = df.groupby('target')[col].transform(
            lambda x: x.fillna(x.median())
        )
        print(f'  Imputé (médiane/classe) : {col:<10} ({n_miss} valeurs)')

# Catégorielles → mode global
cat_imp = ['cp', 'slope', 'thal', 'fbs', 'restecg', 'exang', 'sex']
for col in cat_imp:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f'  Imputé (mode={int(mode_val)})         : {col:<10} ({n_miss} valeurs)')

# Convertir en types propres
int_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'slope', 'ca', 'thal', 'target']
for col in int_cols:
    df[col] = df[col].astype(int)
df['oldpeak'] = df['oldpeak'].round(1)

print(f'\n  Valeurs manquantes restantes : {df[cols_14].isnull().sum().sum()}')
print(f'  Shape final : {len(df)} patients × {len(cols_14)} colonnes')

# ============================================================================
# ÉTAPE 5 : Vérification finale
# ============================================================================
print(f"\n{'=' * 60}")
print("VÉRIFICATION FINALE")
print(f"{'=' * 60}")

tc = df['target'].value_counts().sort_index()
print(f'\nDistribution de la cible :')
print(f'  Sain   (0) : {tc.get(0,0):>4}  ({tc.get(0,0)/len(df)*100:.1f}%)')
print(f'  Malade (1) : {tc.get(1,0):>4}  ({tc.get(1,0)/len(df)*100:.1f}%)')

print(f'\nRépartition par centre :')
print(df.groupby('source')['target'].value_counts().unstack(fill_value=0).to_string())

print(f'\nPlages de valeurs :')
ranges = {
    'age': (29, 77), 'trestbps': (90, 210), 'chol': (100, 600),
    'thalach': (60, 210), 'oldpeak': (0, 7),
}
for col, (lo, hi) in ranges.items():
    print(f'  {col:<10}: {df[col].min()} – {df[col].max()}  (attendu ~{lo}–{hi})')

print(f'\nModalités catégorielles :')
for col in ['cp', 'slope', 'thal', 'ca']:
    print(f'  {col:<8}: {sorted(df[col].unique())}')

# ============================================================================
# ÉTAPE 6 : Vérification de cohérence médicale
# ============================================================================
print(f"\n{'=' * 60}")
print("COHÉRENCE MÉDICALE — CORRÉLATIONS AVEC TARGET")
print(f"{'=' * 60}")

check_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'exang']
corr = df[check_cols + ['target']].corr()['target'].drop('target')

expected = {
    'age':      ('+', 'âge élevé → plus malade'),
    'trestbps': ('+', 'tension élevée → malade'),
    'chol':     ('+', 'cholestérol élevé → malade'),
    'thalach':  ('-', 'FC élevée → sain (bonne capacité)'),
    'oldpeak':  ('+', 'dépression ST → malade'),
    'ca':       ('+', 'vaisseaux obstrués → malade'),
    'exang':    ('+', 'angine effort → malade'),
}

all_ok = True
for col, r in corr.items():
    exp_sign, desc = expected.get(col, ('?', ''))
    actual = '+' if r > 0 else '-'
    ok = '✅' if actual == exp_sign else '❌'
    if actual != exp_sign:
        all_ok = False
    print(f'  {ok} {col:<10} r = {r:+.3f}  (attendu {exp_sign}) — {desc}')

if all_ok:
    print('\n  ✅ Toutes les corrélations sont médicalement cohérentes !')
else:
    print('\n  ❌ Certaines corrélations sont incohérentes — vérifier le dataset.')

# ============================================================================
# ÉTAPE 7 : Sauvegarde
# ============================================================================
df_out = df[cols_14]   # sans la colonne 'source'
df_out.to_csv('../data/heart_uci_combined.csv', index=False)
print(f"\n{'=' * 60}")
print(f'✅ Dataset sauvegardé : ../data/heart_uci_combined.csv')
print(f'   {len(df_out)} patients, {len(df_out.columns)} colonnes')
print(f"{'=' * 60}")
print('\n→ Dans 1_preprocessing.py, remplace :')
print("     df = pd.read_csv('../data/heart.csv')")
print('  par :')
print("     df = pd.read_csv('../data/heart_uci_combined.csv')")