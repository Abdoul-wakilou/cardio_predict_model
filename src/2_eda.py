"""
2_eda.py - Analyse Exploratoire du dataset complet (1025 patients)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

os.makedirs('../images', exist_ok=True)

print("="*60)
print("ANALYSE EXPLORATOIRE - DATASET COMPLET (1025 patients)")
print("="*60)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES NETTOYÉES
# ============================================================================

df = pd.read_csv('../data/heart_clean_full.csv')
print(f"\nDataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ============================================================================
# 2. DISTRIBUTION DE LA CIBLE
# ============================================================================

print("\n" + "="*60)
print("DISTRIBUTION DE LA CIBLE")
print("="*60)

target_counts = df['target'].value_counts().sort_index()
target_pct    = df['target'].value_counts(normalize=True).sort_index() * 100

print(f"   - Sain (0)   : {target_counts[0]} patients ({target_pct[0]:.1f}%)")
print(f"   - Malade (1) : {target_counts[1]} patients ({target_pct[1]:.1f}%)")
print(f"   - Ratio Sain/Malade : {target_counts[0]/target_counts[1]:.2f}")

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(['Sain (0)', 'Malade (1)'], target_counts.values, color=['#2ecc71', '#e74c3c'])
ax.set_title('Distribution de la variable cible', fontsize=14)
ax.set_xlabel('Diagnostic', fontsize=12)
ax.set_ylabel('Nombre de patients', fontsize=12)
for i, (v, pct) in enumerate(zip(target_counts.values, target_pct.values)):
    ax.text(i, v + 10, f'{v}\n({pct:.1f}%)', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../images/distribution_cible.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure sauvegardée : images/distribution_cible.png")

# ============================================================================
# 3. STATISTIQUES DESCRIPTIVES — VARIABLES NUMÉRIQUES
# ============================================================================

print("\n" + "="*60)
print("STATISTIQUES DESCRIPTIVES — VARIABLES NUMÉRIQUES")
print("="*60)

numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

print(f"\n{'Variable':<12} {'Moyenne':<10} {'Médiane':<10} {'Écart-type':<12} {'Min':<6} {'Max':<6} {'p-Shapiro':<12} {'Normale'}")
print("-"*80)

for col in numeric_features:
    m, med, s = df[col].mean(), df[col].median(), df[col].std()
    lo, hi    = df[col].min(), df[col].max()
    _, p      = stats.shapiro(df[col].dropna())
    normal    = "Oui" if p > 0.05 else "Non"
    print(f"{col:<12} {m:<10.2f} {med:<10.2f} {s:<12.2f} {lo:<6} {hi:<6} {p:<12.4f} {normal}")

# Histogrammes
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for i, col in enumerate(numeric_features):
    sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(f'Distribution de {col} (μ={df[col].mean():.1f})')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Fréquence')
axes[-1].set_visible(False)
plt.suptitle('Distribution des variables numériques', fontsize=14)
plt.tight_layout()
plt.savefig('../images/histogrammes_numeriques.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure sauvegardée : images/histogrammes_numeriques.png")

# ============================================================================
# 4. STATISTIQUES DESCRIPTIVES — VARIABLES CATÉGORIELLES / ORDINALES
# ============================================================================

print("\n" + "="*60)
print("STATISTIQUES DESCRIPTIVES — VARIABLES CATÉGORIELLES")
print("="*60)

# ca et thal sont maintenant des entiers propres (plus de float ni de NaN)
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

labels = {
    'sex':    {0: 'Femme', 1: 'Homme'},
    'cp':     {0: 'Asymptomatique', 1: 'Angine typique', 2: 'Angine atypique', 3: 'Douleur non-angineuse'},
    'fbs':    {0: '≤ 120 mg/dl', 1: '> 120 mg/dl'},
    'restecg':{0: 'Normal', 1: 'Anomalie ST-T', 2: 'HVG probable'},
    'exang':  {0: 'Non', 1: 'Oui'},
    'slope':  {0: 'Descendante', 1: 'Plate', 2: 'Ascendante'},
    'ca':     {0: '0 vaisseau', 1: '1 vaisseau', 2: '2 vaisseaux', 3: '3 vaisseaux'},
    'thal':   {1: 'Normal', 2: 'Défaut fixé', 3: 'Défaut réversible'},
}

for col in categorical_cols:
    print(f"\n{col} :")
    counts = df[col].value_counts().sort_index()
    for val, cnt in counts.items():
        label = labels.get(col, {}).get(val, str(val))
        print(f"   - {label} : {cnt} ({cnt/len(df)*100:.1f}%)")

# ============================================================================
# 5. ANALYSE BIVARIÉE — TEST T DE STUDENT
# ============================================================================

print("\n" + "="*60)
print("ANALYSE BIVARIÉE — TEST T DE STUDENT")
print("="*60)

print(f"\n{'Variable':<12} {'Sain (0)':<12} {'Malade (1)':<12} {'Différence':<12} {'p-value':<12} {'Significatif'}")
print("-"*72)

for col in numeric_features:
    g0  = df[df['target'] == 0][col]
    g1  = df[df['target'] == 1][col]
    _, p = stats.ttest_ind(g0, g1)
    diff = g1.mean() - g0.mean()
    sig  = "Oui" if p < 0.05 else "Non"
    print(f"{col:<12} {g0.mean():<12.2f} {g1.mean():<12.2f} {diff:<+12.2f} {p:<12.4f} {sig}")

# Boxplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for i, col in enumerate(numeric_features):
    sns.boxplot(x='target', y=col, data=df, ax=axes[i], palette=['#2ecc71', '#e74c3c'])
    axes[i].set_title(f'{col} vs Diagnostic')
    axes[i].set_xlabel('Diagnostic (0=Sain, 1=Malade)')
    axes[i].set_ylabel(col)
axes[-1].set_visible(False)
plt.suptitle('Distribution des variables numériques par diagnostic', fontsize=14)
plt.tight_layout()
plt.savefig('../images/boxplots_numeriques.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nFigure sauvegardée : images/boxplots_numeriques.png")

# ============================================================================
# 6. MATRICE DE CORRÉLATION
# ============================================================================

print("\n" + "="*60)
print("MATRICE DE CORRÉLATION")
print("="*60)

corr_cols   = numeric_features + ['target']
corr_matrix = df[corr_cols].corr()

target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
print("\nCorrélations avec la cible :")
for feat, corr in target_corr.items():
    print(f"   - {feat} : {corr:.3f}")

plt.figure(figsize=(8, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True)
plt.title('Matrice de corrélation des variables numériques', fontsize=14)
plt.tight_layout()
plt.savefig('../images/matrice_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure sauvegardée : images/matrice_correlation.png")

# ============================================================================
# 7. PRÉVALENCE DE LA MALADIE PAR MODALITÉ
# ============================================================================

print("\n" + "="*60)
print("PRÉVALENCE DE LA MALADIE PAR MODALITÉ")
print("="*60)

for col in categorical_cols:
    print(f"\n{col} :")
    cross = pd.crosstab(df[col], df['target'], normalize='index') * 100
    for cat in cross.index:
        pct   = cross.loc[cat, 1] if 1 in cross.columns else 0
        label = labels.get(col, {}).get(cat, str(cat))
        print(f"   - {label} : {pct:.1f}% de malades")

print("\n" + "="*60)
print("EDA TERMINÉE")
print("="*60)