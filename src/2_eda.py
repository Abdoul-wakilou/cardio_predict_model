"""
2_eda.py - Analyse Exploratoire du dataset complet (1025 patients)
- Statistiques descriptives
- Tests de normalité (Shapiro-Wilk)
- Tests statistiques (t-test)
- Corrélations (Pearson)
- Génération des figures pour le mémoire
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os

# Créer le dossier images
os.makedirs('../images', exist_ok=True)

print("="*60)
print("ANALYSE EXPLORATOIRE - DATASET COMPLET (1025 patients)")
print("="*60)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES NETTOYÉES
# ============================================================================

df = pd.read_csv('../data/heart_clean_full.csv')
print(f"\n📊 Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ============================================================================
# 2. DISTRIBUTION DE LA CIBLE
# ============================================================================

print("\n" + "="*60)
print("DISTRIBUTION DE LA CIBLE")
print("="*60)

target_counts = df['target'].value_counts()
target_pct = df['target'].value_counts(normalize=True) * 100

print(f"   - Sain (0) : {target_counts[0]} patients ({target_pct[0]:.1f}%)")
print(f"   - Malade (1) : {target_counts[1]} patients ({target_pct[1]:.1f}%)")
print(f"   - Ratio Sain/Malade : {target_counts[0]/target_counts[1]:.2f}")

# Figure 1
plt.figure(figsize=(6, 6))
ax = target_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Distribution de la variable cible (0 = Sain, 1 = Malade)', fontsize=14)
plt.xlabel('Diagnostic', fontsize=12)
plt.ylabel('Nombre de patients', fontsize=12)
plt.xticks(rotation=0, ticks=[0, 1], labels=['Sain (0)', 'Malade (1)'])
for i, v in enumerate(target_counts):
    ax.text(i, v + 10, f'{v}\n({target_pct[i]:.1f}%)', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../images/distribution_cible.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : images/distribution_cible.png")

# ============================================================================
# 3. STATISTIQUES DESCRIPTIVES - VARIABLES NUMÉRIQUES
# ============================================================================

print("\n" + "="*60)
print("STATISTIQUES DESCRIPTIVES - VARIABLES NUMÉRIQUES")
print("="*60)

numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
stats_numeric = []

for col in numeric_features:
    mean_val = df[col].mean()
    median_val = df[col].median()
    std_val = df[col].std()
    min_val = df[col].min()
    max_val = df[col].max()
    
    # Test de normalité de Shapiro-Wilk
    stat, p_value = stats.shapiro(df[col].dropna())
    normal = "Oui" if p_value > 0.05 else "Non"
    
    stats_numeric.append([col, f"{mean_val:.2f}", f"{median_val:.2f}", 
                          f"{std_val:.2f}", min_val, max_val, f"{p_value:.4f}", normal])
    
    print(f"\n** {col} **")
    print(f"   - Moyenne : {mean_val:.2f}")
    print(f"   - Médiane : {median_val:.2f}")
    print(f"   - Écart-type : {std_val:.2f}")
    print(f"   - Min : {min_val}, Max : {max_val}")
    print(f"   - Shapiro-Wilk p-value : {p_value:.4f} (distribution {normal} normale)")

# Affichage du tableau
print("\n📊 Tableau récapitulatif :")
print("-"*80)
print(f"{'Variable':<12} {'Moyenne':<10} {'Médiane':<10} {'Écart-type':<12} {'Min':<6} {'Max':<6} {'p-value':<10} {'Normale'}")
print("-"*80)
for row in stats_numeric:
    print(f"{row[0]:<12} {row[1]:<10} {row[2]:<10} {row[3]:<12} {row[4]:<6} {row[5]:<6} {row[6]:<10} {row[7]}")
print("-"*80)

# Figure 2 : Histogrammes
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for i, col in enumerate(numeric_features):
    sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(f'Distribution de {col}\n(Moyenne: {df[col].mean():.1f})')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Fréquence')
if len(numeric_features) < 6:
    axes[-1].set_visible(False)
plt.suptitle('Distribution des variables numériques', fontsize=14)
plt.tight_layout()
plt.savefig('../images/histogrammes_numeriques.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : images/histogrammes_numeriques.png")

# ============================================================================
# 4. STATISTIQUES DESCRIPTIVES - VARIABLES CATÉGORIELLES
# ============================================================================

print("\n" + "="*60)
print("STATISTIQUES DESCRIPTIVES - VARIABLES CATÉGORIELLES")
print("="*60)

categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
labels = {
    'sex': {0: 'Femme', 1: 'Homme'},
    'cp': {0: 'Asymptomatique', 1: 'Angine typique', 2: 'Angine atypique', 3: 'Douleur non-angineuse'},
    'fbs': {0: '≤ 120 mg/dl', 1: '> 120 mg/dl'},
    'exang': {0: 'Non', 1: 'Oui'}
}

for col in categorical_features:
    print(f"\n** {col} **")
    counts = df[col].value_counts().sort_index()
    for val, cnt in counts.items():
        label = labels.get(col, {}).get(val, val)
        print(f"   - {label} : {cnt} ({cnt/len(df)*100:.1f}%)")

# ============================================================================
# 5. ANALYSE BIVARIÉE - TEST T DE STUDENT
# ============================================================================

print("\n" + "="*60)
print("ANALYSE BIVARIÉE - TEST T DE STUDENT")
print("="*60)

print("\n🔍 Comparaison des moyennes entre patients sains (0) et malades (1) :")
print("-"*70)
print(f"{'Variable':<12} {'Sain (0)':<12} {'Malade (1)':<12} {'Différence':<12} {'p-value':<12} {'Significatif'}")
print("-"*70)

for col in numeric_features:
    group_0 = df[df['target'] == 0][col]
    group_1 = df[df['target'] == 1][col]
    stat, p_value = stats.ttest_ind(group_0, group_1)
    diff = group_1.mean() - group_0.mean()
    sig = "Oui" if p_value < 0.05 else "Non"
    print(f"{col:<12} {group_0.mean():<12.2f} {group_1.mean():<12.2f} {diff:<+12.2f} {p_value:<12.4f} {sig}")

# Figure 3 : Boxplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for i, col in enumerate(numeric_features):
    sns.boxplot(x='target', y=col, data=df, ax=axes[i], palette=['#2ecc71', '#e74c3c'])
    axes[i].set_title(f'{col} vs Diagnostic')
    axes[i].set_xlabel('Diagnostic (0 = Sain, 1 = Malade)')
    axes[i].set_ylabel(col)
if len(numeric_features) < 6:
    axes[-1].set_visible(False)
plt.suptitle('Distribution des variables numériques par diagnostic', fontsize=14)
plt.tight_layout()
plt.savefig('../images/boxplots_numeriques.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Figure sauvegardée : images/boxplots_numeriques.png")

# ============================================================================
# 6. MATRICE DE CORRÉLATION
# ============================================================================

print("\n" + "="*60)
print("MATRICE DE CORRÉLATION")
print("="*60)

# Calculer la matrice de corrélation (uniquement variables numériques)
corr_cols = numeric_features + ['target']
corr_matrix = df[corr_cols].corr(numeric_only=True)

# Corrélations avec la cible
target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
print("\n🔗 Corrélations avec la variable cible :")
for feat, corr in target_corr.items():
    print(f"   - {feat} : {corr:.3f}")

# Figure 4 : Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True)
plt.title('Matrice de corrélation des variables numériques', fontsize=14)
plt.tight_layout()
plt.savefig('../images/matrice_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure sauvegardée : images/matrice_correlation.png")

# ============================================================================
# 7. PRÉVALENCE DE LA MALADIE PAR MODALITÉ
# ============================================================================

print("\n" + "="*60)
print("PRÉVALENCE DE LA MALADIE PAR MODALITÉ")
print("="*60)

for col in categorical_features:
    print(f"\n** {col} **")
    cross_tab = pd.crosstab(df[col], df['target'], normalize='index') * 100
    for category in cross_tab.index:
        malade_pct = cross_tab.loc[category, 1] if 1 in cross_tab.columns else 0
        label = labels.get(col, {}).get(category, category)
        print(f"   - {label} : {malade_pct:.1f}% de malades")

print("\n" + "="*60)
print("✅ EDA TERMINÉE")
print("="*60)