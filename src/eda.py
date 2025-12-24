# eda.py

"""
L’analyse exploratoire a permis d’identifier les variables les plus discriminantes 
et d’orienter le choix des algorithmes.

Analyse univariée (moyennes, médianes, Shapiro)

Analyse bivariée (t-test, prévalence)

Analyse multivariée (corrélations)

Graphiques interprétables
"""

# Importations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Charger les données nettoyées
df = pd.read_csv('../data/cleveland_clean.csv')

# !!! CORRECTION CRUCIALE : Transformation de la variable cible en binaire !!!
# Si la valeur originale est 0 -> reste 0 (sain)
# Si la valeur originale est 1, 2, 3, ou 4 -> devient 1 (malade)
print("Valeurs uniques de 'target' AVANT transformation :", sorted(df['target'].unique()))
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)
print("Valeurs uniques de 'target' APRES transformation :", sorted(df['target'].unique()))

# Configuration de l'affichage
plt.style.use('seaborn-v0_8')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (10, 6)

# %% 1. ANALYSE UNIVARIEE - AVEC CHIFFRES

print("\n" + "="*60)
print("ANALYSE UNIVARIÉE - STATISTIQUES DÉTAILLÉES")
print("="*60)

# 1.a) Distribution de la Variable Cible - AVEC CHIFFRES
target_counts = df['target'].value_counts()
target_percentage = df['target'].value_counts(normalize=True) * 100

print(f"\n1. DISTRIBUTION DE LA CIBLE (target):")
print(f"   - Classe 0 (Sain) : {target_counts[0]} patients ({target_percentage[0]:.1f}%)")
print(f"   - Classe 1 (Malade) : {target_counts[1]} patients ({target_percentage[1]:.1f}%)")
print(f"   -> Ratio Sain/Malade : {target_counts[0]}/{target_counts[1]} ≈ {target_counts[0]/target_counts[1]:.2f}")

# Visualisation
plt.figure()
ax = target_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution de la Variable Cible (0 = Sain, 1 = Malade)')
plt.xlabel('Diagnostic de Maladie Cardiaque')
plt.ylabel('Nombre de Patients')
plt.xticks(rotation=0)
for i, v in enumerate(target_counts):
    ax.text(i, v + 3, f'{v}\n({target_percentage[i]:.1f}%)', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# 1.b) Statistiques des Variables Numériques
num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
print(f"\n2. STATISTIQUES DES VARIABLES NUMÉRIQUES:")
for col in num_features:
    print(f"\n   ** {col} **")
    print(f"      - Moyenne = {df[col].mean():.2f}")
    print(f"      - Médiane = {df[col].median():.2f}")
    print(f"      - Écart-type = {df[col].std():.2f}")
    print(f"      - Min = {df[col].min()}, Max = {df[col].max()}")
    # Test de normalité ( Shapiro-Wilk pour échantillons <5000)
    stat, p_value = stats.shapiro(df[col])
    print(f"      - Distribution normale? p-value = {p_value:.4f} {'(Oui)' if p_value > 0.05 else '(Non)'}")

# Visualisation des distributions numériques
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for i, col in enumerate(num_features):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution de {col}\n(Mean: {df[col].mean():.1f})')
if len(num_features) < 6: axes[-1].set_visible(False)
plt.suptitle('Distribution des Variables Numériques')
plt.tight_layout()
plt.show()

# 1.c) Statistiques des Variables Catégorielles
cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
print(f"\n3. DISTRIBUTION DES VARIABLES CATÉGORIELLES:")
for col in cat_features:
    counts = df[col].value_counts().sort_index()
    print(f"\n   ** {col} **")
    for value, count in counts.items():
        print(f"      - {value} : {count} patients ({count/len(df)*100:.1f}%)")

# %% 2. ANALYSE BIVARIEE - AVEC CHIFFRES

print("\n" + "="*60)
print("ANALYSE BIVARIÉE - RELATIONS AVEC LA CIBLE")
print("="*60)

# 2.a) Relation Variables Numériques / Target
print(f"\n4. IMPACT DES VARIABLES NUMÉRIQUES (Groupées par Target):")
for col in num_features:
    group_0 = df[df['target'] == 0][col] # Groupe Sain
    group_1 = df[df['target'] == 1][col] # Groupe Malade
    stat, p_value = stats.ttest_ind(group_0, group_1) # Test t-test
    print(f"\n   ** {col} **")
    print(f"      - Sain (0) : Moyenne = {group_0.mean():.2f}")
    print(f"      - Malade (1) : Moyenne = {group_1.mean():.2f}")
    print(f"      - Différence significative? p-value = {p_value:.4f} {'(Oui)' if p_value < 0.05 else '(Non)'}")

# Visualisation boxplots
plt.figure(figsize=(16, 10))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='target', y=col, data=df)
    plt.title(f'{col} vs. Diagnostic')
plt.tight_layout()
plt.show()

# 2.b) Relation Variables Catégorielles / Target
print(f"\n5. IMPACT DES VARIABLES CATÉGORIELLES (Prévalence de la maladie):")
for col in cat_features:
    print(f"\n   ** {col} **")
    cross_tab = pd.crosstab(df[col], df['target'], normalize='index') * 100
    print(f"      % de malades par catégorie :")
    for category in cross_tab.index:
        malade_percent = cross_tab.loc[category, 1]
        print(f"        - {category} : {malade_percent:.1f}% de malades")

# Visualisation pour 'exang' et 'cp'
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='exang', hue='target', data=df)
plt.title('Maladie par Angine à l\'effort (exang)')
plt.subplot(1, 2, 2)
sns.countplot(x='cp', hue='target', data=df)
plt.title('Maladie par Type de Douleur (cp)')
plt.tight_layout()
plt.show()

# %% 3. ANALYSE MULTIVARIEE : MATRICE DE CORRELATION
corr_matrix = df.corr(numeric_only=True)
target_corr = corr_matrix['target'].sort_values(ascending=False)

print("\n" + "="*60)
print("MATRICE DE CORRÉLATION")
print("="*60)
print(f"\n6. CORRÉLATIONS AVEC LA CIBLE (target):")
for feature, corr in target_corr.items():
    if feature != 'target':
        print(f"   - {feature} : {corr:.3f}")

# Visualisation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Matrice de Corrélation Complète')
plt.tight_layout()
plt.show()