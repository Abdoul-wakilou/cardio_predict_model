# train_and_save_pipeline.py
"""
Script COMPLET d'entraînement et de sauvegarde pour Flask.
Ce script :
1. Charge les données brutes
2. Applique TOUT le prétraitement (comme preprocessing_final.py)
3. Entraîne plusieurs modèles pour comparaison
4. Sauvegarde le MEILLEUR modèle ET le PRÉPROCESSEUR
5. Tout est prêt pour Flask (données brutes → prétraitement → prédiction)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

print("="*60)
print("🚀 TRAIN AND SAVE PIPELINE - POUR FLASK")
print("="*60)

# ============================================================================
# ÉTAPE 1 : CHARGER LES DONNÉES BRUTES
# ============================================================================

print("\n1️⃣ CHARGEMENT DES DONNÉES BRUTES")

# Option A : Charger depuis le fichier déjà nettoyé
df = pd.read_csv('../data/cleveland_clean.csv')
print(f"   ✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Vérifier que target est bien binaire [0, 1]
print(f"   ✅ Target unique values : {sorted(df['target'].unique())}")

# ============================================================================
# ÉTAPE 2 : DÉFINIR ET CRÉER LE PRÉPROCESSEUR (CRITIQUE POUR FLASK)
# ============================================================================

print("\n2️⃣ CRÉATION DU PRÉPROCESSEUR POUR FLASK")

# Définir les types de features (TRÈS IMPORTANT - même ordre que Flutter)
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['cp', 'restecg', 'slope', 'thal']
binary_features = ['sex', 'fbs', 'exang']

# Créer le ColumnTransformer qui sera utilisé dans Flask
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
        ('bin', 'passthrough', binary_features)
    ])

print("   ✅ Préprocesseur créé avec:")
print(f"      - {len(numeric_features)} variables numériques (StandardScaler)")
print(f"      - {len(categorical_features)} variables catégorielles (OneHotEncoder)")
print(f"      - {len(binary_features)} variables binaires (passthrough)")

# ============================================================================
# ÉTAPE 3 : PRÉPARER LES DONNÉES POUR L'ENTRAÎNEMENT
# ============================================================================

print("\n3️⃣ PRÉPARATION DES DONNÉES POUR L'ENTRAÎNEMENT")

# Séparer features (X) et target (y)
X = df.drop('target', axis=1)
y = df['target']

# Séparation train/test (stratifiée pour garder les proportions)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   ✅ Split effectué (70% train, 30% test):")
print(f"      X_train_raw : {X_train_raw.shape}")
print(f"      X_test_raw  : {X_test_raw.shape}")
print(f"      y_train     : {y_train.shape}")
print(f"      y_test      : {y_test.shape}")

# ============================================================================
# ÉTAPE 4 : APPLIQUER LE PRÉTRAITEMENT (comme Flask le fera)
# ============================================================================

print("\n4️⃣ APPLICATION DU PRÉTRAITEMENT")

# Fit sur le train, transform sur train et test
X_train_processed = preprocessor.fit_transform(X_train_raw)
X_test_processed = preprocessor.transform(X_test_raw)

print(f"   ✅ Prétraitement appliqué:")
print(f"      X_train_processed : {X_train_processed.shape}")
print(f"      X_test_processed  : {X_test_processed.shape}")

# Sauvegarder les noms de features après transformation (pour debug)
if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_names) + binary_features
    print(f"   🔍 Noms features après prétraitement: {len(all_feature_names)} features")

# ============================================================================
# ÉTAPE 5 : ENTRAÎNER ET COMPARER LES MODÈLES
# ============================================================================

print("\n5️⃣ ENTRAÎNEMENT ET COMPARAISON DES MODÈLES")

# Initialiser les modèles
models = {
    "Régression Logistique": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "SVM": SVC(random_state=42, probability=True)  # probability=True pour Flask
}

results = {}

for name, model in models.items():
    print(f"\n   🔹 Entraînement {name}...")
    
    # Entraînement
    model.fit(X_train_processed, y_train)
    
    # Prédiction
    y_pred = model.predict(X_test_processed)
    
    # Évaluation
    accuracy = accuracy_score(y_test, y_pred)
    
    # Stocker résultats
    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "y_pred": y_pred
    }
    
    print(f"      Accuracy: {accuracy:.4f}")

# ============================================================================
# ÉTAPE 6 : IDENTIFIER LE MEILLEUR MODÈLE
# ============================================================================

print("\n6️⃣ IDENTIFICATION DU MEILLEUR MODÈLE")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print(f"   🏆 Meilleur modèle: {best_model_name}")
print(f"      Accuracy: {best_accuracy:.4f}")

# Afficher le rapport détaillé pour le meilleur modèle
print(f"\n   📊 Rapport de classification ({best_model_name}):")
y_pred_best = results[best_model_name]['y_pred']
print(classification_report(y_test, y_pred_best, target_names=["Sain", "Malade"]))

# ============================================================================
# ÉTAPE 7 : SAUVEGARDER POUR FLASK (MODÈLE + PRÉPROCESSEUR)
# ============================================================================

print("\n7️⃣ SAUVEGARDE POUR FLASK")

# Décision: Sauvegarder la Régression Logistique (comme décidé)
model_to_save = results["Régression Logistique"]["model"]
print(f"   💾 Modèle à sauvegarder: 'Régression Logistique' (votre choix)")

# Sauvegarde du modèle
joblib.dump(model_to_save, '../model/best_model.joblib')
print(f"   ✅ Modèle sauvegardé: '../model/best_model.joblib'")

# Sauvegarde du préprocesseur (CRITIQUE!)
joblib.dump(preprocessor, '../model/preprocessor.joblib')
print(f"   ✅ Préprocesseur sauvegardé: '../model/preprocessor.joblib'")

# ============================================================================
# ÉTAPE 8 : TEST DE VÉRIFICATION (simule ce que Flask fera)
# ============================================================================

print("\n8️⃣ TEST DE VÉRIFICATION (simulation Flask)")

# Prendre un échantillon brut (comme Flutter l'enverra)
sample_raw = X_test_raw.iloc[0:1].copy()
print(f"   🧪 Donnée brute d'entrée (exemple):")
for col in sample_raw.columns:
    print(f"      {col}: {sample_raw[col].values[0]}")

# Appliquer le préprocesseur (comme Flask le fera)
sample_processed = preprocessor.transform(sample_raw)

# Faire la prédiction avec le modèle sauvegardé
sample_pred = model_to_save.predict(sample_processed)[0]
sample_proba = model_to_save.predict_proba(sample_processed)[0]

print(f"\n   🔮 Prédiction (simulation Flask):")
print(f"      Classe prédite: {sample_pred} ({'Malade' if sample_pred == 1 else 'Sain'})")
print(f"      Probabilité: Sain={sample_proba[0]:.3f}, Malade={sample_proba[1]:.3f}")

# ============================================================================
# ÉTAPE 9 : CRÉER UN FICHIER DE CONFIGURATION POUR FLASK
# ============================================================================

print("\n9️⃣ CRÉATION DE LA CONFIGURATION FLASK")

config_content = f"""
# Configuration pour Flask - CardioPredict
# Généré automatiquement par train_and_save_pipeline.py

FEATURES_ORDER = {list(X.columns)}
NUMERIC_FEATURES = {numeric_features}
CATEGORICAL_FEATURES = {categorical_features}
BINARY_FEATURES = {binary_features}

MODEL_INFO = {{
    'name': 'Régression Logistique',
    'accuracy': {best_accuracy:.4f},
    'model_file': 'best_model.joblib',
    'preprocessor_file': 'preprocessor.joblib'
}}

# Exemple de requête Flask:
"""
example_request = {
    "age": int(sample_raw['age'].values[0]),
    "sex": int(sample_raw['sex'].values[0]),
    "cp": int(sample_raw['cp'].values[0]),
    "trestbps": int(sample_raw['trestbps'].values[0]),
    "chol": int(sample_raw['chol'].values[0]),
    "fbs": int(sample_raw['fbs'].values[0]),
    "restecg": int(sample_raw['restecg'].values[0]),
    "thalach": int(sample_raw['thalach'].values[0]),
    "exang": int(sample_raw['exang'].values[0]),
    "oldpeak": float(sample_raw['oldpeak'].values[0]),
    "slope": int(sample_raw['slope'].values[0]),
    "ca": int(sample_raw['ca'].values[0]),
    "thal": int(sample_raw['thal'].values[0])
}

config_content += f"""
# curl -X POST http://localhost:5000/predict \\
#   -H "Content-Type: application/json" \\
#   -d '{example_request}'
"""

with open('../model/flask_config.py', 'w') as f:
    f.write(config_content)

print(f"   ✅ Configuration sauvegardée: '../model/flask_config.py'")

# ============================================================================
# ÉTAPE 10 : INSTRUCTIONS FINALES
# ============================================================================

print("\n" + "="*60)
print("🎉 PIPELINE COMPLÈTE PRÊTE POUR FLASK")
print("="*60)

print(f"""
📋 RÉSUMÉ :
------------
• Modèle: Régression Logistique
• Accuracy: {best_accuracy:.4f}
• Fichiers créés:
  1. ../model/best_model.joblib     (modèle entraîné)
  2. ../model/preprocessor.joblib   (préprocesseur)
  3. ../model/flask_config.py       (configuration)

🚀 POUR LANCER FLASK :
----------------------
1. Vérifiez les fichiers:
   $ ls -la model/
   
2. Lancez Flask (avec le bon app.py):
   $ python app.py
   
3. Testez avec curl:
   curl -X POST http://localhost:5000/predict \\
     -H "Content-Type: application/json" \\
     -d '{example_request}'

📱 POUR FLUTTER :
-----------------
• Envoyer POST à: http://VOTRE_IP:5000/predict
• Corps JSON avec exactement ces 13 features:
  {list(X.columns)}
• Mêmes types que dans l'exemple ci-dessus

✅ TOUT EST PRÊT ! Le pipeline complète est:
   Flutter (brut) → Flask (prétraitement) → Modèle → Résultat
""")