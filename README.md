# CardioPredict

**CardioPredict** est une application mobile propulsée par l’intelligence artificielle, conçue pour la **détection précoce des risques cardiovasculaires**.  
Elle analyse plusieurs variables cliniques (âge, cholestérol, tension artérielle, etc.) afin de **prédire un score de risque** et de fournir des **conseils personnalisés de prévention**.  

---

## Objectif du projet

Ce projet s’inscrit dans le cadre d’un **mémoire de Master 2** portant sur la conception d’un **modèle prédictif intelligent** basé sur des algorithmes de machine learning (Random Forest, SVM, etc.), intégré dans une application mobile **intuitive et accessible**.

---

## Fonctionnalités principales

- Analyse automatisée des données cliniques.
- Prédiction du risque cardiovasculaire (faible, moyen, élevé).
- Interface conviviale et responsive (Flutter).
- Conseils de prévention personnalisés.
- Historique et suivi des évaluations.
- Architecture sécurisée (API RESTful + IA backend).

---


---
## ⚙️ Technologies utilisées

| Domaine | Outils / Frameworks |
|----------|--------------------|
| Interface mobile | **Flutter (Dart)** |
| Backend / API | **Flask (Python)** |
| IA & Modèle ML | **Python (scikit-learn, pandas, numpy)** |
| Communication | **API RESTful (JSON)** |
| Base de données | **SQLite** *(ou MySQL selon besoin)* |
| Versioning | **Git & GitHub** |

---

## Installation & Lancement

### Prérequis

- **Flutter SDK ≥ 3.0**
- **Python ≥ 3.9**
- **pip** (gestionnaire de paquets Python)
- **SQLite** *(par défaut, déjà intégré à Python)*

---

### Étapes d’installation

#### Cloner le dépôt
```bash
git clone https://github.com/Abdoul-wakilou-Tiga/cardio_predict_model.git
cd cardio_predict_model


cardio_predict_model/
├── 📊 data/                    # Datasets bruts et transformés
├── 📖 documentation/           # Documentation du dataset
├── 🫀 heart+disease/           # Données spécifiques maladies cardiaques  
├── 🤖 model/                   # Modèles entraînés et préprocesseurs
├── 📓 notebooks/               # Notebooks d'analyse et expérimentation
├── 🛠️ src/                     # Code source Python
├── 🌐 static/                  # Fichiers statiques pour Flask
├── 🚀 app.py                   # Application Flask principale
├── 📋 requirements.txt         # Dépendances Python
└── ℹ️ README.md                # Documentation du projet


# 1. Assurez-vous d'avoir les données nettoyées
python src/preprocessingdataset.py

# 2. Exécutez le NOUVEAU pipeline complet
python src/train_and_save_pipeline.py

# 3. Vérifiez les fichiers créés
ls -la model/
# Doit afficher:
# best_model.joblib    preprocessor.joblib    flask_config.py

# 4. Lancez Flask
python app.py

# 5. Testez avec curl (copiez l'exemple du script)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'