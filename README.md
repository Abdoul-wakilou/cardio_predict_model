# 🫀 CardioPredict — Backend IA (Flask & Machine Learning)

**CardioPredict** est un projet de recherche appliquée en intelligence artificielle dédié à la **détection précoce des risques cardiovasculaires** à partir de données cliniques.  
Ce dépôt correspond au **backend IA et à l'API Flask**, utilisés par une application mobile Flutter.

---

## 🎓 Contexte académique

Ce projet est réalisé dans le cadre d'un **mémoire de Master 2**, portant sur :

> **La conception d'un modèle d'intelligence artificielle pour la détection précoce des risques cardiovasculaires.**

L'objectif principal est de concevoir :
- un **pipeline complet de machine learning** (EDA → prétraitement → entraînement → évaluation),
- un **modèle prédictif fiable et interprétable**,
- une **API REST Flask** permettant l'intégration du modèle dans une application mobile.

---

## 🎯 Objectifs du backend

- Explorer et analyser les données cliniques (EDA).
- Nettoyer et prétraiter les données médicales.
- Entraîner et comparer plusieurs algorithmes de machine learning.
- Sélectionner et sauvegarder le **meilleur modèle**.
- Exposer le modèle via une **API Flask** accessible localement.

---

## ⚙️ Technologies utilisées

| Domaine | Technologies |
|-------|-------------|
| Langage | Python 3.9+ |
| API Web | Flask |
| Machine Learning | scikit-learn |
| Analyse de données | pandas, numpy |
| Sérialisation | joblib |
| Communication | API REST (JSON) |
| Versioning | Git & GitHub |

---

## 📁 Structure du projet

```
cardio_predict_model/
├── data/                     # Jeux de données (bruts et transformés)
├── documentation/            # Description du dataset et des variables
├── heart+disease/            # Dataset de maladies cardiovasculaires
├── model/                    # Modèles et préprocesseurs sauvegardés
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   └── flask_config.py
├── notebooks/                # Notebooks (EDA, expérimentations)
├── src/                      # Scripts Python
│   ├── preprocessingdataset.py
│   ├── train_and_save_pipeline.py
│   └── utils.py
├── static/                   # Fichiers statiques Flask (optionnel)
├── app.py                    # Application Flask principale
├── requirements.txt          # Dépendances Python
└── README.md                 # Documentation
```

---

## 🚀 Installation et démarrage

### Prérequis
- **Python 3.9+**
- **Git**
- **pip**

### Étape 1 : Cloner le projet
```bash
git clone https://github.com/Abdoul-wakilou-Tiga/cardio_predict_model.git
cd cardio_predict_model
```

### Étape 2 : Environnement virtuel
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Étape 3 : Dépendances
```bash
pip install -r requirements.txt
```

### Étape 4 : Préparation des données
```bash
python src/preprocessingdataset.py
python src/train_and_save_pipeline.py
```

### Étape 5 : Lancer Flask
```bash
python app.py
```
Serveur disponible sur : **http://localhost:5000**

---

## 🔧 Utilisation de l'API

### Endpoint : `/predict`
**Méthode :** POST  
**Content-Type :** application/json

### Exemple de requête
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

### Exemple de réponse
```json
{
  "success": true,
  "prediction": 1,
  "probability_disease": 0.82,
  "probability_no_disease": 0.18,
  "risk_level": "Élevé",
  "description": "Risque cardiovasculaire élevé nécessitant une consultation médicale",
  "recommendations": [
    "Consulter un cardiologue rapidement",
    "Surveiller régulièrement votre tension artérielle",
    "Adopter une alimentation pauvre en sel"
  ]
}
```

---

## 📊 Variables du modèle

| Variable | Description | Type | Valeurs |
|----------|-------------|------|---------|
| **age** | Âge du patient | numérique | 29–77 ans |
| **sex** | Sexe biologique | catégorique | 0: femme, 1: homme |
| **cp** | Type de douleur thoracique | catégorique | 0–3 |
| **trestbps** | Pression artérielle au repos | numérique | 94–200 mmHg |
| **chol** | Cholestérol sérique | numérique | 126–564 mg/dL |
| **fbs** | Glycémie à jeun > 120 mg/dL | binaire | 0: non, 1: oui |
| **restecg** | Résultats ECG au repos | catégorique | 0–2 |
| **thalach** | Fréquence cardiaque maximale | numérique | 71–202 bpm |
| **exang** | Angine induite par l'effort | binaire | 0: non, 1: oui |
| **oldpeak** | Dépression ST induite | numérique | 0–6.2 |
| **slope** | Pente du segment ST | catégorique | 0–2 |
| **ca** | Nombre de vaisseaux colorés | catégorique | 0–3 |
| **thal** | Résultat du test thalassémie | catégorique | 1–3 |

---

## 🤖 Modèle de Machine Learning

### Algorithme : Régression Logistique
- **Accuracy sur test** : **82%**
- **Précision** : 80%
- **Rappel** : 83%
- **F1-score** : 81%
- **AUC-ROC** : 0.88

### Comparaison des algorithmes
| Algorithme | Accuracy |
|------------|----------|
| **Régression Logistique** | **82%** |
| Random Forest | 81% |
| SVM | 78% |
| KNN | 76% |
| Arbre de Décision | 74% |

### Pipeline de prétraitement
1. **Imputation** des valeurs manquantes
2. **Normalisation** des variables numériques
3. **Encodage one-hot** des variables catégoriques
4. **Sélection des features** basée sur la corrélation

---

## 🧪 Tests et validation

### Tester l'API
```bash
curl http://localhost:5000/health
```
```json
{"status": "healthy", "model_loaded": true, "model_type": "Logistic Regression", "accuracy": 0.82}
```

---

## 🔌 Intégration mobile (Flutter)
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> predictRisk(Map<String, dynamic> patientData) async {
  final response = await http.post(
    Uri.parse('http://10.0.2.2:5000/predict'), // Émulateur Android
    headers: {'Content-Type': 'application/json'},
    body: json.encode(patientData),
  );
  
  if (response.statusCode == 200) {
    return json.decode(response.body);
  } else {
    throw Exception('Failed to get prediction');
  }
}
```

---

## 📈 Analyse exploratoire (EDA)

### Notebooks disponibles
1. **EDA.ipynb** : Analyse statistique complète
   - Distribution des variables
   - Corrélations entre features
   - Visualisation des relations
   - Détection des outliers

2. **Model_Training.ipynb** : Expérimentations ML
   - Comparaison d'algorithmes
   - Optimisation des hyperparamètres
   - Évaluation des performances
   - Analyse d'importance des features

---

## 🔒 Sécurité et confidentialité

- **Aucune donnée patient** n'est stockée sur le téléphone
- **Aucune donnée** n'est persistée côté serveur
- Les données sont utilisées **uniquement** pour la prédiction en temps réel
- **Communication locale** entre mobile et backend (pas de cloud)
- L'application est une **aide à la décision médicale** et ne remplace pas un avis médical professionnel

---

---

## 📚 Ressources et références

### Dataset
- **Source** : UCI Machine Learning Repository - Heart Disease Dataset
- **Lien** : https://archive.ics.uci.edu/dataset/45/heart+disease
- **Patients** : 303
- **Variables** : 13 cliniques + 1 cible

### Documentation
- **Flask** : https://flask.palletsprojects.com/
- **scikit-learn** : https://scikit-learn.org/
- **Logistic Regression** : Hosmer & Lemeshow (2000)

---

## ⚠️ Avertissement médical

**Ce projet est à but de recherche et d'éducation.**  
Les prédictions générées par ce modèle (82% d'accuracy) ne constituent **PAS** un diagnostic médical.  
Toute décision concernant la santé doit être prise en consultation avec un professionnel de santé qualifié.

---

## 👨‍💻 Auteur

**Abdoul-wakilou Tiga**  
Étudiant en Master 2 Génie Logiciel 
Université Université d'Abomey-Calavi
Année académique 2024-2025

### Contact
- 📧 Email : [abdoulwakiloutiga@gmail.com](abdoulwakiloutiga@gmail.com)
- 🔗 LinkedIn : [Abdoul-wakilou-Tiga](https://linkedin.com/in/abdoul-wakilou-tiga)
- 🐙 GitHub : [Abdoul-wakilou-Tiga](https://github.com/Abdoul-wakilou-Tiga)

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**.

---

*Dernière mise à jour : Décembre 2025*  
*Performance du modèle : 82% d'accuracy (Régression Logistique)*