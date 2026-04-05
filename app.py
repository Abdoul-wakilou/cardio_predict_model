"""
app.py - API Flask pour CardioPredict
- Endpoint /predict (POST) pour la prédiction du risque cardiovasculaire
- Génération de recommandations personnalisées
- Endpoint /health pour vérifier l'état du service
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

# Importer les fonctions utilitaires
from src.utils import generate_recommendations, get_risk_level

app = Flask(__name__)

# ============================================================================
# CHARGEMENT DU MODÈLE ET DU PRÉPROCESSEUR
# ============================================================================

MODEL_PATH = 'model/best_model_full.joblib'
PREPROCESSOR_PATH = 'model/preprocessor_full.joblib'

print("="*60)
print("🚀 Démarrage de l'API CardioPredict")
print("="*60)

# Vérification des fichiers
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(f"Préprocesseur introuvable : {PREPROCESSOR_PATH}")

# Chargement
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print(f"✅ Modèle chargé : {type(model).__name__}")
print(f"✅ Préprocesseur chargé")

# ============================================================================
# ENDPOINT /predict
# ============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction.
    Attend un JSON avec les 13 variables cliniques.
    Retourne la prédiction, la probabilité, le niveau de risque et les recommandations.
    """
    try:
        # Récupérer les données
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Aucune donnée fournie'}), 400
        
        # Liste des features attendues (ordre exact)
        expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                             'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        # Vérifier la présence de toutes les features
        missing = [f for f in expected_features if f not in data]
        if missing:
            return jsonify({'error': f'Features manquantes : {missing}'}), 400
        
        # Construire le DataFrame
        input_df = pd.DataFrame([{k: data[k] for k in expected_features}])
        
        # Appliquer le préprocesseur
        input_processed = preprocessor.transform(input_df)
        
        # Prédiction
        prediction = int(model.predict(input_processed)[0])
        probability = float(model.predict_proba(input_processed)[0][1])
        
        # Déterminer le niveau de risque
        risk_level, risk_label, risk_description = get_risk_level(prediction, probability)
        
        # Générer les recommandations personnalisées
        recommendations = generate_recommendations(data, prediction, probability)
        
        # Réponse
        response = {
            'success': True,
            'prediction': prediction,
            'prediction_label': 'Malade' if prediction == 1 else 'Sain',
            'probability_disease': round(probability, 4),
            'probability_no_disease': round(1 - probability, 4),
            'risk_level': risk_level,
            'risk_label': risk_label,
            'risk_description': risk_description,
            'recommendations': recommendations
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ENDPOINT /health
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de vérification de l'état du service."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_type': type(model).__name__
    }), 200


# ============================================================================
# ENDPOINT racine
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """Page d'accueil simple."""
    return jsonify({
        'service': 'CardioPredict API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Prédiction du risque cardiovasculaire',
            '/health': 'GET - Vérification de l\'état du service'
        }
    }), 200


# ============================================================================
# LANCEMENT
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Serveur Flask démarré")
    print("="*60)
    print("📍 Endpoints disponibles :")
    print("   - POST /predict")
    print("   - GET  /health")
    print("   - GET  /")
    print("\n🔧 Pour tester :")
    print("   curl -X POST http://localhost:5000/predict \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"age\":52,\"sex\":1,\"cp\":0,\"trestbps\":125,\"chol\":212,\"fbs\":0,\"restecg\":1,\"thalach\":168,\"exang\":0,\"oldpeak\":1.0,\"slope\":2,\"ca\":2,\"thal\":3}'")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)