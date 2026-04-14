"""
app.py - API Flask CardioPredict avec CORS activé
"""

from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- IMPORTANT : AJOUTER CETTE LIGNE
import joblib
import pandas as pd

from src.utils import generate_recommendations, get_risk_level

app = Flask(__name__)

# ============================================================
# ACTIVER CORS - AJOUTER CETTE LIGNE (TRÈS IMPORTANT)
# ============================================================
CORS(app)  # <-- C'EST CETTE LIGNE QUI MANQUE !

# Alternative plus spécifique si besoin :
# CORS(app, resources={r"/*": {"origins": "*"}})

model = joblib.load('model/best_model_full.joblib')
preprocessor = joblib.load('model/preprocessor_full.joblib')

EXPECTED_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

print("API CardioPredict démarrée — POST /predict | GET /health")


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Gérer la requête preflight CORS (optionnel avec CORS(app))
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Aucune donnée fournie'}), 400
        
        data = normalize_input(data)

        missing = [f for f in EXPECTED_FEATURES if f not in data]
        if missing:
            return jsonify({'error': f'Features manquantes : {missing}'}), 400

        input_df = pd.DataFrame([{k: data[k] for k in EXPECTED_FEATURES}])
        input_processed = preprocessor.transform(input_df)

        prediction = int(model.predict(input_processed)[0])
        probability = float(model.predict_proba(input_processed)[0][1])

        risk_level, risk_label, risk_description = get_risk_level(prediction, probability)
        recommendations = generate_recommendations(data, prediction, probability)

        response = jsonify({
            'success': True,
            'prediction': prediction,
            'prediction_label': 'Malade' if prediction == 1 else 'Sain',
            'probability_disease': round(probability, 4),
            'probability_no_disease': round(1 - probability, 4),
            'risk_level': risk_level,
            'risk_label': risk_label,
            'risk_description': risk_description,
            'recommendations': recommendations
        })
        
        # Ajouter les en-têtes CORS (CORS(app) le fait automatiquement, mais par sécurité)
        response.headers.add('Access-Control-Allow-Origin', '*')
        
        return response, 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _build_cors_preflight_response():
    """Construit la réponse pour la requête preflight OPTIONS"""
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response, 200


def normalize_input(data):
    """Corrige les valeurs des variables catégorielles selon la norme UCI"""
    if data.get('cp') == 0:
        data['cp'] = 4
    if data.get('slope') == 0:
        data['slope'] = 3
    if data.get('thal') == 1:
        data['thal'] = 3
    elif data.get('thal') == 2:
        data['thal'] = 6
    return data


@app.route('/health', methods=['GET'])
def health():
    response = jsonify({'status': 'healthy'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 200


@app.route('/', methods=['GET'])
def index():
    response = jsonify({'service': 'CardioPredict API', 'version': '2.0.0'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)