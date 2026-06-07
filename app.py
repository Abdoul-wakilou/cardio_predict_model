"""
app.py — API Flask CardioPredict v3.0
======================================
Dataset source : UCI Heart Disease combiné (labels corrects, 0=sain, 1=malade)

Encodage attendu côté Flutter (convention 0-based) :
  cp    : 0=typique, 1=atypique, 2=non-angineuse, 3=asymptomatique
  slope : 0=ascendante, 1=plate, 2=descendante
  thal  : 0=normal, 1=défaut fixe, 2=défaut réversible

NOTE : La fonction normalize_input() de la v2 est SUPPRIMÉE.
Le modèle est désormais entraîné avec l'encodage 0-based — aucune
conversion n'est nécessaire côté API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

from src.utils import generate_recommendations, get_risk_level

app = Flask(__name__)
CORS(app)

# Chargement au démarrage (une seule fois)
model        = joblib.load('model/best_model_full.joblib')
preprocessor = joblib.load('model/preprocessor_full.joblib')

# Les 13 features dans l'ordre exact attendu par le préprocesseur
EXPECTED_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Plages de validation pour chaque feature
FEATURE_RANGES = {
    'age':      (1,   120),
    'sex':      (0,   1),
    'cp':       (0,   3),
    'trestbps': (50,  250),
    'chol':     (50,  700),
    'fbs':      (0,   1),
    'restecg':  (0,   2),
    'thalach':  (50,  250),
    'exang':    (0,   1),
    'oldpeak':  (-5,  10),
    'slope':    (0,   2),
    'ca':       (0,   3),
    'thal':     (0,   2),
}

print("=" * 50)
print("API CardioPredict v3.0 démarrée")
print("POST /predict  |  GET /health  |  GET /")
print(f"Modèle : {type(model).__name__}")
print("=" * 50)


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Aucune donnée JSON fournie'}), 400

        # Vérifier les features manquantes
        missing = [f for f in EXPECTED_FEATURES if f not in data]
        if missing:
            return jsonify({'error': f'Features manquantes : {missing}'}), 400

        # Valider les plages
        errors = []
        for feat, (lo, hi) in FEATURE_RANGES.items():
            val = data.get(feat)
            if val is not None and not (lo <= float(val) <= hi):
                errors.append(f'{feat}={val} hors plage [{lo}, {hi}]')
        if errors:
            return jsonify({'error': f'Valeurs hors plage : {errors}'}), 400

        # Construction du DataFrame d'entrée
        input_df = pd.DataFrame([{k: data[k] for k in EXPECTED_FEATURES}])

        # Prétraitement + prédiction
        input_processed = preprocessor.transform(input_df)
        prediction      = int(model.predict(input_processed)[0])
        probability     = float(model.predict_proba(input_processed)[0][1])

        # Niveau de risque et recommandations
        risk_level, risk_label, risk_description = get_risk_level(prediction, probability)
        recommendations = generate_recommendations(data, prediction, probability)

        response = jsonify({
            'success':             True,
            'prediction':          prediction,
            'prediction_label':    'Malade' if prediction == 1 else 'Sain',
            'probability_disease': round(probability, 4),
            'probability_no_disease': round(1 - probability, 4),
            'risk_level':          risk_level,
            'risk_label':          risk_label,
            'risk_description':    risk_description,
            'recommendations':     recommendations,
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _build_cors_preflight_response():
    resp = jsonify({'status': 'ok'})
    resp.headers.add('Access-Control-Allow-Origin',  '*')
    resp.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    resp.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return resp, 200


@app.route('/health', methods=['GET'])
def health():
    resp = jsonify({'status': 'healthy', 'model': type(model).__name__})
    resp.headers.add('Access-Control-Allow-Origin', '*')
    return resp, 200


@app.route('/', methods=['GET'])
def index():
    resp = jsonify({
        'service': 'CardioPredict API',
        'version': '3.0.0',
        'dataset': 'UCI Heart Disease — 4 centers (834 patients)',
        'encoding': {
            'cp':    '0=typique, 1=atypique, 2=non-angineuse, 3=asymptomatique',
            'slope': '0=ascendante, 1=plate, 2=descendante',
            'thal':  '0=normal, 1=défaut fixe, 2=défaut réversible',
            'target': '0=sain, 1=malade',
        }
    })
    resp.headers.add('Access-Control-Allow-Origin', '*')
    return resp, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)