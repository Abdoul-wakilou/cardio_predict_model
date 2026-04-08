"""
app.py - API Flask pour CardioPredict
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

from src.utils import generate_recommendations, get_risk_level

app = Flask(__name__)

# Configuration
MODEL_PATH = 'model/best_model_full.joblib'
PREPROCESSOR_PATH = 'model/preprocessor_full.joblib'

# Chargement silencieux
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("✅ API CardioPredict démarrée")
print("📍 POST /predict | GET /health | GET /")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Aucune donnée fournie'}), 400
        
        expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                             'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        missing = [f for f in expected_features if f not in data]
        if missing:
            return jsonify({'error': f'Features manquantes : {missing}'}), 400
        
        input_df = pd.DataFrame([{k: data[k] for k in expected_features}])
        input_processed = preprocessor.transform(input_df)
        
        prediction = int(model.predict(input_processed)[0])
        probability = float(model.predict_proba(input_processed)[0][1])
        
        risk_level, risk_label, risk_description = get_risk_level(prediction, probability)
        recommendations = generate_recommendations(data, prediction, probability)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'prediction_label': 'Malade' if prediction == 1 else 'Sain',
            'probability_disease': round(probability, 4),
            'probability_no_disease': round(1 - probability, 4),
            'risk_level': risk_level,
            'risk_label': risk_label,
            'risk_description': risk_description,
            'recommendations': recommendations
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


@app.route('/', methods=['GET'])
def index():
    return jsonify({'service': 'CardioPredict API', 'version': '1.0.0'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)